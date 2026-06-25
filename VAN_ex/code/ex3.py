"""Exercise 3: PnP-RANSAC relative motion, supporter filtering, full-sequence tracking."""

import os
import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from dataset import read_images, read_cameras, read_poses
from features import (
    extract_features,
    match_descriptors,
    rectified_stereo_filter,
    consensus_matches,
    pts_from_matches,
)
from geometry import (
    triangulate_linear_lsq,
    camera_center,
    compose_extrinsics,
)
from pnp import solve_pnp, supporters_mask, ransac_pnp
from plot import plot_3d_world, plot_trajectory_xz

DETECTOR = 'AKAZE'
Y_THRESHOLD = 2.0       # px — rectified-stereo vertical-deviation cutoff (ex2)
X_MIN_DISPARITY = 0.0   # px — require positive disparity (rejects x_l ≤ x_r)
PIX_THRESHOLD = 2.0     # px — per-image supporter reprojection cutoff (ex3)
N_FRAMES_FULL = None    # None → use every frame found on disk; integer → cap (debug)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'ex3_figures')


def _save(fig, name):
    """Write a figure to FIGURES_DIR/<name>.png at presentation resolution."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name + '.png'), dpi=150, bbox_inches='tight')


IDENTITY_RT = np.hstack([np.eye(3), np.zeros((3, 1))])


def _stereo_features(img_l, img_r, detector=DETECTOR):
    """Detect features on a stereo pair, best-match, and apply the rectified-stereo filter.

    Returns:
        kp_l, des_l: keypoints + descriptors of the left image (needed for the
            cross-frame match in the next iteration).
        kp_r: keypoints of the right image (right descriptors are consumed
            inside this function and not returned).
        stereo_in: list of inlier DMatch objects (queryIdx ↔ kp_l, trainIdx ↔ kp_r).
    """
    kp_l, des_l = extract_features(img_l, detector=detector)
    kp_r, des_r = extract_features(img_r, detector=detector)
    matches = match_descriptors(des_l, des_r, detector=detector)
    stereo_in, _, _ = rectified_stereo_filter(
        kp_l, kp_r, matches,
        y_threshold=Y_THRESHOLD, x_min_disparity=X_MIN_DISPARITY)
    return kp_l, des_l, kp_r, stereo_in


def _build_consensus(kp_l0, kp_r0, kp_l1, kp_r1, stereo0_in, stereo1_in, cross,
                     P_left, P_right):
    """Bundle the 4-view pixel correspondences and pair-0 triangulation.

    Disparity validity is already enforced by `rectified_stereo_filter`
    (Y-alignment AND minimum X-disparity) on each pair's stereo_in set, so a
    match that reaches this function is already geometrically clean on both
    sides.
    """
    idx0, idx1 = consensus_matches(stereo0_in, stereo1_in, cross)
    s0_sel = [stereo0_in[i] for i in idx0]
    s1_sel = [stereo1_in[i] for i in idx1]
    pts_l0, pts_r0 = pts_from_matches(kp_l0, kp_r0, s0_sel)
    pts_l1, pts_r1 = pts_from_matches(kp_l1, kp_r1, s1_sel)
    X0 = triangulate_linear_lsq(P_left, P_right, pts_l0, pts_r0)
    return dict(X0=X0, pts_l0=pts_l0, pts_r0=pts_r0,
                pts_l1=pts_l1, pts_r1=pts_r1, idx0=idx0, idx1=idx1)


def q1(K, P_left, P_right, kp_l1, kp_r1, stereo1_in, img_shape):
    """Q3.1 — Triangulate the next stereo pair (frame 1) and visualise the cloud."""
    pts_l, pts_r = pts_from_matches(kp_l1, kp_r1, stereo1_in)
    X1 = triangulate_linear_lsq(P_left, P_right, pts_l, pts_r)
    print(f"Frame 1: {len(stereo1_in)} stereo inliers → {len(X1)} 3D points")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_world(X1, ax=ax, K=K, img_shape=img_shape,
                  title='Q3.1 — Point cloud of stereo pair 1')
    fig.tight_layout()
    _save(fig, 'q3_1_pair1_pointcloud')


def q2(img_l0, img_l1, kp_l0, kp_l1, cross, n_display=30):
    """Q3.2 — Match features between left0 and left1; show a random subset as connecting lines."""
    print(f"Cross-frame matches (left0 ↔ left1): {len(cross)}")
    if not cross:
        return

    sample = random.sample(cross, min(n_display, len(cross)))
    cmap = plt.get_cmap('tab20')

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'hspace': 0.18})
    for ax, img, side in [(ax0, img_l0, 'left0'), (ax1, img_l1, 'left1')]:
        ax.imshow(img, cmap='gray')
        ax.set_title(side)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_aspect('equal')

    for i, m in enumerate(sample):
        pt0 = kp_l0[m.queryIdx].pt
        pt1 = kp_l1[m.trainIdx].pt
        c = cmap(i % 20)
        ax0.plot(*pt0, 'o', color=c, markersize=4)
        ax1.plot(*pt1, 'o', color=c, markersize=4)
        fig.add_artist(ConnectionPatch(
            xyA=pt0, coordsA=ax0.transData,
            xyB=pt1, coordsB=ax1.transData,
            color=c, linewidth=0.8, alpha=0.85))

    fig.suptitle(f'Q3.2 — Cross-frame matches  ({len(sample)} shown of {len(cross)})')
    _save(fig, 'q3_2_cross_matches')


def _plot_four_cameras_topdown(centers_4, labels, title, save_name):
    """Top-down (X vs Z) scatter of camera centres with labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'steelblue', 'orange', 'orange']
    markers = ['o', 's', 'o', 's']
    for c, lbl, col, mk in zip(centers_4, labels, colors, markers):
        ax.scatter(c[0], c[2], c=col, marker=mk, s=90, edgecolor='black',
                   linewidth=0.8, zorder=5)
        ax.annotate(lbl, (c[0], c[2]), textcoords='offset points', xytext=(8, 6),
                    fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z — forward (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title(title)
    _save(fig, save_name)


def q3(K, m_right, consensus, rng):
    """Q3.3 — PnP from 4 consensus matches; derive T from left0 to left1; plot 4 cameras."""
    n = len(consensus['X0'])
    if n < 4:
        print(f"Q3.3 skipped — only {n} consensus matches available.")
        return None

    sel = rng.choice(n, 4, replace=False)
    X_sample = consensus['X0'][sel]
    pix_sample = consensus['pts_l1'][sel]
    Rt_left1 = solve_pnp(X_sample, pix_sample, K, flags=cv2.SOLVEPNP_AP3P)
    if Rt_left1 is None:
        print("Q3.3 — PnP failed on the chosen 4 points; pick another seed.")
        return None

    R, t = Rt_left1[:, :3], Rt_left1[:, 3]
    print("Q3.3 — initial PnP on 4 consensus matches:")
    print(f"  R = \n{R}")
    print(f"  t = {t}")
    print(f"  ||t|| = {np.linalg.norm(t):.3f} m   (expected ≈ inter-frame baseline)")

    centers = np.array([
        camera_center(IDENTITY_RT),                            # left0
        camera_center(m_right),                                # right0
        camera_center(Rt_left1),                               # left1
        camera_center(compose_extrinsics(Rt_left1, m_right)),  # right1
    ])
    print(f"  Camera centres (in left0 coords):")
    for name, c in zip(['left0', 'right0', 'left1', 'right1'], centers):
        print(f"    {name}: ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})")

    _plot_four_cameras_topdown(centers, ['left0', 'right0', 'left1', 'right1'],
                               'Q3.3 — Four-camera positions (top-down)',
                               'q3_3_four_cameras')
    return Rt_left1


def _draw_two_class_on_images(img0, img1, pts0, pts1, mask, label_pos, label_neg,
                              title, save_name):
    """Show pts0 on img0 and pts1 on img1 with a two-class colouring (positive in orange)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, img, pts, side in [(axes[0], img0, pts0, 'left0'),
                               (axes[1], img1, pts1, 'left1')]:
        ax.imshow(img, cmap='gray')
        if (~mask).any():
            ax.scatter(pts[~mask, 0], pts[~mask, 1], c='cyan', s=10,
                       label=f'{label_neg} ({int((~mask).sum())})')
        if mask.any():
            ax.scatter(pts[mask, 0], pts[mask, 1], c='orange', s=10,
                       label=f'{label_pos} ({int(mask.sum())})')
        ax.set_title(side)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_name)


def q4(K, m_right, consensus, Rt_left1, img_l0, img_l1):
    """Q3.4 — Find supporters of the initial T via 4-view reprojection (≤ 2 px each)."""
    mask = supporters_mask(consensus['X0'],
                           consensus['pts_l0'], consensus['pts_r0'],
                           consensus['pts_l1'], consensus['pts_r1'],
                           Rt_left1, K, m_right, threshold=PIX_THRESHOLD)
    n_sup = int(mask.sum())
    print(f"Q3.4 — Supporters of initial T: {n_sup} / {len(mask)} "
          f"({100.0 * n_sup / max(1, len(mask)):.1f}%)")

    _draw_two_class_on_images(
        img_l0, img_l1,
        consensus['pts_l0'], consensus['pts_l1'],
        mask, label_pos='supporter', label_neg='non-supporter',
        title='Q3.4 — Supporters (orange) vs non-supporters (cyan) of the initial T',
        save_name='q3_4_supporters')
    return mask


def q5(K, P_left, P_right, m_right, consensus, img_l0, img_l1, img_shape, rng):
    """Q3.5 — RANSAC-PnP; transform pair-0 cloud by T; plot clouds and inliers/outliers."""
    Rt_left1, mask = ransac_pnp(
        consensus['X0'],
        consensus['pts_l0'], consensus['pts_r0'],
        consensus['pts_l1'], consensus['pts_r1'],
        K, m_right, threshold=PIX_THRESHOLD, rng=rng,
    )
    n_in = int(mask.sum())
    n = len(mask)
    print(f"Q3.5 — RANSAC-PnP: {n_in} / {n} inliers "
          f"({100.0 * n_in / max(1, n):.1f}%)")
    assert Rt_left1 is not None, "Q3.5 — RANSAC produced no valid hypothesis"
    R, t = Rt_left1[:, :3], Rt_left1[:, 3]
    print(f"  Refined translation in left0 coords: ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})  "
          f"||t|| = {np.linalg.norm(t):.3f} m")

    # Transform pair-0 cloud by T into pair-1 (left1) coordinates.
    X0 = consensus['X0']
    X0_in_left1 = (R @ X0.T).T + t

    # Triangulate pair 1's own cloud (in left1 coordinates) for the overlay.
    pts_l1, pts_r1 = consensus['pts_l1'], consensus['pts_r1']
    X1 = triangulate_linear_lsq(P_left, P_right, pts_l1, pts_r1)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Use the project's 3D-axes conventions but plot two clouds with different colours.
    ax.computed_zorder = False

    def _scatter(X, color, label):
        Xp = np.column_stack([X[:, 0], X[:, 2], X[:, 1]])
        ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2],  # type: ignore[arg-type]
                   s=6, c=color, alpha=0.7, label=label)

    _scatter(X1, 'steelblue', f'pair 1 cloud (n={len(X1)})')
    _scatter(X0_in_left1, 'orange', f'pair 0 cloud after T (n={len(X0_in_left1)})')

    # Camera frustum at left1 origin.
    from plot import draw_frustum, PLOT_LIMS
    draw_frustum(ax, K, img_shape)
    lims = PLOT_LIMS
    ax.set_xlim(*lims['x']); ax.set_ylim(*lims['y']); ax.set_zlim(*lims['z'])
    rx, ry, rz = (lims[k][1] - lims[k][0] for k in 'xyz')
    ax.set_box_aspect((rx, ry, rz))
    ax.invert_zaxis()
    ax.set_xlabel('X — right (m)')
    ax.set_ylabel('Z — forward (m)')
    ax.set_zlabel('Y (m, +down)')
    ax.view_init(elev=15, azim=-75)
    ax.set_title('Q3.5 — Pair 1 (blue) and pair 0 after T (orange) in left1 coords')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    _save(fig, 'q3_5_two_pointclouds')

    _draw_two_class_on_images(
        img_l0, img_l1,
        consensus['pts_l0'], consensus['pts_l1'],
        mask, label_pos='inlier', label_neg='outlier',
        title='Q3.5 — RANSAC inliers (orange) vs outliers (cyan)',
        save_name='q3_5_inliers_outliers')

    return Rt_left1, mask


def track_sequence(n_frames, K, P_left, P_right, m_right, verbose_every=100):
    """Track frames 0..n_frames−1 with consecutive RANSAC-PnP; return Nx3x4 extrinsics in left0.

    Caches the previous frame's features so each image is processed once.
    """
    Rt_global = [IDENTITY_RT.copy()]
    rng = np.random.default_rng(0)
    start = time.time()

    img_l, img_r = read_images(0)
    kp_l, des_l, kp_r, s_in = _stereo_features(img_l, img_r)

    for i in range(1, n_frames):
        img_l_new, img_r_new = read_images(i)
        kp_l_new, des_l_new, kp_r_new, s_new_in = _stereo_features(
            img_l_new, img_r_new)
        cross = match_descriptors(des_l, des_l_new, DETECTOR)
        consensus = _build_consensus(kp_l, kp_r, kp_l_new, kp_r_new,
                                     s_in, s_new_in, cross, P_left, P_right)

        if len(consensus['X0']) >= 4:
            Rt_rel, _ = ransac_pnp(
                consensus['X0'],
                consensus['pts_l0'], consensus['pts_r0'],
                consensus['pts_l1'], consensus['pts_r1'],
                K, m_right, threshold=PIX_THRESHOLD, rng=rng,
            )
        else:
            Rt_rel = None
        if Rt_rel is None:
            Rt_rel = IDENTITY_RT.copy()
            print(f"  Frame {i}: PnP unrecoverable ({len(consensus['X0'])} consensus matches) "
                  f"— falling back to identity.")
        Rt_global.append(compose_extrinsics(Rt_global[-1], Rt_rel))

        # Roll the cache forward.
        kp_l, des_l, kp_r, s_in = kp_l_new, des_l_new, kp_r_new, s_new_in

        if verbose_every and i % verbose_every == 0:
            elapsed = time.time() - start
            print(f"  Frame {i:>4d}/{n_frames}: elapsed = {elapsed:6.1f}s "
                  f"({elapsed / i:.2f}s/frame)")

    total = time.time() - start
    print(f"Tracking finished: {n_frames} frames in {total:.1f}s "
          f"({total / max(1, n_frames - 1):.2f}s per relative pose).")
    return np.array(Rt_global), total


def q6(K, P_left, P_right, m_right):
    """Q3.6 — Track the whole movie, plot estimated and ground-truth trajectories."""
    # Auto-detect frame count if the user did not cap N_FRAMES_FULL.
    from dataset import DATA_PATH
    img_dir = os.path.join(DATA_PATH, 'image_0')
    n_avail = sum(1 for f in os.listdir(img_dir) if f.endswith('.png'))
    n_frames = N_FRAMES_FULL if N_FRAMES_FULL is not None else n_avail
    print(f"Q3.6 — Tracking {n_frames} frames (sequence 00 has {n_avail} on disk).")

    Rt_seq, elapsed = track_sequence(n_frames, K, P_left, P_right, m_right)
    est_positions = np.array([camera_center(Rt) for Rt in Rt_seq])

    poses = read_poses()[:n_frames]
    gt_positions = np.array([camera_center(Rt) for Rt in poses])

    err = np.linalg.norm(est_positions - gt_positions, axis=1)
    print(f"  Tracking time: {elapsed:.1f}s")
    print(f"  Final-frame position error: {err[-1]:.2f} m")
    print(f"  Median / max trajectory error: {np.median(err):.2f} m / {err.max():.2f} m")

    fig, ax = plt.subplots(figsize=(11, 11))
    plot_trajectory_xz(est_positions, gt=gt_positions, ax=ax,
                       title=(f'Q3.6 — Left-camera trajectory in left0 coords  '
                              f'({n_frames} frames, tracking took {elapsed:.0f}s)'))
    _save(fig, 'q3_6_trajectory')


def main():
    K, m1, m2 = read_cameras()
    P_left, P_right = K @ m1, K @ m2

    img_l0, img_r0 = read_images(0)
    img_l1, img_r1 = read_images(1)
    img_shape = img_l0.shape

    # Feature extraction and stereo matching for both pairs (shared by Q3.1–Q3.5).
    kp_l0, des_l0, kp_r0, s0_in = _stereo_features(img_l0, img_r0)
    kp_l1, des_l1, kp_r1, s1_in = _stereo_features(img_l1, img_r1)

    print("Q3.1")
    q1(K, P_left, P_right, kp_l1, kp_r1, s1_in, img_shape)

    print("\nQ3.2")
    cross = match_descriptors(des_l0, des_l1, DETECTOR)
    q2(img_l0, img_l1, kp_l0, kp_l1, cross)

    consensus = _build_consensus(kp_l0, kp_r0, kp_l1, kp_r1,
                                 s0_in, s1_in, cross, P_left, P_right)
    print(f"Consensus 4-view matches: {len(consensus['X0'])}")

    rng = np.random.default_rng(0)

    print("\nQ3.3")
    Rt_init = q3(K, m2, consensus, rng)

    if Rt_init is not None:
        print("\nQ3.4")
        q4(K, m2, consensus, Rt_init, img_l0, img_l1)

        print("\nQ3.5")
        q5(K, P_left, P_right, m2, consensus, img_l0, img_l1, img_shape, rng)

    print("\nQ3.6")
    q6(K, P_left, P_right, m2)

    plt.show()


if __name__ == '__main__':
    main()
