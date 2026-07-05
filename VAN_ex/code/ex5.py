"""Exercise 5: Bundle adjustment of small windows along the KITTI trajectory.

q1 — single-track sanity check (StereoCamera backproject/project + factor error)
q3 — first bundle window (factor graph, LM, error breakdown, plots)
q4 — all bundle windows + chaining to global frame-0 coordinates
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot

from dataset import read_cameras, read_poses, read_images, DATA_PATH
from geometry import camera_center, compose_extrinsics
from tracking_database import TrackingDB
from bundle import (
    Rt_to_gtsam_pose, gtsam_pose_to_Rt,
    stereo_calibration, stereo_camera, link_to_stereo_point,
    select_keyframes_in_calm_frames, cam_key, lm_key,
    build_bundle_window, optimize_bundle, compose_global_poses,
    compute_feature_sizes,
    STEREO_NOISE,
)


DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
FIGURES_DIR = os.path.join(DOCS_DIR, 'ex5_figures')

TRACKING_DB_BASE = os.path.join(DATA_PATH, 'tracking_db')
PNP_POSES_PATH = os.path.join(DATA_PATH, 'pnp_poses.npy')
FEATURE_SIZES_PATH = os.path.join(DATA_PATH, 'feature_sizes.pkl')

KF_MIN_TRANSLATION = 5.0    # metres — keyframe trigger


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name + '.png'), dpi=150, bbox_inches='tight')


# ex5
def load_or_compute_pnp_poses(n_frames, K, P_left, P_right, m_right):
    """Return Nx3x4 PnP world-to-camera extrinsics, recomputing only if missing."""
    if os.path.exists(PNP_POSES_PATH):
        poses = np.load(PNP_POSES_PATH)
        if poses.shape[0] >= n_frames:
            print(f'Loaded {poses.shape[0]} PnP poses from cache.')
            return poses[:n_frames]
    print(f'PnP poses cache missing — running ex3.track_sequence on {n_frames} frames…')
    from ex3 import track_sequence
    Rt_seq, _ = track_sequence(n_frames, K, P_left, P_right, m_right)
    np.save(PNP_POSES_PATH, Rt_seq)
    print(f'Saved PnP poses to {PNP_POSES_PATH}')
    return Rt_seq


# ex5
def q1(db, pnp_poses, K_stereo, rng=None):
    """Q5.1 — Single-track sanity check.

    Pick a length-≥10 track, build a StereoCamera per frame using the PnP
    GLOBAL poses, triangulate from the last frame, project to all frames, plot
    reprojection-error and factor-error curves.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    candidates = [tid for tid in db.all_tracks() if len(db.frames(tid)) >= 10]
    tid = int(rng.choice(candidates))
    frames = db.frames(tid)
    print(f'  Q5.1 track #{tid}: length {len(frames)}, frames '
          f'{frames[0]}..{frames[-1]}')

    # Build StereoCameras in GLOBAL (frame-0 world) coords.
    stereo_cams = {fid: stereo_camera(Rt_to_gtsam_pose(pnp_poses[fid]), K_stereo)
                   for fid in frames}

    # Triangulate from LAST frame (per spec).
    last_fid = frames[-1]
    last_link = db.link(last_fid, tid)
    X_world = stereo_cams[last_fid].backproject(link_to_stereo_point(last_link))
    print(f'  Triangulated 3D point (global): ({X_world[0]:.2f}, '
          f'{X_world[1]:.2f}, {X_world[2]:.2f})')

    # Reproject onto every frame; record L2 error AND factor error.
    reproj_err = []
    factor_err = []
    for fid in frames:
        link = db.link(fid, tid)
        z = link_to_stereo_point(link)
        try:
            proj = stereo_cams[fid].project(gtsam.Point3(X_world))
        except RuntimeError:
            reproj_err.append(np.nan); factor_err.append(np.nan); continue

        # Reprojection error: L2 distance of (left, right) pixel pairs.
        d_left = np.hypot(proj.uL() - link.x_left, proj.v() - link.y)
        d_right = np.hypot(proj.uR() - link.x_right, proj.v() - link.y)
        reproj_err.append(0.5 * (d_left + d_right))

        # Factor error: build a GenericStereoFactor3D and ask GTSAM.
        factor = gtsam.GenericStereoFactor3D(
            z, STEREO_NOISE, cam_key(fid), lm_key(tid), K_stereo)
        vals = gtsam.Values()
        vals.insert(cam_key(fid), stereo_cams[fid].pose())
        vals.insert(lm_key(tid), gtsam.Point3(X_world))
        factor_err.append(factor.error(vals))

    reproj_err = np.array(reproj_err)
    factor_err = np.array(factor_err)
    print(f'  Reprojection error: ref={reproj_err[-1]:.3f}px, '
          f'far={reproj_err[0]:.3f}px (L2 mean of left/right)')
    print(f'  Factor error:       ref={factor_err[-1]:.3f}, '
          f'far={factor_err[0]:.3f}')

    dist = np.arange(len(frames))[::-1]  # distance from reference (LAST) frame
    order = np.argsort(dist)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(dist[order], reproj_err[order], color='steelblue')
    axes[0].set_xlabel('distance from reference (frames)')
    axes[0].set_ylabel('reprojection error (pixels, L2)')
    axes[0].set_title(f'Q5.1 — reprojection error  (track #{tid})')
    axes[0].grid(alpha=0.3)
    axes[1].plot(dist[order], factor_err[order], color='orange')
    axes[1].set_xlabel('distance from reference (frames)')
    axes[1].set_ylabel('factor error')
    axes[1].set_title(f'Q5.1 — factor error  (track #{tid})')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'q5_1_track_errors')


# ex5
def _largest_initial_error_factor(graph, initial, projection_factors):
    """Return (idx, fid, tid, link, initial_error) for the projection factor
    with the largest error at the initial values."""
    best = (None, None, None, None, -1.0)
    for idx, fid, tid, link in projection_factors:
        err = graph.at(idx).error(initial)
        if err > best[4]:
            best = (idx, fid, tid, link, err)
    return best


def _draw_projection_vs_measurement(fid, link, proj_stereo, save_name, title):
    """Show left+right image cutouts with measurement (cyan) and projection (orange)."""
    img_l, img_r = read_images(fid)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    def crop_around(img, xs, ys, pad=80):
        h, w = img.shape[:2]
        x0 = max(0, int(min(xs)) - pad); y0 = max(0, int(min(ys)) - pad)
        x1 = min(w, int(max(xs)) + pad); y1 = min(h, int(max(ys)) + pad)
        return img[y0:y1, x0:x1], x0, y0

    pts_left = ((link.x_left, proj_stereo.uL()),
                (link.y,      proj_stereo.v()))
    pts_right = ((link.x_right, proj_stereo.uR()),
                 (link.y,        proj_stereo.v()))

    cl, x0l, y0l = crop_around(img_l, pts_left[0], pts_left[1])
    axes[0].imshow(cl, cmap='gray')
    axes[0].scatter(link.x_left - x0l, link.y - y0l,
                    c='cyan', s=80, marker='o', label='measurement', edgecolor='black')
    axes[0].scatter(proj_stereo.uL() - x0l, proj_stereo.v() - y0l,
                    c='orange', s=80, marker='x', label='projection')
    axes[0].set_title(f'frame {fid} — left'); axes[0].axis('off'); axes[0].legend()

    cr, x0r, y0r = crop_around(img_r, pts_right[0], pts_right[1])
    axes[1].imshow(cr, cmap='gray')
    axes[1].scatter(link.x_right - x0r, link.y - y0r,
                    c='cyan', s=80, marker='o', label='measurement', edgecolor='black')
    axes[1].scatter(proj_stereo.uR() - x0r, proj_stereo.v() - y0r,
                    c='orange', s=80, marker='x', label='projection')
    axes[1].set_title(f'frame {fid} — right'); axes[1].axis('off'); axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_name)


# ex5
def q3(db, pnp_poses, K_stereo, keyframes, feature_sizes=None):
    """Q5.3 — First bundle window: keyframes[0] → keyframes[1] inclusive."""
    kf0, kf1 = keyframes[0], keyframes[1]
    frames = list(range(kf0, kf1 + 1))
    print(f'  Q5.3 first bundle: frames {kf0}..{kf1} ({len(frames)} frames)')

    graph, initial, info = build_bundle_window(
        db, pnp_poses, frames, K_stereo, feature_sizes=feature_sizes)
    n_factors = graph.size()
    init_err = graph.error(initial)
    avg_init = init_err / max(1, n_factors)
    print(f'  Factors: {n_factors}')
    print(f'  Total initial error: {init_err:.2f}   '
          f'avg/factor: {avg_init:.4f}')

    result, optimizer = optimize_bundle(graph, initial)
    final_err = graph.error(result)
    avg_final = final_err / max(1, n_factors)
    print(f'  Total final error  : {final_err:.2f}   '
          f'avg/factor: {avg_final:.4f}')

    # Largest-initial-error projection factor.
    idx, fid, tid, link, init_factor_err = _largest_initial_error_factor(
        graph, initial, info['projection_factors'])
    final_factor_err = graph.at(idx).error(result)
    print(f'\n  Largest-initial-error projection: '
          f'frame {fid} ↔ track {tid}')
    print(f'    initial factor error: {init_factor_err:.3f}')

    pose_init = initial.atPose3(info['cam_keys'][fid])
    q_init = initial.atPoint3(info['lm_keys'][tid])
    sc_init = stereo_camera(pose_init, K_stereo)
    try:
        proj_init = sc_init.project(q_init)
        d_l_init = np.hypot(proj_init.uL() - link.x_left, proj_init.v() - link.y)
        d_r_init = np.hypot(proj_init.uR() - link.x_right, proj_init.v() - link.y)
        print(f'    initial proj distance (L/R): {d_l_init:.2f}/{d_r_init:.2f} px')
        _draw_projection_vs_measurement(
            fid, link, proj_init,
            'q5_3_largest_factor_initial',
            f'Q5.3 — largest-error factor (initial): frame {fid}, track {tid}, '
            f'err {init_factor_err:.2f}')
    except RuntimeError:
        print('    [initial projection raised Cheirality — skipping plot]')

    pose_final = result.atPose3(info['cam_keys'][fid])
    q_final = result.atPoint3(info['lm_keys'][tid])
    sc_final = stereo_camera(pose_final, K_stereo)
    proj_final = sc_final.project(q_final)
    d_l_final = np.hypot(proj_final.uL() - link.x_left, proj_final.v() - link.y)
    d_r_final = np.hypot(proj_final.uR() - link.x_right, proj_final.v() - link.y)
    print(f'    final factor error  : {final_factor_err:.4f}')
    print(f'    final proj distance (L/R): {d_l_final:.2f}/{d_r_final:.2f} px')
    _draw_projection_vs_measurement(
        fid, link, proj_final,
        'q5_3_largest_factor_final',
        f'Q5.3 — largest-error factor (after BA): frame {fid}, track {tid}, '
        f'err {final_factor_err:.4f}')

    # 3D plot — conditional covariance per camera given c0 fixed (the
    # "relative" uncertainty growing along the chain, NOT the marginal which
    # is dominated by the loose prior at c0). Per the bundle lecture: take
    # the joint information matrix, erase the conditioned variable's row/
    # column, invert, read the diagonal block of the queried variable.
    marginals = gtsam.Marginals(graph, result)
    COV_VIS_SCALE = 1.0e3  # inflate the (sub-cm) conditional cov so the
                           # ellipsoids are visible at the bundle's metre scale
    res_plot, info_plot = result, info

    c0_key = info['cam_keys'][frames[0]]
    cond_cov = {}
    for fid in frames:
        ck = info['cam_keys'][fid]
        if fid == frames[0]:
            cond_cov[fid] = np.zeros((6, 6))    # c0 is conditioned → no uncertainty
        else:
            kv = gtsam.KeyVector(); kv.append(c0_key); kv.append(ck)
            I_joint = marginals.jointMarginalInformation(kv).fullMatrix()
            # Take the lower 6×6 block (the queried variable's info under
            # conditioning on c0) and invert to get the conditional cov.
            I_kk = I_joint[6:12, 6:12]
            cond_cov[fid] = np.linalg.inv(I_kk)

    # KITTI has X-right, Y-down, Z-forward. Matplotlib 3D renders its own Z
    # as the vertical-on-screen axis, so to display "Y up" we map the data
    # axes:   matplotlib_X = data_X,
    #         matplotlib_Y = data_Z   (depth-into-screen = forward),
    #         matplotlib_Z = −data_Y  (vertical-on-screen = "up").
    # That's a rotation M = [[1,0,0],[0,0,1],[0,−1,0]] applied to every pose
    # and its marginal covariance. We then call plot_pose3_on_axes manually.
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    M6 = np.block([[M, np.zeros((3, 3))], [np.zeros((3, 3)), M]])

    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    for fid in frames:
        ck = info_plot['cam_keys'][fid]
        pose = res_plot.atPose3(ck)
        P = cond_cov[fid]
        new_pose = gtsam.Pose3(gtsam.Rot3(M @ pose.rotation().matrix()),
                               gtsam.Point3(M @ pose.translation()))
        new_P = (M6 @ P @ M6.T) * COV_VIS_SCALE
        gtsam_plot.plot_pose3_on_axes(ax3d, new_pose, axis_length=0.5,
                                      P=new_P)
    ax3d.set_xlabel('X (m, right)')
    ax3d.set_ylabel('Z (m, forward)')
    ax3d.set_zlabel('Y (m, up)')
    ax3d.set_title('Q5.3 — first bundle (3D, with cov)')
    gtsam_plot.set_axes_equal(fig.number)
    _save(fig, 'q5_3_bundle_3d')

    # Top-down (X vs Z) — "view-from-above of the scene, with all cameras
    # and points" (spec wording). All 9 cameras of the first bundle as
    # individual markers, plus every BA landmark.
    est_centres_b = np.array([
        camera_center(gtsam_pose_to_Rt(result.atPose3(info['cam_keys'][f])))
        for f in frames])
    pts3d = np.array([result.atPoint3(k) for k in info['lm_keys'].values()])
    pts_xz = pts3d[:, [0, 2]]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(pts_xz[:, 0], pts_xz[:, 1], s=2, c='lightsteelblue', alpha=0.4,
               label=f'BA landmarks (n={len(pts_xz)})', zorder=1)
    ax.plot(est_centres_b[:, 0], est_centres_b[:, 2], '-',
            color='steelblue', linewidth=0.8, alpha=0.6, zorder=2)
    ax.scatter(est_centres_b[:, 0], est_centres_b[:, 2],
               s=40, c='steelblue', edgecolor='black', linewidth=0.5,
               label=f'cameras (n={len(frames)})', zorder=3)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z — forward (m)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('Q5.3 — first bundle (top-down)')
    ax.legend()
    _save(fig, 'q5_3_bundle_topdown')


# ex5
def q4(db, pnp_poses, K_stereo, keyframes, feature_sizes=None):
    """Q5.4 — solve every bundle window; chain into global keyframe poses."""
    n_bundles = len(keyframes) - 1
    print(f'  Q5.4 — solving {n_bundles} bundle windows')

    relative_kf_Rt = []      # 3x4 world-to-kf_end in kf_start's frame, per bundle
    bundle_internal_Rts = [] # list of dicts: {frame_id -> Rt_rel} per bundle
    bundle_landmarks_local = []  # list of (Nb, 3) arrays in each bundle's local frame
    last_bundle_state = None
    start = time.time()
    for b in range(n_bundles):
        kf_a, kf_b = keyframes[b], keyframes[b + 1]
        frames = list(range(kf_a, kf_b + 1))
        graph, initial, info = build_bundle_window(
            db, pnp_poses, frames, K_stereo, feature_sizes=feature_sizes)
        result, _ = optimize_bundle(graph, initial)
        pose_end = result.atPose3(info['cam_keys'][kf_b])
        relative_kf_Rt.append(gtsam_pose_to_Rt(pose_end))
        bundle_internal_Rts.append({
            fid: gtsam_pose_to_Rt(result.atPose3(info['cam_keys'][fid]))
            for fid in frames
        })
        if info['lm_keys']:
            bundle_landmarks_local.append(np.array([
                result.atPoint3(k) for k in info['lm_keys'].values()]))
        else:
            bundle_landmarks_local.append(np.zeros((0, 3)))
        if b == n_bundles - 1:
            last_bundle_state = (graph, result, info, kf_a, kf_b)
        if (b + 1) % 20 == 0 or b == n_bundles - 1:
            elapsed = time.time() - start
            print(f'    {b + 1}/{n_bundles} bundles done in {elapsed:.1f}s '
                  f'({elapsed / (b + 1):.2f}s each)')

    # Spec 5.4 — last bundle diagnostics.
    graph_last, result_last, info_last, kf_a_last, kf_b_last = last_bundle_state
    pose_first_last_bundle = result_last.atPose3(info_last['cam_keys'][kf_a_last])
    print(f'\n  Last bundle ({kf_a_last}→{kf_b_last}) — '
          f'first-frame pose after optimization:')
    print(f'    rotation diag: '
          f'{np.diag(pose_first_last_bundle.rotation().matrix())}')
    print(f'    translation : {pose_first_last_bundle.translation()}')
    # The anchoring factor is the first one added — by construction it's the prior.
    anchor_factor = graph_last.at(0)
    print(f'  Anchoring factor final error: {anchor_factor.error(result_last):.6e}')

    # Chain to global frame-0 coordinates.
    abs_kf_Rt = compose_global_poses(relative_kf_Rt)
    est_centres = np.array([camera_center(R) for R in abs_kf_Rt])

    # Per-frame BA-estimated centres: compose each frame's bundle-local pose
    # onto its bundle's start keyframe absolute pose. Later bundles overwrite
    # earlier ones on shared boundary keyframes — both should agree.
    n_total_frames = max(keyframes) + 1
    abs_per_frame = [None] * n_total_frames
    for b, rels in enumerate(bundle_internal_Rts):
        abs_kf_a = abs_kf_Rt[b]
        for fid, Rt_local in rels.items():
            abs_per_frame[fid] = compose_extrinsics(abs_kf_a, Rt_local)
    est_centres_full = np.array([camera_center(Rt) for Rt in abs_per_frame
                                 if Rt is not None])

    # Ground truth at every frame and at keyframes (the dots).
    gt = read_poses()
    gt_centres_full = np.array([camera_center(p) for p in gt[:n_total_frames]])
    gt_centres = np.array([camera_center(gt[k]) for k in keyframes])

    # Transform each bundle's optimised landmarks from bundle-local coords
    # (origin = that bundle's first KF) into global frame-0 coords. Each
    # local point x_local satisfies x_local = R_a x_global + t_a, so
    # x_global = R_a^T (x_local - t_a) where [R_a | t_a] = abs_kf_Rt[b].
    all_pts_global = []
    for b, pts_local in enumerate(bundle_landmarks_local):
        if len(pts_local) == 0:
            continue
        R = abs_kf_Rt[b][:, :3]
        t = abs_kf_Rt[b][:, 3]
        all_pts_global.append((pts_local - t) @ R)
    all_pts_global = (np.concatenate(all_pts_global, axis=0)
                      if all_pts_global else np.zeros((0, 3)))
    # Clip the cloud to a plausible scene bounding box: drop wild back-
    # projection outliers (z<0, or absurdly large coords) so they don't
    # explode the axis ranges.
    if len(all_pts_global):
        z_ok = (all_pts_global[:, 2] > est_centres_full[:, 2].min() - 50) & \
               (all_pts_global[:, 2] < est_centres_full[:, 2].max() + 200)
        x_ok = (all_pts_global[:, 0] > est_centres_full[:, 0].min() - 200) & \
               (all_pts_global[:, 0] < est_centres_full[:, 0].max() + 200)
        all_pts_global = all_pts_global[z_ok & x_ok]

    # 2D bird's-eye #1 — full trajectory + landmark cloud + KF markers.
    fig, ax = plt.subplots(figsize=(11, 11))
    if len(all_pts_global):
        ax.scatter(all_pts_global[:, 0], all_pts_global[:, 2],
                   s=1, c='lightsteelblue', alpha=0.25,
                   label=f'BA landmarks (n={len(all_pts_global)})', zorder=1)
    ax.plot(est_centres_full[:, 0], est_centres_full[:, 2], '-',
            color='steelblue', linewidth=1.2,
            label=f'estimated  (all {len(est_centres_full)} frames)',
            zorder=2)
    ax.plot(gt_centres_full[:, 0], gt_centres_full[:, 2], '-',
            color='orange', linewidth=1.2, label='ground truth', zorder=3)
    ax.scatter(est_centres[:, 0], est_centres[:, 2], s=8,
               c='steelblue', edgecolor='black', linewidth=0.3, zorder=4,
               label=f'keyframes (n={len(est_centres)})')
    ax.scatter([0], [0], c='black', s=80, marker='s', zorder=5, label='start')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z — forward (m)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('Q5.4 — full trajectory + landmarks (bundle vs ground truth)')
    ax.legend()
    _save(fig, 'q5_4_keyframe_trajectory')


    # Localization error over time.
    err = np.linalg.norm(est_centres - gt_centres, axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(keyframes, err, color='steelblue', linewidth=0.9)
    ax.set_xlabel('Frame'); ax.set_ylabel('Localization error (m)')
    ax.set_title(f'Q5.4 — keyframe localization error '
                 f'(final={err[-1]:.2f}m, median={np.median(err):.2f}m, '
                 f'max={err.max():.2f}m)')
    ax.grid(alpha=0.3)
    _save(fig, 'q5_4_localization_error')

    print(f'\n  Final-pose error: {err[-1]:.2f} m')
    print(f'  Median / max localization error: '
          f'{np.median(err):.2f} m / {err.max():.2f} m')


def main():
    K, m1, m2 = read_cameras()
    baseline = -m2[0, 3]                # extract positive baseline from data
    print(f'KITTI baseline: {baseline:.4f} m')
    P_left, P_right = K @ m1, K @ m2
    K_stereo = stereo_calibration(K, baseline)

    # Tracking DB from ex4.
    db = TrackingDB()
    db.load(TRACKING_DB_BASE)

    # Frame count: as many as we have images + tracking data for.
    img_dir = os.path.join(DATA_PATH, 'image_0')
    n_frames = sum(1 for f in os.listdir(img_dir) if f.endswith('.png'))
    n_frames = min(n_frames, db.frame_num())
    print(f'Frames in play: {n_frames}')

    pnp_poses = load_or_compute_pnp_poses(n_frames, K, P_left, P_right, m2)

    feature_sizes = compute_feature_sizes(n_frames, FEATURE_SIZES_PATH)

    keyframes = select_keyframes_in_calm_frames(
        pnp_poses,
        min_translation=KF_MIN_TRANSLATION,
        translation_jitter=1.0, seed=42,
        max_frames_gap=19,                  # window count is gap+1, cap at 20
        min_frames_gap=8,                   # window count ≥ 9 (inside 5-20)
        straight_rot_rate_deg=0.5,
    )
    gaps = np.diff(keyframes)
    print(f'Keyframes: {len(keyframes)} '
          f'(first 5: {keyframes[:5]}, last 3: {keyframes[-3:]})')
    print(f'  Window sizes: min={gaps.min() + 1}, median={int(np.median(gaps)) + 1}, '
          f'max={gaps.max() + 1} frames')

    print('\nQ5.1')
    q1(db, pnp_poses, K_stereo)

    print('\nQ5.3')
    q3(db, pnp_poses, K_stereo, keyframes, feature_sizes=feature_sizes)

    print('\nQ5.4')
    q4(db, pnp_poses, K_stereo, keyframes, feature_sizes=feature_sizes)

    plt.show()


if __name__ == '__main__':
    main()
