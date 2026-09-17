"""PnP-based relative pose estimation with four-view supporter filtering and RANSAC."""

import os
import time

import cv2
import numpy as np

from dataset import DATA_PATH, read_images
from features import (extract_features, match_descriptors,
                      rectified_stereo_filter, consensus_matches,
                      pts_from_matches)
from geometry import (rodriguez_to_mat, compose_extrinsics, project,
                      triangulate_linear_lsq)


IDENTITY_RT = np.hstack([np.eye(3), np.zeros((3, 1))])
DETECTOR = 'AKAZE'
Y_THRESHOLD = 2.0       # px — rectified-stereo vertical-deviation cutoff (ex2)
X_MIN_DISPARITY = 0.0   # px — require positive disparity (rejects x_l <= x_r)
PIX_THRESHOLD = 2.0     # px — per-image supporter reprojection cutoff (ex3)
PNP_POSES_PATH = os.path.join(DATA_PATH, 'pnp_poses.npy')


# ex5
def load_or_compute_pnp_poses(n_frames, K, P_left, P_right, m_right):
    """Return Nx3x4 PnP world-to-camera extrinsics, recomputing only if missing."""
    if os.path.exists(PNP_POSES_PATH):
        poses = np.load(PNP_POSES_PATH)
        if poses.shape[0] >= n_frames:
            print(f'Loaded {poses.shape[0]} PnP poses from cache.')
            return poses[:n_frames]
    print(f'PnP poses cache missing — running track_sequence on {n_frames} frames…')
    Rt_seq, _ = track_sequence(n_frames, K, P_left, P_right, m_right)
    np.save(PNP_POSES_PATH, Rt_seq)
    print(f'Saved PnP poses to {PNP_POSES_PATH}')
    return Rt_seq


# ex3
def solve_pnp(X, pts2d, K, flags=cv2.SOLVEPNP_SQPNP):
    """Solve PnP for the extrinsic [R | t] of a camera viewing 3D points X at pixels pts2d.

    Args:
        X: Nx3 array of 3D points in the reference frame to which [R | t] will be relative.
        pts2d: Nx2 array of matching pixel coordinates.
        K: 3x3 intrinsic matrix.
        flags: cv2.solvePnP flag. ``SOLVEPNP_SQPNP`` is the default — it solves the
            problem as a quadratic program and accepts ≥ 3 correspondences, so the
            same call serves both the inner RANSAC hypothesis (4 points) and the
            full-inlier refinement that follows.

    Returns:
        3x4 [R | t] matrix, or None if the solver fails / returns a degenerate result.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 1, 3)
    pts2d = np.asarray(pts2d, dtype=np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = cv2.solvePnP(X, pts2d, K, distCoeffs=None, flags=flags)
    except cv2.error:
        return None
    if not ok:
        return None
    return rodriguez_to_mat(rvec, tvec)


# ex3
def project_to_four_views(X, Rt_left1, K, m_right):
    """Project Nx3 points (in left0 frame) onto left0, right0, left1, right1.

    m_right is the relative extrinsic from left → right of the rectified stereo rig
    (same for every frame); composing it with Rt_left1 gives the extrinsic of
    right1 in left0 coords.
    """
    Rt_left0 = IDENTITY_RT
    Rt_right0 = m_right
    Rt_right1 = compose_extrinsics(Rt_left1, m_right)
    return (project(K, Rt_left0,  X),
            project(K, Rt_right0, X),
            project(K, Rt_left1,  X),
            project(K, Rt_right1, X))


# ex3
def supporters_mask(X0, pts_l0, pts_r0, pts_l1, pts_r1, Rt_left1, K, m_right, threshold=2.0):
    """Boolean mask of points whose reprojection error is within `threshold` on all four images."""
    pl0, pr0, pl1, pr1 = project_to_four_views(X0, Rt_left1, K, m_right)
    e_l0 = np.linalg.norm(pl0 - pts_l0, axis=1)
    e_r0 = np.linalg.norm(pr0 - pts_r0, axis=1)
    e_l1 = np.linalg.norm(pl1 - pts_l1, axis=1)
    e_r1 = np.linalg.norm(pr1 - pts_r1, axis=1)
    return (e_l0 <= threshold) & (e_r0 <= threshold) & (e_l1 <= threshold) & (e_r1 <= threshold)


# ex3
def ransac_pnp(X0, pts_l0, pts_r0, pts_l1, pts_r1, K, m_right,
               threshold=2.0, p_success=0.99, max_iter=1000, min_iter=50,
               refine=True, sample_size=4, rng=None):
    """RANSAC-PnP: find [R | t] of left1 in left0 coords maximizing four-view supporters.

    Implements the textbook RANSAC loop ourselves (the spec forbids ``cv2.solvePnPRansac``):
    sample 4 consensus matches, fit SQPNP, score by the four-view supporters mask, keep
    the best, and adaptively reduce the iteration budget as the best inlier ratio grows.
    Finally refit SQPNP on the full inlier set as the spec's refinement step.

    Returns:
        Rt_left1: best 3x4 extrinsic, or None if no hypothesis ever succeeded.
        mask: boolean array marking supporters of the returned Rt.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(X0)
    if n < sample_size:
        return None, np.zeros(n, dtype=bool)

    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    best_Rt = None
    n_iter = max_iter
    i = 0
    while i < n_iter:
        idx = rng.choice(n, sample_size, replace=False)
        Rt = solve_pnp(X0[idx], pts_l1[idx], K, flags=cv2.SOLVEPNP_SQPNP)
        if Rt is not None:
            mask = supporters_mask(X0, pts_l0, pts_r0, pts_l1, pts_r1,
                                   Rt, K, m_right, threshold)
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_mask = mask
                best_Rt = Rt
                w = count / n
                if 0.0 < w < 1.0:
                    num = np.log(max(1.0 - p_success, 1e-12))
                    den = np.log(max(1.0 - w**sample_size, 1e-12))
                    n_iter = int(min(max_iter, max(min_iter, np.ceil(num / den))))
        i += 1

    if best_Rt is not None and refine and best_count >= sample_size:
        # Spec 3.5: refine the resulting transformation by refitting T on the
        # full inlier set (single pass).
        Rt = solve_pnp(X0[best_mask], pts_l1[best_mask], K,
                       flags=cv2.SOLVEPNP_SQPNP)
        if Rt is not None:
            mask = supporters_mask(X0, pts_l0, pts_r0, pts_l1, pts_r1,
                                   Rt, K, m_right, threshold)
            if mask.sum() >= best_count:
                best_Rt = Rt
                best_mask = mask

    return best_Rt, best_mask


# ex3
def stereo_features(img_l, img_r, detector=DETECTOR):
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


# ex3
def build_consensus(kp_l0, kp_r0, kp_l1, kp_r1, stereo0_in, stereo1_in, cross,
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


# ex3
def track_sequence(n_frames, K, P_left, P_right, m_right, verbose_every=100):
    """Track frames 0..n_frames−1 with consecutive RANSAC-PnP; return Nx3x4 extrinsics in left0.

    Caches the previous frame's features so each image is processed once.
    """
    Rt_global = [IDENTITY_RT.copy()]
    rng = np.random.default_rng(0)
    start = time.time()

    img_l, img_r = read_images(0)
    kp_l, des_l, kp_r, s_in = stereo_features(img_l, img_r)

    for i in range(1, n_frames):
        img_l_new, img_r_new = read_images(i)
        kp_l_new, des_l_new, kp_r_new, s_new_in = stereo_features(
            img_l_new, img_r_new)
        cross = match_descriptors(des_l, des_l_new, DETECTOR)
        consensus = build_consensus(kp_l, kp_r, kp_l_new, kp_r_new,
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
