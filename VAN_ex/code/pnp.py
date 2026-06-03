"""PnP-based relative pose estimation with four-view supporter filtering and RANSAC."""

import cv2
import numpy as np

from geometry import rodriguez_to_mat, compose_extrinsics, project


IDENTITY_RT = np.hstack([np.eye(3), np.zeros((3, 1))])


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
