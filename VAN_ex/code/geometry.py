"""Multi-view geometry — triangulation and related primitives."""

import cv2
import numpy as np


# ex2
def triangulate_linear_lsq(P_left, P_right, pts_left, pts_right):
    """Linear least-squares triangulation.

    For each correspondence (x_l, x_r), build the 4x4 matrix A whose rows are:
        x_l * P_l[2,:] - P_l[0,:]
        y_l * P_l[2,:] - P_l[1,:]
        x_r * P_r[2,:] - P_r[0,:]
        y_r * P_r[2,:] - P_r[1,:]
    and solve A X = 0 via SVD; the world point is the right-singular vector
    of the smallest singular value, dehomogenized.

    Args:
        P_left, P_right: 3x4 projection matrices.
        pts_left, pts_right: Nx2 arrays of pixel correspondences.

    Returns:
        Nx3 array of triangulated world points.
    """
    pts_left = np.asarray(pts_left, dtype=np.float64)
    pts_right = np.asarray(pts_right, dtype=np.float64)
    n = pts_left.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        xl, yl = pts_left[i]
        xr, yr = pts_right[i]
        A = np.vstack([
            xl * P_left[2] - P_left[0],
            yl * P_left[2] - P_left[1],
            xr * P_right[2] - P_right[0],
            yr * P_right[2] - P_right[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out[i] = X[:3] / X[3]
    return out


# ex2
def triangulate_cv2(P_left, P_right, pts_left, pts_right):
    """OpenCV reference triangulation (cv2.triangulatePoints), for comparison.

    Args:
        P_left, P_right: 3x4 projection matrices.
        pts_left, pts_right: Nx2 arrays of pixel correspondences.

    Returns:
        Nx3 array of triangulated world points.
    """
    pts_left = np.asarray(pts_left, dtype=np.float64).T   # 2xN
    pts_right = np.asarray(pts_right, dtype=np.float64).T  # 2xN
    X_h = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)  # 4xN
    X = (X_h[:3] / X_h[3]).T
    return X


# ex3
def rodriguez_to_mat(rvec, tvec):
    """Assemble a 3x4 [R | t] extrinsic from a Rodrigues rotation vector and a translation."""
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    return np.hstack((R, t))


# ex3
def camera_center(Rt):
    """World-frame centre of a camera with extrinsic [R | t]: C = −R^T t.

    The identity [R|t] @ (C, 1)^T = 0 (the camera origin in its own frame is
    zero) gives R C + t = 0, hence C = −R^T t.
    """
    R, t = Rt[:, :3], Rt[:, 3]
    return -R.T @ t


# ex3
def compose_extrinsics(Rt_AB, Rt_BC):
    """Compose two extrinsics so that the result transforms frame A directly into frame C.

    For Rt_AB(x) = R1 x + t1 (A → B) and Rt_BC(x) = R2 x + t2 (B → C),
    the chained map is Rt_AC(x) = R2 R1 x + (R2 t1 + t2), i.e. the extrinsic
    of C expressed in A's coordinates.
    """
    R1, t1 = Rt_AB[:, :3], Rt_AB[:, 3]
    R2, t2 = Rt_BC[:, :3], Rt_BC[:, 3]
    R = R2 @ R1
    t = R2 @ t1 + t2
    return np.hstack((R, t.reshape(3, 1)))


# ex3
def project(K, Rt, X):
    """Project Nx3 points X through a camera with intrinsics K and extrinsic [R | t].

    Points behind the camera (negative depth after the extrinsic transform)
    are still returned with a division-by-z, so callers that care about
    physical validity should pair this with a depth check.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Xh = np.column_stack([X, np.ones(len(X))])
    xh = (K @ Rt @ Xh.T).T
    return xh[:, :2] / xh[:, 2:3]
