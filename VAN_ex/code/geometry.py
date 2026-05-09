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
