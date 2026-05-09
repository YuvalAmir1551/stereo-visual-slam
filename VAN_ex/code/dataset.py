"""KITTI dataset I/O — image reading and camera calibration."""

import os
import cv2
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'sequences', '00')


# ex1
def read_images(idx):
    """Read a stereo pair (left, right) by frame index."""
    img_name = '{:06d}.png'.format(idx)
    img_left = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name), cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name), cv2.IMREAD_GRAYSCALE)
    return img_left, img_right


# ex2
def read_cameras():
    """Read KITTI stereo camera matrices from calib.txt.

    Follows the helper in the ex2 spec PDF: K is factored out of the two
    projection matrices so the returned `m1`, `m2` are pure extrinsics
    ([R | t] form, with R = I for both KITTI rectified cameras).

    Returns:
        K: Shared 3x3 intrinsic matrix.
        m1, m2: 3x4 extrinsics matrices. m1 = [I | 0]; m2 = [I | t] with
            t = (−baseline, 0, 0). To get full projection matrices, use
            `K @ m1` and `K @ m2`.
    """
    with open(os.path.join(DATA_PATH, 'calib.txt')) as f:
        l1 = [float(x) for x in f.readline().split()[1:]]
        l2 = [float(x) for x in f.readline().split()[1:]]
    m1 = np.array(l1).reshape(3, 4)
    m2 = np.array(l2).reshape(3, 4)
    K = m1[:, :3]
    m1 = np.linalg.inv(K) @ m1
    m2 = np.linalg.inv(K) @ m2
    return K, m1, m2
