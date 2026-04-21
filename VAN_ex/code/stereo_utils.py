"""Utility functions for stereo vision processing on the KITTI dataset."""

import os
import cv2
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'sequences', '00')


def read_images(idx):
    """Read a stereo pair (left, right) by frame index."""
    img_name = '{:06d}.png'.format(idx)
    img_left = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name), cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name), cv2.IMREAD_GRAYSCALE)
    return img_left, img_right


def detect_and_compute(img, algorithm='ORB', n_features=5000):
    """Detect keypoints and compute descriptors on an image.

    Args:
        img: Grayscale image.
        algorithm: Feature detection method ('ORB', 'AKAZE', 'SIFT').
        n_features: Maximum number of features to detect (used by ORB/SIFT).

    Returns:
        keypoints: List of cv2.KeyPoint objects.
        descriptors: Numpy array of descriptors.
    """
    if algorithm == 'ORB':
        detector = cv2.ORB_create(nfeatures=n_features)
    elif algorithm == 'AKAZE':
        detector = cv2.AKAZE_create()
    elif algorithm == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=n_features)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


def match_descriptors_knn(desc_left, desc_right, algorithm='ORB', k=2):
    """Match each left descriptor to the k nearest right descriptors.

    Args:
        desc_left: Descriptors from the left image.
        desc_right: Descriptors from the right image.
        algorithm: Feature algorithm used (determines norm type).
        k: Number of nearest neighbors to return per descriptor.

    Returns:
        matches: List of tuples, each containing k DMatch objects.
    """
    if algorithm in ('ORB', 'AKAZE'):
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(desc_left, desc_right, k=k)
    return matches


def apply_ratio_test(knn_matches, ratio=0.7):
    """Apply Lowe's ratio test to filter matches.

    Args:
        knn_matches: KNN matches (k=2).
        ratio: Maximum ratio between best and second-best match distance.

    Returns:
        accepted: List of DMatch objects that passed the test.
        rejected: List of (DMatch, ratio) tuples sorted ascending by ratio,
            so matches that barely failed the test come first.
    """
    accepted = []
    rejected = []
    for m, n in knn_matches:
        r = m.distance / n.distance if n.distance > 0 else float('inf')
        if r < ratio:
            accepted.append(m)
        else:
            rejected.append((m, r))
    rejected.sort(key=lambda x: x[1])
    return accepted, rejected
