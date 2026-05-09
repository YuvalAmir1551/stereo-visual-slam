"""Feature detection, descriptor matching, and match filtering."""

import cv2
import numpy as np


# ex1
def extract_features(img, detector, n_features=5000):
    """Detect keypoints and compute their descriptors on an image.

    Args:
        img: Grayscale image.
        detector: Feature method ('ORB', 'AKAZE', 'SIFT').
        n_features: Maximum number of features (used by ORB/SIFT).

    Returns:
        keypoints: List of cv2.KeyPoint objects.
        descriptors: Numpy array of descriptors.
    """
    if detector == 'ORB':
        det = cv2.ORB_create(nfeatures=n_features)
    elif detector == 'AKAZE':
        det = cv2.AKAZE_create()
    elif detector == 'SIFT':
        det = cv2.SIFT_create(nfeatures=n_features)
    else:
        raise ValueError(f"Unknown detector: {detector}")
    return det.detectAndCompute(img, None)


def _norm_type(detector):
    """Pick the descriptor norm type matching the detector."""
    return cv2.NORM_HAMMING if detector in ('ORB', 'AKAZE') else cv2.NORM_L2


# ex1
def match_descriptors_knn(desc_left, desc_right, detector, k=2):
    """Match each left descriptor to its k nearest right descriptors.

    Args:
        desc_left, desc_right: Descriptor arrays.
        detector: Detector name (selects Hamming vs L2 norm).
        k: Number of nearest neighbors per descriptor.

    Returns:
        matches: List of tuples, each containing k DMatch objects.
    """
    bf = cv2.BFMatcher(_norm_type(detector), crossCheck=False)
    return bf.knnMatch(desc_left, desc_right, k=k)


# ex1
def match_descriptors(desc_left, desc_right, detector):
    """Single-best-match descriptor matching (no ratio test).

    Returns a flat list of DMatch objects, one per left descriptor.
    """
    bf = cv2.BFMatcher(_norm_type(detector), crossCheck=False)
    return bf.match(desc_left, desc_right)


# ex1
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


# ex2
def match_stereo_pair(img_left, img_right, detector):
    """Detect features on a stereo pair and best-match them (no ratio test).

    Returns:
        kp_left, kp_right: Keypoint lists from the two images.
        matches: Flat list of best DMatch per left keypoint.
    """
    kp_l, des_l = extract_features(img_left, detector=detector)
    kp_r, des_r = extract_features(img_right, detector=detector)
    matches = match_descriptors(des_l, des_r, detector=detector)
    return kp_l, kp_r, matches


# ex2
def pts_from_matches(kp_left, kp_right, matches):
    """Extract pixel coordinates of matched keypoints as two Nx2 arrays."""
    pts_l = np.array([kp_left[m.queryIdx].pt for m in matches])
    pts_r = np.array([kp_right[m.trainIdx].pt for m in matches])
    return pts_l, pts_r


# ex2
def rectified_stereo_filter(kp_left, kp_right, matches, y_threshold=2.0):
    """Reject matches that violate the rectified-stereo y-alignment constraint.

    On a rectified pair, corresponding points share the same image row, so
    |y_left - y_right| should be near zero. Matches whose vertical deviation
    exceeds `y_threshold` pixels are treated as outliers.

    Args:
        kp_left, kp_right: Keypoint lists from the two images.
        matches: Iterable of cv2.DMatch (typically the best match per left kp).
        y_threshold: Maximum allowed |Δy| in pixels.

    Returns:
        inliers: List of DMatch with |Δy| <= y_threshold.
        outliers: List of DMatch with |Δy| > y_threshold.
        dy: 1-D numpy array of |Δy| values for every input match (same order).
    """
    inliers, outliers, dy = [], [], []
    for m in matches:
        y_l = kp_left[m.queryIdx].pt[1]
        y_r = kp_right[m.trainIdx].pt[1]
        d = abs(y_l - y_r)
        dy.append(d)
        if d <= y_threshold:
            inliers.append(m)
        else:
            outliers.append(m)
    return inliers, outliers, np.array(dy)
