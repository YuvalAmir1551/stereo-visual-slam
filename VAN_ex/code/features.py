"""Feature detection, descriptor matching, and match filtering."""

import cv2
import numpy as np

# Lower-than-default AKAZE detection threshold (cv2 default 0.001 → 0.0001).
# Yields more far-field keypoints near the vanishing point, which empirically
# tightens the rotation estimate on KITTI 00 and is required to feed enough
# tracks into the ex5/ex6 bundle adjustment.
_AKAZE_THRESHOLD = 0.0001


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
        det = cv2.AKAZE_create(threshold=_AKAZE_THRESHOLD)
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
def match_descriptors(desc_left, desc_right, detector, cross_check=False):
    """Single-best-match descriptor matching (no ratio test).

    Args:
        desc_left, desc_right: Descriptor arrays.
        detector: Detector name (selects Hamming vs L2 norm).
        cross_check: If True, return only mutual best matches (each left
            descriptor must be the best for its matched right descriptor and
            vice versa). Drops ambiguous pairs cheaply.

    Returns a flat list of DMatch objects.
    """
    bf = cv2.BFMatcher(_norm_type(detector), crossCheck=cross_check)
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
def rectified_stereo_filter(kp_left, kp_right, matches,
                            y_threshold=2.0, x_min_disparity=None):
    """Reject matches that violate the rectified-stereo geometry.

    Two checks (the X one is optional):
      • |y_left − y_right| ≤ y_threshold — epipolar alignment. On a rectified
        pair corresponding points share the same image row, so |Δy| should be
        near zero.
      • x_left − x_right > x_min_disparity — positive-disparity sanity check.
        A real point in front of the camera satisfies x_l > x_r (ex2.4);
        requiring a small margin above 0 also drops very-far points whose
        sub-noise disparity would triangulate to garbage. Skipped if
        `x_min_disparity` is None.

    Args:
        kp_left, kp_right: Keypoint lists from the two images.
        matches: Iterable of cv2.DMatch (typically the best match per left kp).
        y_threshold: Maximum allowed |Δy| in pixels.
        x_min_disparity: Minimum required disparity x_l − x_r in pixels, or
            None to disable the X check (ex2 default behaviour).

    Returns:
        inliers: List of DMatch passing both checks.
        outliers: List of DMatch failing at least one check.
        dy: 1-D numpy array of |Δy| values for every input match (same order).
    """
    inliers, outliers, dy = [], [], []
    for m in matches:
        pl = kp_left[m.queryIdx].pt
        pr = kp_right[m.trainIdx].pt
        d_y = abs(pl[1] - pr[1])
        dy.append(d_y)
        if d_y > y_threshold:
            outliers.append(m)
            continue
        if x_min_disparity is not None and (pl[0] - pr[0]) <= x_min_disparity:
            outliers.append(m)
            continue
        inliers.append(m)
    return inliers, outliers, np.array(dy)


# ex3
def consensus_matches(stereo0, stereo1, cross):
    """Find keypoint correspondences visible in all four images of two stereo pairs.

    A consensus is built when:
      • A left0 keypoint has a stereo partner on right0 (entry in `stereo0`).
      • The same left0 keypoint has a cross-frame partner on left1 (entry in `cross`).
      • That left1 keypoint has a stereo partner on right1 (entry in `stereo1`).

    Args:
        stereo0: DMatch list with queryIdx into kp_left0, trainIdx into kp_right0.
        stereo1: DMatch list with queryIdx into kp_left1, trainIdx into kp_right1.
        cross:   DMatch list with queryIdx into kp_left0, trainIdx into kp_left1.

    Returns:
        idx0: indices into `stereo0` of consensus matches.
        idx1: indices into `stereo1`, paired one-to-one with `idx0`.
    """
    l0_to_s0 = {m.queryIdx: i for i, m in enumerate(stereo0)}
    l1_to_s1 = {m.queryIdx: i for i, m in enumerate(stereo1)}
    i0_list, i1_list = [], []
    for m in cross:
        i0 = l0_to_s0.get(m.queryIdx)
        i1 = l1_to_s1.get(m.trainIdx)
        if i0 is not None and i1 is not None:
            i0_list.append(i0)
            i1_list.append(i1)
    return np.array(i0_list, dtype=int), np.array(i1_list, dtype=int)
