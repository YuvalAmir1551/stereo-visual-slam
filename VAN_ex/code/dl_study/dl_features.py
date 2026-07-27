"""Deep-learning front-end: SuperPoint detector + LightGlue matcher.

Learned alternative to the classical detectors/matchers in features.py.
Keypoints come back as cv2.KeyPoint and matches as cv2.DMatch, so every
downstream consumer (rectified_stereo_filter, consensus_matches,
pts_from_matches, the tracking database) works unchanged. The heavy imports
(torch, lightglue) are deferred to first use.

Requires (in the `slam` env): torch, lightglue
(pip install git+https://github.com/cvg/LightGlue.git)
"""
import cv2

# SuperPoint keypoint cap per image. 2048 roughly matches the keypoint count
# the AKAZE/SIFT paths produce on KITTI 00 frames, keeping the two front-ends
# comparable in the report statistics.
_SUPERPOINT_MAX_KEYPOINTS = 2048

# SuperPoint reports position only (no scale); downstream code that reads
# KeyPoint.size still needs a value, so use a nominal diameter in pixels.
_SUPERPOINT_KEYPOINT_SIZE = 8.0

_deep_models = {}  # lazy singleton: {'device', 'extractor', 'matcher'}


# project
def _deep_frontend():
    """Load SuperPoint + LightGlue once and cache them (lazy singleton).

    Picks the best available torch device (CUDA > Apple MPS > CPU).

    Returns:
        Dict with 'device' (str), 'extractor' (SuperPoint module) and
        'matcher' (LightGlue module), both in eval mode on that device.
    """
    if not _deep_models:
        import os
        # Let torch fall back to CPU for any operator MPS doesn't implement;
        # must be set before torch initializes the MPS backend.
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        import torch
        from lightglue import LightGlue, SuperPoint
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        _deep_models['device'] = device
        _deep_models['extractor'] = SuperPoint(
            max_num_keypoints=_SUPERPOINT_MAX_KEYPOINTS).eval().to(device)
        _deep_models['matcher'] = LightGlue(
            features='superpoint').eval().to(device)
    return _deep_models


# project
def extract_features_superpoint(img):
    """Detect SuperPoint keypoints and descriptors on a grayscale image.

    Deep-learning counterpart of features.extract_features(). The image is
    processed at native resolution (no resize), so keypoint coordinates are
    directly in input-pixel units.

    Args:
        img: Grayscale uint8 image (HxW numpy array).

    Returns:
        keypoints: List of cv2.KeyPoint (detection confidence in .response).
        feats: Raw LightGlue feature dict (torch tensors, incl. descriptors);
            pass to match_features_lightglue() wherever the classical path
            would pass a descriptor array.
    """
    import torch
    fe = _deep_frontend()
    tensor = torch.from_numpy(img).float()[None, None] / 255.0
    with torch.no_grad():
        feats = fe['extractor'].extract(tensor.to(fe['device']), resize=None)
    pts = feats['keypoints'][0].cpu().numpy()
    scores = feats['keypoint_scores'][0].cpu().numpy()
    keypoints = [cv2.KeyPoint(x=float(x), y=float(y),
                              size=_SUPERPOINT_KEYPOINT_SIZE,
                              response=float(s))
                 for (x, y), s in zip(pts, scores)]
    return keypoints, feats


# project
def match_features_lightglue(feats0, feats1):
    """Match two SuperPoint feature dicts with LightGlue.

    Deep-learning counterpart of features.match_descriptors(). LightGlue
    attends over both keypoint sets jointly and enforces mutual consistency
    internally, so its output needs no ratio test or cross-check.

    Args:
        feats0, feats1: Feature dicts from extract_features_superpoint().

    Returns:
        Flat list of cv2.DMatch — queryIdx into feats0's keypoints, trainIdx
        into feats1's, distance = 1 − matching confidence (so lower is
        better, consistent with classical DMatch semantics).
    """
    import torch
    fe = _deep_frontend()
    with torch.no_grad():
        out = fe['matcher']({'image0': feats0, 'image1': feats1})
    pairs = out['matches'][0].cpu().numpy()    # (M, 2) keypoint index pairs
    scores = out['scores'][0].cpu().numpy()    # (M,) confidence in [0, 1]
    return [cv2.DMatch(int(i0), int(i1), float(1.0 - s))
            for (i0, i1), s in zip(pairs, scores)]


# project
def match_stereo_pair_superpoint(img_left, img_right):
    """Deep-learning counterpart of features.match_stereo_pair().

    Also returns the raw feature dicts so callers can reuse them for the
    cross-frame (temporal) matching without re-extracting the images.

    Returns:
        kp_left, kp_right: Keypoint lists from the two images.
        matches: Flat list of cv2.DMatch (LightGlue output).
        feats_left, feats_right: Feature dicts for match_features_lightglue().
    """
    kp_l, feats_l = extract_features_superpoint(img_left)
    kp_r, feats_r = extract_features_superpoint(img_right)
    matches = match_features_lightglue(feats_l, feats_r)
    return kp_l, kp_r, matches, feats_l, feats_r
