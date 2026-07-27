"""Head-to-head: classical AKAZE front-end vs SuperPoint+LightGlue.

For frame transitions (i, i+1) sampled evenly across KITTI sequence 00, run
both front-ends through the SAME downstream pipeline stages used by ex4's
build_db: rectified stereo filter -> temporal match -> 4-view consensus ->
triangulation -> ransac_pnp, then compare pose estimates to ground truth.

Both paths use identical filter/RANSAC parameters (Y_THRESHOLD=2.0,
X_MIN_DISPARITY=1.0, reprojection threshold 2.0 px, same RNG seed).
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import cv2

from dataset import read_images, read_cameras, read_poses
from features import (extract_features, match_descriptors, rectified_stereo_filter, consensus_matches)
from dl_features import (extract_features_superpoint, match_features_lightglue, _deep_frontend)
from geometry import triangulate_cv2
from pnp import ransac_pnp

Y_THRESHOLD = 2.0       # px  (ex4 values)
X_MIN_DISPARITY = 1.0   # px
DETECTOR = 'AKAZE'
STEP = 50               # sample every 50th transition -> ~66 samples
OUT_JSON = os.path.join(SCRATCH, "frontend_comparison.json")

K, m1, m2 = read_cameras()
P_left, P_right = K @ m1, K @ m2
gt = read_poses()
n_total = len(gt)
frames = list(range(0, n_total - 1, STEP))
print(f"{len(frames)} transitions, step {STEP}, {n_total} frames total")

_deep_frontend()  # warm up model load before timing


def stereo_classical(img_l, img_r):
    kp_l, des_l = extract_features(img_l, detector=DETECTOR)
    kp_r, des_r = extract_features(img_r, detector=DETECTOR)
    matches = match_descriptors(des_l, des_r, detector=DETECTOR)
    inl, _, _ = rectified_stereo_filter(kp_l, kp_r, matches,
                                        y_threshold=Y_THRESHOLD,
                                        x_min_disparity=X_MIN_DISPARITY)
    return dict(kp_l=kp_l, kp_r=kp_r, des_l=des_l, n_matches=len(matches), inl=inl)


def stereo_deep(img_l, img_r):
    kp_l, f_l = extract_features_superpoint(img_l)
    kp_r, f_r = extract_features_superpoint(img_r)
    matches = match_features_lightglue(f_l, f_r)
    inl, _, _ = rectified_stereo_filter(kp_l, kp_r, matches,
                                        y_threshold=Y_THRESHOLD,
                                        x_min_disparity=X_MIN_DISPARITY)
    return dict(kp_l=kp_l, kp_r=kp_r, feats_l=f_l, n_matches=len(matches), inl=inl)


def rel_gt(i):
    """GT relative extrinsic frame i -> i+1: x_{i+1} = R x_i + t."""
    Ri, ti = gt[i][:, :3], gt[i][:, 3]
    Rj, tj = gt[i + 1][:, :3], gt[i + 1][:, 3]
    R = Rj @ Ri.T
    t = tj - R @ ti
    return R, t


def pose_errors(Rt_est, i):
    Rg, tg = rel_gt(i)
    dR = Rt_est[:, :3] @ Rg.T
    rvec, _ = cv2.Rodrigues(dR)
    ang = float(np.linalg.norm(rvec) * 180 / np.pi)
    loc = float(np.linalg.norm(Rt_est[:, 3] - tg))
    return ang, loc


def run_transition(i, method):
    rng = np.random.default_rng(0)   # same seed for both methods
    img_l0, img_r0 = read_images(i)
    img_l1, img_r1 = read_images(i + 1)

    t0 = time.time()
    if method == 'AKAZE':
        s0 = stereo_classical(img_l0, img_r0)
        s1 = stereo_classical(img_l1, img_r1)
        t_frontend = time.time() - t0
        t0 = time.time()
        cross = match_descriptors(s0['des_l'], s1['des_l'], detector=DETECTOR)
    else:
        s0 = stereo_deep(img_l0, img_r0)
        s1 = stereo_deep(img_l1, img_r1)
        t_frontend = time.time() - t0
        t0 = time.time()
        cross = match_features_lightglue(s0['feats_l'], s1['feats_l'])
    t_temporal = time.time() - t0

    i0, i1 = consensus_matches(s0['inl'], s1['inl'], cross)
    rec = dict(frame=i,
               kp=len(s0['kp_l']),
               stereo_matches=s0['n_matches'],
               stereo_inl=len(s0['inl']),
               stereo_inl_pct=100.0 * len(s0['inl']) / max(s0['n_matches'], 1),
               temporal=len(cross),
               consensus=len(i0),
               t_frontend=t_frontend, t_temporal=t_temporal)

    if len(i0) < 4:
        rec.update(pnp_inl_pct=0.0, ang_err=None, loc_err=None)
        return rec

    m0 = [s0['inl'][k] for k in i0]
    m1_ = [s1['inl'][k] for k in i1]
    pl0 = np.array([s0['kp_l'][m.queryIdx].pt for m in m0])
    pr0 = np.array([s0['kp_r'][m.trainIdx].pt for m in m0])
    pl1 = np.array([s1['kp_l'][m.queryIdx].pt for m in m1_])
    pr1 = np.array([s1['kp_r'][m.trainIdx].pt for m in m1_])
    X0 = triangulate_cv2(P_left, P_right, pl0, pr0)

    Rt, mask = ransac_pnp(X0, pl0, pr0, pl1, pr1, K, m2, rng=rng)
    if Rt is None:
        rec.update(pnp_inl_pct=0.0, ang_err=None, loc_err=None)
        return rec
    ang, loc = pose_errors(Rt, i)
    rec.update(pnp_inl_pct=100.0 * mask.sum() / len(mask), ang_err=ang, loc_err=loc)
    return rec


results = {'AKAZE': [], 'SP+LG': []}
t_start = time.time()
for n, i in enumerate(frames):
    for method in results:
        results[method].append(run_transition(i, method))
    if n % 10 == 0:
        print(f"  {n + 1}/{len(frames)} transitions done ({time.time() - t_start:.0f}s)")

with open(OUT_JSON, 'w') as f:
    json.dump(results, f)

def summarize(recs):
    g = lambda key: np.array([r[key] for r in recs if r[key] is not None], dtype=float)
    return {
        'keypoints/img': g('kp').mean(),
        'stereo matches': g('stereo_matches').mean(),
        'stereo inliers': g('stereo_inl').mean(),
        'stereo inlier %': g('stereo_inl_pct').mean(),
        'temporal matches': g('temporal').mean(),
        'consensus (4-view)': g('consensus').mean(),
        'PnP inlier %': g('pnp_inl_pct').mean(),
        'rot err median (deg)': float(np.median(g('ang_err'))),
        'rot err p95 (deg)': float(np.percentile(g('ang_err'), 95)),
        'loc err median (m)': float(np.median(g('loc_err'))),
        'loc err p95 (m)': float(np.percentile(g('loc_err'), 95)),
        'front-end s/frame-pair': g('t_frontend').mean() / 2,
        'temporal match s': g('t_temporal').mean(),
    }

print(f"\n{'metric':<26}{'AKAZE':>12}{'SP+LG':>12}")
sa, sd = summarize(results['AKAZE']), summarize(results['SP+LG'])
for k in sa:
    print(f"{k:<26}{sa[k]:>12.3f}{sd[k]:>12.3f}")
