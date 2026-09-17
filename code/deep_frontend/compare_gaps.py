"""Front-end comparison at increasing frame gaps (wide-baseline robustness).

Same pipeline as compare_frontends.py but matching frame i to i+gap for
gap in (1, 5, 10, 20, 40). This emulates keyframe-to-keyframe and
loop-closure-like viewpoint changes, where learned matching should
outperform binary-descriptor best-matching.
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

Y_THRESHOLD = 2.0
X_MIN_DISPARITY = 1.0
DETECTOR = 'AKAZE'
GAPS = [1, 5, 10, 20, 40]
STARTS = list(range(0, 3200, 100))   # 32 start frames
OUT_JSON = os.path.join(SCRATCH, "gap_comparison.json")

K, m1, m2 = read_cameras()
P_left, P_right = K @ m1, K @ m2
gt = read_poses()
_deep_frontend()

_cache = {}  # (frame, method) -> stereo dict; reused across gaps

def stereo(frame, method):
    key = (frame, method)
    if key in _cache:
        return _cache[key]
    img_l, img_r = read_images(frame)
    if method == 'AKAZE':
        kp_l, des_l = extract_features(img_l, detector=DETECTOR)
        kp_r, des_r = extract_features(img_r, detector=DETECTOR)
        matches = match_descriptors(des_l, des_r, detector=DETECTOR)
        extra = dict(des_l=des_l)
    else:
        kp_l, f_l = extract_features_superpoint(img_l)
        kp_r, f_r = extract_features_superpoint(img_r)
        matches = match_features_lightglue(f_l, f_r)
        extra = dict(feats_l=f_l)
    inl, _, _ = rectified_stereo_filter(kp_l, kp_r, matches,
                                        y_threshold=Y_THRESHOLD,
                                        x_min_disparity=X_MIN_DISPARITY)
    rec = dict(kp_l=kp_l, kp_r=kp_r, inl=inl, **extra)
    _cache[key] = rec
    return rec


def rel_gt(i, j):
    Ri, ti = gt[i][:, :3], gt[i][:, 3]
    Rj, tj = gt[j][:, :3], gt[j][:, 3]
    R = Rj @ Ri.T
    return R, tj - R @ ti


def run(i, j, method):
    s0, s1 = stereo(i, method), stereo(j, method)
    if method == 'AKAZE':
        cross = match_descriptors(s0['des_l'], s1['des_l'], detector=DETECTOR)
    else:
        cross = match_features_lightglue(s0['feats_l'], s1['feats_l'])
    i0, i1 = consensus_matches(s0['inl'], s1['inl'], cross)
    rec = dict(start=i, gap=j - i, consensus=len(i0))
    if len(i0) < 4:
        rec.update(pnp_inl_pct=None, ang_err=None, loc_err=None, failed=True)
        return rec
    m0 = [s0['inl'][k] for k in i0]
    m1_ = [s1['inl'][k] for k in i1]
    pl0 = np.array([s0['kp_l'][m.queryIdx].pt for m in m0])
    pr0 = np.array([s0['kp_r'][m.trainIdx].pt for m in m0])
    pl1 = np.array([s1['kp_l'][m.queryIdx].pt for m in m1_])
    pr1 = np.array([s1['kp_r'][m.trainIdx].pt for m in m1_])
    X0 = triangulate_cv2(P_left, P_right, pl0, pr0)
    Rt, mask = ransac_pnp(X0, pl0, pr0, pl1, pr1, K, m2,
                          rng=np.random.default_rng(0))
    if Rt is None:
        rec.update(pnp_inl_pct=None, ang_err=None, loc_err=None, failed=True)
        return rec
    Rg, tg = rel_gt(i, j)
    dR = Rt[:, :3] @ Rg.T
    rvec, _ = cv2.Rodrigues(dR)
    rec.update(pnp_inl_pct=100.0 * mask.sum() / len(mask),
               ang_err=float(np.linalg.norm(rvec) * 180 / np.pi),
               loc_err=float(np.linalg.norm(Rt[:, 3] - tg)),
               failed=False)
    return rec


results = {'AKAZE': [], 'SP+LG': []}
t0 = time.time()
for n, s in enumerate(STARTS):
    for gap in GAPS:
        if s + gap >= len(gt):
            continue
        for method in results:
            results[method].append(run(s, s + gap, method))
    _cache.clear()  # frames are reused only within one start
    # keep the start frame cached across gaps only; safe to clear per start
    if n % 8 == 0:
        print(f"  start {n + 1}/{len(STARTS)} ({time.time() - t0:.0f}s)")

with open(OUT_JSON, 'w') as f:
    json.dump(results, f)

print(f"\n{'gap':>4} {'metric':<22}{'AKAZE':>10}{'SP+LG':>10}")
for gap in GAPS:
    for method in ('AKAZE', 'SP+LG'):
        rs = [r for r in results[method] if r['gap'] == gap]
        cons = np.mean([r['consensus'] for r in rs])
        ok = [r for r in rs if not r.get('failed')]
        fail = len(rs) - len(ok)
        ang = np.median([r['ang_err'] for r in ok]) if ok else float('nan')
        loc = np.median([r['loc_err'] for r in ok]) if ok else float('nan')
        inl = np.mean([r['pnp_inl_pct'] for r in ok]) if ok else float('nan')
        print(f"{gap:>4} {method:<8} cons={cons:7.1f} inl%={inl:5.1f} "
              f"rot_med={ang:7.3f}deg loc_med={loc:7.3f}m fails={fail}/{len(rs)}")
