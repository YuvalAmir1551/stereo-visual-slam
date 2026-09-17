"""GT-anomaly check with four front-ends: SIFT / AKAZE / ORB / SP+LG.

For each test window (kf_a -> kf_b): stereo features at both keyframes,
temporal match, 4-view consensus, triangulate at a, RANSAC-PnP -> relative
pose. All four front-ends share the identical geometry stage. Report each
pose's rotation deviation from GT and the max pairwise spread between the
four estimates.
"""
import sys, os, pickle, json
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the code/ dir
sys.path.insert(0, os.path.join(CODE, "deep_frontend"))
sys.path.insert(0, CODE)

import numpy as np
import cv2

from dataset import read_images, read_cameras, read_poses
from features import (extract_features, match_descriptors,
                      rectified_stereo_filter, consensus_matches,
                      pts_from_matches)
from dl_features import extract_features_superpoint, match_features_lightglue
from geometry import triangulate_cv2
from bundle import relative_extrinsic
from pnp import ransac_pnp
from ex7 import LC_CACHE

Y_T, X_MIN, PIX_T = 2.0, 1.0, 2.0
SUSPECT = [17, 18, 174, 175, 203, 204, 205, 246, 271, 314]
CONTROL = [30, 50, 70, 90, 120, 150, 190, 220, 260, 280]

K, m1, m2 = read_cameras()
P_left, P_right = K @ m1, K @ m2
gt = read_poses()
with open(LC_CACHE, 'rb') as f:
    keyframes = pickle.load(f)['keyframes']

def stereo_classical(img_l, img_r, det):
    kp_l, des_l = extract_features(img_l, detector=det)
    kp_r, des_r = extract_features(img_r, detector=det)
    m = match_descriptors(des_l, des_r, detector=det)
    inl, _, _ = rectified_stereo_filter(kp_l, kp_r, m, y_threshold=Y_T,
                                        x_min_disparity=X_MIN)
    return kp_l, des_l, kp_r, inl

def pose_classical(det, kf_a, kf_b):
    il_a, ir_a = read_images(kf_a); il_b, ir_b = read_images(kf_b)
    kl0, dl0, kr0, in0 = stereo_classical(il_a, ir_a, det)
    kl1, dl1, kr1, in1 = stereo_classical(il_b, ir_b, det)
    cross = match_descriptors(dl0, dl1, detector=det)
    i0, i1 = consensus_matches(in0, in1, cross)
    if len(i0) < 8:
        return None, len(i0)
    m0 = [in0[k] for k in i0]; m1_ = [in1[k] for k in i1]
    pl0, pr0 = pts_from_matches(kl0, kr0, m0)
    pl1, pr1 = pts_from_matches(kl1, kr1, m1_)
    X0 = triangulate_cv2(P_left, P_right, pl0, pr0)
    Rt, mask = ransac_pnp(X0, pl0, pr0, pl1, pr1, K, m2, threshold=PIX_T,
                          rng=np.random.default_rng(0))
    return Rt, int(mask.sum()) if mask is not None else 0

_sp = {}
def sp_stereo(kf):
    if kf in _sp:
        return _sp[kf]
    img_l, img_r = read_images(kf)
    kp_l, f_l = extract_features_superpoint(img_l)
    kp_r, f_r = extract_features_superpoint(img_r)
    m = match_features_lightglue(f_l, f_r)
    inl, _, _ = rectified_stereo_filter(kp_l, kp_r, m, y_threshold=Y_T,
                                        x_min_disparity=X_MIN)
    _sp[kf] = (kp_l, f_l, kp_r, inl)
    return _sp[kf]

def pose_sp(kf_a, kf_b):
    kl0, f0, kr0, in0 = sp_stereo(kf_a)
    kl1, f1, kr1, in1 = sp_stereo(kf_b)
    cross = match_features_lightglue(f0, f1)
    i0, i1 = consensus_matches(in0, in1, cross)
    if len(i0) < 8:
        return None, len(i0)
    m0 = [in0[k] for k in i0]; m1_ = [in1[k] for k in i1]
    pl0, pr0 = pts_from_matches(kl0, kr0, m0)
    pl1, pr1 = pts_from_matches(kl1, kr1, m1_)
    X0 = triangulate_cv2(P_left, P_right, pl0, pr0)
    Rt, mask = ransac_pnp(X0, pl0, pr0, pl1, pr1, K, m2, threshold=PIX_T,
                          rng=np.random.default_rng(0))
    return Rt, int(mask.sum()) if mask is not None else 0

def rot_diff(Ra, Rb):
    rvec, _ = cv2.Rodrigues(Ra @ Rb.T)
    return float(np.degrees(np.linalg.norm(rvec)))

rows = []
for b in SUSPECT + CONTROL:
    kf_a, kf_b = keyframes[b], keyframes[b + 1]
    g = relative_extrinsic(gt[kf_a], gt[kf_b])
    poses = {}
    for det in ('SIFT', 'AKAZE', 'ORB'):
        Rt, n_in = pose_classical(det, kf_a, kf_b)
        if Rt is not None:
            poses[det] = Rt
    Rt, n_in = pose_sp(kf_a, kf_b)
    if Rt is not None:
        poses['SP+LG'] = Rt
    errs = {d: rot_diff(p[:, :3], g[:, :3]) for d, p in poses.items()}
    dets = list(poses)
    spread = max((rot_diff(poses[a][:, :3], poses[b_][:, :3])
                  for i, a in enumerate(dets) for b_ in dets[i+1:]), default=np.nan)
    rows.append(dict(b=b, suspect=b in SUSPECT, n=len(poses),
                     spread=spread, errs=errs))
    e = " ".join(f"{d}:{v:.2f}" for d, v in errs.items())
    print(f"b={b:3d} {'SUSPECT' if b in SUSPECT else 'control'} "
          f"spread={spread:5.2f}  vsGT: {e}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "gt_multi_detector.json"), "w") as f:
    json.dump(rows, f, indent=1)

sus = [r for r in rows if r['suspect'] and r['n'] == 4]
ctl = [r for r in rows if not r['suspect'] and r['n'] == 4]
def agg(rs, key):
    return np.median([r[key] if key == 'spread' else np.median(list(r['errs'].values())) for r in rs])
print(f"\nSUSPECT (n={len(sus)}): median spread {agg(sus,'spread'):.2f} deg, "
      f"median vs-GT {agg(sus,'gt'):.2f} deg")
print(f"control (n={len(ctl)}): median spread {agg(ctl,'spread'):.2f} deg, "
      f"median vs-GT {agg(ctl,'gt'):.2f} deg")
