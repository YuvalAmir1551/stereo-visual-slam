"""Oracle-retrieval ceiling: how good could loop closure get?

Emulates a perfect place-recognition candidate generator: propose ALL
GT-true revisit pairs (<7 m, >10 kf apart), verify with the existing
AKAZE consensus match + acceptance rule, add mini-bundle factors for the
survivors, optimize. Compares trajectory error against the current
Mahalanobis-gated result (12 LCs) and no-LC baseline.
"""
import os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import gtsam
import cv2

import loop_closure as lc
from run_loop_closure import build_pose_graph, RELATIVES_CACHE, LC_CACHE
from dataset import read_cameras, read_poses
from geometry import camera_center
from bundle import stereo_calibration, cam_key, Rt_to_gtsam_pose

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset', 'sequences', '00', 'report_data')
CLOSE_M = 7.0

K, m1, m2 = read_cameras()
P_left, P_right = K @ m1, K @ m2
K_stereo = stereo_calibration(K, -m2[0, 3])

with open(LC_CACHE, 'rb') as f:
    d = pickle.load(f)
keyframes, accepted_ak = d['keyframes'], d['accepted']
with open(RELATIVES_CACHE, 'rb') as f:
    d = pickle.load(f)
rel_poses, rel_covs = d['rel_poses'], d['rel_covs']
N = len(keyframes)

gt_poses = read_poses()
centres = np.array([camera_center(gt_poses[k]) for k in keyframes])
gt_R_c2w = [gt_poses[k][:, :3].T for k in keyframes]

pos_pairs = [(n, i) for n in range(lc.K_SKIP_RECENT_KFS, N)
             for i in range(0, n - lc.K_SKIP_RECENT_KFS)
             if np.linalg.norm(centres[n] - centres[i]) < CLOSE_M]
print(f"{len(pos_pairs)} oracle candidate pairs")

with open(lc.KF_FEATURES_CACHE, 'rb') as f:
    d = pickle.load(f)
assert d['keyframes'] == keyframes
ak_feats = d['features']

oracle_lcs = []
t0 = time.time()
for j, (n, i) in enumerate(pos_pairs):
    res = lc.loop_consensus_match(keyframes[n], keyframes[i], ak_feats,
                                  K, m2, P_left, P_right)
    if res is None:
        continue
    Rt_rel, mask, cross, q, t = res
    if Rt_rel is None or mask is None:
        continue
    n_in = int(mask.sum())
    if n_in / len(cross) < lc.MIN_INLIER_PCT:
        continue
    rel_pose, rel_cov = lc.loop_mini_bundle(
        keyframes[n], keyframes[i], ak_feats, cross, q, t, mask,
        K_stereo, Rt_rel)
    if rel_pose is None:
        continue
    oracle_lcs.append(dict(n=n, i=i, rel_pose=rel_pose, rel_cov=rel_cov,
                           inliers=n_in))
    if (j + 1) % 10 == 0:
        print(f"  {j + 1}/{len(pos_pairs)} pairs, {len(oracle_lcs)} verified "
              f"({time.time() - t0:.0f}s)")
print(f"Oracle LCs verified: {len(oracle_lcs)}/{len(pos_pairs)}")

def eval_traj(lc_list, pose_getter):
    graph, initial = build_pose_graph(keyframes, rel_poses, rel_covs)
    result = lc.optimize(graph, initial)
    for rec in lc_list:
        noise = gtsam.noiseModel.Gaussian.Covariance(rec['rel_cov'])
        graph.add(gtsam.BetweenFactorPose3(
            cam_key(keyframes[rec['i']]), cam_key(keyframes[rec['n']]),
            pose_getter(rec), noise))
        result = lc.optimize(graph, result)
    cent = np.array([result.atPose3(cam_key(k)).translation()
                     for k in keyframes])
    loc = np.linalg.norm(cent - centres, axis=1)
    ang = []
    for j, k in enumerate(keyframes):
        R_est = result.atPose3(cam_key(k)).rotation().matrix()
        rvec, _ = cv2.Rodrigues(R_est @ gt_R_c2w[j].T)
        ang.append(np.linalg.norm(rvec) * 180 / np.pi)
    return cent, loc, np.array(ang)

print("Optimizing pose graphs...")
cent_or, loc_or, ang_or = eval_traj(oracle_lcs, lambda r: r['rel_pose'])
cent_ak, loc_ak, ang_ak = eval_traj(accepted_ak,
                                    lambda r: Rt_to_gtsam_pose(r['rel_pose']))
cent_no, loc_no, ang_no = eval_traj([], None)

np.savez(f"{SCRATCH}/lc_ceiling.npz",
         keyframes=np.array(keyframes), gt=centres,
         cent_no=cent_no, cent_ak=cent_ak, cent_or=cent_or,
         loc_no=loc_no, loc_ak=loc_ak, loc_or=loc_or,
         ang_no=ang_no, ang_ak=ang_ak, ang_or=ang_or,
         n_oracle=len(oracle_lcs))
cov_kfs = sorted({r['n'] for r in oracle_lcs} | {r['i'] for r in oracle_lcs})
print(f"\noracle LC pairs span {len(cov_kfs)} distinct keyframes "
      f"(baseline: {len({r['n'] for r in accepted_ak} | {r['i'] for r in accepted_ak})})")
for label, loc, ang in (('no LC          ', loc_no, ang_no),
                        (f'gate    ({len(accepted_ak):2d} LC)', loc_ak, ang_ak),
                        (f'oracle  ({len(oracle_lcs):2d} LC)', loc_or, ang_or)):
    print(f"{label}: loc median {np.median(loc):6.2f}m  max {loc.max():6.2f}m  "
          f"final {loc[-1]:6.2f}m | ang median {np.median(ang):5.2f}deg "
          f"max {ang.max():5.2f}deg")
