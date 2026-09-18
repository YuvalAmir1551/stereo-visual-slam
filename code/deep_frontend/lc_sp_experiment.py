"""Loop closure with SuperPoint+LightGlue vs the AKAZE baseline.

Everything upstream is held fixed (same tracking DB, bundle relatives,
keyframes, pose graph, Mahalanobis pre-filter, mini-bundle, thresholds
from loop_closure.py). ONLY the consensus-match front-end changes:
  AKAZE  : best-match over stereo-inlier AKAZE descriptors (baseline,
           loaded from the existing loop_closures.pkl cache)
  SP+LG  : LightGlue match of full SuperPoint feature sets, restricted
           to stereo-inlier keypoints.

Acceptance rule is kept semantically identical: RANSAC inliers must be
>= MIN_INLIER_PCT of kf_i's stereo-inlier count (for AKAZE, len(cross)
equals kf_i's stereo-inlier count because best-match emits one match
per query descriptor).
"""
import os, sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import gtsam
import cv2

import loop_closure as lc
from run_loop_closure import build_pose_graph, RELATIVES_CACHE, LC_CACHE
from dataset import read_images, read_cameras, read_poses
from geometry import triangulate_linear_lsq, camera_center
from features import (rectified_stereo_filter)
from dl_features import (extract_features_superpoint, match_features_lightglue)
from pnp import ransac_pnp
from bundle import stereo_calibration, cam_key, gtsam_pose_to_Rt, Rt_to_gtsam_pose

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset', 'sequences', '00', 'report_data')
SP_LC_CACHE = f"{SCRATCH}/loop_closures_sp.pkl"

K, m1, m2 = read_cameras()
P_left, P_right = K @ m1, K @ m2
K_stereo = stereo_calibration(K, -m2[0, 3])

with open(LC_CACHE, 'rb') as f:
    d = pickle.load(f)
keyframes, accepted_ak = d['keyframes'], d['accepted']
with open(RELATIVES_CACHE, 'rb') as f:
    d = pickle.load(f)
rel_poses, rel_covs = d['rel_poses'], d['rel_covs']
assert len(rel_poses) == len(keyframes) - 1
print(f"{len(keyframes)} keyframes; AKAZE baseline: {len(accepted_ak)} LCs")

# ---------------------------------------------------------------------------
# SP+LG keyframe features (RAM cache only - does not touch project caches)
# ---------------------------------------------------------------------------
kf_feats = {}    # kf -> (None, pts_l, pts_r) stereo-inlier arrays (mini-bundle format)
sp_feats = {}    # kf -> LightGlue feature dict of the LEFT image
rowmap = {}      # kf -> array: left-kp index -> stereo-inlier row (or -1)

def build_sp_features():
    t0 = time.time()
    for k, kf in enumerate(keyframes):
        img_l, img_r = read_images(kf)
        kp_l, f_l = extract_features_superpoint(img_l)
        kp_r, f_r = extract_features_superpoint(img_r)
        matches = match_features_lightglue(f_l, f_r)
        inl, _, _ = rectified_stereo_filter(
            kp_l, kp_r, matches,
            y_threshold=lc.Y_THRESHOLD, x_min_disparity=lc.X_MIN_DISPARITY)
        pts_l = np.array([kp_l[m.queryIdx].pt for m in inl]).reshape(-1, 2)
        pts_r = np.array([kp_r[m.trainIdx].pt for m in inl]).reshape(-1, 2)
        rm = -np.ones(len(kp_l), dtype=int)
        for row, m in enumerate(inl):
            rm[m.queryIdx] = row
        kf_feats[kf] = (None, pts_l, pts_r)
        sp_feats[kf] = f_l
        rowmap[kf] = rm
        if (k + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  SP features {k + 1}/{len(keyframes)} ({dt / (k + 1):.2f}s each)")

def consensus_sp(kf_n, kf_i):
    """SP+LG counterpart of lc.loop_consensus_match. Returns
    (Rt_rel, mask, denom, q, t) or None."""
    _, pl_i, pr_i = kf_feats[kf_i]
    _, pl_n, pr_n = kf_feats[kf_n]
    if len(pl_i) < 4 or len(pl_n) < 4:
        return None
    pairs = match_features_lightglue(sp_feats[kf_i], sp_feats[kf_n])
    q, t = [], []
    for m in pairs:
        ri = rowmap[kf_i][m.queryIdx]
        rn = rowmap[kf_n][m.trainIdx]
        if ri >= 0 and rn >= 0:
            q.append(ri); t.append(rn)
    if len(q) < 4:
        return None
    q, t = np.array(q), np.array(t)
    X_i = triangulate_linear_lsq(P_left, P_right, pl_i[q], pr_i[q])
    Rt_rel, mask = ransac_pnp(
        X_i, pl_i[q], pr_i[q], pl_n[t], pr_n[t],
        K, m2, threshold=lc.PIX_THRESHOLD, rng=np.random.default_rng(0))
    denom = len(pl_i)   # same denominator semantics as the AKAZE path
    return Rt_rel, mask, denom, q, t

# ---------------------------------------------------------------------------
# LC search driver — identical logic to lc.run_loop_closure_search /
# lc._single_pass, with only the consensus call swapped.
# ---------------------------------------------------------------------------
def run_lc_search_sp(graph, result):
    loop_edges, accepted, tried = [], [], set()
    nx_g = lc.pose_graph_to_nx(len(keyframes), rel_covs, loop_edges=loop_edges)
    for p in range(lc.MAX_PASSES):
        print(f"  Pass {p + 1}/{lc.MAX_PASSES}:")
        t0 = time.time()
        candidates = lc._score_candidates(result, nx_g, keyframes, tried)
        print(f"    {len(candidates)} candidates pass Mahalanobis "
              f"({time.time() - t0:.1f}s)")
        added = 0
        for idx, (m, n, i) in enumerate(candidates):
            tried.add((n, i))
            res = consensus_sp(keyframes[n], keyframes[i])
            if res is None:
                continue
            Rt_rel, mask, denom, q, t = res
            if Rt_rel is None or mask is None:
                continue
            n_in = int(mask.sum())
            if n_in / denom < lc.MIN_INLIER_PCT:
                continue
            rel_pose, rel_cov = lc.loop_mini_bundle(
                keyframes[n], keyframes[i], kf_feats,
                None, q, t, mask, K_stereo, Rt_rel)
            if rel_pose is None:
                continue
            noise = gtsam.noiseModel.Gaussian.Covariance(rel_cov)
            graph.add(gtsam.BetweenFactorPose3(
                cam_key(keyframes[i]), cam_key(keyframes[n]), rel_pose, noise))
            result = lc.optimize(graph, result)
            loop_edges.append((i, n, rel_cov))
            nx_g.add_edge(i, n, cov=rel_cov)
            accepted.append({
                'n': n, 'i': i,
                'kf_n': keyframes[n], 'kf_i': keyframes[i],
                'rel_pose': gtsam_pose_to_Rt(rel_pose),
                'rel_cov': rel_cov,
                'inliers': n_in, 'cross_pre': denom, 'mahalanobis': m,
                'matches': len(q),
            })
            added += 1
            print(f"    LC {len(accepted):3d}: c_{n}->c_{i} "
                  f"(frames {keyframes[n]}<->{keyframes[i]}, "
                  f"{n_in}/{len(q)} match inliers, mah {m:.1f})")
            if (idx + 1) % 200 == 0:
                print(f"    ...{idx + 1}/{len(candidates)} tried, {added} added")
        print(f"  Pass {p + 1} added {added} ({len(accepted)} total)")
        if added == 0:
            break
    return accepted, result

# ---------------------------------------------------------------------------
# Run (cached in scratchpad)
# ---------------------------------------------------------------------------
if os.path.exists(SP_LC_CACHE):
    with open(SP_LC_CACHE, 'rb') as f:
        d = pickle.load(f)
    accepted_sp = d['accepted']
    print(f"Loaded SP LC cache: {len(accepted_sp)} LCs")
else:
    print("Extracting SP+LG keyframe features...")
    build_sp_features()
    print("Running SP+LG loop-closure search...")
    graph, initial = build_pose_graph(keyframes, rel_poses, rel_covs)
    result = lc.optimize(graph, initial)
    accepted_sp, _ = run_lc_search_sp(graph, result)
    with open(SP_LC_CACHE, 'wb') as f:
        pickle.dump({'keyframes': keyframes, 'accepted': accepted_sp}, f)
    print(f"Saved {len(accepted_sp)} SP LCs to {SP_LC_CACHE}")

# ---------------------------------------------------------------------------
# Evaluation: trajectory error with each LC set
# ---------------------------------------------------------------------------
gt_poses = read_poses()
gt_centres = np.array([camera_center(gt_poses[k]) for k in keyframes])
gt_R_c2w = [gt_poses[k][:, :3].T for k in keyframes]

def eval_traj(accepted):
    graph, initial = build_pose_graph(keyframes, rel_poses, rel_covs)
    result = lc.optimize(graph, initial)
    for rec in accepted:
        noise = gtsam.noiseModel.Gaussian.Covariance(rec['rel_cov'])
        graph.add(gtsam.BetweenFactorPose3(
            cam_key(keyframes[rec['i']]), cam_key(keyframes[rec['n']]),
            Rt_to_gtsam_pose(rec['rel_pose']), noise))
        result = lc.optimize(graph, result)
    cent = np.array([result.atPose3(cam_key(k)).translation() for k in keyframes])
    loc_err = np.linalg.norm(cent - gt_centres, axis=1)
    ang_err = []
    for j, k in enumerate(keyframes):
        R_est = result.atPose3(cam_key(k)).rotation().matrix()
        rvec, _ = cv2.Rodrigues(R_est @ gt_R_c2w[j].T)
        ang_err.append(np.linalg.norm(rvec) * 180 / np.pi)
    return cent, loc_err, np.array(ang_err)

print("\nEvaluating trajectories (no LC / AKAZE / SP+LG)...")
cent_no, loc_no, ang_no = eval_traj([])
cent_ak, loc_ak, ang_ak = eval_traj(accepted_ak)
cent_sp, loc_sp, ang_sp = eval_traj(accepted_sp)

np.savez(f"{SCRATCH}/lc_eval.npz",
         keyframes=np.array(keyframes), gt=gt_centres,
         cent_no=cent_no, cent_ak=cent_ak, cent_sp=cent_sp,
         loc_no=loc_no, loc_ak=loc_ak, loc_sp=loc_sp,
         ang_no=ang_no, ang_ak=ang_ak, ang_sp=ang_sp)

def lc_stats(accepted, label):
    if not accepted:
        print(f"{label}: 0 LCs"); return
    inl = np.array([r['inliers'] for r in accepted])
    pct = np.array([100.0 * r['inliers'] / r['cross_pre'] for r in accepted])
    pairs = sorted({(r['kf_i'], r['kf_n']) for r in accepted})
    print(f"{label}: {len(accepted)} LCs, median inliers {np.median(inl):.0f}, "
          f"median inlier% {np.median(pct):.1f}")
    print(f"  pairs: {pairs}")

print()
lc_stats(accepted_ak, 'AKAZE')
lc_stats(accepted_sp, 'SP+LG')
for label, loc, ang in (('no LC ', loc_no, ang_no),
                        ('AKAZE ', loc_ak, ang_ak),
                        ('SP+LG ', loc_sp, ang_sp)):
    print(f"{label}: loc err median {np.median(loc):6.2f}m  max {loc.max():6.2f}m  "
          f"final {loc[-1]:6.2f}m | ang err median {np.median(ang):5.2f}deg  "
          f"max {ang.max():5.2f}deg")
