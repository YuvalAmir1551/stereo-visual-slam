"""Wide-baseline anchor bundle refinement — the adopted deep-learning
improvement (report section 4.3), applied one bundle window at a time.

WHAT PROBLEM THIS SOLVES
------------------------
The odometry drift on KITTI 00 is dominated by ROTATION error, and that error
is concentrated in the fast-turn bundle windows. Inside a turn the features are
tracked frame-by-frame, so each small tracking error compounds and the tracks
"slide" systematically across the window. Bundle adjustment cannot see this
bias, because within the window the sliding stays self-consistent (the
reprojection residuals remain low). The window's total rotation ends up
slightly wrong, and that error accumulates into the global trajectory.

THE IDEA
--------
Add a direct, long-range constraint that pins the window's two ends together in
ONE hop, instead of trusting the drifting frame-by-frame chain. We match the
window's FIRST keyframe (kf_a) to its LAST keyframe (kf_b) — a wide baseline
(~8-19 frames apart) — using a LEARNED matcher, which is far more reliable than
AKAZE at wide baselines. Each such match becomes an extra 3D landmark observed
in both end keyframes (two stereo factors), injected into the bundle. These
"anchors" out-vote the biased AKAZE tracks and correct the window's rotation.

HOW run_window() WORKS (per window kf_a..kf_b)
----------------------------------------------
1. Build anchor matches between a keyframe pair (pair_anchors), from the UNION
   of two independent learned sources, each RANSAC-verified:
     a. LoFTR (kornia 'outdoor') matches the two LEFT images; the right-image x
        of each match is recovered by a ZNCC sub-pixel template search along the
        epipolar row (_zncc_xr) -> a stereo observation at both ends.
     b. SuperPoint + LightGlue stereo consensus -> a second, independent set.
   Both are verified with pnp.ransac_pnp (threshold retry 2.0 -> 3.5 px, keep if
   >= 12 inliers AND >= 25% inlier ratio), merged (SP points within 2 px of a
   LoFTR point are dropped as duplicates), and capped at 150 per pair.
2. Hybrid pairing rule (GT-free — which pairs get anchored):
     - If the END pair (kf_a, kf_b) verifies -> anchor ONLY the end pair.
     - Else fall back to a half-window CHAIN (~10-frame segments: the
       consecutive segment pairs plus the end pair), keeping every pair that
       verifies. This rescues turns too wide for a single end-to-end match.
     - If nothing verifies -> return rel=None; the caller keeps the plain
       bundle relative pose for that window.
3. Solve the bundle:
     - build_bundle_window as usual, but the AKAZE track measurement sigmas are
       DOUBLED (down-weighted x2) so the few anchors out-weigh the many biased
       tracks.
     - Inject each anchor as one landmark + two GenericStereoFactor3D (on kf_a
       and kf_b), tight sigma 0.25 px, wrapped in a Huber(1.345) robust kernel.
     - Optimize; kf_a is the window's local identity, so the optimized kf_b pose
       IS the relative motion, returned as a 3x4 world-to-camera Rt.

CONFIG NAME 'hybrid_c025dw' (the winner of the ~90-config search):
  hybrid = the end-pair / half-chain fallback rule;  c025 = anchor sigma 0.25 px;
  dw = AKAZE tracks down-weighted x2.

RESULT: applied to all 317 windows this halves the open-loop trajectory drift
(median 14.47 -> 7.16 m). With loop closures the final gain is small, because
the closures already absorb most of that drift.

NOTE — interaction with the Link.size refactor: the x2 track down-weight below
works by doubling `feature_sizes`, which build_bundle_window uses only as a
FALLBACK. After the refactor that stores keypoint size on each Link,
build_bundle_window prefers `link.size`, so the down-weight is bypassed on a
size-carrying DB. The reported campaign numbers were produced on the pre-refactor
DB (links without size), where the down-weight was active. To re-run this method
on the migrated DB and reproduce it, the down-weight must be reapplied through
the Link sizes (or via a track-sigma scale on build_bundle_window).
"""
import os, sys, pickle
import numpy as np
import cv2
import gtsam
import torch
import kornia.feature as KF

CODE = "/Users/YuvalA/Documents - Yuval/GitHub/Slam - Video Navigation/Slam Project/VAN_ex/code"
sys.path.insert(0, os.path.join(CODE, "dl_study"))
sys.path.insert(0, CODE)

from dataset import read_images, read_cameras, DATA_PATH
from features import rectified_stereo_filter
from geometry import triangulate_linear_lsq
from dl_features import extract_features_superpoint, match_features_lightglue
from pnp import ransac_pnp
from bundle import (stereo_calibration, build_bundle_window, optimize_bundle,
                    gtsam_pose_to_Rt, lm_key)

# ---- fixed hyper-params (run_midchain.py defaults) --------------------------
HUBER = 1.345
CONF_MIN = 0.5
ZNCC_MIN = 0.7
MIN_INLIERS = 12
MIN_RATIO = 0.25
MAX_PER_PAIR = 150
DEDUP_PX = 2.0
SEG_FALLBACK = 10          # half-window chain segment length
ANCHOR_TID0 = 10_000_000
Y_THRESHOLD, X_MIN_DISPARITY = 2.0, 1.0
RANSAC_THRS = (2.0, 3.5)
SIGMA = 0.25               # c025
DOWNWEIGHT = True          # dw

K, m1, m2 = read_cameras()
P_left, P_right = K @ m1, K @ m2
K_stereo = stereo_calibration(K, -m2[0, 3])

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
_matcher = None
def matcher():
    global _matcher
    if _matcher is None:
        _matcher = KF.LoFTR(pretrained='outdoor').eval().to(DEVICE)
    return _matcher


# ---- LoFTR anchor source ----------------------------------------------------
def _to_t(img):
    return torch.from_numpy(img).float()[None, None].to(DEVICE) / 255.0

def _loftr(imgA, imgB):
    with torch.inference_mode():
        out = matcher()({'image0': _to_t(imgA), 'image1': _to_t(imgB)})
    return (out['keypoints0'].cpu().numpy(), out['keypoints1'].cpu().numpy(),
            out['confidence'].cpu().numpy())

def _zncc_xr(img_l, img_r, pts, half=5, dmax=128, dmin=1.0):
    H, W = img_l.shape
    xr = np.full(len(pts), np.nan); sc = np.full(len(pts), np.nan)
    for i, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        if xi - half - 1 < 0 or xi + half + 1 >= W or yi - half < 0 or yi + half >= H:
            continue
        tpl = img_l[yi - half:yi + half + 1, xi - half:xi + half + 1]
        x0 = max(0, xi - int(dmax) - half); x1 = xi + half + 1
        strip = img_r[yi - half:yi + half + 1, x0:x1]
        if strip.shape[1] < tpl.shape[1] + 3:
            continue
        res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)[0]
        j = int(np.argmax(res)); s = float(res[j]); j_sub = float(j)
        if 0 < j < len(res) - 1:
            den = res[j - 1] - 2 * res[j] + res[j + 1]
            if abs(den) > 1e-9:
                j_sub = j + 0.5 * (res[j - 1] - res[j + 1]) / den
        cx_r = x0 + j_sub + half; d = xi - cx_r
        if d < dmin or d > dmax:
            continue
        xr[i] = x - d; sc[i] = s
    return xr, sc

def _ransac_verify(X_a, pl_a, pr_a, pl_b, pr_b):
    for thr in RANSAC_THRS:
        rng = np.random.default_rng(0)
        Rt, mask = ransac_pnp(X_a, pl_a, pr_a, pl_b, pr_b, K, m2,
                              threshold=thr, rng=rng)
        n_inl = int(mask.sum())
        if Rt is not None and n_inl >= MIN_INLIERS and n_inl / len(X_a) >= MIN_RATIO:
            return np.where(mask)[0], thr
    return None, None

def _loftr_anchors(fa, fb):
    la, ra = read_images(fa); lb, rb = read_images(fb)
    kp0, kp1, conf = _loftr(la, lb)
    keep = conf >= CONF_MIN
    kp0, kp1 = kp0[keep], kp1[keep]
    if len(kp0) < 8:
        return None
    xr_a, s_a = _zncc_xr(la, ra, kp0); xr_b, s_b = _zncc_xr(lb, rb, kp1)
    ok = (np.isfinite(xr_a) & np.isfinite(xr_b) & (s_a >= ZNCC_MIN) & (s_b >= ZNCC_MIN))
    if ok.sum() < 8:
        return None
    pl_a, pl_b = kp0[ok], kp1[ok]
    pr_a = np.column_stack([xr_a[ok], pl_a[:, 1]])
    pr_b = np.column_stack([xr_b[ok], pl_b[:, 1]])
    X_a = triangulate_linear_lsq(P_left, P_right, pl_a, pr_a)
    depth_ok = (X_a[:, 2] > 0.5) & (X_a[:, 2] < 200.0)
    pl_a, pr_a, pl_b, pr_b, X_a = (pl_a[depth_ok], pr_a[depth_ok],
                                   pl_b[depth_ok], pr_b[depth_ok], X_a[depth_ok])
    if len(X_a) < 8:
        return None
    sel, thr = _ransac_verify(X_a, pl_a, pr_a, pl_b, pr_b)
    if sel is None:
        return None
    return dict(pl_a=pl_a[sel], pr_a=pr_a[sel], pl_b=pl_b[sel],
                pr_b=pr_b[sel], X_a=X_a[sel], n=len(sel), thr=thr)

# ---- SP+LG anchor source ----------------------------------------------------
_sp_cache = {}
def _sp_stereo(frame):
    if frame in _sp_cache:
        return _sp_cache[frame]
    img_l, img_r = read_images(frame)
    kp_l, f_l = extract_features_superpoint(img_l)
    kp_r, f_r = extract_features_superpoint(img_r)
    matches = match_features_lightglue(f_l, f_r)
    inl, _, _ = rectified_stereo_filter(kp_l, kp_r, matches,
                                        y_threshold=Y_THRESHOLD,
                                        x_min_disparity=X_MIN_DISPARITY)
    pts_l = np.array([kp_l[m.queryIdx].pt for m in inl]).reshape(-1, 2)
    pts_r = np.array([kp_r[m.trainIdx].pt for m in inl]).reshape(-1, 2)
    rm = -np.ones(len(kp_l), dtype=int)
    for row, m in enumerate(inl):
        rm[m.queryIdx] = row
    _sp_cache[frame] = (pts_l, pts_r, f_l, rm)
    if len(_sp_cache) > 8:
        for k in list(_sp_cache):
            if k != frame and len(_sp_cache) > 8:
                del _sp_cache[k]
    return _sp_cache[frame]

def _splg_anchors(fa, fb):
    pl_a, pr_a, f_a, rm_a = _sp_stereo(fa)
    pl_b, pr_b, f_b, rm_b = _sp_stereo(fb)
    if len(pl_a) < 8 or len(pl_b) < 8:
        return None
    pairs = match_features_lightglue(f_a, f_b)
    qa, qb = [], []
    for m in pairs:
        ra, rb = rm_a[m.queryIdx], rm_b[m.trainIdx]
        if ra >= 0 and rb >= 0:
            qa.append(ra); qb.append(rb)
    if len(qa) < 8:
        return None
    qa, qb = np.array(qa), np.array(qb)
    A_l, A_r, B_l, B_r = pl_a[qa], pr_a[qa], pl_b[qb], pr_b[qb]
    X_a = triangulate_linear_lsq(P_left, P_right, A_l, A_r)
    depth_ok = (X_a[:, 2] > 0.5) & (X_a[:, 2] < 200.0)
    A_l, A_r, B_l, B_r, X_a = (A_l[depth_ok], A_r[depth_ok],
                               B_l[depth_ok], B_r[depth_ok], X_a[depth_ok])
    if len(X_a) < 8:
        return None
    sel, thr = _ransac_verify(X_a, A_l, A_r, B_l, B_r)
    if sel is None:
        return None
    return dict(pl_a=A_l[sel], pr_a=A_r[sel], pl_b=B_l[sel],
                pr_b=B_r[sel], X_a=X_a[sel], n=len(sel), thr=thr)

def _merge_anchors(lo, sp, fa, fb):
    if lo is None and sp is None:
        return None
    if lo is None or sp is None:
        one = lo if lo is not None else sp
        anc = dict(one)
        anc['stats'] = dict(fa=fa, fb=fb, n_loftr=lo['n'] if lo else 0,
                            n_splg=sp['n'] if sp else 0, n_union=one['n'])
    else:
        keep = np.ones(len(sp['X_a']), dtype=bool)
        d2 = ((sp['pl_a'][:, None, :] - lo['pl_a'][None, :, :]) ** 2).sum(-1)
        keep &= d2.min(axis=1) > DEDUP_PX ** 2
        anc = {k: np.concatenate([lo[k], sp[k][keep]])
               for k in ('pl_a', 'pr_a', 'pl_b', 'pr_b', 'X_a')}
        anc['stats'] = dict(fa=fa, fb=fb, n_loftr=lo['n'], n_splg=sp['n'],
                            n_dup=int((~keep).sum()), n_union=len(anc['X_a']))
    if len(anc['X_a']) > MAX_PER_PAIR:
        rng = np.random.default_rng(1)
        sel = rng.choice(len(anc['X_a']), MAX_PER_PAIR, replace=False)
        for k in ('pl_a', 'pr_a', 'pl_b', 'pr_b', 'X_a'):
            anc[k] = anc[k][sel]
        anc['stats']['n_used'] = MAX_PER_PAIR
    else:
        anc['stats']['n_used'] = len(anc['X_a'])
    return anc

# chain_cache maps (fa,fb) -> anchor dict or None
def pair_anchors(fa, fb, chain_cache):
    key = (fa, fb)
    if key in chain_cache:
        return chain_cache[key]
    lo = _loftr_anchors(fa, fb)
    sp = _splg_anchors(fa, fb)
    anc = _merge_anchors(lo, sp, fa, fb)
    chain_cache[key] = anc
    return anc

# ---- bundle solving ---------------------------------------------------------
def _chain_nodes(kf_a, kf_b, seg):
    n = kf_b - kf_a
    k = max(1, int(round(n / seg)))
    return [kf_a + int(round(i * n / k)) for i in range(k + 1)]

def _inject(graph, initial, info, fa, fb, anc, tid_start, noise):
    pose_fa = initial.atPose3(info['cam_keys'][fa])
    n = 0
    for i in range(len(anc['X_a'])):
        X_local = pose_fa.transformFrom(gtsam.Point3(anc['X_a'][i]))
        Lk = lm_key(tid_start + i)
        initial.insert(Lk, X_local)
        z_a = gtsam.StereoPoint2(anc['pl_a'][i, 0], anc['pr_a'][i, 0], anc['pl_a'][i, 1])
        z_b = gtsam.StereoPoint2(anc['pl_b'][i, 0], anc['pr_b'][i, 0], anc['pl_b'][i, 1])
        graph.add(gtsam.GenericStereoFactor3D(z_a, noise, info['cam_keys'][fa], Lk, K_stereo))
        graph.add(gtsam.GenericStereoFactor3D(z_b, noise, info['cam_keys'][fb], Lk, K_stereo))
        n += 1
    return n

def _make_noise(sigma):
    base = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma] * 3))
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(HUBER), base)


def run_window(kf_a, kf_b, db, pnp_poses, feature_sizes, chain_cache):
    """Compute the hybrid_c025dw relative pose (kf_b in kf_a frame, 3x4 w2c Rt).

    Returns dict: rel (3x4 ndarray) or None if the method fails, plus
    mode ('end'|'half'|None), n_pairs, n_anchors, pair_stats.
    """
    frames = list(range(kf_a, kf_b + 1))
    # --- hybrid pairing rule ---
    end_anc = pair_anchors(kf_a, kf_b, chain_cache)
    if end_anc is not None:
        pair_specs = [(kf_a, kf_b, end_anc)]
        mode = 'end'
    else:
        nodes = _chain_nodes(kf_a, kf_b, SEG_FALLBACK)
        pairs = list(zip(nodes[:-1], nodes[1:]))
        if len(nodes) > 2:
            pairs.append((kf_a, kf_b))
        pair_specs = []
        for fa, fb in pairs:
            anc = pair_anchors(fa, fb, chain_cache)
            if anc is not None:
                pair_specs.append((fa, fb, anc))
        mode = 'half'
    if not pair_specs:
        return dict(rel=None, mode=None, n_pairs=0, n_anchors=0,
                    pair_stats=[], reason='no verified anchors on any pair')

    # --- bundle with AKAZE down-weight x2 ---
    fs = feature_sizes
    if DOWNWEIGHT:
        fs = dict(feature_sizes)
        for fid in frames:
            arr = feature_sizes[fid].copy()
            arr[:, 2] *= 2.0
            fs[fid] = arr
    graph0, initial0, info = build_bundle_window(
        db, pnp_poses, frames, K_stereo, feature_sizes=fs)
    graph = gtsam.NonlinearFactorGraph(graph0)
    initial = gtsam.Values(initial0)
    noise = _make_noise(SIGMA)
    tid = ANCHOR_TID0
    n_anchors = 0
    for fa, fb, anc in pair_specs:
        n_anchors += _inject(graph, initial, info, fa, fb, anc, tid, noise)
        tid += 1_000_000
    result, _ = optimize_bundle(graph, initial)
    rel = gtsam_pose_to_Rt(result.atPose3(info['cam_keys'][kf_b]))
    return dict(rel=rel, mode=mode, n_pairs=len(pair_specs), n_anchors=n_anchors,
                pair_stats=[a['stats'] for _, _, a in pair_specs])
