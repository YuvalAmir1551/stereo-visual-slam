"""Loop closure detection for pose graph SLAM (VAN Ex7).

Spec-mandated flow:
  7.1  Shortest-path Σ_n|i estimate + Mahalanobis pre-filter
  7.2  RANSAC-PnP consensus match between distant keyframes
  7.3  Two-frame mini-bundle → conditional Σ of c_n | c_i
  7.4  BetweenFactorPose3 → pose graph re-optimisation

Within each pass, the SET of consensus-matched pairs is snapshotted at
pass start (compute Mahalanobis for every eligible (n, i), keep those
under threshold). Which pairs get tested therefore does not depend on
within-pass acceptance events, so LC count is monotonic in MIN_INLIER_COUNT.
Candidates are only re-scored between passes.
"""

import os
import heapq
import pickle
import time
import numpy as np
import gtsam
import networkx as nx

from dataset import read_images, DATA_PATH
from geometry import triangulate_linear_lsq
from features import (
    extract_features, match_descriptors, rectified_stereo_filter,
)
from pnp import ransac_pnp
from bundle import (
    cam_key, lm_key, Rt_to_gtsam_pose, gtsam_pose_to_Rt,
    stereo_camera, PRIOR_NOISE, STEREO_NOISE,
)

# ----- Hyperparameters -----
K_SKIP_RECENT_KFS = 10        # skip the last N keyframes when searching for LC
MAH_THRESHOLD = 1000.0        # Mahalanobis quad-form pre-filter threshold
MIN_INLIER_COUNT = 100        # consensus-match acceptance: ≥ this many
                              # RANSAC-PnP inliers to accept the LC
MAX_PASSES = 5                # upper bound; loop breaks on convergence
PIX_THRESHOLD = 2.0           # RANSAC reprojection threshold (px)
Y_THRESHOLD = 2.0             # rectified-stereo |Δy| (px)
X_MIN_DISPARITY = 0.5         # minimum stereo disparity (px)
DETECTOR = 'AKAZE'

KF_FEATURES_CACHE = os.path.join(DATA_PATH, 'keyframe_features.pkl')


# ex7
def optimize(graph, initial):
    """Levenberg–Marquardt optimisation of a NonlinearFactorGraph."""
    return gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()


# ===========================================================================
#  Pose-graph → networkx helpers  (7.1)
# ===========================================================================
# ex7
def pose_graph_to_nx(n_keyframes, rel_covs, loop_edges=()):
    """Undirected networkx graph; edges carry the bundle's 6×6 cov as the
    edge attribute 'cov'. Nodes are keyframe indices 0..N-1.
    """
    g = nx.Graph()
    for k in range(n_keyframes):
        g.add_node(k)
    for b, cov in enumerate(rel_covs):
        g.add_edge(b, b + 1, cov=cov)
    for (i, j, cov) in loop_edges:
        g.add_edge(i, j, cov=cov)
    return g


# ex7
def shortest_path_cov(nx_graph, src, dst):
    """Course-spec Dijkstra with matrix-valued (6×6) edge weights.

    From VAN 07 § "Shortest Path" (David Arnon slides):
      • Σ[s] = 0₆; Σ[v] = ∞·I initially (nodes not yet reached).
      • EXTRACT-MIN picks the vertex with smallest √det(Σ[u]).
      • RELAX(u, v): if √det(Σ[u] + w(u,v)) < √det(Σ[v]),
                     set Σ[v] ← Σ[u] + w(u,v) and add v to the frontier.
      • w(u, v) = the full 6×6 covariance of edge (u, v).

    We use √det rather than raw det: it's a monotone transformation so
    the path ranking is identical, but it keeps the scalar cost in a
    friendlier float64 dynamic range (per-edge det is ~1e-30 and grows
    ~exponentially with path length).

    Returns (Σ_dst, path). By construction Σ_dst is the *sum* of edge
    covariances along the min-cost path — i.e. Σ_n|i, exactly as the
    ex7 spec asks.
    """
    Sigma = {src: np.zeros((6, 6))}
    costs = {src: 0.0}
    parent = {src: None}
    heap = [(0.0, src)]
    visited = set()

    while heap:
        _, u = heapq.heappop(heap)
        if u in visited:
            continue
        if u == dst:
            break
        visited.add(u)
        for v in nx_graph.neighbors(u):
            if v in visited:
                continue
            new_Sigma = Sigma[u] + nx_graph[u][v]['cov']
            new_cost = float(np.sqrt(max(np.linalg.det(new_Sigma), 0.0)))
            if v not in costs or new_cost < costs[v]:
                Sigma[v] = new_Sigma
                costs[v] = new_cost
                parent[v] = u
                heapq.heappush(heap, (new_cost, v))

    if dst not in Sigma:
        raise nx.NetworkXNoPath(f'No path from {src} to {dst}')

    path = []
    node = dst
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return Sigma[dst], path


# ex7
def mahalanobis_pose3(pose_delta, sigma):
    """6-DOF Mahalanobis quadratic form for a Pose3 measurement.

    Logmap(Pose3) gives a 6-vector in se(3) tangent coords (ω₃, ρ₃).
    """
    v = gtsam.Pose3.Logmap(pose_delta)
    try:
        invS = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return np.inf
    return float(v @ invS @ v)


# ===========================================================================
#  Keyframe feature cache (stereo-inlier features per keyframe)
# ===========================================================================
# ex7
def _extract_kf_features(kf_id):
    """Stereo-pair feature extraction + filter; return (descriptors, pts_l, pts_r).

    pts_l, pts_r are Nx2 pixel arrays of LEFT/RIGHT image locations for each
    stereo-inlier. descriptors[i] corresponds to (pts_l[i], pts_r[i]).
    """
    img_l, img_r = read_images(kf_id)
    kp_l, des_l = extract_features(img_l, DETECTOR)
    kp_r, des_r = extract_features(img_r, DETECTOR)
    matches = match_descriptors(des_l, des_r, DETECTOR)
    inliers, _, _ = rectified_stereo_filter(
        kp_l, kp_r, matches,
        y_threshold=Y_THRESHOLD, x_min_disparity=X_MIN_DISPARITY)
    if not inliers:
        return np.zeros((0, des_l.shape[1]), dtype=des_l.dtype), \
               np.zeros((0, 2)), np.zeros((0, 2))
    desc = np.array([des_l[m.queryIdx] for m in inliers])
    pts_l = np.array([kp_l[m.queryIdx].pt for m in inliers], dtype=np.float64)
    pts_r = np.array([kp_r[m.trainIdx].pt for m in inliers], dtype=np.float64)
    return desc, pts_l, pts_r


# ex7
def load_or_build_kf_features(keyframes):
    """Disk-backed cache of per-keyframe stereo features."""
    if os.path.exists(KF_FEATURES_CACHE):
        with open(KF_FEATURES_CACHE, 'rb') as f:
            d = pickle.load(f)
        if d.get('keyframes') == keyframes:
            print(f'Loaded keyframe features cache ({len(keyframes)} kfs).')
            return d['features']
        print('Keyframe-feature cache keyframe-list mismatch — rebuilding.')

    print(f'Extracting features for {len(keyframes)} keyframes…')
    features = {}
    t0 = time.time()
    for k, kf in enumerate(keyframes):
        features[kf] = _extract_kf_features(kf)
        if (k + 1) % 50 == 0:
            dt = time.time() - t0
            print(f'  {k + 1}/{len(keyframes)} done in {dt:.1f}s '
                  f'({dt / (k + 1):.2f}s each)')
    with open(KF_FEATURES_CACHE, 'wb') as f:
        pickle.dump({'keyframes': keyframes, 'features': features}, f)
    print(f'Saved keyframe features to {KF_FEATURES_CACHE}')
    return features


# ===========================================================================
#  Q7.2 — consensus match between two (distant) keyframes
# ===========================================================================
# ex7
def loop_consensus_match(kf_n, kf_i, kf_features, K, m_right, P_left, P_right):
    """RANSAC-PnP between two distant keyframes.

    Returns (Rt_rel, inlier_mask, cross, q, t) where Rt_rel is the 3×4
    extrinsic of c_n in c_i's frame, cross the raw cross-frame DMatch list,
    q/t the queryIdx/trainIdx arrays. Returns None if the match failed.
    """
    desc_n, pts_l_n, pts_r_n = kf_features[kf_n]
    desc_i, pts_l_i, pts_r_i = kf_features[kf_i]
    if len(desc_n) < 4 or len(desc_i) < 4:
        return None
    cross = match_descriptors(desc_i, desc_n, DETECTOR, cross_check=False)
    if len(cross) < 4:
        return None
    q = np.array([m.queryIdx for m in cross], dtype=int)
    t = np.array([m.trainIdx for m in cross], dtype=int)

    # Triangulate each cross-match landmark from c_i's stereo (c_i ⇒ identity).
    X_i = triangulate_linear_lsq(P_left, P_right, pts_l_i[q], pts_r_i[q])
    rng = np.random.default_rng(0)
    Rt_rel, mask = ransac_pnp(
        X_i, pts_l_i[q], pts_r_i[q],
        pts_l_n[t], pts_r_n[t],
        K, m_right, threshold=PIX_THRESHOLD, rng=rng,
    )
    return Rt_rel, mask, cross, q, t


# ===========================================================================
#  Q7.3 — mini-bundle for a loop pair
# ===========================================================================
# ex7
def loop_mini_bundle(kf_n, kf_i, kf_features, cross, q, t, mask,
                     K_stereo, Rt_rel_init):
    """Two-frame BA for the loop pair.

    Anchor c_i at identity; initialise c_n at the RANSAC PnP extrinsic;
    triangulate landmarks from c_i's stereo; add stereo factors on c_i and
    c_n; optimise.

    Returns (rel_pose, rel_cov) where rel_cov is the conditional covariance
    of c_n given c_i.
    """
    desc_n, pts_l_n, pts_r_n = kf_features[kf_n]
    desc_i, pts_l_i, pts_r_i = kf_features[kf_i]

    sel = np.where(mask)[0]
    pl_i = pts_l_i[q[sel]]; pr_i = pts_r_i[q[sel]]
    pl_n = pts_l_n[t[sel]]; pr_n = pts_r_n[t[sel]]

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    A_KEY, B_KEY = cam_key(kf_i), cam_key(kf_n)

    # Anchor c_i at identity.
    graph.add(gtsam.PriorFactorPose3(A_KEY, gtsam.Pose3(), PRIOR_NOISE))
    initial.insert(A_KEY, gtsam.Pose3())

    # c_n at the RANSAC init.
    pose_n_init = Rt_to_gtsam_pose(Rt_rel_init)
    initial.insert(B_KEY, pose_n_init)

    # Triangulate landmarks in c_i's local frame using a StereoCamera at identity.
    sc_i = stereo_camera(gtsam.Pose3(), K_stereo)

    used = 0
    for xli, xri, yi, xln, xrn, yn in zip(
            pl_i[:, 0], pr_i[:, 0], (pl_i[:, 1] + pr_i[:, 1]) / 2,
            pl_n[:, 0], pr_n[:, 0], (pl_n[:, 1] + pr_n[:, 1]) / 2):
        z_i = gtsam.StereoPoint2(xli, xri, yi)
        z_n = gtsam.StereoPoint2(xln, xrn, yn)
        try:
            X = sc_i.backproject(z_i)
        except RuntimeError:
            continue
        if not np.isfinite(X).all() or X[2] <= 0 or X[2] > 200:
            continue
        L = lm_key(used)
        initial.insert(L, gtsam.Point3(X))
        graph.add(gtsam.GenericStereoFactor3D(z_i, STEREO_NOISE, A_KEY, L, K_stereo))
        graph.add(gtsam.GenericStereoFactor3D(z_n, STEREO_NOISE, B_KEY, L, K_stereo))
        used += 1

    if used < 4:
        return None, None
    result = optimize(graph, initial)
    rel_pose = result.atPose3(B_KEY)
    # Conditional cov of c_n given c_i fixed — same Schur trick as ex6.
    marg = gtsam.Marginals(graph, result)
    kv = gtsam.KeyVector(); kv.append(A_KEY); kv.append(B_KEY)
    I_joint = marg.jointMarginalInformation(kv).fullMatrix()
    rel_cov = np.linalg.inv(I_joint[6:12, 6:12])
    return rel_pose, rel_cov


# ===========================================================================
#  LC search
# ===========================================================================
# ex7
def _score_candidates(result, nx_g, keyframes, tried_pairs):
    """Score every eligible (n, i) pair by Mahalanobis; return the survivors
    (those below MAH_THRESHOLD) in natural (n, i) order.

    Snapshotted once per pass — the survivor list is fixed even as later
    within-pass acceptances change the pose graph. This decouples the SET
    of consensus-matched pairs from the within-pass acceptance filter and
    makes the LC count monotonic in MIN_INLIER_COUNT.
    """
    N = len(keyframes)
    candidates = []
    for n in range(K_SKIP_RECENT_KFS, N):
        pose_n = result.atPose3(cam_key(keyframes[n]))
        for i in range(0, n - K_SKIP_RECENT_KFS):
            if (n, i) in tried_pairs:
                continue
            Sigma, _ = shortest_path_cov(nx_g, n, i)
            pose_i = result.atPose3(cam_key(keyframes[i]))
            Delta = pose_i.between(pose_n)
            m = mahalanobis_pose3(Delta, Sigma)
            if m > MAH_THRESHOLD:
                continue
            candidates.append((m, n, i))
    return candidates


# ex7
def _single_pass(graph, result, nx_g, loop_edges, accepted,
                 tried_pairs, kf_features, keyframes,
                 K, m_right, P_left, P_right, K_stereo):
    """One sweep over the pass-start candidate list. Returns (# LCs added,
    updated result)."""
    t0 = time.time()
    print('    Scoring candidates by Mahalanobis…')
    candidates = _score_candidates(result, nx_g, keyframes, tried_pairs)
    print(f'    {len(candidates)} candidates pass Mahalanobis '
          f'(in {time.time() - t0:.1f}s); processing…')

    added = 0
    for idx, (m, n, i) in enumerate(candidates):
        res = loop_consensus_match(
            keyframes[n], keyframes[i], kf_features,
            K, m_right, P_left, P_right)
        tried_pairs.add((n, i))
        if res is None:
            continue
        Rt_rel, mask, cross, q, t = res
        if Rt_rel is None or mask is None:
            continue
        n_in = int(mask.sum())
        if n_in < MIN_INLIER_COUNT:
            continue

        rel_pose, rel_cov = loop_mini_bundle(
            keyframes[n], keyframes[i], kf_features,
            cross, q, t, mask, K_stereo, Rt_rel)
        if rel_pose is None:
            continue

        noise = gtsam.noiseModel.Gaussian.Covariance(rel_cov)
        graph.add(gtsam.BetweenFactorPose3(
            cam_key(keyframes[i]), cam_key(keyframes[n]),
            rel_pose, noise))
        result = optimize(graph, result)
        loop_edges.append((i, n, rel_cov))
        nx_g.add_edge(i, n, cov=rel_cov)

        accepted.append({
            'n': n, 'i': i,
            'kf_n': keyframes[n], 'kf_i': keyframes[i],
            'rel_pose': gtsam_pose_to_Rt(rel_pose),
            'rel_cov': rel_cov,
            'inliers': n_in,
            'cross_pre': len(cross),
            'mahalanobis': m,
        })
        added += 1
        print(f'    LC {len(accepted):3d}: c_{n}→c_{i}  '
              f'(frames {keyframes[n]}↔{keyframes[i]}, '
              f'inliers {n_in}, mah {m:.1f})')

        if (idx + 1) % 200 == 0:
            dt = time.time() - t0
            print(f'    …tried {idx + 1}/{len(candidates)} candidates in '
                  f'{dt:.1f}s, {added} LCs so far')
    return added, result


# ex7
def run_loop_closure_search(graph, result, keyframes, rel_covs,
                            K, m_right, P_left, P_right, K_stereo):
    """Multi-pass loop-closure search over the given pose graph.

    Pass 1 finds the obvious closures. Subsequent passes re-score all
    still-unattempted pairs against the improved pose graph, so mid-
    trajectory pairs whose Mahalanobis dropped after earlier passes get
    a fresh chance.

    Returns (loop_edges, accepted, result). Modifies `graph` in place.
    """
    loop_edges = []
    accepted = []
    tried_pairs = set()
    nx_g = pose_graph_to_nx(len(keyframes), rel_covs, loop_edges=loop_edges)
    kf_features = load_or_build_kf_features(keyframes)

    for p in range(MAX_PASSES):
        print(f'  Pass {p + 1}/{MAX_PASSES}:')
        added, result = _single_pass(
            graph, result, nx_g, loop_edges, accepted,
            tried_pairs, kf_features, keyframes,
            K, m_right, P_left, P_right, K_stereo)
        print(f'  Pass {p + 1} added {added} loop closures '
              f'({len(accepted)} total).')
        if added == 0:
            break

    return loop_edges, accepted, result


# ex7
def build_pose_graph(keyframes, rel_poses, rel_covs):
    """Chain pose graph over keyframes: tight prior on c_0 + BetweenFactors."""
    graph = gtsam.NonlinearFactorGraph()
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 6))
    graph.add(gtsam.PriorFactorPose3(cam_key(keyframes[0]),
                                     gtsam.Pose3(), anchor_noise))
    for b, rp in enumerate(rel_poses):
        noise = gtsam.noiseModel.Gaussian.Covariance(rel_covs[b])
        graph.add(gtsam.BetweenFactorPose3(
            cam_key(keyframes[b]), cam_key(keyframes[b + 1]),
            rp, noise))
    initial = gtsam.Values()
    initial.insert(cam_key(keyframes[0]), gtsam.Pose3())
    cur = gtsam.Pose3()
    for b, rp in enumerate(rel_poses):
        cur = cur.compose(rp)
        initial.insert(cam_key(keyframes[b + 1]), cur)
    return graph, initial
