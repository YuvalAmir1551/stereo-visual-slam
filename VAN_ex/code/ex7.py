"""Exercise 7: Loop closure for the pose graph.

Orchestration only — algorithm primitives (Mahalanobis pre-filter,
consensus match, mini-bundle, LC search) live in `loop_closure.py`.
This file builds the initial pose graph, calls the LC search, and
renders Q7.2 / Q7.5 plots.

State persisted across runs:
  loop_closures.pkl     — list of accepted (n, i, rel_pose, rel_cov, n_inliers)
  keyframe_features.pkl — per-keyframe stereo-inlier features
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from matplotlib.patches import Ellipse, ConnectionPatch

from dataset import read_cameras, read_poses, read_images, DATA_PATH
from geometry import camera_center
from tracking_database import TrackingDB
from bundle import (
    stereo_calibration, select_keyframes_in_calm_frames,
    cam_key, Rt_to_gtsam_pose,
)
from loop_closure import (
    run_loop_closure_search, load_or_build_kf_features,
    loop_consensus_match, optimize,
)

# ----- Paths -----
DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
FIGURES_DIR = os.path.join(DOCS_DIR, 'ex7_figures')
TRACKING_DB_BASE = os.path.join(DATA_PATH, 'tracking_db')
PNP_POSES_PATH = os.path.join(DATA_PATH, 'pnp_poses.npy')
RELATIVES_CACHE = os.path.join(DATA_PATH, 'pose_graph_relatives.pkl')
LC_CACHE = os.path.join(DATA_PATH, 'loop_closures.pkl')


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name + '.png'),
                dpi=150, bbox_inches='tight')


# ===========================================================================
#  Pose-graph construction (chain from bundle relatives)
# ===========================================================================
def build_pose_graph(keyframes, rel_poses, rel_covs):
    """Replicate ex6's chain pose graph: tight prior on c_0 + BetweenFactors."""
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


# ===========================================================================
#  Plot helpers
# ===========================================================================
def _trajectory_centres(result, keyframes):
    return np.array([result.atPose3(cam_key(k)).translation() for k in keyframes])


def _draw_cov_ellipses(ax, centres, marginals, keyframes, nsigma, color='red'):
    for idx, k in enumerate(keyframes):
        P = marginals.marginalCovariance(cam_key(k))
        Pxz = P[3:6, 3:6][np.ix_([0, 2], [0, 2])]
        vals, vecs = np.linalg.eigh(Pxz)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * nsigma * np.sqrt(np.maximum(vals, 0))
        ax.add_patch(Ellipse((centres[idx, 0], centres[idx, 2]), w, h,
                             angle=angle, facecolor='none',
                             edgecolor=color, linewidth=0.6, alpha=0.7))


def _replay_to_milestone(keyframes, rel_poses, rel_covs, lc_list, k_stop):
    """Replay the first k_stop loop closures and return (graph, result, marg)."""
    graph, initial = build_pose_graph(keyframes, rel_poses, rel_covs)
    result = optimize(graph, initial)
    for rec in lc_list[:k_stop]:
        n, i = rec['n'], rec['i']
        rel_pose = Rt_to_gtsam_pose(rec['rel_pose'])
        noise = gtsam.noiseModel.Gaussian.Covariance(rec['rel_cov'])
        graph.add(gtsam.BetweenFactorPose3(
            cam_key(keyframes[i]), cam_key(keyframes[n]), rel_pose, noise))
        result = optimize(graph, result)
    marg = gtsam.Marginals(graph, result)
    return graph, result, marg


# ===========================================================================
#  Q7.2 — consensus match visualisation for one accepted LC
# ===========================================================================
def q2_visualise(record, kf_features, K, m_right, P_left, P_right):
    """Re-run the consensus match for one accepted LC and plot the matches
    with connecting lines between inlier keypoints (outliers shown as red
    dots, no line)."""
    kf_n, kf_i = record['kf_n'], record['kf_i']
    img_l_i, _ = read_images(kf_i)
    img_l_n, _ = read_images(kf_n)

    res = loop_consensus_match(kf_n, kf_i, kf_features, K, m_right,
                               P_left, P_right)
    if res is None:
        print('Q7.2 viz: consensus match re-run failed unexpectedly'); return
    _Rt, mask, _cross, q, t = res
    _, pts_l_i, _ = kf_features[kf_i]
    _, pts_l_n, _ = kf_features[kf_n]

    in_pts_i = pts_l_i[q[mask]]
    in_pts_n = pts_l_n[t[mask]]
    out_pts_i = pts_l_i[q[~mask]]
    out_pts_n = pts_l_n[t[~mask]]

    fig, (ax_i, ax_n) = plt.subplots(2, 1, figsize=(13, 9))
    for ax, img, pts_in, pts_out, side in [
            (ax_i, img_l_i, in_pts_i, out_pts_i,
             f'left frame {kf_i}  (c_i, earlier)'),
            (ax_n, img_l_n, in_pts_n, out_pts_n,
             f'left frame {kf_n}  (c_n, later)'),
    ]:
        ax.imshow(img, cmap='gray')
        if len(pts_out):
            ax.scatter(pts_out[:, 0], pts_out[:, 1], c='red', s=6,
                       label=f'outliers ({len(pts_out)})')
        ax.scatter(pts_in[:, 0], pts_in[:, 1], c='cyan', s=6,
                   label=f'inliers ({len(pts_in)})')
        ax.set_title(side); ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)'); ax.set_aspect('equal'); ax.legend(loc='upper right')

    # Connect each inlier pair with a cyan line between the two subplots.
    rng = np.random.default_rng(0)
    n_lines = min(40, len(in_pts_i))
    sample = rng.choice(len(in_pts_i), n_lines, replace=False)
    for idx in sample:
        pa = (in_pts_i[idx, 0], in_pts_i[idx, 1])
        pb = (in_pts_n[idx, 0], in_pts_n[idx, 1])
        fig.add_artist(ConnectionPatch(
            xyA=pa, coordsA=ax_i.transData,
            xyB=pb, coordsB=ax_n.transData,
            color='cyan', linewidth=0.6, alpha=0.7))

    fig.suptitle(f'Q7.2 — consensus match c_{record["n"]} ↔ c_{record["i"]}  '
                 f'({int(mask.sum())} inliers of {len(mask)}, '
                 f'{n_lines} inlier links shown)')
    fig.tight_layout()
    _save(fig, 'q7_2_consensus_match')


# ===========================================================================
#  Q7.5 — full deliverable plots
# ===========================================================================
def q5_plots(keyframes, rel_poses, rel_covs, accepted):
    """Q7.5 — snapshots + GT/uncertainty/error comparisons."""
    gt_poses = read_poses()
    gt_centres = np.array([camera_center(gt_poses[k]) for k in keyframes])

    # Without LC: just the initial pose graph.
    graph0, init0 = build_pose_graph(keyframes, rel_poses, rel_covs)
    result_no = optimize(graph0, init0)
    marg_no = gtsam.Marginals(graph0, result_no)
    cent_no = _trajectory_centres(result_no, keyframes)

    # With LC: final state.
    _, result_f, marg_f = _replay_to_milestone(
        keyframes, rel_poses, rel_covs, accepted, len(accepted))
    cent_with = _trajectory_centres(result_f, keyframes)

    # ----- (1) Four-snapshot plot -----
    n_lc = len(accepted)
    milestones = [0,
                  1 if n_lc >= 1 else 0,
                  max(1, n_lc // 2) if n_lc >= 2 else 0,
                  n_lc]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, k_stop in zip(axes.ravel(), milestones):
        _, res_k, marg_k = _replay_to_milestone(
            keyframes, rel_poses, rel_covs, accepted, k_stop)
        cent_k = _trajectory_centres(res_k, keyframes)
        ax.plot(gt_centres[:, 0], gt_centres[:, 2], ':',
                color='orange', linewidth=1.4, label='ground truth')
        ax.plot(cent_k[:, 0], cent_k[:, 2], '-', color='steelblue',
                linewidth=1.2, label='pose graph')
        _draw_cov_ellipses(ax, cent_k, marg_k, keyframes,
                           nsigma=10, color='red')
        for rec in accepted[:k_stop]:
            xn = cent_k[rec['n'], 0]; zn = cent_k[rec['n'], 2]
            xi = cent_k[rec['i'], 0]; zi = cent_k[rec['i'], 2]
            ax.plot([xn, xi], [zn, zi], '-', color='crimson',
                    alpha=0.6, linewidth=0.8)
        ax.set_title(f'after {k_stop} loop closures')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(loc='lower left')
    fig.suptitle('Q7.5 — pose-graph evolution (with 10σ ellipses)')
    fig.tight_layout()
    _save(fig, 'q7_5_snapshots')

    # ----- (2) Final trajectory: with vs without LC vs GT -----
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.plot(cent_no[:, 0], cent_no[:, 2], '-', color='steelblue',
            linewidth=2, label=f'pose graph WITHOUT loop closures')
    ax.plot(cent_with[:, 0], cent_with[:, 2], '--', color='crimson',
            linewidth=1.8, label=f'pose graph WITH {n_lc} loop closures')
    ax.plot(gt_centres[:, 0], gt_centres[:, 2], ':', color='orange',
            linewidth=1.8, label='ground truth')
    ax.scatter([0], [0], c='black', s=80, marker='s', zorder=5, label='c_0')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
    ax.set_title('Q7.5 — final trajectory vs ground truth')
    _save(fig, 'q7_5_trajectory_with_without')

    # ----- (3) Absolute location error -----
    err_no = np.linalg.norm(cent_no - gt_centres, axis=1)
    err_with = np.linalg.norm(cent_with - gt_centres, axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(keyframes, err_no, color='steelblue', label='without LC')
    ax.plot(keyframes, err_with, color='crimson', label='with LC')
    ax.set_xlabel('keyframe (absolute frame id)')
    ax.set_ylabel('absolute location error (m)')
    ax.set_title(f'Q7.5 — abs. location error  '
                 f'(median {np.median(err_no):.2f}m → {np.median(err_with):.2f}m, '
                 f'max {err_no.max():.2f}m → {err_with.max():.2f}m)')
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, 'q7_5_abs_error')

    # ----- (4) Uncertainty size: √det(Σ_t) per keyframe -----
    # Same metric as the pose-graph Dijkstra edge weight — volume of the
    # 1σ translation-uncertainty ellipsoid (units: m³).
    def _unc(marg, k):
        P = marg.marginalCovariance(cam_key(k))
        return float(np.sqrt(max(np.linalg.det(P[3:6, 3:6]), 0.0)))
    unc_no = np.array([_unc(marg_no, k) for k in keyframes])
    unc_with = np.array([_unc(marg_f, k) for k in keyframes])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(keyframes, unc_no, color='steelblue', label='without LC')
    ax.plot(keyframes, unc_with, color='crimson', label='with LC')
    ax.set_yscale('log')
    ax.set_xlabel('keyframe (absolute frame id)')
    ax.set_ylabel(r'uncertainty size  $\sqrt{\det\Sigma_t}$  (m³, log scale)')
    ax.set_title('Q7.5 — location uncertainty size')
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, 'q7_5_uncertainty')

    print(f'\nQ7.5 summary:')
    print(f'  Loop closures accepted: {n_lc}')
    print(f'  Median abs error: {np.median(err_no):.2f} m  →  '
          f'{np.median(err_with):.2f} m')
    print(f'  Max    abs error: {err_no.max():.2f} m  →  '
          f'{err_with.max():.2f} m')
    print(f'  Final  abs error: {err_no[-1]:.2f} m  →  '
          f'{err_with[-1]:.2f} m')


# ===========================================================================
#  main
# ===========================================================================
def main():
    K, m1, m2 = read_cameras()
    P_left, P_right = K @ m1, K @ m2
    K_stereo = stereo_calibration(K, -m2[0, 3])

    db = TrackingDB(); db.load(TRACKING_DB_BASE)
    img_dir = os.path.join(DATA_PATH, 'image_0')
    n_frames = min(sum(1 for f in os.listdir(img_dir) if f.endswith('.png')),
                   db.frame_num())
    pnp_poses = np.load(PNP_POSES_PATH)[:n_frames]

    keyframes = select_keyframes_in_calm_frames(
        pnp_poses, min_translation=5.0,
        translation_jitter=1.0, seed=42,
        max_frames_gap=19, min_frames_gap=8,
        straight_rot_rate_deg=0.5,
    )
    print(f'Keyframes: {len(keyframes)}')

    with open(RELATIVES_CACHE, 'rb') as f:
        d = pickle.load(f)
    rel_poses, rel_covs = d['rel_poses'], d['rel_covs']
    print(f'Loaded {len(rel_poses)} bundle relatives from cache.')

    # ----- Q7.2–7.4 main loop (cached) -----
    accepted = None
    if os.path.exists(LC_CACHE):
        with open(LC_CACHE, 'rb') as f:
            d = pickle.load(f)
        if d.get('keyframes') == keyframes:
            accepted = d['accepted']
            print(f'\nLoaded {len(accepted)} loop closures from cache.')

    if accepted is None:
        print('\nQ7.2–7.4 — running loop closure search…')
        graph, initial = build_pose_graph(keyframes, rel_poses, rel_covs)
        result = optimize(graph, initial)
        _, accepted, _ = run_loop_closure_search(
            graph, result, keyframes, rel_covs,
            K, m2, P_left, P_right, K_stereo)
        with open(LC_CACHE, 'wb') as f:
            pickle.dump({'keyframes': keyframes, 'accepted': accepted}, f)
        print(f'Saved {len(accepted)} loop closures to {LC_CACHE}')

    # ----- Q7.2 visualisation of one accepted match -----
    if accepted:
        kf_features = load_or_build_kf_features(keyframes)
        # Pick the LC with the largest |n - i| as the most visually striking.
        best = max(accepted, key=lambda r: r['n'] - r['i'])
        print(f"\nQ7.2 viz — c_{best['n']}↔c_{best['i']} "
              f"({best['inliers']} inliers, mah {best['mahalanobis']:.1f})")
        q2_visualise(best, kf_features, K, m2, P_left, P_right)

    # ----- Q7.5 deliverable plots -----
    print('\nQ7.5 — building deliverable plots')
    q5_plots(keyframes, rel_poses, rel_covs, accepted or [])


if __name__ == '__main__':
    main()
