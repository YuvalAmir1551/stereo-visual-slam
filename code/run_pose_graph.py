"""Exercise 6: Pose graph from bundle adjustment results.

q1 — Extract the relative pose + conditional covariance between every
     consecutive pair of keyframes (using the per-bundle optimisation
     marginals). Plot the first bundle's frames with their conditional
     covariances.
q2 — Build a pose graph (BetweenFactorPose3 chain + gauge prior) using the
     extracted relatives, optimise it, and plot the keyframe trajectory
     before / after / with marginals.
"""

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot

from dataset import read_cameras, read_poses, DATA_PATH
from geometry import camera_center
from tracking_database import TrackingDB
from bundle import (
    stereo_calibration, select_keyframes_in_calm_frames,
    build_bundle_window, optimize_bundle,
    compute_feature_sizes,
    cam_key,
    conditional_cov, solve_bundle, solve_all_bundles,
)

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
FIGURES_DIR = os.path.join(DOCS_DIR, 'ex6_figures')
TRACKING_DB_BASE = os.path.join(DATA_PATH, 'tracking_db')
PNP_POSES_PATH = os.path.join(DATA_PATH, 'pnp_poses.npy')
FEATURE_SIZES_PATH = os.path.join(DATA_PATH, 'feature_sizes.pkl')
RELATIVES_CACHE_PATH = os.path.join(DATA_PATH, 'pose_graph_relatives.pkl')

# Y-up axis remap for 3D plots: matplotlib_X = data_X, matplotlib_Y = data_Z,
# matplotlib_Z = -data_Y. Same trick we used in ex5's q5_3 3D plot.
M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
M6 = np.block([[M, np.zeros((3, 3))], [np.zeros((3, 3)), M]])


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name + '.png'), dpi=150,
                bbox_inches='tight')


# ex6
def _plot_pose3_with_remap(ax, pose, P, axis_length=0.5, cov_scale=1.0):
    """Apply the M6 axis-remap to a (pose, cov) pair and plot the triad."""
    new_pose = gtsam.Pose3(
        gtsam.Rot3(M @ pose.rotation().matrix()),
        gtsam.Point3(M @ pose.translation()))
    new_P = (M6 @ P @ M6.T) * cov_scale
    gtsam_plot.plot_pose3_on_axes(ax, new_pose, axis_length=axis_length,
                                  P=new_P)


# ex6
def q1(db, pnp_poses, keyframes, K_stereo, feature_sizes):
    """Q6.1 — relative pose + cov for the first bundle; 3D plot per-frame cov."""
    kf_a, kf_b = keyframes[0], keyframes[1]
    frames = list(range(kf_a, kf_b + 1))
    print(f'  Q6.1 first bundle: frames {kf_a}..{kf_b}')

    rel_pose, rel_cov, result, info, marginals = solve_bundle(
        db, pnp_poses, frames, K_stereo, feature_sizes)

    print(f'\n  Relative pose c_{kf_a} → c_{kf_b}:')
    R, t = rel_pose.rotation().matrix(), rel_pose.translation()
    print(f'    R =\n{np.array_str(R, precision=4, suppress_small=True)}')
    print(f'    t = {np.array_str(t, precision=4, suppress_small=True)}')
    print(f'    ||t|| = {np.linalg.norm(t):.4f} m')

    print(f'\n  Relative covariance (6×6, order = rot[3] | trans[3]):')
    with np.printoptions(precision=2, suppress=False, formatter={'float': '{: 0.2e}'.format}):
        print(rel_cov)
    sigma_diag = np.sqrt(np.diag(rel_cov))
    print(f'  σ (per dim) = {np.array_str(sigma_diag, precision=4, suppress_small=True)}')

    # 3D plot: every frame in the bundle, with its CONDITIONAL covariance
    # given c_start fixed (Schur per frame).
    COV_VIS_NSIGMA = 30  # σ-multiplier — matches the Q6.2 ellipse scale
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    kf_a_key = info['cam_keys'][kf_a]
    for fid in frames:
        ck = info['cam_keys'][fid]
        pose = result.atPose3(ck)
        if fid == kf_a:
            P = np.zeros((6, 6))         # c_a is the conditioned variable
        else:
            P = conditional_cov(marginals, kf_a_key, ck)
        _plot_pose3_with_remap(ax, pose, P, axis_length=0.5,
                               cov_scale=COV_VIS_NSIGMA ** 2)
    ax.set_xlabel('X (m)', labelpad=8)
    ax.set_ylabel('Z (m)', labelpad=8)
    ax.set_zlabel('Y (m)', labelpad=8)
    ax.set_title('Q6.1 — first bundle frames with conditional covariance')
    gtsam_plot.set_axes_equal(fig.number)
    fig.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.05)
    fig.savefig(os.path.join(FIGURES_DIR, 'q6_1_first_bundle_3d.png'),
                dpi=150, pad_inches=0.4)


# ex6
def q2(db, pnp_poses, keyframes, K_stereo, feature_sizes):
    """Q6.2 — build pose graph from all bundles, optimise, plot."""
    n_kf = len(keyframes)
    n_bundles = n_kf - 1

    rel_poses, rel_covs = solve_all_bundles(
        db, pnp_poses, keyframes, K_stereo, feature_sizes)

    # Initial guess: chain relative poses to get absolute poses in c_0's frame.
    # Reasonable because every BetweenFactor is exactly satisfied by the
    # chained values, so the optimiser starts at zero relative-residual.
    initial = gtsam.Values()
    initial.insert(cam_key(keyframes[0]), gtsam.Pose3())
    cur = gtsam.Pose3()
    for b, rp in enumerate(rel_poses):
        cur = cur.compose(rp)
        initial.insert(cam_key(keyframes[b + 1]), cur)

    # Pose graph: tight gauge prior on c_0 + Between factors per bundle.
    graph = gtsam.NonlinearFactorGraph()
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 6))
    graph.add(gtsam.PriorFactorPose3(cam_key(keyframes[0]),
                                     gtsam.Pose3(), anchor_noise))
    for b in range(n_bundles):
        noise = gtsam.noiseModel.Gaussian.Covariance(rel_covs[b])
        graph.add(gtsam.BetweenFactorPose3(
            cam_key(keyframes[b]), cam_key(keyframes[b + 1]),
            rel_poses[b], noise))

    err_before = graph.error(initial)
    print(f'\n  Pose graph: {graph.size()} factors '
          f'({n_bundles} BetweenFactor + 1 prior).')
    print(f'  Total error BEFORE optimization: {err_before:.4e}')

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    err_after = graph.error(result)
    print(f'  Total error AFTER optimization:  {err_after:.4e}')

    # Ground-truth keyframe centres (KITTI frame-0 frame == c_0 frame).
    gt_poses = read_poses()
    gt_centres = np.array([camera_center(gt_poses[k]) for k in keyframes])

    init_centres = np.array([initial.atPose3(cam_key(k)).translation()
                             for k in keyframes])
    final_centres = np.array([result.atPose3(cam_key(k)).translation()
                              for k in keyframes])

    # Merged before / after / GT plot (top-down X-Z).
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(init_centres[:, 0], init_centres[:, 2], '-',
            color='steelblue', linewidth=2.2, label='initial (chained)')
    ax.plot(final_centres[:, 0], final_centres[:, 2], '--',
            color='crimson', linewidth=1.6, label='after optimisation')
    ax.plot(gt_centres[:, 0], gt_centres[:, 2], ':',
            color='orange', linewidth=1.8, label='ground truth')
    ax.scatter([0], [0], c='black', s=80, marker='s', zorder=5, label='c_0')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z — forward (m)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('Q6.2 — pose-graph trajectory')
    ax.legend()
    _save(fig, 'q6_2_trajectory')

    # Marginal covariance ellipses on the optimised trajectory + GT overlay.
    marginals = gtsam.Marginals(graph, result)
    from matplotlib.patches import Ellipse
    ELLIPSE_NSIGMA = 30.0  # 30σ — purely a visual gain
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.plot(final_centres[:, 0], final_centres[:, 2], '-', color='steelblue',
            linewidth=1.0, alpha=0.8, label='optimised trajectory')
    ax.plot(gt_centres[:, 0], gt_centres[:, 2], '--',
            color='orange', linewidth=1.2, alpha=0.9, label='ground truth')
    for k in keyframes:
        P = marginals.marginalCovariance(cam_key(k))    # 6×6: rot[3] | trans[3]
        Ptxz = P[3:6, 3:6][np.ix_([0, 2], [0, 2])]      # 2×2 (X, Z) trans block
        cx, cz = final_centres[keyframes.index(k), 0], final_centres[keyframes.index(k), 2]
        vals, vecs = np.linalg.eigh(Ptxz)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * ELLIPSE_NSIGMA * np.sqrt(np.maximum(vals, 0))
        ax.add_patch(Ellipse((cx, cz), w, h, angle=angle,
                             facecolor='none', edgecolor='red',
                             linewidth=0.6, alpha=0.7))
    ax.scatter([0], [0], c='black', s=80, marker='s', zorder=5, label='c_0')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z — forward (m)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('Q6.2 — keyframes with marginal covariances')
    ax.legend()
    _save(fig, 'q6_2_final_with_cov')


def main():
    K, m1, m2 = read_cameras()
    K_stereo = stereo_calibration(K, -m2[0, 3])

    db = TrackingDB(); db.load(TRACKING_DB_BASE)
    img_dir = os.path.join(DATA_PATH, 'image_0')
    n_frames = min(sum(1 for f in os.listdir(img_dir) if f.endswith('.png')),
                   db.frame_num())
    pnp_poses = np.load(PNP_POSES_PATH)[:n_frames]
    # The tracking DB now stores each keypoint's size on its Link, so the
    # separate feature-size detection pass is only needed for older DBs.
    _lk = next(iter(db.linkId_to_link.values()), None)
    feature_sizes = (None if _lk is not None and getattr(_lk, 'size', None)
                     else compute_feature_sizes(n_frames, FEATURE_SIZES_PATH))

    keyframes = select_keyframes_in_calm_frames(
        pnp_poses, min_translation=5.0,
        translation_jitter=1.0, seed=42,
        max_frames_gap=19, min_frames_gap=8,
        straight_rot_rate_deg=0.5,
    )
    print(f'Keyframes: {len(keyframes)}')

    print('\nQ6.1')
    q1(db, pnp_poses, keyframes, K_stereo, feature_sizes)

    print('\nQ6.2')
    q2(db, pnp_poses, keyframes, K_stereo, feature_sizes)


if __name__ == '__main__':
    main()
