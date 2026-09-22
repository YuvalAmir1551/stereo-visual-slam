"""Figures summarising the SLAM pipeline results.

Reads cached analysis data (under ``<dataset>/report_data/``) and renders every
summary figure into ``docs/``. Each ``figNN_*`` function creates
exactly one figure.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import DATA_PATH

DATA_DIR = os.path.join(DATA_PATH, 'report_data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')

# Categorical palette (fixed role -> hue assignment across the figures)
C_PNP = '#2a78d6'      # blue    — PnP estimation
C_BA = '#eda100'       # yellow  — bundle adjustment
C_PG = '#4a3aa7'       # violet  — pose graph without LC
C_LC = '#e34948'       # red     — pose graph with LC
C_GT = '#008300'       # green   — ground truth
C_2ND = '#1baf7a'      # aqua    — second series (right image / after-opt)
INK, INK2 = '#0b0b0b', '#52514e'

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.titlesize': 11, 'legend.frameon': False,
})


# project
def _load(name):
    return np.load(os.path.join(DATA_DIR, name), allow_pickle=True)


# project
def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name + '.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}.png')


# project
def fig01_matches_per_frame():
    d = _load('report_tracking.npz')
    y = d['links_per_frame']
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(np.arange(len(y)), y, color=C_PNP, lw=0.9)
    ax.axhline(y.mean(), color=INK2, lw=1.2, ls='--',
               label=f'mean {y.mean():.0f}')
    ax.set_xlabel('frame [index]')
    ax.set_ylabel('matches [count]')
    ax.set_title('Stereo-inlier matches per frame')
    ax.legend()
    _save(fig, 'fig01_matches_per_frame')


# project
def fig02_inlier_pct_per_frame():
    d = _load('report_tracking.npz')
    y = d['inlier_pct']
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(np.arange(len(y)), y, color=C_PNP, lw=0.9)
    ax.axhline(np.mean(y), color=INK2, lw=1.2, ls='--',
               label=f'mean {np.mean(y):.1f}%')
    ax.set_xlabel('frame [index]')
    ax.set_ylabel('inliers [%]')
    ax.set_title('PnP-RANSAC inlier rate per frame')
    ax.legend()
    _save(fig, 'fig02_inlier_pct_per_frame')


# project
def fig03_connectivity():
    d = _load('report_tracking.npz')
    y = d['connectivity']
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(np.arange(len(y)), y, color=C_PNP, lw=0.9)
    ax.axhline(y.mean(), color=INK2, lw=1.2, ls='--',
               label=f'mean {y.mean():.0f}')
    ax.set_xlabel('frame [index]')
    ax.set_ylabel('outgoing tracks [count]')
    ax.set_title('Connectivity: outgoing tracks per frame')
    ax.legend()
    _save(fig, 'fig03_connectivity')


# project
def fig04_track_length_histogram():
    d = _load('report_tracking.npz')
    tl = d['track_lengths']
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(2, tl.max() + 2) - 0.5
    ax.hist(tl, bins=bins, color=C_PNP, edgecolor='white', linewidth=0.3)
    ax.set_yscale('log')
    ax.set_xlabel('track length [frames]')
    ax.set_ylabel('tracks [count]')
    ax.set_title('Track-length histogram (log scale)')
    _save(fig, 'fig04_track_length_histogram')


# project
def fig05_trajectories():
    e = np.load(os.path.join(DATA_DIR, 'lc_eval.npz'))
    from geometry import camera_center
    pnp = np.load(os.path.join(DATA_PATH, 'pnp_poses.npy'))
    pnp_cent = np.array([camera_center(p) for p in pnp])
    fig, ax = plt.subplots(figsize=(9, 9))
    gt = e['gt']
    ax.plot(gt[:, 0], gt[:, 2], color=C_GT, lw=1.8, ls='--',
            label='ground truth', zorder=5)
    ax.plot(pnp_cent[:, 0], pnp_cent[:, 2], color=C_PNP, lw=1.2,
            label='PnP (per frame)')
    ax.plot(e['cent_no'][:, 0], e['cent_no'][:, 2], color=C_BA, lw=1.2,
            label='bundle / pose graph, no LC (keyframes)')
    ax.plot(e['cent_ak'][:, 0], e['cent_ak'][:, 2], color=C_LC, lw=1.4,
            label='pose graph with loop closures')
    ax.scatter([0], [0], c=INK, marker='s', s=60, zorder=6, label='start')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.set_title('Estimated trajectories vs ground truth (top-down view)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3)
    _save(fig, 'fig05_trajectories')


# project
def fig06_mean_factor_error():
    d = _load('report_bundles.npz')
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d['win_start'], d['fact_err0'], color=C_BA, lw=1.2,
            label='initial (before optimization)')
    ax.plot(d['win_start'], d['fact_err1'], color=C_PNP, lw=1.2,
            label='optimized')
    ax.set_yscale('log')
    ax.set_xlabel('bundle window starting keyframe [frame id]')
    ax.set_ylabel('factor error [—]')
    ax.set_title('Mean factor error per bundle window (log scale)')
    ax.legend()
    _save(fig, 'fig06_mean_factor_error')


# project
def fig07_median_projection_error():
    d = _load('report_bundles.npz')
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d['win_start'], d['proj_err0'], color=C_BA, lw=1.2,
            label='initial (before optimization)')
    ax.plot(d['win_start'], d['proj_err1'], color=C_PNP, lw=1.2,
            label='optimized')
    ax.set_yscale('log')
    ax.set_ylim(*_projerr_bounds())
    ax.set_xlabel('bundle window starting keyframe [frame id]')
    ax.set_ylabel('projection error [px]')
    ax.set_title('Median projection error per bundle window (log scale)')
    ax.legend()
    _save(fig, 'fig07_median_projection_error')


# project
def _projerr_bounds():
    """Shared log-scale y-limits for every median-projection-error plot
    (figs 7-9; spec: same scale for comparable graphs)."""
    d = _load('report_projdist.npz')
    b = _load('report_bundles.npz')
    sel = d['pnp_dists'] <= 40
    vals = np.concatenate([d['pnp_med_l'][sel], d['pnp_med_r'][sel],
                           d['ba_median'],
                           b['proj_err0'][np.isfinite(b['proj_err0'])],
                           b['proj_err1'][np.isfinite(b['proj_err1'])]])
    # exclude degenerate near-zero values (e.g. reprojection at distance 0
    # into the triangulation frame itself is exact by construction)
    vals = vals[vals > 1e-3]
    return 0.85 * vals.min(), 1.15 * vals.max()


# project
def fig08_projdist_pnp():
    d = _load('report_projdist.npz')
    # skip distance 0: reprojection into the triangulation frame itself is
    # exact by construction and carries no information
    s = d['pnp_dists'] >= 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d['pnp_dists'][s], d['pnp_med_l'][s], color=C_PNP, lw=1.6,
            label='left')
    ax.plot(d['pnp_dists'][s], d['pnp_med_r'][s], color=C_2ND, lw=1.6,
            label='right')
    ax.set_yscale('log')
    ax.set_xlabel('distance from triangulation frame [frames]')
    ax.set_ylabel('projection error [px]')
    ax.set_xlim(0, 40)
    ax.set_ylim(*_projerr_bounds())
    ax.set_title('PnP: median projection error vs distance (log scale)')
    ax.legend()
    _save(fig, 'fig08_projdist_pnp')


# project
def fig09_projdist_bundle():
    d = _load('report_projdist.npz')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d['ba_dists'], d['ba_median'], color=C_PNP, lw=1.6,
            label='after optimization (left+right combined)')
    ax.set_yscale('log')
    ax.set_xlabel('distance from first frame of bundle window [frames]')
    ax.set_ylabel('projection error [px]')
    ax.set_ylim(*_projerr_bounds())
    ax.set_title('Bundle: median projection error vs distance (log scale)')
    ax.legend()
    _save(fig, 'fig09_projdist_bundle')


# project
def _abs_error_panels(xyz_err, ang_err, x, xlabel, name, title,
                      ylim_loc=None, ylim_ang=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    norm = np.linalg.norm(xyz_err, axis=1)
    for i, (lbl, color) in enumerate((('X', C_PNP), ('Y', C_2ND), ('Z', C_BA))):
        axes[0].plot(x, np.abs(xyz_err[:, i]), color=color, lw=1.0, label=lbl)
    axes[0].plot(x, norm, color=C_LC, lw=1.4, label='norm')
    axes[0].set_ylabel('location error [m]')
    axes[0].set_title(title)
    axes[0].legend(ncol=4)
    axes[1].plot(x, ang_err, color=C_PNP, lw=1.2)
    axes[1].set_ylabel('angle error [deg]')
    axes[1].set_xlabel(xlabel)
    if ylim_loc:
        axes[0].set_ylim(0, ylim_loc)
    if ylim_ang:
        axes[1].set_ylim(0, ylim_ang)
    _save(fig, name)


# project
def _pg_shared_ylims():
    """Shared y-limits for figs 11-12 (same scale for comparable graphs)."""
    d = _load('report_abs_errors.npz')
    loc = 1.05 * np.linalg.norm(d['pg_no_xyz'], axis=1).max()
    ang = 1.05 * d['pg_no_ang'].max()
    return loc, ang


# project
def fig10_abs_pnp_error():
    d = _load('report_abs_errors.npz')
    _abs_error_panels(d['pnp_err_xyz'], d['pnp_ang'], d['frames'],
                      'frame [index]', 'fig10_abs_pnp_error',
                      'Absolute PnP estimation error')


# project
def fig11_abs_pg_no_lc_error():
    d = _load('report_abs_errors.npz')
    loc, ang = _pg_shared_ylims()
    _abs_error_panels(d['pg_no_xyz'], d['pg_no_ang'], d['keyframes'],
                      'keyframe [frame id]', 'fig11_abs_pg_no_lc_error',
                      'Absolute pose-graph error, without loop closures',
                      ylim_loc=loc, ylim_ang=ang)


# project
def fig12_abs_pg_lc_error():
    d = _load('report_abs_errors.npz')
    loc, ang = _pg_shared_ylims()
    _abs_error_panels(d['pg_lc_xyz'], d['pg_lc_ang'], d['keyframes'],
                      'keyframe [frame id]', 'fig12_abs_pg_lc_error',
                      'Absolute pose-graph error, with loop closures '
                      '(same scale as the previous figure)',
                      ylim_loc=loc, ylim_ang=ang)


# project
def fig13_relative_kf_errors():
    d = _load('report_rel_errors.npz')
    x = d['kf_pairs_start']
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(x, d['loc_pnp'], color=C_PNP, lw=1.2, label='PnP (accumulated)')
    axes[0].plot(x, d['loc_ba'], color=C_BA, lw=1.2, label='bundle')
    axes[0].set_ylabel('relative location error [m]')
    axes[0].set_title('Relative pose error between consecutive keyframes',
                      loc='left')
    axes[0].legend()
    axes[1].plot(x, d['ang_pnp'], color=C_PNP, lw=1.2, label='PnP (accumulated)')
    axes[1].plot(x, d['ang_ba'], color=C_BA, lw=1.2, label='bundle')
    axes[1].set_ylabel('relative angle error [deg]')
    axes[1].set_xlabel('first keyframe of pair [frame id]')
    axes[1].legend()
    _save(fig, 'fig13_relative_kf_errors')


# project
def _subsection_panels(d, prefix, name, title):
    colors = {100: C_PNP, 400: C_2ND, 800: C_BA}
    # shared y-limits across the PnP and Bundle figures (same scale rule)
    loc_max = 1.05 * max(d[f'{p}_{L}_loc'].max()
                         for p in ('pnp', 'ba') for L in (100, 400, 800))
    ang_max = 1.05 * max(d[f'{p}_{L}_ang'].max()
                         for p in ('pnp', 'ba') for L in (100, 400, 800))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for L in (100, 400, 800):
        s = d[f'{prefix}_{L}_starts']; loc = d[f'{prefix}_{L}_loc']
        ang = d[f'{prefix}_{L}_ang']
        axes[0].plot(s, loc, color=colors[L], lw=1.1,
                     label=f'len {L} (avg {loc.mean():.2f}%)')
        axes[1].plot(s, ang, color=colors[L], lw=1.1,
                     label=f'len {L} (avg {ang.mean():.4f} deg/m)')
    axes[0].set_ylabel('location error [%]')
    axes[0].set_ylim(0, loc_max)
    axes[0].set_title(title)
    axes[1].set_ylabel('angle error [deg/m]')
    axes[1].set_ylim(0, ang_max)
    axes[1].set_xlabel('sequence start frame [index]')
    for ax in axes:
        ax.legend(ncol=3, fontsize=9)
    _save(fig, name)


# project
def fig14_subsections_pnp():
    _subsection_panels(_load('report_subsections.npz'), 'pnp',
                       'fig14_subsections_pnp',
                       'Relative PnP error over sub-sections (KITTI metric)')


# project
def fig15_subsections_bundle():
    _subsection_panels(_load('report_subsections.npz'), 'ba',
                       'fig15_subsections_bundle',
                       'Relative bundle error over sub-sections (KITTI metric)')


# project
def fig16_lc_stats():
    d = _load('report_lc_stats.npz')
    labels = [f"{i}↔{n}" for i, n in
              zip(d['lc_frames_i'], d['lc_frames_n'])]
    x = np.arange(len(labels))
    pct = 100.0 * d['lc_inliers'] / d['lc_matches']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, d['lc_matches'], color=C_PNP, label='matches')
    axes[0].bar(x, d['lc_inliers'], color=C_2ND, label='RANSAC inliers')
    axes[0].set_ylabel('matches [count]')
    axes[0].legend()
    axes[1].bar(x, pct, color=C_PNP)
    axes[1].set_ylabel('inliers [% of matches]')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, fontsize=7.5)
        ax.set_xlabel('loop closure pair [frame ids]')
    _save(fig, 'fig16_lc_stats')


# project
def fig17_uncertainty():
    d = _load('report_uncertainty.npz')
    lcs = _load('report_lc_stats.npz')
    lc_kfs = sorted(set(lcs['lc_frames_i'].tolist())
                    | set(lcs['lc_frames_n'].tolist()))
    kf = d['keyframes']
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(kf, d['loc_u_no'], color=C_PG, lw=1.3, label='without LC')
    axes[0].plot(kf, d['loc_u_lc'], color=C_LC, lw=1.3, label='with LC')
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$\sqrt{\det\Sigma_{t}}$ [m$^3$]')
    axes[0].set_title('Location uncertainty per keyframe (log scale)')
    axes[1].plot(kf, np.degrees(d['ang_u_no'] ** (1 / 3)) ** 3, color=C_PG,
                 lw=1.3, label='without LC')
    axes[1].plot(kf, np.degrees(d['ang_u_lc'] ** (1 / 3)) ** 3, color=C_LC,
                 lw=1.3, label='with LC')
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$\sqrt{\det\Sigma_{R}}$ [deg$^3$]')
    axes[1].set_title('Angle uncertainty per keyframe (log scale)')
    axes[1].set_xlabel('keyframe [frame id]')
    for ax in axes:
        lo, _ = ax.get_ylim()
        ax.scatter(lc_kfs, [lo * 1.5] * len(lc_kfs), marker='o', s=22,
                   color=INK, zorder=5, label='loop closure location')
        ax.legend()
    _save(fig, 'fig17_uncertainty')


# project
def fig18_dl_campaign():
    d = _load('campaign_eval.npz')
    kf, gt = d['keyframes'], d['gt']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    ax = axes[0]
    ax.plot(gt[:, 0], gt[:, 2], '--', color=C_GT, lw=1.6, label='ground truth')
    ax.plot(d['cent_old_nolc'][:, 0], d['cent_old_nolc'][:, 2], color=C_BA,
            lw=1.3, label='odometry, AKAZE tracks only')
    ax.plot(d['cent_new_nolc'][:, 0], d['cent_new_nolc'][:, 2], color=C_PG,
            lw=1.3, label='odometry + learned anchors')
    ax.scatter([0], [0], c=INK, marker='s', s=50, zorder=5, label='start')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.set_title('Open-loop trajectory (no loop closures)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=8.5)
    ax = axes[1]
    ax.plot(kf, d['err_old_nolc'], color=C_BA, lw=1.2,
            label='no LC, AKAZE tracks only')
    ax.plot(kf, d['err_new_nolc'], color=C_PG, lw=1.2,
            label='no LC, + learned anchors')
    ax.plot(kf, d['err_old_lc'], color=C_LC, lw=1.2, ls=':',
            label='with LC, AKAZE tracks only')
    ax.plot(kf, d['err_new_lc'], color=C_LC, lw=1.2,
            label='with LC, + learned anchors')
    ax.set_xlabel('keyframe [frame id]')
    ax.set_ylabel('absolute location error [m]')
    ax.set_title('Absolute error per keyframe')
    ax.legend(fontsize=8.5)
    _save(fig, 'fig18_dl_campaign')


# project
def main():
    for fn in sorted(k for k in globals() if k.startswith('fig')):
        globals()[fn]()


if __name__ == '__main__':
    main()
