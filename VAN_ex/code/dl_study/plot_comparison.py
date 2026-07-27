"""Comparison figures: AKAZE vs SuperPoint+LightGlue front-ends."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset', 'sequences', '00', 'report_data')
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'project_figures')

C_AKAZE = '#2a78d6'   # categorical slot 1 (blue)
C_SPLG = '#1baf7a'    # categorical slot 2 (aqua)
INK = '#0b0b0b'
INK2 = '#52514e'
SURFACE = '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.titlesize': 11,
})

with open(f"{SCRATCH}/frontend_comparison.json") as f:
    per_frame = json.load(f)

def series(method, key):
    xs = [r['frame'] for r in per_frame[method] if r[key] is not None]
    ys = [r[key] for r in per_frame[method] if r[key] is not None]
    return np.array(xs), np.array(ys)

fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
panels = [
    ('stereo_inl_pct', 'Stereo inlier rate (rectified filter)', 'inliers [%]', False),
    ('consensus', '4-view consensus matches', 'matches [count]', False),
    ('pnp_inl_pct', 'PnP-RANSAC inlier rate', 'inliers [%]', False),
    ('ang_err', 'Relative rotation error vs GT (per transition)', 'rotation error [deg]', True),
]
for ax, (key, title, ylabel, logy) in zip(axes.flat, panels):
    for method, color in (('AKAZE', C_AKAZE), ('SP+LG', C_SPLG)):
        x, y = series(method, key)
        ax.plot(x, y, color=color, lw=1.6, label=method)
    ax.set_title(title)
    ax.set_xlabel('frame index')
    ax.set_ylabel(ylabel)
    ax.margins(x=0.08)
    if logy:
        ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=8.5, loc='best')
fig.suptitle('Front-end comparison on KITTI 00 — 66 consecutive-frame transitions (every 50th frame)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{FIGS}/fig_frontend_per_frame.png", dpi=150)
print("wrote fig_frontend_per_frame.png")

# ---- gap sweep figure (run after compare_gaps.py finishes) ----
try:
    with open(f"{SCRATCH}/gap_comparison.json") as f:
        gaps_data = json.load(f)
except FileNotFoundError:
    print("gap_comparison.json not there yet - skipping gap figure")
    raise SystemExit

GAPS = sorted({r['gap'] for r in gaps_data['AKAZE']})

def gap_stats(method):
    cons, ang, loc, fails = [], [], [], []
    for g in GAPS:
        rs = [r for r in gaps_data[method] if r['gap'] == g]
        ok = [r for r in rs if not r.get('failed')]
        cons.append(np.mean([r['consensus'] for r in rs]))
        ang.append(np.median([r['ang_err'] for r in ok]) if ok else np.nan)
        loc.append(np.median([r['loc_err'] for r in ok]) if ok else np.nan)
        fails.append(100.0 * (len(rs) - len(ok)) / len(rs))
    return map(np.array, (cons, ang, loc, fails))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
titles = [('Consensus matches vs frame gap', 'matches [count]'),
          ('Median rotation error vs frame gap', 'rotation error [deg]'),
          ('Median location error vs frame gap', 'location error [m]')]
for method, color in (('AKAZE', C_AKAZE), ('SP+LG', C_SPLG)):
    cons, ang, loc, fails = gap_stats(method)
    for ax, y in zip(axes, (cons, ang, loc)):
        ax.plot(GAPS, y, color=color, lw=1.8, marker='o', ms=5, label=method)
for ax, (title, ylabel) in zip(axes, titles):
    ax.set_title(title)
    ax.set_xlabel('frame gap [frames]')
    ax.set_ylabel(ylabel)
    ax.set_xticks(GAPS)
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=8.5)
fig.suptitle('Wide-baseline robustness — 32 start frames per gap, KITTI 00 (log scale)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{FIGS}/fig_gap_sweep.png", dpi=150)
print("wrote fig_gap_sweep.png")
