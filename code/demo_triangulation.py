"""Rectified-stereo outlier rejection and 3D triangulation."""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from dataset import read_images, read_cameras
from features import (
    match_stereo_pair,
    rectified_stereo_filter,
    pts_from_matches,
)
from geometry import triangulate_linear_lsq, triangulate_cv2
from plot import plot_3d_world

DETECTOR = 'AKAZE'
Y_THRESHOLD = 2.0          # px — rectified-stereo deviation cutoff (Q2.1, Q2.2)
Q4_FRAMES = [0, 500, 700]  # frames to triangulate for Q2.4
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'ex2_figures')


def _save(fig, name):
    """Write a figure to FIGURES_DIR/<name>.png at presentation resolution."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name + '.png'), dpi=150, bbox_inches='tight')


def plot_y_dev_histogram(dy, threshold=Y_THRESHOLD, bins=80):
    """Histogram of |y_left − y_right| across all matches."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dy, bins=bins, color='steelblue', edgecolor='white')
    ax.axvline(threshold, color='red', linestyle='--',
               label=f'threshold = {threshold} px')
    ax.set_xlabel('|y_left − y_right|  (pixels)')
    ax.set_ylabel('Number of matches')
    ax.set_title(f'Q2.1 — Deviation from rectified stereo pattern  (n = {len(dy)})')
    ax.legend()
    return fig


def plot_inlier_outlier(img_left, img_right, kp_left, kp_right,
                        inliers, outliers, suptitle=''):
    """Show all matches as dots: outliers cyan (drawn first), inliers orange (on top)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    in_pts_l, in_pts_r = pts_from_matches(kp_left, kp_right, inliers)
    out_pts_l, out_pts_r = pts_from_matches(kp_left, kp_right, outliers) if outliers \
        else (np.zeros((0, 2)), np.zeros((0, 2)))

    for ax, img, side, pts_in, pts_out in [
        (axes[0], img_left,  'Left',  in_pts_l, out_pts_l),
        (axes[1], img_right, 'Right', in_pts_r, out_pts_r),
    ]:
        ax.imshow(img, cmap='gray')
        if len(pts_out):
            ax.scatter(pts_out[:, 0], pts_out[:, 1], c='cyan', s=8,
                       label=f'outliers ({len(pts_out)})')
        ax.scatter(pts_in[:, 0], pts_in[:, 1], c='orange', s=8,
                   label=f'inliers ({len(pts_in)})')
        ax.set_title(f'{side} image')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig

def q1(kp_left, kp_right, matches):
    """Q2.1 — Histogram of y-deviations and percent of matches deviating > 2 px.

    On a rectified pair the epipolar lines are horizontal, so corresponding
    points share the same image row. Correct matches therefore concentrate
    near |Δy| = 0; everything else is matching error.
    """
    _, _, dy = rectified_stereo_filter(kp_left, kp_right, matches, y_threshold=Y_THRESHOLD)
    pct_above = 100.0 * (dy > Y_THRESHOLD).sum() / len(dy)
    print(f"Total matches: {len(dy)}")
    print(f"Matches with |Δy| > {Y_THRESHOLD} px: {(dy > Y_THRESHOLD).sum()} "
          f"({pct_above:.2f}%)")
    print(f"|Δy| stats — median: {np.median(dy):.2f} px, max: {dy.max():.2f} px")
    fig = plot_y_dev_histogram(dy, threshold=Y_THRESHOLD)
    _save(fig, 'q2_1_y_dev_histogram')


def q2(img_left, img_right, kp_left, kp_right, matches):
    """Q2.2 — Reject matches by rectified-stereo pattern; reasoning about uniform errors.

    Returns the accepted (inlier) matches for re-use in Q2.3.
    """
    inliers, outliers, _ = rectified_stereo_filter(
        kp_left, kp_right, matches, y_threshold=Y_THRESHOLD)
    print(f"Inliers: {len(inliers)}, outliers: {len(outliers)}")

    fig = plot_inlier_outlier(img_left, img_right, kp_left, kp_right, inliers, outliers,
                              suptitle='Q2.2 — Inliers (orange) on top of outliers (cyan)')
    _save(fig, 'q2_2_inliers_outliers')
    return inliers


def q3(K, P_left, P_right, kp_left, kp_right, inliers, img_shape):
    """Q2.3 — Linear-LSQ triangulation vs cv2.triangulatePoints."""
    pts_l, pts_r = pts_from_matches(kp_left, kp_right, inliers)
    X_lsq = triangulate_linear_lsq(P_left, P_right, pts_l, pts_r)
    X_cv2 = triangulate_cv2(P_left, P_right, pts_l, pts_r)

    diff = np.linalg.norm(X_lsq - X_cv2, axis=1)
    print(f"Triangulated {len(X_lsq)} points. "
          f"|Linear LSQ − cv2| per point: "
          f"median={np.median(diff):.2e} m, mean={np.mean(diff):.2e} m, "
          f"max={np.max(diff):.2e} m")

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_world(X_lsq, ax=ax1, K=K, img_shape=img_shape, title='Linear LSQ')
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_world(X_cv2, ax=ax2, K=K, img_shape=img_shape, title='OpenCV')
    fig.suptitle('Q2.3 — Triangulation comparison (Linear LSQ vs OpenCV)')
    fig.tight_layout()
    _save(fig, 'q2_3_triangulation_compare')


def _draw_matches_3class(ax, img, y_out, x_out, x_in, side):
    """Overlay matches on an image with 3-class coloring.

    Drawing order (back-to-front): cyan Y-outliers, red X-outliers, orange good.

    Args:
        y_out: Nx2 left/right pixel coords of matches that fail the Y filter.
        x_out: Nx2 of matches that pass Y but have x_left ≤ x_right (X outlier).
        x_in:  Nx2 of matches that pass both checks.
    """
    ax.imshow(img, cmap='gray')
    if len(y_out):
        ax.scatter(y_out[:, 0], y_out[:, 1], c='cyan', s=6,
                   label=f'Y-outlier ({len(y_out)})')
    if len(x_out):
        ax.scatter(x_out[:, 0], x_out[:, 1], c='red', s=8,
                   label=f'X-outlier (Z≤0) ({len(x_out)})')
    ax.scatter(x_in[:, 0], x_in[:, 1], c='orange', s=6,
               label=f'good ({len(x_in)})')
    ax.set_title(f'{side} image')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)


def q4(K, P_left, P_right, img_shape):
    """Q2.4 — Run match-and-triangulate over a few frames; observe erroneous points.

    Each frame's figure has three panels:
      • top-left  — left image, matches in 3 classes:
                    cyan = Y-outlier, red = passes Y but x_l ≤ x_r → Z ≤ 0,
                    orange = passes both checks.
      • top-right — right image, same coloring.
      • bottom    — 3D cloud of the Y-inliers; orange = good, red = X-outlier.
    """
    for idx in Q4_FRAMES:
        img_l, img_r = read_images(idx)
        kp_l, kp_r, matches = match_stereo_pair(img_l, img_r, detector=DETECTOR)
        inliers, outliers, _ = rectified_stereo_filter(kp_l, kp_r, matches, Y_THRESHOLD)
        in_l, in_r = pts_from_matches(kp_l, kp_r, inliers)
        y_out_l, y_out_r = (pts_from_matches(kp_l, kp_r, outliers) if outliers
                            else (np.empty((0, 2)), np.empty((0, 2))))
        # X-direction sanity check: for KITTI rectified stereo, real points
        # require x_left > x_right.
        x_diff = in_l[:, 0] - in_r[:, 0]
        x_in_mask = x_diff > 0

        X = triangulate_linear_lsq(P_left, P_right, in_l, in_r)

        n_x_in = int(x_in_mask.sum())
        n_x_out = int((~x_in_mask).sum())
        print(f"Frame {idx:>4d}: {len(matches)} matches → {len(inliers)} Y-inliers "
              f"({n_x_in} good + {n_x_out} X-outliers) + {len(outliers)} Y-outliers")

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.2])
        ax_l = fig.add_subplot(gs[0, 0])
        ax_r = fig.add_subplot(gs[0, 1])
        ax_3d = fig.add_subplot(gs[1, :], projection='3d')

        _draw_matches_3class(ax_l, img_l, y_out_l, in_l[~x_in_mask], in_l[x_in_mask], 'Left')
        _draw_matches_3class(ax_r, img_r, y_out_r, in_r[~x_in_mask], in_r[x_in_mask], 'Right')
        plot_3d_world(X, ax=ax_3d, K=K, img_shape=img_shape, title='3D point cloud',
                      highlight_mask=~x_in_mask)

        fig.suptitle(f'Q2.4 — Frame {idx}  (3-class colouring: cyan = Y-out, '
                     f'red = X-out, orange = good)')
        fig.tight_layout()
        _save(fig, f'q2_4_frame_{idx:04d}')


def main():
    K, m1, m2 = read_cameras()
    P_left, P_right = K @ m1, K @ m2
    img_left, img_right = read_images(0)
    kp_left, kp_right, matches = match_stereo_pair(img_left, img_right, detector=DETECTOR)

    print("Q2.1")
    q1(kp_left, kp_right, matches)
    print("\nQ2.2")
    inliers = q2(img_left, img_right, kp_left, kp_right, matches)
    print("\nQ2.3")
    q3(K, P_left, P_right, kp_left, kp_right, inliers, img_shape=img_left.shape)
    print(f"\nQ2.4 — frames {Q4_FRAMES}")
    q4(K, P_left, P_right, img_shape=img_left.shape)

    plt.show()


if __name__ == '__main__':
    main()
