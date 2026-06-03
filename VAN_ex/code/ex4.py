"""Exercise 4: Multi-frame feature tracking database built on ex3's RANSAC PnP."""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from dataset import read_images, read_cameras, read_poses, DATA_PATH
from features import extract_features, match_descriptors, rectified_stereo_filter
from geometry import (
    triangulate_linear_lsq,
    compose_extrinsics,
    project,
)
from pnp import ransac_pnp
from tracking_database import TrackingDB

DETECTOR = 'AKAZE'
Y_THRESHOLD = 2.0       # px — rectified-stereo vertical-deviation cutoff (ex2)
X_MIN_DISPARITY = 1.0   # px — minimum stereo disparity; drops far points whose
                        # sub-pixel disparity noise dominates triangulated depth.
PIX_THRESHOLD = 2.0     # px — per-image supporter reprojection cutoff (ex3 RANSAC)
N_FRAMES_FULL = None    # None → use every frame found on disk; integer → cap (debug)

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
FIGURES_DIR = os.path.join(DOCS_DIR, 'ex4_figures')
DB_BASE = os.path.join(DATA_PATH, 'tracking_db')

def _save(fig, name):
    """Write a figure to FIGURES_DIR/<name>.png at presentation resolution."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name + '.png'), dpi=150, bbox_inches='tight')

def _stereo_for_db(img_l, img_r, detector=DETECTOR):
    """Stereo-pair feature extraction + rectified filter, packaged for TrackingDB.add_frame.

    Returns:
        features: Nx<desc_dim> descriptor array, one row per surviving feature
            on the left image (rows are dropped if the feature has no stereo inlier).
        links: List[Link] of length N — the (x_l, x_r, y) record for each row.
    """
    kp_l, des_l = extract_features(img_l, detector=detector)
    kp_r, des_r = extract_features(img_r, detector=detector)
    matches = match_descriptors(des_l, des_r, detector=detector)
    inliers, _, _ = rectified_stereo_filter(
        kp_l, kp_r, matches,
        y_threshold=Y_THRESHOLD, x_min_disparity=X_MIN_DISPARITY)
    filtered_features, links = TrackingDB.create_links(des_l, kp_l, kp_r, inliers)
    return filtered_features, links


def _links_to_pixels(links):
    """Convert List[Link] → (Nx2 left-pixel array, Nx2 right-pixel array)."""
    pl = np.array([[lk.x_left, lk.y] for lk in links], dtype=np.float64)
    pr = np.array([[lk.x_right, lk.y] for lk in links], dtype=np.float64)
    return pl, pr


def build_db(n_frames, verbose_every=200):
    """Build the tracking DB for the first n_frames stereo pairs of KITTI sequence 00.

    Per frame transition i → i+1:
      1. Stereo-filter frame i+1 → filtered descriptor matrix + Link list.
      2. Match frame i's filtered descriptors against frame i+1's (cross-frame).
      3. Triangulate the prev links to get 3D points in frame-i coords.
      4. Run ex3.ransac_pnp on the cross-matched 4-view consensus.
      5. Hand the inlier mask + cross matches to TrackingDB.add_frame.

    Returns:
        db: the populated TrackingDB.
        inliers_per_frame: 1-D float array, RANSAC inlier % per transition.
    """
    K, m1, m2 = read_cameras()
    P_left, P_right = K @ m1, K @ m2

    db = TrackingDB()
    rng = np.random.default_rng(0)
    inliers_per_frame = []
    start = time.time()

    img_l, img_r = read_images(0)
    feats_prev, links_prev = _stereo_for_db(img_l, img_r)
    db.add_frame(links_prev, feats_prev,
                 matches_to_previous_left=None, inliers=None)

    for i in range(1, n_frames):
        img_l, img_r = read_images(i)
        feats_curr, links_curr = _stereo_for_db(img_l, img_r)

        # Cross-frame match with mutual best-match (crossCheck) — kills
        # ambiguous pairs cheaply; RANSAC then handles the remaining outliers.
        cross = match_descriptors(feats_prev, feats_curr, DETECTOR,
                                  cross_check=True)

        pts_l_prev, pts_r_prev = _links_to_pixels(links_prev)
        pts_l_curr, pts_r_curr = _links_to_pixels(links_curr)
        X_prev = triangulate_linear_lsq(P_left, P_right, pts_l_prev, pts_r_prev)

        # Reindex by cross-match: prev[query] ↔ curr[train].
        q_idx = np.array([m.queryIdx for m in cross], dtype=int)
        t_idx = np.array([m.trainIdx for m in cross], dtype=int)
        Rt, mask = ransac_pnp(
            X_prev[q_idx], pts_l_prev[q_idx], pts_r_prev[q_idx],
            pts_l_curr[t_idx], pts_r_curr[t_idx],
            K, m2, threshold=PIX_THRESHOLD, rng=rng,
        )

        inliers = mask.tolist() if mask is not None else [False] * len(cross)
        inlier_pct = 100.0 * sum(inliers) / max(1, len(inliers))
        inliers_per_frame.append(inlier_pct)

        db.add_frame(links_curr, feats_curr,
                     matches_to_previous_left=cross, inliers=inliers)

        feats_prev, links_prev = feats_curr, links_curr

        if verbose_every and i % verbose_every == 0:
            elapsed = time.time() - start
            print(f'  Frame {i:>4d}/{n_frames}: '
                  f'inliers={inlier_pct:5.1f}%  elapsed={elapsed:6.1f}s '
                  f'({elapsed / i:.2f}s/frame)')

    total = time.time() - start
    print(f'DB built: {db.frame_num()} frames, {db.track_num()} tracks, '
          f'{db.link_num()} links in {total:.1f}s.')
    return db, np.array(inliers_per_frame)


def _track_lengths(db):
    """Length of every track in the DB. Tracks of length < 2 are skipped."""
    return np.array([len(db.frames(tid))
                     for tid in db.all_tracks()
                     if len(db.frames(tid)) >= 2])


def q2(db):
    """Q4.2 — Tracking statistics (excluding trivial length-1 tracks)."""
    track_lens = _track_lengths(db)
    per_frame_counts = np.array([len(db.tracks(fid)) for fid in db.all_frames()])

    print(f'  Total tracks (len ≥ 2): {len(track_lens)}')
    print(f'  Number of frames:       {db.frame_num()}')
    print(f'  Mean track length:      {track_lens.mean():.2f}')
    print(f'  Min track length:       {track_lens.min()}')
    print(f'  Max track length:       {track_lens.max()}')
    print(f'  Mean tracks per frame:  {per_frame_counts.mean():.1f}')
    return track_lens, per_frame_counts


def q3(db, min_length=6, max_length=8):
    """Q4.3 — Pick a track of length ≥ min_length; show the full LEFT frame +
    the tight 20×20 cutout, both from the LEFT camera, for every frame.
    """
    candidates = [tid for tid in db.all_tracks()
                  if min_length <= len(db.frames(tid)) <= max_length]
    if not candidates:
        candidates = [tid for tid in db.all_tracks()
                      if len(db.frames(tid)) >= min_length]
    if not candidates:
        print(f'  No track of length ≥ {min_length} found.')
        return

    candidates.sort(key=lambda t: -len(db.frames(t)))
    tid = candidates[0]
    frames = db.frames(tid)
    if len(frames) > max_length:
        frames = frames[:max_length]
    print(f'  Track #{tid}: length {len(db.frames(tid))} (showing {len(frames)}), '
          f'frames {frames[0]}..{frames[-1]}')

    n = len(frames)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.4 * n),
                             gridspec_kw={'width_ratios': [6, 1]})
    if n == 1:
        axes = axes[np.newaxis, :]

    HALF = 10  # 20×20 cutout

    for i, fid in enumerate(frames):
        link = db.link(fid, tid)
        img_l, _ = read_images(fid)
        h, w = img_l.shape[:2]

        # Full LEFT frame with the feature marked.
        axes[i, 0].imshow(img_l, cmap='gray')
        axes[i, 0].plot(link.x_left, link.y, 'rx', markersize=10, mew=1.5)
        axes[i, 0].axis('off')

        # Tight 20×20 cutout from the same image.
        x0 = max(0, int(round(link.x_left - HALF)))
        y0 = max(0, int(round(link.y - HALF)))
        x1 = min(w, x0 + 2 * HALF)
        y1 = min(h, y0 + 2 * HALF)
        cutout = img_l[y0:y1, x0:x1]
        axes[i, 1].imshow(cutout, cmap='gray', interpolation='nearest')
        axes[i, 1].plot(link.x_left - x0, link.y - y0,
                        'rx', markersize=10, mew=1.5)
        axes[i, 1].axis('off')

    fig.suptitle(f'track #{tid}, frame #{frames[0]}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, 'q4_3_track_cutouts')


def q4(db):
    """Q4.4 — Connectivity graph: count of tracks alive in both frame i and frame i+1."""
    outgoing = []
    for fid in range(db.frame_num() - 1):
        a = set(db.tracks(fid))
        b = set(db.tracks(fid + 1))
        outgoing.append(len(a & b))
    outgoing = np.array(outgoing)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(outgoing, color='steelblue', linewidth=0.7)
    ax.axhline(outgoing.mean(), color='red', linestyle='--',
               label=f'mean = {outgoing.mean():.0f}')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Outgoing tracks')
    ax.set_title('Q4.4 — Connectivity (tracks present in both frame i and i+1)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'q4_4_connectivity')


def q5(inliers_per_frame):
    """Q4.5 — RANSAC inlier percentage per frame transition."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(inliers_per_frame, color='steelblue', linewidth=0.7)
    ax.axhline(inliers_per_frame.mean(), color='red', linestyle='--',
               label=f'mean = {inliers_per_frame.mean():.1f}%')
    ax.set_xlabel('Frame transition i → i+1')
    ax.set_ylabel('Inlier %')
    ax.set_title('Q4.5 — RANSAC inlier percentage per frame transition')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'q4_5_inlier_pct')


def q6(track_lens):
    """Q4.6 — Track-length histogram on a log y-scale."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(2, track_lens.max() + 2)
    ax.hist(track_lens, bins=bins, color='steelblue', edgecolor='white')
    ax.set_yscale('log')
    ax.set_xlabel('Track length')
    ax.set_ylabel('Number of tracks (log scale)')
    ax.set_title('Q4.6 — Track-length histogram (excluding length 1)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'q4_6_track_length_hist')


def _triangulate_from_frame(link, fid, K, m_right, poses_gt):
    """Triangulate a 3D world point from a single stereo link, using GT poses."""
    Rt_left = poses_gt[fid]
    Rt_right = compose_extrinsics(Rt_left, m_right)
    P_left = K @ Rt_left
    P_right = K @ Rt_right
    pl = np.array([[link.x_left, link.y]], dtype=np.float64)
    pr = np.array([[link.x_right, link.y]], dtype=np.float64)
    return triangulate_linear_lsq(P_left, P_right, pl, pr)[0]


def _reproject_to_track(X_world, frames, db, tid, K, m_right, poses_gt):
    """Project a world point onto every frame of a track; return per-frame (err_left, err_right)."""
    err_l, err_r = [], []
    for fid in frames:
        link = db.link(fid, tid)
        Rt_left = poses_gt[fid]
        Rt_right = compose_extrinsics(Rt_left, m_right)
        proj_l = project(K, Rt_left, X_world[np.newaxis, :])[0]
        proj_r = project(K, Rt_right, X_world[np.newaxis, :])[0]
        err_l.append(np.linalg.norm(proj_l - np.array([link.x_left, link.y])))
        err_r.append(np.linalg.norm(proj_r - np.array([link.x_right, link.y])))
    return np.array(err_l), np.array(err_r)


def q7(db, min_length=10, rng=None):
    """Q4.7 — Triangulate from the LAST frame's stereo (using GT cameras), project to every frame.

    Also runs the "what if we triangulated from the FIRST frame instead" comparison —
    the discussion-question half of 4.7.
    """
    K, _, m2 = read_cameras()
    poses_gt = read_poses()
    if rng is None:
        rng = np.random.default_rng(42)

    candidates = [tid for tid in db.all_tracks() if len(db.frames(tid)) >= min_length]
    if not candidates:
        print(f'  No track of length ≥ {min_length} found.')
        return

    tid = int(rng.choice(candidates))
    frames = db.frames(tid)
    print(f'  Track #{tid}: length {len(frames)}, frames {frames[0]}..{frames[-1]}')

    # Triangulate from the LAST frame's stereo pair (as the spec asks).
    X_last = _triangulate_from_frame(db.link(frames[-1], tid), frames[-1], K, m2, poses_gt)
    err_l, err_r = _reproject_to_track(X_last, frames, db, tid, K, m2, poses_gt)

    # Distance from the reference (last) frame: 0 at the reference, growing
    # backwards through the track. Plot in that order so the curve reads
    # 0 at the left and grows to the right (matches the spec figure).
    dist = np.arange(len(frames))[::-1]
    order = np.argsort(dist)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dist[order], err_l[order], '-', color='steelblue', label='Left')
    ax.plot(dist[order], err_r[order], '-', color='orange', label='Right')
    ax.set_xlabel('distance from reference')
    ax.set_ylabel('projection error (pixels)')
    ax.set_title(f'PnP - projection error vs track length (track #{tid})')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    _save(fig, 'q4_7_reprojection_error')

    print(f'  Track #{tid}: ref=LAST(frame {frames[-1]}); '
          f'err at ref={err_l[-1]:.3f}/{err_r[-1]:.3f}; '
          f'err at farthest={err_l[0]:.3f}/{err_r[0]:.3f} (L/R)')


def main():
    from dataset import DATA_PATH
    img_dir = os.path.join(DATA_PATH, 'image_0')
    n_avail = sum(1 for f in os.listdir(img_dir) if f.endswith('.png'))
    n_frames = N_FRAMES_FULL if N_FRAMES_FULL is not None else n_avail

    db_pkl = DB_BASE + '.pkl'
    inliers_npy = DB_BASE + '_inliers.npy'
    if os.path.exists(db_pkl) and os.path.exists(inliers_npy):
        print(f'Loading cached DB from {db_pkl}')
        db = TrackingDB()
        db.load(DB_BASE)
        inliers_per_frame = np.load(inliers_npy)
    else:
        print(f'Building tracking DB for {n_frames} frames…')
        db, inliers_per_frame = build_db(n_frames)
        os.makedirs(os.path.dirname(DB_BASE), exist_ok=True)
        db.serialize(DB_BASE)
        np.save(inliers_npy, inliers_per_frame)

    print('\nQ4.2 — Tracking statistics')
    track_lens, _ = q2(db)

    print('\nQ4.3 — Length-≥6 track visualisation')
    q3(db)

    print('\nQ4.4 — Connectivity graph')
    q4(db)

    print('\nQ4.5 — Inlier % per frame transition')
    q5(inliers_per_frame)

    print('\nQ4.6 — Track-length histogram')
    q6(track_lens)

    print('\nQ4.7 — Reprojection error along a length-≥10 track')
    q7(db)

    plt.show()


if __name__ == '__main__':
    main()
