"""GTSAM helpers for bundle adjustment over the KITTI tracking DB."""

import os
import pickle
import numpy as np
import gtsam

from geometry import compose_extrinsics, camera_center


# AKAZE keypoint diameter → pixel sigma. Rule of thumb: ±3σ ≈ feature radius,
# so σ ≈ diameter / 6. For KITTI AKAZE this maps size 6 → σ≈1 px (sharp corner),
# size 24 → σ≈4 px (large blob).
SIZE_TO_SIGMA = 1.0 / 6.0
SIGMA_FLOOR = 0.5  # px — never trust a measurement more than half a pixel

# Noise models. Pose3 order is (azimuth, pitch, roll, x, y, z).
PRIOR_SIGMAS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
STEREO_SIGMAS = np.array([1.0, 1.0, 1.0])

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_SIGMAS)
STEREO_NOISE = gtsam.noiseModel.Diagonal.Sigmas(STEREO_SIGMAS)


# ex5
def stereo_calibration(K, baseline):
    """Wrap KITTI intrinsics + baseline into a gtsam.Cal3_S2Stereo.

    The ex5 spec writes the last arg as ``-baseline``. That holds when
    ``baseline`` is taken from the right-camera extrinsic ``m2[0, 3]``, which
    on KITTI is the SIGNED value −0.54 m, so ``-baseline = +0.54``. Passing
    +baseline matches GTSAM's convention (right camera at +x relative to left).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return gtsam.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, float(baseline))


# ex5
def Rt_to_gtsam_pose(Rt):
    """Convert our world-to-camera extrinsic [R | t] to a gtsam.Pose3.

    gtsam.Pose3(R, t) represents the camera-to-world transform x_w = R x_c + t,
    so we invert: R_g = R^T, t_g = -R^T t.
    """
    R, t = Rt[:, :3], Rt[:, 3]
    return gtsam.Pose3(gtsam.Rot3(R.T), gtsam.Point3(-R.T @ t))


# ex5
def gtsam_pose_to_Rt(pose):
    """Inverse of Rt_to_gtsam_pose — return a 3x4 world-to-camera extrinsic."""
    R_g = pose.rotation().matrix()
    t_g = pose.translation()
    R = R_g.T
    t = -R @ t_g
    return np.hstack([R, t.reshape(3, 1)])


# ex5
def relative_extrinsic(Rt_ref, Rt_i):
    """Compute Rt of frame i expressed in frame ref's local coordinate system.

    Given world-to-frame extrinsics ``Rt_ref`` and ``Rt_i``::

        R_rel = R_i R_ref^T
        t_rel = t_i − R_i R_ref^T t_ref
    """
    R_ref, t_ref = Rt_ref[:, :3], Rt_ref[:, 3]
    R_i, t_i = Rt_i[:, :3], Rt_i[:, 3]
    R_rel = R_i @ R_ref.T
    t_rel = t_i - R_rel @ t_ref
    return np.hstack([R_rel, t_rel.reshape(3, 1)])


# ex5
def _frame_to_frame_rotation_rate(pnp_poses):
    """Per-frame rotation magnitude (rad) between consecutive PnP poses."""
    rots = pnp_poses[:, :3, :3]
    n = len(rots)
    rate = np.zeros(n)
    for i in range(1, n):
        R_rel = rots[i] @ rots[i - 1].T
        cos_theta = (np.trace(R_rel) - 1.0) * 0.5
        cos_theta = max(-1.0, min(1.0, cos_theta))
        rate[i] = float(np.arccos(cos_theta))
    return rate


# ex5
def select_keyframes_in_calm_frames(pnp_poses,
                                    min_translation=5.0,
                                    max_frames_gap=19,
                                    min_frames_gap=8,
                                    straight_rot_rate_deg=0.5,
                                    smooth=5):
    """Pick keyframes at STRAIGHT-section frames; defer when in a corner.

    A bundle's relative pose between its two boundary keyframes is what gets
    chained into the global trajectory, so the boundary keyframes should sit
    on frames whose PnP pose is the most reliable — frames in near-rectilinear
    motion, where there's no fast yaw change to confuse the PnP. Corners then
    naturally land INSIDE bundles, where BA can refine them using observations
    from both sides.

    Algorithm: scan forward. Mark a keyframe when:
      • cumulative translation since the last KF ≥ min_translation, AND
      • min_frames_gap frames have passed, AND
      • the current rotation rate is below the straight threshold.
    If max_frames_gap frames pass without such a moment (long curve), force
    one to keep the window inside the spec's 5–20 frame range.

    Args:
        pnp_poses: Nx3x4 world-to-camera extrinsics.
        min_translation: meters of motion required before the next KF.
        max_frames_gap: hard cap on frames between KFs.
        min_frames_gap: floor on frames between KFs.
        straight_rot_rate_deg: rotation rate (deg / frame) below which the
            current frame is considered "straight" enough to anchor.
        smooth: moving-average window for the rotation rate (so a single
            noisy frame doesn't disqualify a clean section).
    """
    centres = np.array([camera_center(p) for p in pnp_poses])
    rot_rate = _frame_to_frame_rotation_rate(pnp_poses)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        rot_rate = np.convolve(rot_rate, kernel, mode='same')
    straight_threshold = np.deg2rad(straight_rot_rate_deg)

    n = len(centres)
    keyframes = [0]
    last_pos = centres[0]
    last_idx = 0
    for i in range(1, n):
        gap = i - last_idx
        dist = float(np.linalg.norm(centres[i] - last_pos))
        in_straight = rot_rate[i] <= straight_threshold
        force = gap >= max_frames_gap
        soft = (gap >= min_frames_gap and dist >= min_translation and in_straight)
        if force or soft:
            keyframes.append(i)
            last_pos = centres[i]
            last_idx = i
    # Tail handling: always end on frame n-1 without creating a degenerate
    # window. If sliding would over-stretch the previous window past
    # max_frames_gap, redistribute via a midpoint.
    if keyframes[-1] != n - 1:
        last = keyframes[-1]
        gap_to_end = (n - 1) - last
        if gap_to_end >= min_frames_gap:
            keyframes.append(n - 1)
        else:
            prev = keyframes[-2] if len(keyframes) >= 2 else -1
            if (n - 1) - prev <= max_frames_gap:
                keyframes[-1] = n - 1
            else:
                keyframes[-1] = (prev + n - 1) // 2
                keyframes.append(n - 1)
    return keyframes


# ex5 — variable-key conventions used across the bundle layer.
def cam_key(frame_id):
    return gtsam.symbol('c', frame_id)


def lm_key(track_id):
    return gtsam.symbol('q', track_id)


# ex5
def stereo_camera(gtsam_pose, K_stereo):
    """Shorthand: gtsam.StereoCamera(gtsam_pose, K_stereo)."""
    return gtsam.StereoCamera(gtsam_pose, K_stereo)


# ex5
def link_to_stereo_point(link):
    """A TrackingDB Link is (x_left, x_right, y); wrap as a StereoPoint2."""
    return gtsam.StereoPoint2(float(link.x_left),
                              float(link.x_right),
                              float(link.y))


# ex5
def compute_feature_sizes(n_frames, cache_path,
                          detector_threshold=0.0001, verbose_every=200):
    """Run AKAZE on every frame, save (x, y, size) per keypoint to a pickle.

    Returns a dict {frame_id -> Nx3 array}. Reads the cache if present.
    Re-creating it from scratch takes ~5 min for 3300 KITTI frames.
    """
    import cv2
    from dataset import read_images
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            sizes = pickle.load(f)
        if (all(fid in sizes for fid in range(n_frames))
                and sizes[next(iter(sizes))].shape[1] == 3):
            print(f'Loaded feature-size cache from {cache_path}')
            return sizes
        print('Cache schema outdated — rebuilding (need 3 cols: x, y, size)')
    print(f'Building feature-size cache for {n_frames} frames…')
    sizes = {}
    det = cv2.AKAZE_create(threshold=detector_threshold)
    for fid in range(n_frames):
        img_l, _ = read_images(fid)
        kps = det.detect(img_l, None)
        sizes[fid] = np.array(
            [[k.pt[0], k.pt[1], k.size] for k in kps],
            dtype=np.float64) if kps else np.zeros((0, 3))
        if verbose_every and (fid + 1) % verbose_every == 0:
            print(f'  feature sizes: frame {fid + 1}/{n_frames}')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(sizes, f)
    print(f'Saved feature-size cache to {cache_path}')
    return sizes


# ex5
def lookup_feature_size(frame_sizes, x, y, fallback=8.0, tol=2.0):
    """Find the AKAZE keypoint closest to (x, y); return its size in pixels.

    ``frame_sizes`` is the Nx3 array for that frame (cols: x, y, size). Returns
    ``fallback`` if no keypoint is within ``tol`` pixels.
    """
    if len(frame_sizes) == 0:
        return fallback
    diffs = frame_sizes[:, :2] - np.array([x, y])
    d2 = (diffs * diffs).sum(axis=1)
    idx = int(np.argmin(d2))
    if d2[idx] <= tol * tol:
        return float(frame_sizes[idx, 2])
    return fallback


# ex5
def build_bundle_window(db, pnp_poses, frames, K_stereo,
                        feature_sizes,
                        prior_noise=PRIOR_NOISE,
                        size_scale=6.0):
    """Construct the factor graph and initial values for a single bundle window.

    Coordinate system: the first frame in ``frames`` becomes the local origin
    (identity Pose3). Every other frame's initial pose is its PnP global pose
    expressed relative to that first frame.

    For every track that appears in ≥2 of ``frames`` we triangulate an initial
    3D landmark in local coords from the LAST frame in which it appears, then
    add one GenericStereoFactor3D per appearance. Per-link covariance comes
    from the AKAZE keypoint size at that (frame, track) appearance:
    σ = max(SIGMA_FLOOR, size / size_scale).

    A PriorFactorPose3 on the first frame anchors the gauge.

    Args:
        db: TrackingDB.
        pnp_poses: Nx3x4 world-to-camera extrinsics from ex3 PnP.
        frames: list of frame ids in the window (sorted ascending).
        K_stereo: gtsam.Cal3_S2Stereo.
        feature_sizes: dict {frame_id -> Nx3 array of (x, y, size)} from
            compute_feature_sizes.
        prior_noise: gauge-prior noise model.
        size_scale: pixel-σ = max(SIGMA_FLOOR, AKAZE size / size_scale).

    Returns:
        graph, initial, info dict (cam_keys, lm_keys, projection_factors,
        first_frame).
    """
    first = frames[0]
    Rt_ref = pnp_poses[first]

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    cam_keys = {}
    for fid in frames:
        Rt_rel = relative_extrinsic(Rt_ref, pnp_poses[fid])
        pose = Rt_to_gtsam_pose(Rt_rel)
        ck = cam_key(fid)
        initial.insert(ck, pose)
        cam_keys[fid] = ck

    # Gauge prior — pins the first frame at identity in bundle-local coords.
    graph.add(gtsam.PriorFactorPose3(cam_keys[first], gtsam.Pose3(), prior_noise))

    # Gather tracks that appear in ≥ 2 frames of this window.
    track_appearances = {}
    for fid in frames:
        for tid in db.tracks(fid):
            track_appearances.setdefault(tid, []).append(fid)
    track_appearances = {tid: sorted(fids)
                         for tid, fids in track_appearances.items()
                         if len(fids) >= 2}

    lm_keys = {}
    projection_factors = []
    for tid, fids in track_appearances.items():
        # Triangulate from the LAST appearance — most reliable depth.
        last_fid = fids[-1]
        link_last = db.link(last_fid, tid)
        sc_last = stereo_camera(initial.atPose3(cam_keys[last_fid]), K_stereo)
        try:
            X_local = sc_last.backproject(link_to_stereo_point(link_last))
        except RuntimeError:
            continue  # cheirality (point behind camera)
        qk = lm_key(tid)
        initial.insert(qk, gtsam.Point3(X_local))
        lm_keys[tid] = qk
        for fid in fids:
            link = db.link(fid, tid)
            size = lookup_feature_size(
                feature_sizes.get(fid, np.zeros((0, 3))),
                link.x_left, link.y)
            sigma = max(SIGMA_FLOOR, size / size_scale)
            noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([sigma, sigma, sigma]))
            factor = gtsam.GenericStereoFactor3D(
                link_to_stereo_point(link), noise,
                cam_keys[fid], qk, K_stereo)
            idx = graph.size()
            graph.add(factor)
            projection_factors.append((idx, fid, tid, link))

    return graph, initial, dict(cam_keys=cam_keys,
                                lm_keys=lm_keys,
                                projection_factors=projection_factors,
                                first_frame=first)


# ex5
def optimize_bundle(graph, initial):
    """Run Levenberg-Marquardt on the bundle and return (result, optimizer)."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    return result, optimizer


# ex5
def compose_global_poses(relative_keyframe_poses, anchor_Rt=None):
    """Chain per-bundle relative poses into absolute world-to-camera extrinsics.

    Args:
        relative_keyframe_poses: list of 3x4 extrinsics. Item i is the
            world-to-camera extrinsic of keyframe ``kf_i+1`` expressed in
            keyframe ``kf_i``'s coordinate frame.
        anchor_Rt: 3x4 extrinsic of the first keyframe in the global (frame-0)
            world. If None, defaults to identity.

    Returns:
        Nx3x4 array of world-to-camera extrinsics for every keyframe.
    """
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    abs_poses = [identity if anchor_Rt is None else anchor_Rt.copy()]
    for Rt_rel in relative_keyframe_poses:
        abs_poses.append(compose_extrinsics(abs_poses[-1], Rt_rel))
    return np.array(abs_poses)
