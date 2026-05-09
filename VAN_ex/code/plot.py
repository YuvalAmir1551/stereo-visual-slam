"""Shared plotting utilities — consistent 3D-world conventions for the project.

Conventions used here (and expected by callers):
  • Data is in the camera/CV frame: X right, Y +down, Z forward.
    Triangulated points and camera matrices are stored this way unmodified.
  • The matplotlib 3D axes use the permutation (X_data, Z_data, Y_data) and
    have their third axis inverted (`invert_zaxis`) so that screen 'up'
    corresponds to −Y. Data values are NOT transformed.
  • Equal physical metre scale on every axis via `set_box_aspect`.

Every exercise that visualises 3D world points should use these helpers
so that figures across the project (ex2, ex3, ex4) share the same look.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Default viewing window (metres). 100 m on every axis → cube view.
PLOT_LIMS = dict(x=(-50, 50), y=(0, 100), z=(-50, 50))

# Default depth (m) of the image plane when drawing a camera frustum.
FRUSTUM_DEPTH = 5.0


# ex2
def draw_frustum(ax, K, img_shape, depth=FRUSTUM_DEPTH, origin=(0., 0., 0.)):
    """Draw a Blender/pytransform3d-style camera frustum at `origin`.

    Wireframe pyramid (apex at the optical centre + the four image-corner
    rays projected to `depth`) plus a small red triangle sitting on the top
    edge of the image plane to indicate the camera's 'up' direction.
    """
    H, W = img_shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    corners_uv = [(0, 0), (W, 0), (W, H), (0, H)]   # TL, TR, BR, BL
    base_cam = np.array([
        [(u - cx) * depth / fx, (v - cy) * depth / fy, depth]
        for u, v in corners_uv
    ]) + np.asarray(origin, dtype=float)
    apex_cam = np.asarray(origin, dtype=float)

    # 'Up' triangle sitting above the top image edge (camera up = −Y).
    TL_cam, TR_cam = base_cam[0], base_cam[1]
    width = np.linalg.norm(TR_cam - TL_cam)
    tri_base_l = TL_cam + 0.30 * (TR_cam - TL_cam)
    tri_base_r = TL_cam + 0.70 * (TR_cam - TL_cam)
    tri_apex_cam = (TL_cam + TR_cam) / 2 + np.array([0., -0.18 * width, 0.])

    # Permute (X, Y, Z) → (X, Z, Y) to match plot_3d_world's scatter axes.
    def _to_plot(p):
        return np.array([p[0], p[2], p[1]])
    apex = _to_plot(apex_cam)
    base = np.array([_to_plot(p) for p in base_cam])
    tri = [_to_plot(p) for p in (tri_base_l, tri_base_r, tri_apex_cam)]

    wire = [[apex, b] for b in base]
    wire += [[base[i], base[(i + 1) % 4]] for i in range(4)]
    up_edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]

    wf = Line3DCollection(wire, colors='black', linewidths=1.4)
    up = Line3DCollection(up_edges, colors='black', linewidths=2.0)
    wf.set_zorder(10); up.set_zorder(11)
    ax.add_collection3d(wf)
    ax.add_collection3d(up)


# ex2
def plot_3d_world(X, ax, K=None, img_shape=None, title='',
                  highlight_mask=None, lims=PLOT_LIMS):
    """Scatter Nx3 camera-frame points on `ax` using the project's conventions.

    Args:
        X: Nx3 array of triangulated points in the camera/CV frame.
        ax: existing 3D axis (created by the caller via add_subplot(..., projection='3d')).
        K, img_shape: if both provided, draw the camera frustum at origin.
        title: subplot title.
        highlight_mask: boolean mask over X. True entries are drawn red and
            larger to flag erroneous points.
        lims: dict with 'x', 'y', 'z' axis limits in metres.
    """
    # Disable matplotlib's 3D depth-based ordering so the frustum stays on top
    # regardless of view angle (otherwise it gets hidden when looking from behind).
    ax.computed_zorder = False

    Xp = np.column_stack([X[:, 0], X[:, 2], X[:, 1]])     # (X, Z, Y)
    if highlight_mask is None:
        ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2],
                   s=6, c='steelblue', alpha=0.7, zorder=1)
    else:
        good = ~highlight_mask
        ax.scatter(Xp[good, 0], Xp[good, 1], Xp[good, 2],
                   s=6, c='steelblue', alpha=0.7, zorder=1,
                   label=f'kept ({good.sum()})')
        if highlight_mask.any():
            ax.scatter(Xp[highlight_mask, 0], Xp[highlight_mask, 1], Xp[highlight_mask, 2],
                       s=22, c='red', marker='x', zorder=2,
                       label=f'erroneous ({highlight_mask.sum()})')
        ax.legend(loc='upper right', fontsize=8)

    if K is not None and img_shape is not None:
        draw_frustum(ax, K, img_shape)

    ax.set_xlim(*lims['x']); ax.set_ylim(*lims['y']); ax.set_zlim(*lims['z'])
    rx, ry, rz = (lims[k][1] - lims[k][0] for k in 'xyz')
    ax.set_box_aspect((rx, ry, rz))
    ax.invert_zaxis()

    ax.set_xlabel('X — right (m)')
    ax.set_ylabel('Z — forward (m)')
    ax.set_zlabel('Y (m, +down)')
    ax.view_init(elev=15, azim=-75)
    if title:
        ax.set_title(title)
    return ax
