"""Depth image to the robot-centric occupancy array the planner consumes.

The tracker already computes a full disparity image and then samples one pixel
of it. This is the rest of that map put to work: ground-plane removal, then a
projection down to the 2-D grid ``PathPlanner.submit_scan`` expects.

Three things here are easy to leave out and expensive to leave out.

**Carve free space, not just obstacles.** A depth camera reports emptiness as
well as returns. Everything along a ray up to the first obstacle is FREE;
beyond it is UNKNOWN. Skip this and the map only ever gains obstacles -- a
person who walks through the view is branded LETHAL forever, because nothing
ever observes that cell as clear again.

**Respect the cone.** A 74.5-degree camera sees about a sixth of the array. The
rest must stay UNKNOWN rather than defaulting to free, or the planner will
confidently route the robot sideways into terrain it has never looked at.

**Account for camera tilt.** Three degrees of downward tilt puts the floor
21 cm up in the image at 4 m. Uncorrected, the floor reads as a wall and the
robot refuses to move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .types import CameraModel, VisionConfig

FREE = 0
UNKNOWN = 254
LETHAL = 255


@dataclass(frozen=True)
class ProjectionStats:
    """What a single depth frame actually contributed."""

    valid_pixels: int = 0
    obstacle_points: int = 0
    ground_points: int = 0
    overhead_points: int = 0
    masked_pixels: int = 0
    occupied_cells: int = 0
    free_cells: int = 0
    unknown_fraction: float = 1.0
    #: The floor as this frame saw it: ``height = floor_offset +
    #: floor_slope * distance from the camera``, in metres, from the points in
    #: the lower half
    #: of the image near where the floor should be. Both 0 when the camera's
    #: mounting is right. A constant offset is ``mount_height`` wrong; a slope
    #: is ``mount_pitch`` wrong by that many radians. NaN without enough floor
    #: in view. Either one past ``ground_height`` turns bare floor into
    #: obstacles.
    floor_offset: float = float("nan")
    floor_slope: float = float("nan")
    floor_points: int = 0

    def summary(self) -> str:
        return (
            f"valid={self.valid_pixels:6d} obstacle={self.obstacle_points:6d} "
            f"ground={self.ground_points:6d} overhead={self.overhead_points:5d} "
            f"masked={self.masked_pixels:5d} | cells occupied={self.occupied_cells:5d} "
            f"free={self.free_cells:6d} unknown={self.unknown_fraction * 100:.0f}%"
        )


class DepthProjector:
    """Turns depth frames into planner scans.

    Per-cell range and bearing are precomputed once, so a frame costs one
    vectorised pass over the depth image and a couple of lookups over the grid.
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        self.cfg = config or VisionConfig()
        self.cfg.validate()
        self._build_grid_tables()
        self.last_stats = ProjectionStats()

    # ------------------------------------------------------------------ #

    def _build_grid_tables(self) -> None:
        cfg = self.cfg
        half = cfg.grid_half
        res = cfg.grid_resolution

        offsets = (np.arange(cfg.grid_size, dtype=np.float32) - half) * res
        # scan[iy, ix]: +ix forward (x), +iy left (y)
        x = offsets[None, :]
        y = offsets[:, None]
        self._cell_x = np.broadcast_to(x, (cfg.grid_size, cfg.grid_size)).copy()
        self._cell_y = np.broadcast_to(y, (cfg.grid_size, cfg.grid_size)).copy()
        self._cell_range = np.hypot(self._cell_x, self._cell_y)
        self._cell_bearing = np.arctan2(self._cell_y, self._cell_x)

        self._set_fov(self.cfg.camera.hfov)

        # The head-centred versions, cached for the common case of a head that
        # is not moving. Recomputing them is two passes over the grid, which is
        # cheap, but not free at camera rate.
        self._cached_yaw: Optional[float] = None
        self._in_cone = np.zeros(self._cell_range.shape, dtype=bool)
        self._cell_bin = np.zeros(self._cell_range.shape, dtype=np.int32)
        self._cone_for(0.0)

    def _set_fov(self, hfov: float) -> None:
        """Size the bearing bins to the camera's field of view."""
        self._hfov = hfov
        self._half_fov = hfov / 2.0
        self._n_bins = max(8, int(math.ceil(hfov / self.cfg.bearing_bin)))
        self._bin_edge = -self._half_fov
        self._bin_width = hfov / self._n_bins
        self._cached_yaw = None

    def _cone_for(self, head_yaw: float) -> None:
        """Point the cone at ``head_yaw`` and re-bin the cells behind it.

        Every cell's bearing is fixed in the robot frame; what moves is which
        of them the camera can see and which bin each falls in. Rotating the
        *question* rather than the grid keeps the output array in the robot
        frame, which is what the planner's contract requires -- a scan that
        silently arrived in the head's frame would put every obstacle off by
        the head angle, and look exactly like odometry drift.
        """
        if self._cached_yaw is not None and abs(head_yaw - self._cached_yaw) < 1e-6:
            return
        relative = _wrap(self._cell_bearing - head_yaw)
        self._in_cone = np.abs(relative) <= self._half_fov
        cell_bin = np.floor((relative - self._bin_edge) / self._bin_width).astype(np.int32)
        self._cell_bin = np.clip(cell_bin, 0, self._n_bins - 1)
        self._cached_yaw = head_yaw

    def _bin_of(self, bearing: np.ndarray, head_yaw: float = 0.0) -> np.ndarray:
        relative = _wrap(bearing - head_yaw)
        idx = np.floor((relative - self._bin_edge) / self._bin_width).astype(np.int32)
        return np.clip(idx, 0, self._n_bins - 1)

    # ------------------------------------------------------------------ #

    def project(
        self,
        depth: np.ndarray,
        camera: Optional[CameraModel] = None,
        exclude: Optional[np.ndarray] = None,
        stride: int = 2,
        head_yaw: float = 0.0,
    ) -> np.ndarray:
        """Depth in metres -> the ``uint8`` scan the planner takes.

        ``camera`` must describe the depth image's own resolution. ``exclude``
        is a boolean image of pixels carrying no information -- the tracked
        person, principally. Excluded pixels contribute neither obstacles nor
        free space, so the floor in front of the person still carves out to
        their feet and the person's own cell stays UNKNOWN rather than LETHAL.

        ``head_yaw`` is where the pan servo had the camera pointing **when
        this frame was captured**, positive to the robot's left. The output is
        always in the robot frame regardless, so a sweeping head simply fills
        in a wider arc of the same array over successive frames.
        """
        cfg = self.cfg
        camera = camera or cfg.camera
        if abs(camera.hfov - self._hfov) > 1e-4:
            # The focal length changed under us -- the range calibration on
            # the live page fits it -- and the cone has to follow, or the
            # edges of the view are binned as if they were somewhere else.
            self._set_fov(camera.hfov)
        self._cone_for(head_yaw)
        grid = np.full((cfg.grid_size, cfg.grid_size), UNKNOWN, dtype=np.uint8)

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth must be 2-D, got {depth.shape}")

        view = depth[::stride, ::stride]
        rows = np.arange(0, depth.shape[0], stride, dtype=np.float32)
        cols = np.arange(0, depth.shape[1], stride, dtype=np.float32)
        u = np.broadcast_to(cols[None, :], view.shape)
        v = np.broadcast_to(rows[:, None], view.shape)

        valid = np.isfinite(view) & (view > cfg.near_limit) & (view < cfg.max_range)

        trim = cfg.border_trim
        if trim > 0:
            valid &= (
                (u >= trim)
                & (u < depth.shape[1] - trim)
                & (v >= trim)
                & (v < depth.shape[0] - trim)
            )

        masked_count = 0
        if exclude is not None:
            blocked = np.asarray(exclude)[::stride, ::stride].astype(bool)
            masked_count = int((valid & blocked).sum())
            valid &= ~blocked

        if not valid.any():
            self.last_stats = ProjectionStats(masked_pixels=masked_count)
            return grid

        z = view[valid]
        du = u[valid] - camera.cx
        dv = v[valid] - camera.cy

        # Camera optical frame -> robot-aligned axes, before tilt.
        forward = z
        left = -du * z / camera.focal_px
        up = -dv * z / camera.focal_px

        # Undo the mount pitch (positive = looking down). This lands in the
        # *head* frame: x along the head's forward axis, y to its left.
        cp = math.cos(camera.mount_pitch)
        sp = math.sin(camera.mount_pitch)
        x_h = forward * cp + up * sp + camera.mount_forward
        y_h = left
        z_r = -forward * sp + up * cp + camera.mount_height

        # Head frame -> robot frame. The camera swings around the pan axis, so
        # this is a rotation followed by a translation, not a rotation alone.
        if head_yaw:
            cy = math.cos(head_yaw)
            sy = math.sin(head_yaw)
            x_r = x_h * cy - y_h * sy + camera.pan_offset
            y_r = x_h * sy + y_h * cy
        else:
            x_r = x_h + camera.pan_offset
            y_r = y_h

        rng = np.hypot(x_r, y_r)
        brg = np.arctan2(y_r, x_r)
        bins = self._bin_of(brg, head_yaw)

        # Distance measured from the camera, not the wheel axis: the slope is
        # a rotation about the lens, and measured from anywhere else it leaks
        # into the offset.
        floor_offset, floor_slope, floor_n = _fit_floor(
            x_h - camera.mount_forward, y_h, z_r, v[valid], depth.shape[0])

        is_ground = z_r < cfg.ground_height
        is_overhead = z_r > cfg.ceiling_height
        is_obstacle = ~is_ground & ~is_overhead

        # Nearest obstacle and furthest floor, per bearing bin.
        obstacle_range = np.full(self._n_bins, np.inf, dtype=np.float32)
        if is_obstacle.any():
            np.minimum.at(obstacle_range, bins[is_obstacle], rng[is_obstacle])
        floor_range = np.zeros(self._n_bins, dtype=np.float32)
        if is_ground.any():
            np.maximum.at(floor_range, bins[is_ground], rng[is_ground])

        # How far we can see along each bearing: up to the nearest obstacle,
        # or as far as bare floor was observed when nothing blocked the view.
        visible = np.where(np.isfinite(obstacle_range), obstacle_range, floor_range)

        # -- free space ------------------------------------------------- #
        reach = visible[self._cell_bin]
        free_mask = self._in_cone & (self._cell_range < reach - 0.5 * cfg.grid_resolution)
        grid[free_mask] = FREE

        # -- obstacles -------------------------------------------------- #
        occupied = 0
        if is_obstacle.any():
            half = cfg.grid_half
            res = cfg.grid_resolution
            ix = np.rint(x_r[is_obstacle] / res).astype(np.int32) + half
            iy = np.rint(y_r[is_obstacle] / res).astype(np.int32) + half
            inside = (ix >= 0) & (ix < cfg.grid_size) & (iy >= 0) & (iy < cfg.grid_size)
            if inside.any():
                counts = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.int32)
                np.add.at(counts, (iy[inside], ix[inside]), 1)
                hit = counts >= cfg.min_points_per_cell
                grid[hit] = LETHAL
                occupied = int(hit.sum())

        self.last_stats = ProjectionStats(
            valid_pixels=int(valid.sum()),
            obstacle_points=int(is_obstacle.sum()),
            ground_points=int(is_ground.sum()),
            overhead_points=int(is_overhead.sum()),
            masked_pixels=masked_count,
            occupied_cells=occupied,
            free_cells=int((grid == FREE).sum()),
            unknown_fraction=float((grid == UNKNOWN).mean()),
            floor_offset=floor_offset,
            floor_slope=floor_slope,
            floor_points=floor_n,
        )
        return grid


#: Floor candidates: this far ahead of the camera, this near the centre line,
#: within this of where the configured mounting says the floor is.
FLOOR_AHEAD = (0.3, 3.0)
FLOOR_SIDE = 0.8
FLOOR_BAND = 0.25
FLOOR_MIN_POINTS = 200


def _fit_floor(x, y, z, rows, height):
    """Least-squares ``z = a + b*x`` over the points that look like floor."""
    near = ((rows > 0.5 * height) & (x > FLOOR_AHEAD[0]) & (x < FLOOR_AHEAD[1])
            & (np.abs(y) < FLOOR_SIDE) & (np.abs(z) < FLOOR_BAND))
    n = int(near.sum())
    if n < FLOOR_MIN_POINTS:
        return float("nan"), float("nan"), n
    xs, zs = x[near].astype(np.float64), z[near].astype(np.float64)
    # Twice, the second time without the worst quarter: a box or a foot in
    # the band would otherwise drag the line towards itself.
    for _ in range(2):
        A = np.stack([np.ones_like(xs), xs], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, zs, rcond=None)
        resid = np.abs(zs - (a + b * xs))
        keep = resid <= np.percentile(resid, 75)
        if keep.sum() < FLOOR_MIN_POINTS // 2:
            break
        xs, zs = xs[keep], zs[keep]
    return float(a), float(b), n


def _wrap(angle):
    """Fold an angle (or an array of them) into (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def box_mask(
    shape: Tuple[int, int],
    box: Optional[Tuple[int, int, int, int]],
    padding: int = 0,
) -> np.ndarray:
    """Boolean image with a padded box set True. Empty when ``box`` is None."""
    mask = np.zeros(shape, dtype=bool)
    if box is None:
        return mask
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(shape[1], int(x2) + padding + 1)
    y2 = min(shape[0], int(y2) + padding + 1)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


__all__ = ["DepthProjector", "ProjectionStats", "box_mask", "FREE", "LETHAL", "UNKNOWN"]
