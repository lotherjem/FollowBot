from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .types import LETHAL, MAX_TRAVERSAL, UNKNOWN, PlannerConfig, Pose

#: Full confidence in an obstacle. Plain uint8 headroom -- the value itself is
#: arbitrary, only the ratio to ``obstacle_confirmations`` matters.
CONF_MAX = 255

try:  # pragma: no cover - exercised implicitly by whichever branch runs
    from scipy.ndimage import distance_transform_edt as _scipy_edt

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _scipy_edt = None
    _HAVE_SCIPY = False


class WorldCostmap:
    """A growable, odometry-anchored occupancy grid with inflation.

    Not thread-safe by design: it is owned exclusively by the planner's worker
    thread. Callers hand scans in through :class:`~carrybot_nav.planner.PathPlanner`.
    """

    def __init__(self, config: PlannerConfig):
        config.validate()
        self.cfg = config
        self.res = config.resolution

        side = max(3, config.cells(config.initial_world_size_m) | 1)
        self._h = side
        self._w = side

        # World coordinates of the centre of cell (0, 0).
        self._origin_x = -0.5 * (self._w - 1) * self.res
        self._origin_y = -0.5 * (self._h - 1) * self.res

        self.raw = np.full((self._h, self._w), UNKNOWN, dtype=np.uint8)
        self.cost = np.full((self._h, self._w), UNKNOWN, dtype=np.uint8)
        # How sure we are that each LETHAL cell is still occupied, 0..CONF_MAX.
        # Only meaningful where ``raw`` is LETHAL; zero everywhere else.
        self.confidence = np.zeros((self._h, self._w), dtype=np.uint8)

        self._infl_cells = config.cells(config.inflation_radius)
        self._decay_cells = config.cells(config.cost_decay_distance)
        self.updates = 0
        self.expired_cells = 0
        #: Obstacles cost ``stale_obstacle_cost`` instead of blocking; see
        #: :meth:`set_soft`.
        self.soft = False

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> Tuple[int, int]:
        return self._h, self._w

    @property
    def origin(self) -> Tuple[float, float]:
        """World coordinates of the centre of cell (0, 0)."""
        return self._origin_x, self._origin_y

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """World metres -> (row, col). May fall outside the grid."""
        col = int(round((x - self._origin_x) / self.res))
        row = int(round((y - self._origin_y) / self.res))
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        return self._origin_x + col * self.res, self._origin_y + row * self.res

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._h and 0 <= col < self._w

    def cost_at(self, x: float, y: float) -> int:
        row, col = self.world_to_cell(x, y)
        if not self.in_bounds(row, col):
            return UNKNOWN
        return int(self.cost[row, col])

    # ------------------------------------------------------------------ #
    # Growth
    # ------------------------------------------------------------------ #

    def _ensure_capacity(self, min_x: float, min_y: float, max_x: float, max_y: float) -> None:
        """Grow the grid so the given world box fits, keeping cells aligned."""
        pad = self.cfg.cells(self.cfg.world_growth_m)

        r0, c0 = self.world_to_cell(min_x, min_y)
        r1, c1 = self.world_to_cell(max_x, max_y)

        grow_top = max(0, -r0 + pad) if r0 < 0 else 0
        grow_left = max(0, -c0 + pad) if c0 < 0 else 0
        grow_bottom = max(0, r1 - self._h + 1 + pad) if r1 >= self._h else 0
        grow_right = max(0, c1 - self._w + 1 + pad) if c1 >= self._w else 0

        if not (grow_top or grow_left or grow_bottom or grow_right):
            return

        new_h = self._h + grow_top + grow_bottom
        new_w = self._w + grow_left + grow_right

        for name, fill in (("raw", UNKNOWN), ("cost", UNKNOWN), ("confidence", 0)):
            old = getattr(self, name)
            new = np.full((new_h, new_w), fill, dtype=np.uint8)
            new[grow_top : grow_top + self._h, grow_left : grow_left + self._w] = old
            setattr(self, name, new)

        self._origin_x -= grow_left * self.res
        self._origin_y -= grow_top * self.res
        self._h, self._w = new_h, new_w

    # ------------------------------------------------------------------ #
    # Scan integration
    # ------------------------------------------------------------------ #

    def integrate(
        self, scan: np.ndarray, robot_pose: Pose, elapsed: float = 0.0
    ) -> int:
        """Fuse a robot-centric scan taken at ``robot_pose`` (world frame).

        ``scan`` is indexed ``scan[iy, ix]`` with the robot at the exact centre
        cell, ``+ix`` pointing along the robot's forward axis and ``+iy`` to its
        left. Values follow the cost encoding in :mod:`carrybot_nav.types`;
        UNKNOWN cells are ignored rather than overwriting prior knowledge.

        ``elapsed`` is the time since the previous scan. It ages obstacles that
        sit inside this scan's footprint and were not seen again; see the
        lifetime block below. Passing 0 disables that, as does
        ``PlannerConfig.obstacle_ttl = 0``. Returns the number of cells that
        expired.
        """
        scan = np.asarray(scan)
        if scan.ndim != 2:
            raise ValueError(f"scan must be 2-D, got shape {scan.shape}")
        if scan.dtype != np.uint8:
            scan = np.clip(scan, 0, 255).astype(np.uint8)

        sh, sw = scan.shape
        scy = (sh - 1) * 0.5
        scx = (sw - 1) * 0.5

        cos_t = math.cos(robot_pose.theta)
        sin_t = math.sin(robot_pose.theta)

        # World-frame bounding box of the (rotated) scan footprint.
        half_x = scx * self.res
        half_y = scy * self.res
        corners = [(-half_x, -half_y), (-half_x, half_y), (half_x, -half_y), (half_x, half_y)]
        xs, ys = [], []
        for bx, by in corners:
            xs.append(robot_pose.x + bx * cos_t - by * sin_t)
            ys.append(robot_pose.y + bx * sin_t + by * cos_t)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        self._ensure_capacity(min_x, min_y, max_x, max_y)

        r0, c0 = self.world_to_cell(min_x, min_y)
        r1, c1 = self.world_to_cell(max_x, max_y)
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(self._h - 1, r1)
        c1 = min(self._w - 1, c1)
        if r1 < r0 or c1 < c0:
            return 0

        # Inverse mapping: for every world cell in the box, find its scan cell.
        rows = np.arange(r0, r1 + 1, dtype=np.float64)
        cols = np.arange(c0, c1 + 1, dtype=np.float64)
        wx = self._origin_x + cols[None, :] * self.res - robot_pose.x
        wy = self._origin_y + rows[:, None] * self.res - robot_pose.y

        # Rotate world delta into the robot frame: R(-theta).
        bx = wx * cos_t + wy * sin_t
        by = -wx * sin_t + wy * cos_t

        ix = np.rint(bx / self.res + scx).astype(np.int32)
        iy = np.rint(by / self.res + scy).astype(np.int32)

        valid = (ix >= 0) & (ix < sw) & (iy >= 0) & (iy < sh)
        if not valid.any():
            return 0

        sampled = np.full(valid.shape, UNKNOWN, dtype=np.uint8)
        sampled[valid] = scan[iy[valid], ix[valid]]

        window = self.raw[r0 : r1 + 1, c0 : c1 + 1]
        observed = valid & (sampled != UNKNOWN)
        np.copyto(window, sampled, where=observed)

        # -- obstacle lifetimes ---------------------------------------- #
        # Only cells this scan actually covered are touched. That scoping is
        # the whole design: a wall the robot drove past and left behind is not
        # evidence of anything, so its confidence is frozen rather than ticking
        # down, and the map keeps the building. What does tick down is a cell
        # the sensor is pointed straight at and reports nothing for -- an
        # untextured wall the stereo matcher fails on, a person who has walked
        # out of the frame -- which is the one case free-space carving cannot
        # clear, because carving needs a reading to carve with.
        conf = self.confidence[r0 : r1 + 1, c0 : c1 + 1]
        window_lethal = window == LETHAL

        hit = observed & (sampled == LETHAL)
        if hit.any():
            gain = CONF_MAX // max(1, self.cfg.obstacle_confirmations)
            np.copyto(
                conf,
                np.minimum(conf.astype(np.int16) + gain, CONF_MAX).astype(np.uint8),
                where=hit,
            )
        cleared = observed & (sampled != LETHAL)
        if cleared.any():
            # Carved free or re-read as traversable: no credit to carry over if
            # something reappears here later.
            np.copyto(conf, np.zeros_like(conf), where=cleared)

        expired = 0
        ttl = self.cfg.obstacle_ttl
        if ttl > 0.0 and elapsed > 0.0:
            # In the footprint, still lethal, and this scan had nothing to say
            # about it -- but only where the sensor plainly *could* have seen
            # it. A cell is unobserved for two very different reasons: the
            # sensor looked and got nothing back, or something in front of it
            # blocked the view. Only the first is evidence of absence.
            #
            # Touching known-free ground separates them. A stereo hole in an
            # untextured wall sits in carved floor; the back face of a wall is
            # surrounded by the wall's own cells and by the shadow behind it.
            # Without this the far side of every two-cell-thick wall quietly
            # expires, the planner starts costing routes *through* walls at
            # unknown_cost, and the robot drives into them -- measured, in the
            # maze scenario, as 129 wall contacts against a baseline of none.
            touches_free = np.zeros_like(observed)
            seen_free = observed & (sampled <= MAX_TRAVERSAL)
            touches_free[1:, :] |= seen_free[:-1, :]
            touches_free[:-1, :] |= seen_free[1:, :]
            touches_free[:, 1:] |= seen_free[:, :-1]
            touches_free[:, :-1] |= seen_free[:, 1:]

            silent = valid & ~observed & window_lethal & touches_free
            if silent.any():
                drop = max(1, int(round(CONF_MAX * elapsed / ttl)))
                faded = np.maximum(conf.astype(np.int16) - drop, 0).astype(np.uint8)
                np.copyto(conf, faded, where=silent)
                gone = silent & (conf == 0)
                if gone.any():
                    # UNKNOWN, not FREE. Forgetting is not the same as
                    # clearing, and claiming ground is passable because a
                    # reading got old is how a robot drives into a wall it saw
                    # a moment ago.
                    np.copyto(window, np.full_like(window, UNKNOWN), where=gone)
                    expired = int(gone.sum())
                    self.expired_cells += expired

        self._reinflate(r0, c0, r1, c1)
        self.updates += 1
        return expired

    def load_absolute(self, grid: np.ndarray, origin_x: float, origin_y: float) -> None:
        """Replace the map wholesale. Mostly useful for tests and replay."""
        grid = np.ascontiguousarray(grid, dtype=np.uint8)
        self._h, self._w = grid.shape
        self._origin_x = origin_x
        self._origin_y = origin_y
        self.raw = grid.copy()
        self.cost = grid.copy()
        # A wholesale load is an assertion about the world, not an observation
        # of it, so everything it declares lethal starts at full confidence.
        self.confidence = np.where(self.raw == LETHAL, CONF_MAX, 0).astype(np.uint8)
        self._reinflate(0, 0, self._h - 1, self._w - 1)

    # ------------------------------------------------------------------ #
    # Inflation
    # ------------------------------------------------------------------ #

    @property
    def soft_cost(self) -> int:
        return int(round(self.cfg.stale_obstacle_cost))

    def set_soft(self, soft: bool) -> None:
        """Make every obstacle, and its inflation, expensive instead of lethal.

        For when the map has stopped being updated -- terrain mapping switched
        off -- so what is on it is a memory, not an observation. ``raw`` is
        untouched: switching back restores every obstacle exactly.
        """
        soft = bool(soft)
        if soft != self.soft:
            self.soft = soft
            self._reinflate(0, 0, self._h - 1, self._w - 1)

    def _reinflate(self, r0: int, c0: int, r1: int, c1: int) -> None:
        """Recompute the cost layer over a dirty box, plus a halo.

        The halo matters: an obstacle just outside the box still throws a skirt
        into it, and an obstacle that was *removed* inside the box must have its
        skirt cleaned up outside it.
        """
        halo = self._infl_cells + self._decay_cells + 1
        rr0 = max(0, r0 - halo)
        cc0 = max(0, c0 - halo)
        rr1 = min(self._h - 1, r1 + halo)
        cc1 = min(self._w - 1, c1 + halo)

        raw = self.raw[rr0 : rr1 + 1, cc0 : cc1 + 1]
        out = raw.copy()

        lethal = raw == LETHAL
        if lethal.any():
            dist = self._distance_to_lethal(lethal)

            if self._decay_cells > 0 and self.cfg.decay_peak_cost > 0:
                band = self._infl_cells + self._decay_cells
                # Linear falloff from decay_peak_cost at the skirt edge to 0.
                t = np.clip((band - dist) / max(1e-9, self._decay_cells), 0.0, 1.0)
                decay = (t * self.cfg.decay_peak_cost).astype(np.uint8)
                # max() preserves UNKNOWN (254) and LETHAL (255) automatically,
                # since both sit above every traversal cost.
                np.maximum(out, decay, out=out)

            out[dist <= self._infl_cells] = LETHAL

        # Raw obstacles always win.
        out[lethal] = LETHAL
        if self.soft:
            out[out == LETHAL] = self.soft_cost
        self.cost[rr0 : rr1 + 1, cc0 : cc1 + 1] = out

    def _distance_to_lethal(self, lethal: np.ndarray) -> np.ndarray:
        """Distance, in cells, from every cell to the nearest lethal cell."""
        if _HAVE_SCIPY:
            return _scipy_edt(~lethal)
        return _chamfer_distance(lethal)

    # ------------------------------------------------------------------ #
    # Queries used by the search
    # ------------------------------------------------------------------ #

    def frontier_mask(self) -> np.ndarray:
        """Known-traversable cells that touch unknown space.

        These are the places worth driving to when the goal itself cannot be
        reached or has never been observed.
        """
        known_free = self.cost <= MAX_TRAVERSAL
        unknown = self.raw == UNKNOWN

        neighbour_unknown = np.zeros_like(unknown)
        neighbour_unknown[1:, :] |= unknown[:-1, :]
        neighbour_unknown[:-1, :] |= unknown[1:, :]
        neighbour_unknown[:, 1:] |= unknown[:, :-1]
        neighbour_unknown[:, :-1] |= unknown[:, 1:]

        return known_free & neighbour_unknown & ~unknown

    def is_frontier(self, row: int, col: int) -> bool:
        """O(1) version of :meth:`frontier_mask` for a single cell."""
        if not self.in_bounds(row, col):
            return False
        if self.cost[row, col] > MAX_TRAVERSAL:
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = row + dr, col + dc
            if self.in_bounds(rr, cc) and self.raw[rr, cc] == UNKNOWN:
                return True
        return False

    def nearest_free_cell(self, row: int, col: int, radius_cells: int) -> Optional[Tuple[int, int]]:
        """Spiral outward for a traversable cell near (row, col)."""
        if self.in_bounds(row, col) and self.cost[row, col] <= MAX_TRAVERSAL:
            return row, col
        for r in range(1, radius_cells + 1):
            best = None
            best_d = float("inf")
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if max(abs(dr), abs(dc)) != r:
                        continue
                    rr, cc = row + dr, col + dc
                    if not self.in_bounds(rr, cc):
                        continue
                    if self.cost[rr, cc] > MAX_TRAVERSAL:
                        continue
                    d = dr * dr + dc * dc
                    if d < best_d:
                        best_d = d
                        best = (rr, cc)
            if best is not None:
                return best
        return None

    def segment_is_clear(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Supercover line check between two world points on the cost layer."""
        r0, c0 = self.world_to_cell(x0, y0)
        r1, c1 = self.world_to_cell(x1, y1)
        # Soft obstacles block a straight-line shortcut even though the search
        # may cross them: a shortcut ignores cost, and would otherwise cut
        # straight through what the search deliberately went round.
        block = self.soft_cost if self.soft else LETHAL
        for row, col in bresenham(r0, c0, r1, c1):
            if not self.in_bounds(row, col):
                return False
            cell = self.cost[row, col]
            if cell == LETHAL or cell == block:
                return False
        return True

    def stats(self) -> dict:
        known = int(np.count_nonzero(self.raw != UNKNOWN))
        return {
            "shape": self.shape,
            "origin": self.origin,
            "known_cells": known,
            "known_fraction": known / float(self._h * self._w),
            "lethal_cells": int(np.count_nonzero(self.raw == LETHAL)),
            "updates": self.updates,
            "expired_cells": self.expired_cells,
            "backend": "scipy" if _HAVE_SCIPY else "numpy-chamfer",
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _chamfer_distance(seed: np.ndarray) -> np.ndarray:
    """Two-pass chamfer approximation of the Euclidean distance transform.

    Used only when SciPy is unavailable. Accurate to a few percent, which is
    well inside the safety margin, and still fully vectorised per row.
    """
    big = np.float32(1e6)
    d = np.where(seed, np.float32(0.0), big).astype(np.float32)
    h, w = d.shape
    diag = np.float32(math.sqrt(2.0))
    one = np.float32(1.0)

    for r in range(h):
        row = d[r]
        if r > 0:
            prev = d[r - 1]
            np.minimum(row, prev + one, out=row)
            np.minimum(row[1:], prev[:-1] + diag, out=row[1:])
            np.minimum(row[:-1], prev[1:] + diag, out=row[:-1])
        for c in range(1, w):
            if row[c] > row[c - 1] + one:
                row[c] = row[c - 1] + one

    for r in range(h - 1, -1, -1):
        row = d[r]
        if r < h - 1:
            nxt = d[r + 1]
            np.minimum(row, nxt + one, out=row)
            np.minimum(row[1:], nxt[:-1] + diag, out=row[1:])
            np.minimum(row[:-1], nxt[1:] + diag, out=row[:-1])
        for c in range(w - 2, -1, -1):
            if row[c] > row[c + 1] + one:
                row[c] = row[c + 1] + one

    return d


def bresenham(r0: int, c0: int, r1: int, c1: int):
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        yield r0, c0
        if r0 == r1 and c0 == c1:
            return
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


__all__ = ["WorldCostmap", "bresenham"]
