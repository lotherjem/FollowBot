from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .costmap import WorldCostmap
from .types import LETHAL, UNKNOWN, PlannerConfig, PlanStatus

INF = float("inf")
SQRT2 = math.sqrt(2.0)

# (dr, dc) for the eight neighbours, ordered anticlockwise from +x.
# Row increases with +y and column with +x, so direction i sits at i * 45 deg.
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
STEP_LENGTH: Tuple[float, ...] = tuple(SQRT2 if dr and dc else 1.0 for dr, dc in DIRECTIONS)
NDIR = 8


def direction_from_angle(theta: float) -> int:
    """Nearest of the eight grid directions to a heading in radians."""
    return int(round(theta / (math.pi / 4.0))) % NDIR


def angle_of_direction(index: int) -> float:
    return (index % NDIR) * (math.pi / 4.0)


def turn_steps(a: int, b: int) -> int:
    """Angular separation between two direction indices, in 45-degree steps."""
    d = abs(a - b) % NDIR
    return min(d, NDIR - d)


@dataclass
class SearchResult:
    """Raw output of a grid search, in world-costmap cell coordinates."""

    cells: List[Tuple[int, int]] = field(default_factory=list)
    status: PlanStatus = PlanStatus.NO_PATH
    expansions: int = 0
    #: The cell actually planned to: the goal, a snapped goal, or a frontier.
    target_cell: Optional[Tuple[int, int]] = None
    #: Cost-to-come of the final cell, in the search's internal units.
    cost: float = 0.0


class GridSearch(Protocol):
    """Anything that can turn a costmap and two cells into a path."""

    def plan(
        self,
        costmap: WorldCostmap,
        start_cell: Tuple[int, int],
        goal_cell: Optional[Tuple[int, int]],
        start_theta: Optional[float] = None,
        escape_lethal: bool = False,
        corridor: Optional[np.ndarray] = None,
    ) -> SearchResult: ...


@dataclass
class _Pass:
    """Bookkeeping from one sweep of the priority queue."""

    expansions: int = 0
    found_state: int = -1
    best_state: int = -1
    best_h: float = INF
    start_h: float = INF
    exhausted: bool = True  # heap emptied rather than budget running out
    frontier_states: List[Tuple[int, int]] = field(default_factory=list)  # (state, idx)


# --------------------------------------------------------------------------- #


class AStarSearch:
    """Weighted A* over the grid, with turn penalties and a frontier fallback."""

    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self._buf_size = 0
        self._g: List[float] = []
        self._parent: List[int] = []
        self._closed: List[bool] = []
        self._dir: List[int] = []
        self._touched: List[int] = []
        self._turn_table: List[float] = []
        self._turn_cost_cached: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Buffers
    # ------------------------------------------------------------------ #

    def _reset_buffers(self, n_states: int) -> None:
        """Clear only what the previous sweep dirtied.

        Blanket-clearing a million-entry buffer costs more than the search
        itself at these window sizes, so only the states that were actually
        written get reset.
        """
        if n_states > self._buf_size:
            self._g = [INF] * n_states
            self._parent = [-1] * n_states
            self._closed = [False] * n_states
            self._dir = [0] * n_states
            self._buf_size = n_states
            self._touched = []
            return

        g, parent, closed = self._g, self._parent, self._closed
        for s in self._touched:
            g[s] = INF
            parent[s] = -1
            closed[s] = False
        self._touched = []

    def _turn_costs(self) -> List[float]:
        """Flat 8x8 table of rotation penalties, indexed ``a * 8 + b``."""
        if self._turn_cost_cached != self.cfg.turn_cost:
            self._turn_table = [
                self.cfg.turn_cost * 0.5 * turn_steps(a, b)
                for a in range(NDIR)
                for b in range(NDIR)
            ]
            self._turn_cost_cached = self.cfg.turn_cost
        return self._turn_table

    # ------------------------------------------------------------------ #
    # Window
    # ------------------------------------------------------------------ #

    def _window(
        self,
        costmap: WorldCostmap,
        start_cell: Tuple[int, int],
        goal_cell: Optional[Tuple[int, int]],
    ) -> Tuple[int, int, int, int]:
        h, w = costmap.shape
        if self.cfg.search_radius_m is None:
            return 0, 0, h - 1, w - 1

        rows = [start_cell[0]]
        cols = [start_cell[1]]
        if goal_cell is not None:
            rows.append(goal_cell[0])
            cols.append(goal_cell[1])
        bh = max(rows) - min(rows) + 1
        bw = max(cols) - min(cols) + 1

        pad = self.cfg.cells(self.cfg.search_radius_m)

        # Cap the window area. Latency scales with it, and a planner that
        # occasionally blows its deadline is worse than one that occasionally
        # has to make progress via a frontier.
        cap = self.cfg.max_window_cells
        if cap and (bh + 2 * pad) * (bw + 2 * pad) > cap:
            b = 2.0 * (bh + bw)
            disc = b * b - 16.0 * (bh * bw - cap)
            allowed = int((-b + math.sqrt(max(0.0, disc))) / 8.0)
            pad = max(2, min(pad, allowed))

        r0 = max(0, min(rows) - pad)
        c0 = max(0, min(cols) - pad)
        r1 = min(h - 1, max(rows) + pad)
        c1 = min(w - 1, max(cols) + pad)
        return r0, c0, r1, c1

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def plan(
        self,
        costmap: WorldCostmap,
        start_cell: Tuple[int, int],
        goal_cell: Optional[Tuple[int, int]],
        start_theta: Optional[float] = None,
        escape_lethal: bool = False,
        corridor: Optional[np.ndarray] = None,
    ) -> SearchResult:
        """Search from ``start_cell`` to ``goal_cell``.

        ``start_theta`` is the robot's current heading in radians. Every one of
        the eight departure directions is seeded with the real cost of turning
        to it, rather than snapping the heading to the nearest of the eight.
        Snapping makes the plan a step function of heading: a couple of degrees
        of rotation flips which way the robot is told to go round an obstacle,
        and it stands there oscillating.

        ``corridor`` is a boolean map the same shape as the costmap marking the
        previous route. Steps inside it are discounted, which keeps the planner
        committed to a choice it has already made instead of flipping between
        two near-identical ways around an obstacle.

        ``escape_lethal`` makes LETHAL cells traversable at a punitive cost. It
        exists for one situation: the robot is standing inside inflated space
        with no clear neighbour. Without it the search cannot expand at all and
        the robot sits there reporting NO_PATH forever.
        """
        cfg = self.cfg
        r0, c0, r1, c1 = self._window(costmap, start_cell, goal_cell)
        sub = costmap.cost[r0 : r1 + 1, c0 : c1 + 1]
        wh, ww = sub.shape

        # bytes indexing is markedly faster than numpy scalar indexing in the
        # inner loop, and the window is read-only during a search.
        grid = sub.tobytes()

        corridor_bytes: Optional[bytes] = None
        if corridor is not None and cfg.path_hysteresis > 0.0:
            corridor_bytes = np.ascontiguousarray(
                corridor[r0 : r1 + 1, c0 : c1 + 1], dtype=np.uint8
            ).tobytes()

        sr = start_cell[0] - r0
        sc = start_cell[1] - c0
        if not (0 <= sr < wh and 0 <= sc < ww):
            return SearchResult(status=PlanStatus.NO_PATH)
        start_idx = sr * ww + sc

        gr = gc = -1
        goal_idx = -1
        if goal_cell is not None:
            gr = goal_cell[0] - r0
            gc = goal_cell[1] - c0
            if 0 <= gr < wh and 0 <= gc < ww:
                goal_idx = gr * ww + gc

        total_expansions = 0

        # -- pass 1: straight at the goal ------------------------------- #
        if goal_idx >= 0:
            p = self._sweep(
                grid,
                wh,
                ww,
                start_idx,
                start_theta,
                goal_idx,
                gr,
                gc,
                escape_lethal,
                cfg.max_expansions,
                frontier=None,
                frontier_limit=0,
                corridor=corridor_bytes,
            )
            total_expansions += p.expansions

            if p.found_state >= 0:
                return SearchResult(
                    cells=self._reconstruct(p.found_state, ww, r0, c0),
                    status=PlanStatus.PATH_OK,
                    expansions=total_expansions,
                    target_cell=goal_cell,
                    cost=self._g[p.found_state],
                )

            # Budget ran out while still closing on the goal: drive what we
            # have. Natural exhaustion means the goal is walled off, and no
            # amount of extra budget would help -- fall through to frontiers.
            if not p.exhausted and p.best_h < p.start_h - 1e-9:
                cells = self._reconstruct(p.best_state, ww, r0, c0)
                if len(cells) > 1:
                    return SearchResult(
                        cells=cells,
                        status=PlanStatus.PARTIAL,
                        expansions=total_expansions,
                        target_cell=cells[-1],
                        cost=self._g[p.best_state],
                    )

        # -- pass 2: where can we usefully go instead? ------------------ #
        mask = costmap.frontier_mask()[r0 : r1 + 1, c0 : c1 + 1]
        mask = _drop_small_clusters(mask, cfg.min_frontier_cells)
        if not mask.any():
            return SearchResult(status=PlanStatus.NO_PATH, expansions=total_expansions)

        p = self._sweep(
            grid,
            wh,
            ww,
            start_idx,
            start_theta,
            -1,  # no goal: Dijkstra, so it expands outward from the robot
            -1,
            -1,
            escape_lethal,
            cfg.max_expansions,
            frontier=mask.tobytes(),
            frontier_limit=cfg.frontier_sample,
            corridor=corridor_bytes,
        )
        total_expansions += p.expansions

        best = self._score_frontiers(p.frontier_states, ww, r0, c0, goal_cell)
        if best is not None:
            state, cell = best
            cells = self._reconstruct(state, ww, r0, c0)
            if len(cells) > 1:
                return SearchResult(
                    cells=cells,
                    status=PlanStatus.FRONTIER,
                    expansions=total_expansions,
                    target_cell=cell,
                    cost=self._g[state],
                )

        return SearchResult(status=PlanStatus.NO_PATH, expansions=total_expansions)

    # ------------------------------------------------------------------ #
    # The sweep
    # ------------------------------------------------------------------ #

    def _sweep(
        self,
        grid: bytes,
        wh: int,
        ww: int,
        start_idx: int,
        start_theta: Optional[float],
        goal_idx: int,
        gr: int,
        gc: int,
        escape_lethal: bool,
        budget: int,
        frontier: Optional[bytes],
        frontier_limit: int,
        corridor: Optional[bytes] = None,
    ) -> _Pass:
        """One priority-queue sweep.

        With ``goal_idx >= 0`` this is weighted A* toward that cell. With
        ``goal_idx < 0`` the heuristic is zero and it degenerates to Dijkstra,
        which is what the frontier pass wants: nearest-first expansion.
        """
        cfg = self.cfg
        # One state per cell, or one per (cell, heading)? The directional form
        # accounts for turns exactly but multiplies the state space by eight,
        # and expansions -- hence latency -- scale with it. The compact form
        # keeps the best heading reached per cell and charges the turn from
        # there: a few percent worse on paths, several times faster.
        dirstates = cfg.directional_states
        shift = 3 if dirstates else 0
        self._reset_buffers(wh * ww * (NDIR if dirstates else 1))
        g = self._g
        parent = self._parent
        closed = self._closed
        came = self._dir
        touched = self._touched

        # Local bindings and flat tables: this loop is the whole latency
        # budget, and every attribute lookup inside it is paid ~10^6 times.
        turn_table = self._turn_costs()
        cost_weight = cfg.cost_weight
        unknown_penalty = cost_weight * cfg.unknown_cost if cfg.allow_unknown else None
        lethal_penalty = cost_weight * cfg.lethal_escape_cost if escape_lethal else None
        hw = cfg.heuristic_weight
        step_length = STEP_LENGTH
        row_off = tuple(dr for dr, _ in DIRECTIONS)
        col_off = tuple(dc for _, dc in DIRECTIONS)
        heap: List[Tuple[float, float, int]] = []
        heappush = heapq.heappush
        heappop = heapq.heappop
        k = SQRT2 - 2.0
        has_goal = goal_idx >= 0
        discount = min(0.9, max(0.0, cfg.path_hysteresis)) if corridor else 0.0

        out = _Pass()

        sr, sc = divmod(start_idx, ww)
        if has_goal:
            dr0, dc0 = abs(sr - gr), abs(sc - gc)
            out.start_h = (dr0 + dc0) + k * (dr0 if dr0 < dc0 else dc0)
        else:
            out.start_h = 0.0
        out.best_h = out.start_h

        # Seed every departure direction, each charged the true cost of
        # rotating to it from the current heading. With one state per cell the
        # cheapest seed wins outright, so seed in order of increasing turn.
        seeds: List[Tuple[float, int]] = []
        if start_theta is None:
            seeds = [(0.0, d) for d in range(NDIR)]
        else:
            per_rad = cfg.turn_cost / (math.pi * 0.5)
            for d in range(NDIR):
                delta = angle_of_direction(d) - start_theta
                delta = math.atan2(math.sin(delta), math.cos(delta))
                seeds.append((abs(delta) * per_rad, d))
            seeds.sort()

        for cost, d in seeds:
            state = (start_idx << shift) | d if dirstates else start_idx
            if not dirstates and g[state] <= cost:
                continue
            if g[state] == INF:
                touched.append(state)
            g[state] = cost
            came[state] = d
            heappush(heap, (cost + hw * out.start_h, -cost, state))
        out.best_state = (start_idx << shift) | seeds[0][1] if dirstates else start_idx

        expansions = 0
        while heap:
            _, _, state = heappop(heap)
            if closed[state]:
                continue
            closed[state] = True
            expansions += 1

            idx = state >> shift
            if idx == goal_idx:
                out.found_state = state
                break

            if frontier is not None and frontier[idx]:
                out.frontier_states.append((state, idx))
                if len(out.frontier_states) >= frontier_limit:
                    out.exhausted = False
                    break

            if expansions >= budget:
                out.exhausted = False
                break

            cur_g = g[state]
            row = idx // ww
            col = idx - row * ww

            if has_goal:
                dr0 = row - gr
                if dr0 < 0:
                    dr0 = -dr0
                dc0 = col - gc
                if dc0 < 0:
                    dc0 = -dc0
                h_here = (dr0 + dc0) + k * (dr0 if dr0 < dc0 else dc0)
                if h_here < out.best_h:
                    out.best_h = h_here
                    out.best_state = state

            turn_row = ((state & 7) if dirstates else came[state]) << 3

            for d in range(NDIR):
                nrow = row + row_off[d]
                if nrow < 0 or nrow >= wh:
                    continue
                ncol = col + col_off[d]
                if ncol < 0 or ncol >= ww:
                    continue

                nidx = nrow * ww + ncol
                cell = grid[nidx]
                if cell == LETHAL:
                    if lethal_penalty is None:
                        continue
                    penalty = lethal_penalty
                elif cell == UNKNOWN:
                    if unknown_penalty is None:
                        continue
                    penalty = unknown_penalty
                else:
                    penalty = cost_weight * cell

                # Corner-cutting guard: a diagonal step is legal only if both
                # orthogonal cells beside it are clear too. Without this the
                # robot clips the corner of a wall it is inflated against.
                if d & 1 and lethal_penalty is None:
                    if grid[row * ww + ncol] == LETHAL or grid[nrow * ww + col] == LETHAL:
                        continue

                nstate = (nidx << shift) | d if dirstates else nidx
                if closed[nstate]:
                    continue

                weight = 1.0 + penalty
                if discount and corridor[nidx]:
                    weight -= discount
                ng = cur_g + step_length[d] * weight + turn_table[turn_row + d]
                if ng < g[nstate]:
                    if g[nstate] == INF:
                        touched.append(nstate)
                    g[nstate] = ng
                    parent[nstate] = state
                    came[nstate] = d
                    if has_goal:
                        dr1 = nrow - gr
                        if dr1 < 0:
                            dr1 = -dr1
                        dc1 = ncol - gc
                        if dc1 < 0:
                            dc1 = -dc1
                        hn = (dr1 + dc1) + k * (dr1 if dr1 < dc1 else dc1)
                        # Second element breaks f-ties toward the deeper node,
                        # which stops A* fanning sideways across open ground.
                        heappush(heap, (ng + hw * hn, -ng, nstate))
                    else:
                        heappush(heap, (ng, -ng, nstate))

        out.expansions = expansions
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _reconstruct(self, state: int, ww: int, r0: int, c0: int) -> List[Tuple[int, int]]:
        parent = self._parent
        shift = 3 if self.cfg.directional_states else 0
        cells: List[Tuple[int, int]] = []
        while state >= 0:
            idx = state >> shift
            cells.append((idx // ww + r0, idx % ww + c0))
            state = parent[state]
        cells.reverse()
        # Consecutive duplicates appear where only the direction changed.
        deduped = [cells[0]]
        for cell in cells[1:]:
            if cell != deduped[-1]:
                deduped.append(cell)
        return deduped

    def _score_frontiers(
        self,
        candidates: Sequence[Tuple[int, int]],
        ww: int,
        r0: int,
        c0: int,
        goal_cell: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, Tuple[int, int]]]:
        """Trade travel cost against progress toward the real goal."""
        if not candidates:
            return None

        g = self._g
        bias = self.cfg.frontier_goal_bias
        best_score = INF
        best: Optional[Tuple[int, Tuple[int, int]]] = None

        for state, idx in candidates:
            row = idx // ww + r0
            col = idx % ww + c0
            score = g[state]
            if goal_cell is not None:
                dr = abs(row - goal_cell[0])
                dc = abs(col - goal_cell[1])
                score += bias * ((dr + dc) + (SQRT2 - 2.0) * min(dr, dc))
            if score < best_score:
                best_score = score
                best = (state, (row, col))
        return best


def _drop_small_clusters(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove frontier specks caused by sensor noise."""
    if min_size <= 1 or not mask.any():
        return np.ascontiguousarray(mask)
    try:
        from scipy.ndimage import label
    except ImportError:
        return np.ascontiguousarray(mask)
    labels, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.ascontiguousarray(mask)
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(sizes.shape[0], dtype=bool)
    keep[1:] = sizes[1:] >= min_size
    return np.ascontiguousarray(keep[labels])


__all__ = [
    "DIRECTIONS",
    "NDIR",
    "AStarSearch",
    "GridSearch",
    "SearchResult",
    "angle_of_direction",
    "direction_from_angle",
    "turn_steps",
]
