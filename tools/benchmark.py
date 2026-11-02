"""Latency and path-quality benchmark for the CarryBot planner.

Exploration behaviour is measured by ``tools/simulate.py``. This measures the
thing underneath it: given a map, how long does a plan take and how good is it?

    python -m tools.benchmark
    python -m tools.benchmark --sizes 201 401 801 --trials 40
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carrybot_nav import FREE, LETHAL, PlannerConfig, PlanStatus, WorldCostmap
from carrybot_nav.path import (
    assign_headings,
    cells_to_points,
    enforce_spacing,
    path_length,
    shortcut,
    simplify,
    total_rotation,
)
from carrybot_nav.search import AStarSearch


def random_map(size: int, rng: np.random.Generator, clutter: float = 0.10) -> np.ndarray:
    """Blocky obstacles on an open field, walled at the edges."""
    grid = np.full((size, size), FREE, dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = LETHAL

    area = 0
    target = clutter * size * size
    while area < target:
        h = int(rng.integers(size // 40 + 2, size // 8 + 3))
        w = int(rng.integers(size // 40 + 2, size // 8 + 3))
        r = int(rng.integers(1, size - h - 1))
        c = int(rng.integers(1, size - w - 1))
        grid[r : r + h, c : c + w] = LETHAL
        area += h * w
    return grid


def _endpoints(
    costmap: WorldCostmap, size: int, rng: np.random.Generator
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """A start and goal on clear ground, far enough apart to be interesting."""
    for _ in range(400):
        a = (int(rng.integers(2, size - 2)), int(rng.integers(2, size - 2)))
        b = (int(rng.integers(2, size - 2)), int(rng.integers(2, size - 2)))
        if math.dist(a, b) < size * 0.5:
            continue
        fa = costmap.nearest_free_cell(*a, 6)
        fb = costmap.nearest_free_cell(*b, 6)
        if fa and fb:
            return fa, fb
    return None


def run(sizes: List[int], trials: int, seed: int, directional: Optional[bool]) -> None:
    variants = [directional] if directional is not None else [False, True]

    for size in sizes:
        for variant in variants:
            cfg = PlannerConfig()
            cfg.directional_states = variant
            cfg.search_radius_m = None  # benchmark the whole map, no windowing

            rng = np.random.default_rng(seed)
            times: List[float] = []
            expansions: List[int] = []
            lengths: List[float] = []
            ratios: List[float] = []
            rotations: List[float] = []
            statuses: List[str] = []

            for _ in range(trials):
                grid = random_map(size, rng)
                cm = WorldCostmap(cfg)
                cm.load_absolute(grid, 0.0, 0.0)
                ends = _endpoints(cm, size, rng)
                if ends is None:
                    continue
                start_cell, goal_cell = ends

                search = AStarSearch(cfg)
                t0 = time.perf_counter()
                result = search.plan(cm, start_cell, goal_cell, start_theta=0.0)
                elapsed = (time.perf_counter() - t0) * 1e3

                statuses.append(result.status.value)
                if result.status is not PlanStatus.PATH_OK:
                    continue

                points = cells_to_points(cm, result.cells)
                points = shortcut(points, cm)
                points = simplify(points, cfg.simplify_tolerance, cm)
                points = enforce_spacing(points, cfg.min_waypoint_spacing, cm)
                poses = assign_headings(points, start_theta=0.0)

                straight = (
                    math.dist(start_cell, goal_cell) * cfg.resolution
                )
                length = path_length(poses)
                times.append(elapsed)
                expansions.append(result.expansions)
                lengths.append(length)
                ratios.append(length / max(1e-9, straight))
                rotations.append(total_rotation(poses))

            if not times:
                print(f"size {size:>4} directional={variant}: no solvable trials")
                continue

            label = "directional" if variant else "compact    "
            print(
                f"grid {size:>4}x{size:<4} {label}  "
                f"plan ms: mean {statistics.mean(times):7.1f}  "
                f"p95 {np.percentile(times, 95):7.1f}  max {max(times):7.1f}   "
                f"expansions mean {statistics.mean(expansions):8.0f}   "
                f"len/straight {statistics.mean(ratios):.3f}   "
                f"turn rad {statistics.mean(rotations):5.2f}   "
                f"solved {len(times)}/{len(statuses)}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[201, 401, 801])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--states",
        choices=["compact", "directional", "both"],
        default="both",
        help="which state-space variant to measure",
    )
    args = ap.parse_args(argv)

    directional = {"compact": False, "directional": True, "both": None}[args.states]
    print(
        "Whole-map search, no windowing -- this is the worst case. "
        "The planner windows by default, which bounds it further.\n"
    )
    run(args.sizes, args.trials, args.seed, directional)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
