from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .costmap import WorldCostmap, bresenham
from .types import Pose, wrap_angle

Point = Tuple[float, float]


def cells_to_points(costmap: WorldCostmap, cells: Sequence[Tuple[int, int]]) -> List[Point]:
    return [costmap.cell_to_world(r, c) for r, c in cells]


def shortcut(points: Sequence[Point], costmap: WorldCostmap) -> List[Point]:
    """Greedy line-of-sight shortcutting.

    From each kept point, reach as far forward as the map allows, then keep the
    furthest visible point and repeat. O(n^2) worst case but n is small once
    the path is a handful of corners, and the early-exit keeps it near-linear
    in open space.
    """
    if len(points) < 3:
        return list(points)

    out = [points[0]]
    i = 0
    n = len(points)
    while i < n - 1:
        best = i + 1
        for j in range(n - 1, i + 1, -1):
            if costmap.segment_is_clear(points[i][0], points[i][1], points[j][0], points[j][1]):
                best = j
                break
        out.append(points[best])
        i = best
    return out


def simplify(points: Sequence[Point], tolerance: float, costmap: WorldCostmap) -> List[Point]:
    """Drop waypoints that add nothing, without cutting into obstacles."""
    if len(points) < 3 or tolerance <= 0.0:
        return list(points)

    out = [points[0]]
    for i in range(1, len(points) - 1):
        prev = out[-1]
        nxt = points[i + 1]
        if _point_line_distance(points[i], prev, nxt) <= tolerance and costmap.segment_is_clear(
            prev[0], prev[1], nxt[0], nxt[1]
        ):
            continue
        out.append(points[i])
    out.append(points[-1])
    return out


def enforce_spacing(
    points: Sequence[Point], min_spacing: float, costmap: WorldCostmap
) -> List[Point]:
    """Merge waypoints that sit on top of each other.

    A first waypoint 5 cm ahead of the robot is worse than useless: a follower
    will pivot to face it, overshoot, and pivot back. Anything closer together
    than ``min_spacing`` collapses, and the final waypoint is always kept so
    the path still ends where the search said it should.
    """
    if len(points) < 3 or min_spacing <= 0.0:
        return list(points)

    out = [points[0]]
    for i in range(1, len(points) - 1):
        p = points[i]
        close = math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < min_spacing
        # Dropping a waypoint moves the segment, so it is only safe if the line
        # that replaces it is clear. Skipping this check quietly cuts corners
        # through doorframes, which is exactly where waypoints bunch up.
        if close and costmap.segment_is_clear(
            out[-1][0], out[-1][1], points[i + 1][0], points[i + 1][1]
        ):
            continue
        out.append(p)

    last = points[-1]
    too_close = math.hypot(last[0] - out[-1][0], last[1] - out[-1][1]) < min_spacing
    if too_close and len(out) > 1 and costmap.segment_is_clear(
        out[-2][0], out[-2][1], last[0], last[1]
    ):
        out[-1] = last
    elif out[-1] != last:
        out.append(last)
    return out


def assign_headings(
    points: Sequence[Point],
    start_theta: float = 0.0,
    goal_theta: Optional[float] = None,
    heading_epsilon: float = 0.05,
) -> List[Pose]:
    """Attach the heading the robot needs on arrival at each waypoint.

    ``goal_theta`` is optional. When given and it differs from the approach
    heading, a final pose is appended at the same position -- a pure
    turn-in-place once the robot has arrived.
    """
    if not points:
        return []
    if len(points) == 1:
        return [Pose(points[0][0], points[0][1], start_theta)]

    poses = [Pose(points[0][0], points[0][1], start_theta)]
    for i in range(1, len(points)):
        px, py = points[i - 1]
        cx, cy = points[i]
        heading = math.atan2(cy - py, cx - px)
        poses.append(Pose(cx, cy, heading))

    if goal_theta is not None:
        final = poses[-1]
        if abs(wrap_angle(goal_theta - final.theta)) > heading_epsilon:
            poses.append(Pose(final.x, final.y, wrap_angle(goal_theta)))

    return poses


def path_length(poses: Sequence[Pose]) -> float:
    return sum(
        math.hypot(poses[i].x - poses[i - 1].x, poses[i].y - poses[i - 1].y)
        for i in range(1, len(poses))
    )


def total_rotation(poses: Sequence[Pose]) -> float:
    """Total in-place rotation the path demands, in radians.

    Handy when tuning ``turn_cost``: watch this against ``path_length``.
    """
    return sum(
        abs(wrap_angle(poses[i].theta - poses[i - 1].theta))
        for i in range(1, len(poses))
    )


def resample(poses: Sequence[Pose], spacing: float, horizon: float) -> List[Point]:
    """Points every ``spacing`` metres along the first ``horizon`` metres."""
    if len(poses) < 2 or spacing <= 0.0:
        return []
    out: List[Point] = []
    travelled = 0.0
    ax, ay = poses[0].x, poses[0].y
    for i in range(1, len(poses)):
        bx, by = poses[i].x, poses[i].y
        segment = math.hypot(bx - ax, by - ay)
        if segment < 1e-9:
            continue
        t = 0.0
        while t + spacing <= segment and travelled < horizon:
            t += spacing
            travelled += spacing
            out.append((ax + (bx - ax) * t / segment, ay + (by - ay) * t / segment))
        ax, ay = bx, by
        if travelled >= horizon:
            break
    return out


def point_to_path(point: Point, poses: Sequence[Pose]) -> float:
    """Shortest distance from a point to a path, treated as line segments."""
    best = float("inf")
    for i in range(1, len(poses)):
        ax, ay = poses[i - 1].x, poses[i - 1].y
        bx, by = poses[i].x, poses[i].y
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            best = min(best, math.hypot(point[0] - ax, point[1] - ay))
            continue
        t = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / denom))
        best = min(best, math.hypot(point[0] - (ax + t * dx), point[1] - (ay + t * dy)))
    return best


def path_deviation(
    candidate: Sequence[Pose],
    incumbent: Sequence[Pose],
    horizon: float = 2.0,
    spacing: float = 0.1,
) -> float:
    """How far a new path strays from the one being followed, in metres.

    Mean distance from samples of ``candidate`` to the nearest point on
    ``incumbent``, over the near field only -- the part the follower is
    actually steering by. Measured in the world frame, so a path that is simply
    the same route with the robot further along it scores near zero. That is
    the distinction worth making: re-running A* every cycle produces a slightly
    different discretisation of the *same* route, and swapping to it achieves
    nothing except moving the follower's target.
    """
    if len(candidate) < 2 or len(incumbent) < 2:
        return float("inf")
    samples = resample(candidate, spacing, horizon)
    if not samples:
        return float("inf")
    return sum(point_to_path(p, incumbent) for p in samples) / len(samples)


def plan_cost(
    poses: Sequence[Pose],
    costmap: WorldCostmap,
    config,
    start_theta: float,
) -> float:
    """Estimate what an existing plan would cost the search, in its own units.

    Used to compare an incumbent route against a freshly searched one. It walks
    the same cells with the same per-cell penalties and the same turn cost, so
    the two numbers are comparable to within the discretisation of the line
    rasteriser -- close enough to decide whether a new plan is *meaningfully*
    better, which is all it is asked to do.
    """
    from .types import LETHAL, UNKNOWN

    if len(poses) < 2:
        return 0.0

    cost_layer = costmap.cost
    h, w = costmap.shape
    cw = config.cost_weight
    unknown_pen = cw * config.unknown_cost
    lethal_pen = cw * config.lethal_escape_cost
    per_rad = config.turn_cost / (math.pi * 0.5)

    total = 0.0
    theta = start_theta
    for i in range(1, len(poses)):
        a, b = poses[i - 1], poses[i]
        dx, dy = b.x - a.x, b.y - a.y
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            heading = math.atan2(dy, dx)
            total += abs(wrap_angle(heading - theta)) * per_rad
            theta = heading

        r0, c0 = costmap.world_to_cell(a.x, a.y)
        r1, c1 = costmap.world_to_cell(b.x, b.y)
        prev: Optional[Tuple[int, int]] = None
        for row, col in bresenham(r0, c0, r1, c1):
            if prev is None:
                prev = (row, col)
                continue
            step = math.sqrt(2.0) if (row != prev[0] and col != prev[1]) else 1.0
            prev = (row, col)
            if not (0 <= row < h and 0 <= col < w):
                total += step * (1.0 + lethal_pen)
                continue
            value = int(cost_layer[row, col])
            if value == LETHAL:
                pen = lethal_pen
            elif value == UNKNOWN:
                pen = unknown_pen
            else:
                pen = cw * value
            total += step * (1.0 + pen)
    return total


def path_is_clear(poses: Sequence[Pose], costmap: WorldCostmap) -> bool:
    """Re-validate an existing plan against the current map."""
    for i in range(1, len(poses)):
        a, b = poses[i - 1], poses[i]
        if not costmap.segment_is_clear(a.x, a.y, b.x, b.y):
            return False
    return True


def _point_line_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


__all__ = [
    "assign_headings",
    "cells_to_points",
    "enforce_spacing",
    "path_is_clear",
    "path_deviation",
    "path_length",
    "plan_cost",
    "point_to_path",
    "resample",
    "shortcut",
    "simplify",
    "total_rotation",
]
