from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .odometry import Pose, wrap_angle


@dataclass
class FollowerConfig:
    #: How far along the path to aim. Too short and the robot weaves; too long
    #: and it cuts corners. Roughly two robot lengths is a sane starting point.
    lookahead: float = 0.35
    #: Above this bearing error the robot pivots instead of driving. A
    #: differential drive can turn in place, so there is no reason to arc
    #: through a large heading error.
    pivot_threshold: float = 0.55
    max_linear: float = 0.30
    max_angular: float = 2.0
    #: Proportional gain on bearing error, rad/s per rad.
    heading_gain: float = 2.5
    #: Damping on the robot's *measured* turn rate, rad/s per rad/s. Without
    #: it the follower is pure proportional and assumes the turn rate it asks
    #: for is the turn rate it gets. When the velocity loop lags -- which it
    #: does badly with feedforward uncalibrated -- the robot sails past the
    #: heading it was aiming at, the follower reverses, and it spins on the
    #: spot indefinitely. Subtracting the turn already underway is what stops
    #: it asking for more of a turn it is going to overshoot anyway.
    heading_damping: float = 0.45
    #: Start easing off the throttle this far from the end of the path.
    approach_distance: float = 0.30
    #: Distance at which the path counts as finished.
    arrival_tolerance: float = 0.05
    #: Final heading tolerance, when the path specifies one.
    heading_tolerance: float = 0.10
    #: Floor on the throttle scale while turning, so a tight corner still
    #: creeps forward rather than stalling into a pure pivot.
    min_throttle: float = 0.15


@dataclass(frozen=True)
class FollowResult:
    linear: float
    angular: float
    finished: bool
    #: Distance from the robot to the end of the path, in metres.
    remaining: float
    #: Bearing to the lookahead point, in radians.
    bearing: float


class PathFollower:
    """Stateless: hand it the path in the robot's frame, get velocities back."""

    def __init__(self, config: Optional[FollowerConfig] = None):
        self.cfg = config or FollowerConfig()

    def compute(
        self, poses: Sequence[Pose], angular_velocity: float = 0.0
    ) -> FollowResult:
        """``poses`` are robot-relative, poses[0] at the robot.

        ``angular_velocity`` is the robot's measured turn rate in rad/s. Pass
        it: it is what turns this from a proportional controller into a PD one,
        and the D term is what keeps it stable when the wheels are slow to
        respond.
        """
        cfg = self.cfg

        if len(poses) < 2:
            return FollowResult(0.0, 0.0, True, 0.0, 0.0)

        end = poses[-1]
        straight_line = math.hypot(end.x, end.y)

        if straight_line <= cfg.arrival_tolerance:
            # Arrived. If the path asked for a final heading, turn to it.
            error = wrap_angle(end.theta)
            if abs(error) > cfg.heading_tolerance:
                command = cfg.heading_gain * error - cfg.heading_damping * angular_velocity
                return FollowResult(
                    0.0,
                    _clamp(command, -cfg.max_angular, cfg.max_angular),
                    False,
                    straight_line,
                    error,
                )
            return FollowResult(0.0, 0.0, True, straight_line, 0.0)

        target = self._lookahead_point(poses)
        bearing = math.atan2(target[1], target[0])
        angular = _clamp(
            cfg.heading_gain * bearing - cfg.heading_damping * angular_velocity,
            -cfg.max_angular,
            cfg.max_angular,
        )

        if abs(bearing) > cfg.pivot_threshold:
            return FollowResult(0.0, angular, False, straight_line, bearing)

        # Ease off for the corner and again for the end of the path.
        turn_scale = max(cfg.min_throttle, 1.0 - abs(bearing) / cfg.pivot_threshold)
        approach = min(1.0, max(0.0, straight_line / cfg.approach_distance))
        linear = cfg.max_linear * turn_scale * max(approach, cfg.min_throttle)

        return FollowResult(linear, angular, False, straight_line, bearing)

    def _lookahead_point(self, poses: Sequence[Pose]) -> Tuple[float, float]:
        """Walk the polyline and interpolate at ``lookahead`` metres along it."""
        remaining = self.cfg.lookahead
        for i in range(1, len(poses)):
            ax, ay = poses[i - 1].x, poses[i - 1].y
            bx, by = poses[i].x, poses[i].y
            segment = math.hypot(bx - ax, by - ay)
            if segment <= 1e-9:
                continue
            if segment >= remaining:
                t = remaining / segment
                return ax + t * (bx - ax), ay + t * (by - ay)
            remaining -= segment
        return poses[-1].x, poses[-1].y


def _path_length(poses: Sequence[Pose]) -> float:
    return sum(
        math.hypot(poses[i].x - poses[i - 1].x, poses[i].y - poses[i - 1].y)
        for i in range(1, len(poses))
    )


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


__all__ = ["FollowResult", "FollowerConfig", "PathFollower"]
