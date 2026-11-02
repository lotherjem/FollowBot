from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import DriveConfig


def wrap_angle(theta: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    theta = math.fmod(theta + math.pi, 2.0 * math.pi)
    if theta <= 0.0:
        theta += 2.0 * math.pi
    return theta - math.pi


@dataclass(frozen=True)
class Pose:
    """Planar pose in metres and radians.

    Field-compatible with ``carrybot_nav.types.Pose`` -- ``Pose(*state.pose)``
    converts between them -- but defined here so the drive layer stays pure
    stdlib and can run on a board with no numpy.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.theta

    def distance_to(self, other: "Pose") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


class DifferentialOdometry:
    """Accumulates a pose from wheel counts. Thread-safe."""

    def __init__(self, config: DriveConfig):
        self.cfg = config
        self._lock = threading.Lock()
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._last_left: Optional[int] = None
        self._last_right: Optional[int] = None
        self._distance = 0.0
        self._rotation = 0.0

    def reset(self, pose: Optional[Pose] = None) -> None:
        """Zero the pose. Counts are re-baselined on the next update."""
        pose = pose or Pose()
        with self._lock:
            self._x, self._y, self._theta = pose.x, pose.y, pose.theta
            self._last_left = None
            self._last_right = None
            self._distance = 0.0
            self._rotation = 0.0

    def set_heading(self, theta: float) -> None:
        """Overwrite the heading, e.g. with a fused IMU estimate."""
        with self._lock:
            self._theta = wrap_angle(theta)

    def update(self, left_counts: int, right_counts: int) -> Pose:
        with self._lock:
            if self._last_left is None:
                self._last_left = left_counts
                self._last_right = right_counts
                return Pose(self._x, self._y, self._theta)

            d_left = (left_counts - self._last_left) * self.cfg.left_metres_per_count
            d_right = (right_counts - self._last_right) * self.cfg.right_metres_per_count
            self._last_left = left_counts
            self._last_right = right_counts

            d_centre = 0.5 * (d_left + d_right)
            d_theta = (d_right - d_left) / self.cfg.track_width_m

            # Heading at the midpoint of the step, not the start of it.
            mid = self._theta + 0.5 * d_theta
            self._x += d_centre * math.cos(mid)
            self._y += d_centre * math.sin(mid)
            self._theta = wrap_angle(self._theta + d_theta)

            self._distance += abs(d_centre)
            self._rotation += abs(d_theta)

            return Pose(self._x, self._y, self._theta)

    @property
    def pose(self) -> Pose:
        with self._lock:
            return Pose(self._x, self._y, self._theta)

    @property
    def travelled(self) -> Tuple[float, float]:
        """Total path length in metres and total rotation in radians."""
        with self._lock:
            return self._distance, self._rotation


__all__ = ["DifferentialOdometry", "Pose", "wrap_angle"]
