from __future__ import annotations

import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

from carrybot_nav.types import PlanStatus

from .controller import DriveController
from .follower import FollowerConfig, FollowResult, PathFollower
from .odometry import Pose


@dataclass(frozen=True)
class BridgeStatus:
    stamp: float = 0.0
    plan_seq: int = 0
    plan_status: str = "IDLE"
    linear: float = 0.0
    angular: float = 0.0
    remaining: float = 0.0
    finished: bool = False
    #: True on the cycles the robot is sitting at its goal.
    at_goal: bool = False
    #: How many goals have been reached since the bridge started.
    arrivals: int = 0
    stale_plans: int = 0


class NavBridge:
    """Drives a :class:`DriveController` from a ``carrybot_nav`` planner."""

    def __init__(
        self,
        planner,
        drive: DriveController,
        follower: Optional[PathFollower] = None,
        rate_hz: float = 20.0,
        max_plan_age: float = 1.0,
        on_arrival: Optional[Callable[[], None]] = None,
    ):
        self.planner = planner
        #: Called once each time the robot reaches its goal, on the bridge
        #: thread. This is the hook for a mission layer: set the next goal from
        #: it and the robot carries straight on without anything being
        #: restarted. Keep it quick -- it holds up the control cadence.
        self.on_arrival = on_arrival
        self._was_at_goal = False
        self._arrivals = 0
        self.drive = drive
        self.follower = follower or PathFollower(
            FollowerConfig(
                max_linear=drive.cfg.max_linear_speed,
                max_angular=drive.cfg.max_angular_speed,
            )
        )
        self.period = 1.0 / rate_hz
        #: Plans older than this are not driven. A planner that has stopped
        #: publishing is a planner that no longer knows where the robot is.
        self.max_plan_age = max_plan_age

        if drive.cfg.left_gains.kv == 0.0 or drive.cfg.right_gains.kv == 0.0:
            warnings.warn(
                "Velocity feedforward is uncalibrated (kv = 0). The wheels will "
                "take roughly two seconds to reach a commanded speed, and the "
                "robot may spin on the spot instead of following the path. Run "
                "the 'f' command in carrybot_drive.cli first.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status = BridgeStatus()
        self._stale = 0
        self.last_error: Optional[str] = None

    def start(self) -> "NavBridge":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="carrybot-bridge", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.drive.stop()

    def __enter__(self) -> "NavBridge":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def status(self) -> BridgeStatus:
        with self._lock:
            return self._status

    def _run(self) -> None:
        from carrybot_nav.types import Pose as NavPose

        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += self.period
            try:
                self._tick(NavPose)
            except Exception:
                self.last_error = traceback.format_exc()
                self.drive.stop()

            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def _tick(self, NavPose) -> None:
        state = self.drive.state()
        pose = NavPose(state.pose.x, state.pose.y, state.pose.theta)
        self.planner.update_odometry_absolute(pose)

        snapshot = self.planner.latest()
        now = time.monotonic()
        age = now - snapshot.stamp if snapshot.stamp else float("inf")

        at_goal = snapshot.status is PlanStatus.AT_GOAL
        if at_goal and not self._was_at_goal:
            self._arrivals += 1
            self.drive.stop()
            if self.on_arrival is not None:
                try:
                    self.on_arrival()
                except Exception:
                    self.last_error = traceback.format_exc()
        self._was_at_goal = at_goal

        if not snapshot.is_drivable or age > self.max_plan_age:
            if snapshot.is_drivable:
                self._stale += 1
            self.drive.stop()
            self._publish(snapshot, FollowResult(0.0, 0.0, False, 0.0, 0.0), now)
            return

        relative = [Pose(p.x, p.y, p.theta) for p in snapshot.in_frame_of(pose)]
        result = self.follower.compute(relative, state.angular_velocity)

        if result.finished:
            self.drive.stop()
        else:
            self.drive.set_velocity(result.linear, result.angular)

        self._publish(snapshot, result, now)

    def _publish(self, snapshot, result: FollowResult, now: float) -> None:
        with self._lock:
            self._status = BridgeStatus(
                at_goal=snapshot.status is PlanStatus.AT_GOAL,
                arrivals=self._arrivals,
                stamp=now,
                plan_seq=snapshot.seq,
                plan_status=snapshot.status.value,
                linear=result.linear,
                angular=result.angular,
                remaining=result.remaining,
                finished=result.finished,
                stale_plans=self._stale,
            )


__all__ = ["BridgeStatus", "NavBridge"]
