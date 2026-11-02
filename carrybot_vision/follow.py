from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Deque, Optional, Tuple

from .types import TargetObservation


def _wrap(angle: float) -> float:
    angle = math.fmod(angle + math.pi, 2.0 * math.pi)
    if angle <= 0.0:
        angle += 2.0 * math.pi
    return angle - math.pi


class FollowState(Enum):
    #: No target has ever been seen, or the last one is long gone.
    SEARCHING = "SEARCHING"
    #: Driving toward the standoff point.
    FOLLOWING = "FOLLOWING"
    #: At the standoff; tracking but not moving.
    HOLDING = "HOLDING"
    #: Briefly out of sight. The last goal still stands.
    COASTING = "COASTING"
    #: Out of sight for too long to trust the estimate, but still driving to
    #: where it said they went, in the hope of seeing them from there.
    PURSUING = "PURSUING"
    #: Out of sight, and done looking.
    LOST = "LOST"


@dataclass
class FollowConfig:
    #: How far behind the person to sit, in metres.
    standoff: float = 1.20
    #: Extra distance before a held robot starts moving again. Without this
    #: band the robot creeps and stops continuously as the person shifts.
    resume_band: float = 0.35
    #: Keep driving to the last goal for this long after losing sight.
    coast_time: float = 0.60
    #: After this long, stop and report LOST.
    lost_time: float = 2.50
    #: Refuse sightings further than this; the range error out there is larger
    #: than the reading.
    max_range: float = 5.00
    #: Refuse sightings closer than this -- inside the stereo near limit the
    #: numbers are meaningless.
    min_range: float = 0.45
    #: Correct the bearing for robot motion between capture and use.
    motion_compensate: bool = True
    #: Seconds of pose history kept for that correction.
    pose_history: float = 3.0

    # -- carrying the target through a gap ------------------------------- #
    #: Extrapolate the target along their last measured velocity for this long
    #: after the tracker loses them, in seconds. 0 disables it.
    #:
    #: People do not stop existing when a detector drops a frame, and they
    #: rarely stop moving either. Freezing the target where they were last seen
    #: makes the robot steer at a spot the person has already left, and -- much
    #: worse -- makes the head search the place they are *not*. Carrying the
    #: last velocity forward for a beat keeps the aim roughly right through a
    #: dropout, and is the difference between reacquiring someone who walked
    #: behind a pillar and starting a blind sweep.
    #:
    #: Bounded deliberately. Past about a second the extrapolation is fiction:
    #: they may have stopped, turned, or sat down. Beyond this horizon the
    #: estimate is held where the prediction left it rather than run on.
    predict_horizon: float = 1.20
    #: Seconds of target history used to measure their velocity. Long enough to
    #: average out tracker jitter, short enough to follow a change of direction.
    velocity_window: float = 0.70
    #: Sightings needed before a velocity is trusted at all.
    velocity_samples: int = 3
    #: Reject speeds above this as tracker noise rather than athletics, m/s.
    max_target_speed: float = 3.0

    # -- going after them ------------------------------------------------ #
    #: How much of the way to the last estimated position the robot will
    #: travel once it has properly lost the target, as a fraction of the
    #: distance. The distance is measured to the end of the prediction, so it
    #: includes however far the target was carried after the tracker dropped
    #: them.
    #:
    #: 0 stops the robot dead where it stood, which is the safe, useless
    #: answer: standing still is the one thing guaranteed not to bring anybody
    #: back into view. 1 drives the whole way to the spot the estimate ran out
    #: at, which is what you want for someone who went behind a pillar or round
    #: a doorway -- the view from there is the view that finds them.
    #:
    #: The point is fixed in the world the moment the pursuit starts, not
    #: recomputed from the shrinking distance each cycle. Re-taking a fraction
    #: of what is left every time is how a robot ends up creeping towards a
    #: place it never reaches.
    pursue_fraction: float = 0.75
    #: Give up the pursuit after this long, seconds. 0 means no limit.
    pursue_time: float = 8.0
    #: Close enough to the pursuit point to call it arrived, metres.
    pursue_arrive: float = 0.30


@dataclass(frozen=True)
class FollowCommand:
    state: FollowState = FollowState.SEARCHING
    #: Goal in the robot frame, metres, or None when there is nothing to drive
    #: to. ``+x`` forward, ``+y`` left -- the planner's convention.
    goal: Optional[Tuple[float, float]] = None
    #: True when the caller should actively stop rather than merely not move.
    stop: bool = True
    #: Range to the target as last known, metres.
    target_range: float = 0.0
    #: Bearing to the target as last known, radians, positive to the right.
    #: Reported even when holding or coasting, because "do not drive" is not
    #: the same as "stop caring where they are" -- something still has to keep
    #: them in view.
    target_bearing: float = 0.0
    #: Age of the sighting the command is based on, seconds.
    age: float = 0.0
    #: True when the position behind this command is extrapolated rather than
    #: measured. Worth surfacing: a goal built on a prediction deserves less
    #: confidence than one built on a sighting, and a log that does not say
    #: which is which is very hard to read afterwards.
    predicted: bool = False
    #: The target's measured speed over the ground, m/s. 0 when unknown.
    target_speed: float = 0.0
    reason: str = "no target"

    @property
    def is_driving(self) -> bool:
        return self.goal is not None and not self.stop


class FollowController:
    """Sightings in, standoff goals out."""

    def __init__(self, config: Optional[FollowConfig] = None):
        self.cfg = config or FollowConfig()
        self.state = FollowState.SEARCHING
        self._poses: Deque[Tuple[float, float, float, float]] = deque()
        self._last_valid: Optional[TargetObservation] = None
        self._last_goal: Optional[Tuple[float, float]] = None
        #: (stamp, world x, world y) for recent sightings, for the velocity fit.
        self._track: Deque[Tuple[float, float, float]] = deque()
        self._velocity = (0.0, 0.0)
        #: Where the pursuit is headed, in world coordinates, and when it began.
        self._pursuit: Optional[Tuple[float, float]] = None
        self._pursuit_from: float = 0.0
        #: Seen again, but not ranged: no more chasing a prediction until a
        #: proper sighting -- they are not where the prediction went.
        self._pursuit_off = False
        self.arrivals = 0
        self.pursuits = 0

    # ------------------------------------------------------------------ #

    def note_pose(self, stamp: float, x: float, y: float, heading: float) -> None:
        """Record where the robot was, so old sightings can be corrected.

        Call this every control cycle with the odometry pose. Keeping the whole
        pose rather than just the heading also corrects for the metre or so the
        robot may have driven while a frame was being processed.
        """
        self._poses.append((stamp, x, y, heading))
        cutoff = stamp - self.cfg.pose_history
        while self._poses and self._poses[0][0] < cutoff:
            self._poses.popleft()

    def _pose_at(self, stamp: float) -> Optional[Tuple[float, float, float]]:
        if not self._poses:
            return None
        best = min(self._poses, key=lambda p: abs(p[0] - stamp))
        return best[1], best[2], best[3]

    def reset(self) -> None:
        self.state = FollowState.SEARCHING
        self._last_valid = None
        self._last_goal = None
        self._track.clear()
        self._velocity = (0.0, 0.0)
        self._pursuit = None
        self._pursuit_off = False

    # ------------------------------------------------------------------ #
    # Velocity
    # ------------------------------------------------------------------ #

    @property
    def velocity(self) -> Tuple[float, float]:
        """The target's measured velocity over the ground, m/s, world frame."""
        return self._velocity

    @property
    def speed(self) -> float:
        return math.hypot(*self._velocity)

    def _note_target(self, observation: TargetObservation) -> None:
        """Record where a sighting puts the target in the world, and re-fit."""
        placed = self._world_of(observation)
        if placed is None:
            return

        # A sighting that arrives long after the last one is the far side of a
        # dropout, not the next point on a trajectory. Fitting a line across
        # the gap divides a short displacement by a long interval and reports
        # somebody strolling at a third of their actual speed -- which then
        # under-predicts exactly when the prediction was needed most. Start
        # again instead.
        if self._track and observation.stamp - self._track[-1][0] > self.cfg.velocity_window:
            self._track.clear()
            self._velocity = (0.0, 0.0)

        self._track.append((observation.stamp, placed[0], placed[1]))
        cutoff = observation.stamp - self.cfg.velocity_window
        while len(self._track) > 1 and self._track[0][0] < cutoff:
            self._track.popleft()
        self._velocity = self._fit_velocity()

    def _fit_velocity(self) -> Tuple[float, float]:
        """Least-squares velocity over the window.

        A straight-line fit rather than differencing the two ends: the tracker's
        range is noisy at a few centimetres, and over a 0.7 s window two-point
        differencing turns that noise into a velocity error of tenths of a metre
        per second, which then gets multiplied by the prediction horizon.
        """
        cfg = self.cfg
        if len(self._track) < max(2, cfg.velocity_samples):
            return (0.0, 0.0)

        stamps = [p[0] for p in self._track]
        span = stamps[-1] - stamps[0]
        if span < 1e-3:
            return (0.0, 0.0)

        mean_t = sum(stamps) / len(stamps)
        denom = sum((t - mean_t) ** 2 for t in stamps)
        if denom < 1e-9:
            return (0.0, 0.0)

        mean_x = sum(p[1] for p in self._track) / len(self._track)
        mean_y = sum(p[2] for p in self._track) / len(self._track)
        vx = sum((t - mean_t) * (p[1] - mean_x) for t, p in zip(stamps, self._track)) / denom
        vy = sum((t - mean_t) * (p[2] - mean_y) for t, p in zip(stamps, self._track)) / denom

        speed = math.hypot(vx, vy)
        if speed > cfg.max_target_speed:
            # Somebody teleported, which means the tracker swapped identity or
            # the range estimate glitched. Either way it is not a velocity.
            scale = cfg.max_target_speed / speed
            return (vx * scale, vy * scale)
        return (vx, vy)

    # ------------------------------------------------------------------ #

    def update(
        self, observation: Optional[TargetObservation], now: Optional[float] = None
    ) -> FollowCommand:
        now = time.monotonic() if now is None else now
        cfg = self.cfg

        if observation is not None and observation.valid:
            if not (cfg.min_range <= observation.distance <= cfg.max_range):
                # Seen, if not ranged. Whatever the prediction said, they are
                # here: stop chasing it -- driving on to where they were
                # supposed to be took the robot straight past them -- and say
                # where they are, so the robot can at least turn to them.
                self._pursuit = None
                self._pursuit_off = True
                command = self._stale(
                    now,
                    f"range {observation.distance:.2f} m outside "
                    f"{cfg.min_range:.2f}-{cfg.max_range:.1f} m",
                )
                return replace(command, goal=None, stop=True, age=0.0,
                               target_bearing=self._bearing_now(observation))
            self._last_valid = observation
            self._note_target(observation)
            self._pursuit = None       # found them; nothing left to chase
            self._pursuit_off = False

        if self._last_valid is None:
            self.state = FollowState.SEARCHING
            return FollowCommand(state=self.state, reason="no target yet")

        age = now - self._last_valid.stamp
        x, y = self._target_now(self._last_valid, now)
        distance = math.hypot(x, y)
        bearing = -math.atan2(y, x)     # right-positive, as sightings report
        speed = self.speed
        predicted = age > 1e-3 and self.cfg.predict_horizon > 0.0 and speed > 0.02

        if age > cfg.lost_time:
            self._last_goal = None
            # Still report the bearing. Where the prediction ran out is a much
            # better place to start looking than where they were last actually
            # seen -- somebody who walked past is not still standing there.
            chase = self._pursue(x, y, now)
            if chase is not None:
                self.state = FollowState.PURSUING
                goal, left, elapsed = chase
                return FollowCommand(
                    state=self.state,
                    goal=goal,
                    stop=False,
                    target_range=distance,
                    target_bearing=bearing,
                    age=age,
                    predicted=predicted,
                    target_speed=speed,
                    reason=(
                        f"unseen for {age:.1f} s, going {left:.2f} m further to "
                        f"where they were headed ({elapsed:.1f} s in)"
                    ),
                )

            self.state = FollowState.LOST
            return FollowCommand(
                state=self.state,
                target_range=distance,
                target_bearing=bearing,
                age=age,
                predicted=predicted,
                target_speed=speed,
                reason=f"target unseen for {age:.1f} s",
            )

        if age > cfg.coast_time:
            # Dead reckoning, and the goal moves with it. Freezing the last
            # goal here steers the robot at a spot the person left seconds ago;
            # carrying their velocity keeps it pointed where they are going.
            self.state = FollowState.COASTING
            goal = None
            if distance > cfg.standoff:
                scale = (distance - cfg.standoff) / distance
                goal = (x * scale, y * scale)
                self._last_goal = goal
            return FollowCommand(
                state=self.state,
                goal=goal,
                stop=goal is None,
                target_range=distance,
                target_bearing=bearing,
                age=age,
                predicted=predicted,
                target_speed=speed,
                reason=(
                    f"coasting on a {age:.1f} s old sighting"
                    + (f", carried at {speed:.2f} m/s" if predicted else "")
                ),
            )

        # -- hysteresis --------------------------------------------------- #
        if distance <= cfg.standoff:
            if self.state is not FollowState.HOLDING:
                self.arrivals += 1
            self.state = FollowState.HOLDING
        elif distance >= cfg.standoff + cfg.resume_band:
            self.state = FollowState.FOLLOWING
        elif self.state in (FollowState.SEARCHING, FollowState.LOST, FollowState.COASTING):
            # Re-acquired inside the band: hold rather than lurch forward.
            self.state = FollowState.HOLDING

        if self.state is FollowState.HOLDING:
            self._last_goal = None
            return FollowCommand(
                state=self.state,
                stop=True,
                target_range=distance,
                target_bearing=bearing,
                age=age,
                predicted=predicted,
                target_speed=speed,
                reason=f"at standoff ({distance:.2f} m)",
            )

        # -- goal: short of the person, along the line to them ------------ #
        scale = (distance - cfg.standoff) / distance
        goal = (x * scale, y * scale)
        self._last_goal = goal
        return FollowCommand(
            state=FollowState.FOLLOWING,
            goal=goal,
            stop=False,
            target_range=distance,
            target_bearing=bearing,
            age=age,
            predicted=predicted,
            target_speed=speed,
            reason=f"{distance:.2f} m away, driving to {cfg.standoff:.2f} m",
        )

    # ------------------------------------------------------------------ #

    def _pursue(
        self, x: float, y: float, now: float
    ) -> Optional[Tuple[Tuple[float, float], float, float]]:
        """Keep going towards where the estimate said they went.

        Returns ``(goal, metres remaining, seconds elapsed)``, or None when
        there is no pursuit to make -- disabled, finished, or timed out.

        ``x, y`` are the last estimate in the robot's *current* frame, so the
        pursuit point is fixed in the world on the first call and then simply
        re-expressed as the robot drives towards it.
        """
        cfg = self.cfg
        if cfg.pursue_fraction <= 0.0 or self._pursuit_off:
            return None

        current = self._poses[-1][1:] if self._poses else None
        if current is None:
            return None
        cx, cy, ctheta = current

        if self._pursuit is None:
            # Fix the point, once. The fraction is of the distance from where
            # the robot stood when it gave up to the end of the prediction.
            wx = cx + x * math.cos(ctheta) - y * math.sin(ctheta)
            wy = cy + x * math.sin(ctheta) + y * math.cos(ctheta)
            frac = min(1.0, cfg.pursue_fraction)
            self._pursuit = (cx + (wx - cx) * frac, cy + (wy - cy) * frac)
            self._pursuit_from = now
            self.pursuits += 1

        elapsed = now - self._pursuit_from
        if cfg.pursue_time > 0.0 and elapsed > cfg.pursue_time:
            return None

        dx, dy = self._pursuit[0] - cx, self._pursuit[1] - cy
        left = math.hypot(dx, dy)
        if left <= cfg.pursue_arrive:
            # Arrived, and they are not here. Nothing more to be gained by
            # driving; the head can go on sweeping from a better vantage point.
            return None

        cos_t, sin_t = math.cos(-ctheta), math.sin(-ctheta)
        goal = (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)
        return goal, left, elapsed

    def _bearing_now(self, observation: TargetObservation) -> float:
        """A sighting's bearing, moved into the robot's current frame by the
        turn made since it was captured -- like every other bearing the
        follower reports."""
        then = self._pose_at(observation.stamp)
        if not self.cfg.motion_compensate or then is None or not self._poses:
            return observation.bearing
        turned = self._poses[-1][3] - then[2]
        turned = math.atan2(math.sin(turned), math.cos(turned))
        return observation.bearing + turned     # right-positive; a left turn moves them right

    def _stale(self, now: float, reason: str) -> FollowCommand:
        age = now - self._last_valid.stamp if self._last_valid else 0.0
        return FollowCommand(
            state=self.state,
            goal=self._last_goal,
            stop=self._last_goal is None,
            age=age,
            reason=reason,
        )

    def _world_of(self, observation: TargetObservation) -> Optional[Tuple[float, float]]:
        """Where a sighting puts the target in the world frame.

        Needs the robot pose at *capture* time, which is why poses are kept.
        Returns None when there is no pose history to place it against.
        """
        then = self._pose_at(observation.stamp)
        if then is None:
            return None
        x, y = observation.position
        tx, ty, ttheta = then
        return (
            tx + x * math.cos(ttheta) - y * math.sin(ttheta),
            ty + x * math.sin(ttheta) + y * math.cos(ttheta),
        )

    def _target_now(
        self, observation: TargetObservation, now: float
    ) -> Tuple[float, float]:
        """Where the target is in the robot's frame *now*.

        Two corrections, in order. The sighting was taken from a pose the robot
        has since left, so it is placed in the world using the pose at capture
        time and re-expressed from the current pose -- that handles both the
        rotation and the translation the robot made in between. Then, if the
        sighting is stale, the target is carried along its own measured
        velocity, because the person kept walking while the tracker was not
        looking.
        """
        x, y = observation.position          # robot frame at capture time
        if not self.cfg.motion_compensate:
            return x, y

        placed = self._world_of(observation)
        current = self._poses[-1][1:] if self._poses else None
        if placed is None or current is None:
            return x, y
        wx, wy = placed

        # Carry them forward, for a bounded while.
        age = now - observation.stamp
        horizon = self.cfg.predict_horizon
        if horizon > 0.0 and age > 0.0:
            lead = min(age, horizon)
            wx += self._velocity[0] * lead
            wy += self._velocity[1] * lead

        # world -> current robot frame
        cx, cy, ctheta = current
        dx, dy = wx - cx, wy - cy
        cos_t, sin_t = math.cos(-ctheta), math.sin(-ctheta)
        return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t


__all__ = ["FollowCommand", "FollowConfig", "FollowController", "FollowState"]
