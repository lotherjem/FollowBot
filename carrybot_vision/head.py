from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .costmap import FREE, LETHAL, UNKNOWN
from .follow import FollowState
from .types import TargetObservation, VisionConfig


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class HeadMode(Enum):
    CENTRED = "CENTRED"
    TRACK = "TRACK"
    #: Following an extrapolated position -- the tracker has dropped them, but
    #: their velocity says where they should be. Distinct from TRACK on
    #: purpose: a log that does not separate "I can see them" from "I think I
    #: know where they went" is very hard to read after the fact.
    COAST = "COAST"
    SEARCH = "SEARCH"
    SURVEY = "SURVEY"
    #: Pointed by hand from the live interface. The controller is bypassed.
    MANUAL = "MANUAL"


@dataclass
class HeadConfig:
    """Tuning for the pan head. Everything here is off or neutral by default."""

    #: Master switch. Off means the head never leaves centre, and the rest of
    #: the stack behaves exactly as it does on a robot with no servo.
    enabled: bool = False
    #: Hard limit on head angle either side of straight ahead, radians. This is
    #: the slider. Keep it inside the servo's mechanical travel.
    max_angle: float = math.radians(60.0)

    # -- tracking -------------------------------------------------------- #
    #: Fraction of the bearing error corrected per update. 1.0 snaps to the
    #: target and overshoots on a noisy bearing; this is the damping.
    track_gain: float = 0.6
    #: Do not move the head for errors smaller than this. Below about three
    #: degrees the person is comfortably in frame already and moving costs a
    #: blurred frame for nothing.
    track_deadband: float = math.radians(3.0)
    #: Keep the target inside this fraction of the half-cone. Tracking to dead
    #: centre fights every step the person takes; tracking to "comfortably
    #: inside the frame" does not.
    track_margin: float = 0.55

    # -- asking the body to help ----------------------------------------- #
    #: May the controller ask the body to turn when the head cannot reach?
    #: This is the answer to "the person walked past me": a head with 60
    #: degrees of travel cannot see something at 150, and a robot that holds
    #: station at its standoff has no path to drive, so nothing else will ever
    #: turn it. Off, the robot watches them walk away and loses them.
    #:
    #: It stays a *hint*. The head never turns the body itself -- the caller
    #: decides whether now is a good moment, and by default acts on it only
    #: when there is no path being driven, so this cannot fight the planner.
    body_assist: bool = True
    #: Do not ask the body to turn for less than this while *driving*: rotating
    #: a robot that is going somewhere, to save the head a few degrees, is not
    #: worth fighting the path for.
    body_deadband: float = math.radians(12.0)
    #: The same, while stationary -- far smaller, because a robot standing
    #: still has nothing to fight. Turning on the spot costs a little wheel
    #: slip and buys keeping the person in frame.
    body_deadband_still: float = math.radians(3.0)
    #: Fastest the body should be asked to turn on the head's behalf while
    #: driving, rad/s. Modest, because it is competing with the planner.
    body_turn_speed: float = 0.9
    #: The same, while stationary. Much higher: keeping up with someone walking
    #: past at arm's length needs real angular rate, and a stationary robot has
    #: no path to disturb. It is still bounded -- a body that whips round
    #: smears every frame of the search that justified turning it -- but the
    #: bound is "fast enough to keep up with a person", not "gentle".
    body_turn_speed_still: float = 2.5
    #: Proportional gain from "how far off they are" to "how fast to turn",
    #: 1/s. Lives here rather than in each consumer so the simulator and the
    #: robot cannot quietly disagree about how eager the robot is.
    body_gain: float = 2.5
    #: Fraction of the head's travel the target may occupy before the body is
    #: asked to help, while *driving*. Below this the head handles it alone.
    #:
    #: Waiting for the head to saturate before turning the body is too late:
    #: by then the target is at the edge of a cone they are still moving out
    #: of, and the next thing that happens is a search. Starting at about
    #: two-thirds of travel gives the head room to keep working while the body
    #: slowly takes over the coarse part -- which is what a person does when
    #: they watch someone walk past, eyes first and then the neck.
    body_follow_ratio: float = 0.6
    #: The same, while stationary -- and much lower, so the body starts coming
    #: round almost as soon as the target is off-centre. A robot holding at its
    #: standoff has literally nothing else to do, and the one thing that loses
    #: the target is being slow to turn.
    body_follow_ratio_still: float = 0.20
    #: Seconds of fruitless head-only searching before the body is asked to
    #: extend the sweep. Long enough that a target who merely stepped behind a
    #: doorframe is found without the robot spinning.
    search_body_after: float = 1.5
    #: How fast the body-assisted search widens, radians per second. Gentle on
    #: purpose: a body that swings quickly smears every frame it is supposed to
    #: be searching with, and the sweep outruns the sensor that justifies it.
    search_body_rate: float = 0.30
    #: Seconds for one body sweep, there and back.
    search_body_period: float = 12.0
    #: Give up asking the body to turn after this long unsuccessfully
    #: searching. A robot that pirouettes indefinitely because someone left the
    #: room is worse company than one that waits where it was last useful.
    #: 0 means never give up.
    search_timeout: float = 25.0

    # -- searching ------------------------------------------------------- #
    #: Seconds for one there-and-back sweep while searching.
    search_period: float = 2.4
    #: The sweep starts this wide around the last known bearing and grows to
    #: ``max_angle`` over ``search_expand``.
    search_start: float = math.radians(15.0)
    search_expand: float = 4.0

    # -- surveying ------------------------------------------------------- #
    #: Look around when there is nothing to follow. Off leaves the head centred
    #: whenever it is not tracking, which keeps the depth map consistent frame
    #: to frame at the cost of never filling in the sides.
    survey: bool = True
    #: Hold a chosen survey bearing at least this long. Without it the head
    #: chases the score function and never dwells long enough to see anything.
    survey_dwell: float = 0.8
    #: A bearing is fully stale after this long unobserved, in seconds.
    staleness_horizon: float = 8.0
    #: How much the interest score can outweigh staleness. 0 surveys purely by
    #: what has not been looked at lately.
    interest_weight: float = 2.0
    #: Width of the world-bearing bins the survey memory is kept in, radians.
    memory_bin: float = math.radians(5.0)
    #: Never survey while the robot is turning faster than this, rad/s. Both
    #: motions at once smears the depth map and confuses the memory about what
    #: was actually observed.
    survey_max_spin: float = 0.8
    #: While moving faster than this, a survey may not turn the head so far
    #: that the direction of travel leaves the camera's cone -- "you may look
    #: around, but never stop watching where you are going". Stationary, or
    #: below this speed, the head may use its full travel.
    #:
    #: Measured: in the `rooms` scenario the rule took the robot from wandering
    #: 40.9 m with 2 blocked-motion cycles to covering the same ground in
    #: 15.0 m with none. It does *not* rescue the `maze` scenario, where an
    #: unconstrained survey scored 828 blocked cycles and a constrained one
    #: still scores 810 -- see VISION.md, because the reason turns out to be
    #: that surveying unsticks a robot that otherwise barely moves at all.
    survey_creep_speed: float = 0.05
    #: How far inside the cone edge the forward axis must stay, radians.
    survey_forward_margin: float = math.radians(8.0)
    #: How much better a candidate bearing must score than where the head is
    #: already pointing before it is worth moving. Without this the head
    #: re-points for a score of 0.13 immediately after looking at that very
    #: spot: every move costs a smeared frame and a settling time, so a survey
    #: that twitches sees less than one that sits still.
    survey_margin: float = 0.35


@dataclass(frozen=True)
class HeadCommand:
    #: Where to point, radians, positive left, already clamped to max_angle.
    angle: float = 0.0
    mode: HeadMode = HeadMode.CENTRED
    reason: str = "disabled"
    #: Bearing of whatever the head is aiming at, in the world frame. None in
    #: CENTRED. Useful for drawing, and for telling tracking from sweeping.
    world_bearing: Optional[float] = None
    #: The head has run out of travel and still cannot frame the target.
    at_limit: bool = False
    #: Angular velocity to turn the body at, rad/s, positive anticlockwise.
    #: Derived from ``body_hint`` and already limited to something the sensor
    #: can survive, so a caller can apply it directly after clamping to its own
    #: chassis limit. Zero whenever no help is wanted.
    body_rate: float = 0.0
    #: How far the *body* would have to rotate to re-centre the head, radians,
    #: positive anticlockwise. Zero unless the head needs help.
    #:
    #: Deliberately a hint and not an action. The whole point of a pan head is
    #: that the body drives the route instead of aiming the sensor, and a
    #: controller that quietly rotated the body whenever the head moved would
    #: hand back the smooth paths it was added to buy. Only when the head
    #: physically cannot reach is the body the answer, and even then it is the
    #: caller's decision -- mid-doorway is not the moment to spin.
    body_hint: float = 0.0


class HeadController:
    """Sightings and scans in, head angles out."""

    def __init__(self, config: Optional[HeadConfig] = None,
                 vision: Optional[VisionConfig] = None):
        self.cfg = config or HeadConfig()
        self.vision = vision or VisionConfig()

        bins = max(8, int(round(2.0 * math.pi / self.cfg.memory_bin)))
        self._bins = bins
        self._bin_width = 2.0 * math.pi / bins
        #: Last time each world bearing was inside the camera cone.
        self._seen_at = np.zeros(bins, dtype=np.float64)
        #: How much there was worth looking at, last time we looked. Bearings
        #: that have never been examined start at maximum: "I have no idea
        #: what is over there" is the strongest reason to look there is, and
        #: starting them at zero makes the head prefer re-examining somewhere
        #: it already knows is interesting over ever turning round.
        self._interest = np.ones(bins, dtype=np.float32)
        self._started = False

        self.mode = HeadMode.CENTRED
        self._angle = 0.0
        self._lost_since: Optional[float] = None
        self._last_bearing = 0.0
        #: Last known direction to the target in the **world** frame. Kept in
        #: world terms so that a hint derived from it shrinks to nothing as the
        #: body turns towards it. A robot-relative memory would keep asking for
        #: the same rotation it had just performed, and spin for ever.
        self._last_world: Optional[float] = None
        self._last_world_at: Optional[float] = None
        #: Sign of the target's drift across the field of view. When someone
        #: walks past, this is what says which way they went -- and searching
        #: the way they were travelling beats sweeping symmetrically.
        self._drift = 1.0
        self._survey_pick: Optional[float] = None
        self._survey_until = 0.0
        self._last_heading = 0.0
        self._last_heading_at: Optional[float] = None
        self._spin = 0.0
        self._speed = 0.0
        self.surveys = 0
        self.searches = 0

    # ------------------------------------------------------------------ #
    # Memory of where we have looked
    # ------------------------------------------------------------------ #

    def _bin_of(self, world_bearing: float) -> int:
        idx = int(math.floor((_wrap(world_bearing) + math.pi) / self._bin_width))
        return min(self._bins - 1, max(0, idx))

    def note_scan(
        self,
        scan: np.ndarray,
        robot_heading: float,
        head_yaw: float,
        now: Optional[float] = None,
    ) -> None:
        """Record what the last scan showed, indexed by *world* bearing.

        World bearing rather than robot bearing, deliberately. The robot turns;
        a memory kept relative to its nose would say a patch had just been
        looked at when in fact the robot had merely rotated to face somewhere
        it examined a minute ago.
        """
        now = time.monotonic() if now is None else now
        if not self._started:
            self._seen_at[:] = now
            self._started = True

        cfg = self.vision
        half = cfg.grid_half
        res = cfg.grid_resolution

        # Where the camera could see, this frame.
        half_fov = cfg.camera.hfov / 2.0
        lo = head_yaw - half_fov
        hi = head_yaw + half_fov

        # What was worth seeing: unknown ground beside known-clear ground, and
        # the boundaries of obstacles. Both are places where one more look
        # changes what the planner can conclude.
        unknown = scan == UNKNOWN
        free = scan == FREE
        lethal = scan == LETHAL
        touches_free = np.zeros_like(free)
        touches_free[1:, :] |= free[:-1, :]
        touches_free[:-1, :] |= free[1:, :]
        touches_free[:, 1:] |= free[:, :-1]
        touches_free[:, :-1] |= free[:, 1:]
        interesting = (unknown & touches_free) | (lethal & touches_free)

        ys, xs = np.nonzero(interesting)
        if len(xs):
            fx = (xs - half) * res
            fy = (ys - half) * res
            bearings = np.arctan2(fy, fx)
            within = (bearings >= lo) & (bearings <= hi)
            bearings = bearings[within]
        else:
            bearings = np.empty(0)

        # Mark the whole cone as freshly observed, then write what it held.
        steps = max(8, int(round((hi - lo) / self._bin_width)) + 1)
        for b in np.linspace(lo, hi, steps):
            self._seen_at[self._bin_of(robot_heading + b)] = now

        if len(bearings):
            counts = np.zeros(self._bins, dtype=np.float32)
            for b in bearings:
                counts[self._bin_of(robot_heading + b)] += 1.0
            # Saturating: a bin with 200 frontier cells is not twenty times
            # more worth a look than one with 10.
            fresh = np.tanh(counts / 40.0)
            covered = np.zeros(self._bins, dtype=bool)
            for b in np.linspace(lo, hi, steps):
                covered[self._bin_of(robot_heading + b)] = True
            self._interest[covered] = fresh[covered]

    def note_motion(self, linear_speed: float) -> None:
        """How fast the body is travelling, metres per second.

        The head's freedom to look away depends on it: a stopped robot can look
        wherever it likes, a moving one must keep its direction of travel in
        frame.
        """
        self._speed = abs(linear_speed)

    def note_heading(self, heading: float, now: Optional[float] = None) -> None:
        """Track how fast the body is turning, so survey can stand down."""
        now = time.monotonic() if now is None else now
        if self._last_heading_at is not None:
            dt = now - self._last_heading_at
            if dt > 1e-3:
                self._spin = abs(_wrap(heading - self._last_heading)) / dt
        self._last_heading = heading
        self._last_heading_at = now

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.mode = HeadMode.CENTRED
        self._lost_since = None
        self._survey_pick = None
        self._angle = 0.0
        self._last_world = None
        self._last_world_at = None
        self._drift = 1.0

    def update(
        self,
        observation: Optional[TargetObservation],
        state: FollowState,
        current: float = 0.0,
        robot_heading: float = 0.0,
        now: Optional[float] = None,
        predicted_bearing: Optional[float] = None,
    ) -> HeadCommand:
        """Decide where to point. ``current`` is the head's *measured* angle.

        ``predicted_bearing`` is where the follower's dead reckoning says the
        target should be, right-positive like a sighting. Supplied while
        coasting, it keeps the head moving with them instead of staring at the
        spot they were last actually seen -- and when it finally does run out,
        the search starts from where the prediction got to rather than from
        history.
        """
        now = time.monotonic() if now is None else now
        cfg = self.cfg

        if not cfg.enabled:
            self.mode = HeadMode.CENTRED
            self._angle = 0.0
            return HeadCommand(0.0, HeadMode.CENTRED, "head disabled")

        visible = observation is not None and observation.valid
        if visible:
            self._lost_since = None
            # Everything in this module is in head-yaw convention: positive to
            # the *left*, like the servo, the projector and the navigation
            # frame. The observation's bearing is positive to the right, so it
            # is negated once, here.
            #
            # Note also that the head angle which centres the target is that
            # bearing outright, not the current angle plus it: the bearing is
            # already expressed in the robot frame, and adding the head's own
            # angle to it again is what makes a tracking head run away to its
            # stop and sit there.
            target = -observation.bearing
            world = _wrap(robot_heading + target)
            if self._last_world is not None and self._last_world_at is not None:
                dt = now - self._last_world_at
                if dt > 1e-3:
                    rate = _wrap(world - self._last_world) / dt
                    if abs(rate) > 0.05:
                        self._drift = math.copysign(1.0, rate)
            self._last_bearing = target
            self._last_world = world
            self._last_world_at = now
            return self._track(target, current, robot_heading, now)

        # A prediction is not a sighting, but it is a great deal better than
        # the last one. Adopting it keeps the search honest about where the
        # person should be by now.
        if predicted_bearing is not None:
            self._last_bearing = -predicted_bearing
            self._last_world = _wrap(robot_heading - predicted_bearing)

        if state is FollowState.COASTING and predicted_bearing is not None:
            self._lost_since = None
            return self._coast(-predicted_bearing, current, robot_heading)

        # Search only for someone we have actually seen. SEARCHING before the
        # first sighting is not a loss, it is a cold start, and sweeping a
        # narrow arc around a bearing we never measured is worse than
        # surveying -- which sweeps anyway, and builds the map while it does.
        lost = state in (
            FollowState.COASTING, FollowState.PURSUING, FollowState.LOST
        ) or (
            state is FollowState.SEARCHING and self._lost_since is not None
        )
        if lost:
            if self._lost_since is None:
                self._lost_since = now
            return self._search(now, robot_heading)

        return self._survey(now, robot_heading)

    # ------------------------------------------------------------------ #

    def _clamp(self, angle: float) -> float:
        return max(-self.cfg.max_angle, min(self.cfg.max_angle, angle))

    # -- how eager to be, given what the body is already doing ---------- #
    #
    # Every one of these is a pair: a conservative value while the robot is
    # driving a path, and a far more aggressive one while it is standing still.
    # The reason is simply that the costs differ. A moving robot that rotates
    # for the camera's benefit fights its own planner and stops watching where
    # it is going. A stationary one gives up nothing at all.

    def _moving(self) -> bool:
        return self._speed > self.cfg.survey_creep_speed

    def _follow_ratio(self) -> float:
        cfg = self.cfg
        return cfg.body_follow_ratio if self._moving() else cfg.body_follow_ratio_still

    def _deadband(self) -> float:
        cfg = self.cfg
        return cfg.body_deadband if self._moving() else cfg.body_deadband_still

    def _turn_speed(self) -> float:
        cfg = self.cfg
        return cfg.body_turn_speed if self._moving() else cfg.body_turn_speed_still

    def _gate(self, hint: float) -> float:
        """Suppress body requests too small to be worth the wheel slip."""
        if not self.cfg.body_assist or abs(hint) < self._deadband():
            return 0.0
        return hint

    def _body_hint_for(self, bearing: float) -> float:
        """How far the body should come round for a target at ``bearing``.

        Whatever the bearing exceeds the comfort band by, so help ramps in
        smoothly as they move out rather than arriving all at once when the
        head hits its stop.
        """
        comfort = self._follow_ratio() * self.cfg.max_angle
        if abs(bearing) <= comfort:
            return 0.0
        return bearing - math.copysign(comfort, bearing)

    def _rate(self, hint: float) -> float:
        """Turn a rotation request into the angular velocity to ask for."""
        if not hint:
            return 0.0
        limit = self._turn_speed()
        return max(-limit, min(limit, self.cfg.body_gain * hint))

    def _survey_limit(self) -> float:
        """How far a survey may turn the head right now.

        Tracking and searching are exempt: those are about the target, and a
        robot that loses the person rather than glance away from its path has
        failed at its one job. Surveying is discretionary, so it yields.
        """
        cfg = self.cfg
        if self._speed <= cfg.survey_creep_speed:
            return cfg.max_angle
        forward_safe = self.vision.camera.hfov / 2.0 - cfg.survey_forward_margin
        return max(0.0, min(cfg.max_angle, forward_safe))

    def _track(
        self, bearing: float, current: float, robot_heading: float, now: float
    ) -> HeadCommand:
        """``bearing`` is the target's direction in head-yaw convention."""
        cfg = self.cfg
        self.mode = HeadMode.TRACK

        # Whether or not the head moves, the body may still want to come round:
        # these are separate questions. The head is about keeping the target in
        # frame; the body is about keeping it *reachable* as they keep going.
        # Computing this before the early exit matters -- otherwise a person
        # standing a little off to one side leaves the head settled, the hint
        # unset, and the robot permanently addressing them over its shoulder.
        hint = self._body_hint_for(bearing)

        comfortable = cfg.track_margin * self.vision.camera.hfov / 2.0
        error = bearing - current
        if abs(bearing) <= comfortable and abs(error) < cfg.track_deadband:
            # Already framed. Moving the head now buys nothing and costs a
            # smeared frame, which is the one thing the tracker cannot afford.
            self._angle = current
            return HeadCommand(
                self._clamp(current), HeadMode.TRACK,
                f"framed at {math.degrees(bearing):+.0f} deg",
                robot_heading + bearing,
                body_hint=self._gate(hint),
                body_rate=self._rate(self._gate(hint)),
            )

        wanted = self._clamp(current + cfg.track_gain * error)
        self._angle = wanted
        # "At its limit" means clamped *and* still short of the target. A head
        # sitting exactly on its stop because the target happens to be there
        # is doing its job, and asking the body to spin for it would be wrong.
        at_limit = (
            abs(wanted) >= cfg.max_angle - 1e-6
            and abs(bearing) > cfg.max_angle + cfg.track_deadband
        )
        reason = (
            f"target {math.degrees(bearing):+.0f} deg"
            + (" -- head at its limit, the body must turn" if at_limit else "")
        )

        return HeadCommand(
            wanted, HeadMode.TRACK, reason, robot_heading + bearing,
            at_limit=at_limit,
            body_hint=self._gate(hint),
            body_rate=self._rate(self._gate(hint)),
        )

    def _coast(
        self, bearing: float, current: float, robot_heading: float
    ) -> HeadCommand:
        """Aim at an extrapolated position. Same geometry as tracking.

        The body is asked for help on the same terms, and deliberately so: the
        case this exists for is someone walking past, where the prediction is
        at its most reliable precisely while their bearing is sweeping fastest
        and the head is running out of travel.
        """
        cfg = self.cfg
        self.mode = HeadMode.COAST
        wanted = self._clamp(current + cfg.track_gain * (bearing - current))
        self._angle = wanted

        at_limit = (
            abs(wanted) >= cfg.max_angle - 1e-6
            and abs(bearing) > cfg.max_angle + cfg.track_deadband
        )
        hint = self._body_hint_for(bearing)
        return HeadCommand(
            wanted, HeadMode.COAST,
            f"predicted at {math.degrees(bearing):+.0f} deg",
            robot_heading + bearing,
            at_limit=at_limit,
            body_hint=self._gate(hint),
            body_rate=self._rate(self._gate(hint)),
        )

    def _search(self, now: float, robot_heading: float) -> HeadCommand:
        cfg = self.cfg
        if self.mode is not HeadMode.SEARCH:
            self.searches += 1
        self.mode = HeadMode.SEARCH

        elapsed = now - (self._lost_since or now)
        # Start narrow around where they were, widen as hope fades.
        grow = min(1.0, elapsed / max(1e-6, cfg.search_expand))
        amplitude = cfg.search_start + grow * (cfg.max_angle - cfg.search_start)
        centre = self._clamp(self._last_bearing)

        # Triangle wave: constant angular speed, so the dwell is even across
        # the sweep instead of bunching at the ends the way a sine does.
        phase = (elapsed % cfg.search_period) / cfg.search_period
        offset = (4.0 * phase - 1.0) if phase < 0.5 else (3.0 - 4.0 * phase)
        wanted = self._clamp(centre + offset * amplitude)
        self._angle = wanted

        # Once the head has swept its whole travel without finding them, the
        # only place left to look is where the head cannot reach. Rotate the
        # search outward past the head's limit and let the body carry it,
        # starting in the direction the target was last drifting -- someone who
        # walked past the robot went one particular way, and sweeping the other
        # way first wastes the seconds that matter.
        hint = 0.0
        note = ""
        overrun = elapsed - cfg.search_body_after
        timed_out = cfg.search_timeout > 0.0 and elapsed > cfg.search_timeout
        if cfg.body_assist and overrun > 0.0 and self._last_world is not None:
            if timed_out:
                note = f", gave up asking the body after {cfg.search_timeout:.0f} s"
            else:
                # The body sweeps rather than aiming. Converging on a single
                # bearing and stopping there is what a robot does when it
                # thinks it knows where they are; having already failed to find
                # them, what it actually needs is to keep looking. The sweep
                # starts in the direction they were last drifting and widens,
                # so the head's own sweep rides on top of a slowly turning
                # body and the pair between them eventually cover the circle.
                amplitude = min(
                    math.pi - cfg.max_angle,
                    cfg.search_body_rate * overrun,
                )
                phase = 2.0 * math.pi * overrun / max(1e-6, cfg.search_body_period)
                aim = _wrap(
                    self._last_world + self._drift * amplitude * math.sin(phase)
                )
                hint = _wrap(aim - robot_heading)
                note = (f", body sweep +/-{math.degrees(amplitude):.0f} deg "
                        f"({'left' if self._drift > 0 else 'right'} first)")

        return HeadCommand(
            wanted, HeadMode.SEARCH,
            f"searching +/-{math.degrees(amplitude):.0f} deg around "
            f"{math.degrees(centre):+.0f} deg, {elapsed:.1f} s lost{note}",
            robot_heading + wanted,
            body_hint=self._gate(hint),
            body_rate=self._rate(self._gate(hint)),
        )

    def _survey(self, now: float, robot_heading: float) -> HeadCommand:
        cfg = self.cfg
        if not cfg.survey:
            self.mode = HeadMode.CENTRED
            self._angle = 0.0
            return HeadCommand(0.0, HeadMode.CENTRED, "survey off")

        if self._spin > cfg.survey_max_spin:
            # The body is already sweeping the sensor for us, and doing both at
            # once smears the frame and lies to the memory about what was seen.
            self.mode = HeadMode.CENTRED
            self._angle = 0.0
            return HeadCommand(
                0.0, HeadMode.CENTRED,
                f"body turning at {self._spin:.1f} rad/s, holding centre",
            )

        limit = self._survey_limit()
        if limit <= 1e-6:
            self.mode = HeadMode.CENTRED
            self._angle = 0.0
            return HeadCommand(
                0.0, HeadMode.CENTRED,
                "moving, and the cone is too narrow to look away safely",
            )

        if self._survey_pick is not None and now < self._survey_until:
            wanted = max(-limit, min(limit, _wrap(self._survey_pick - robot_heading)))
            self.mode = HeadMode.SURVEY
            self._angle = wanted
            return HeadCommand(
                wanted, HeadMode.SURVEY, "dwelling", self._survey_pick
            )

        pick, score = self._best_bearing(robot_heading, now, limit)
        if pick is None:
            self.mode = HeadMode.CENTRED
            self._angle = 0.0
            return HeadCommand(0.0, HeadMode.CENTRED, "nothing worth a look")

        here = self._score_at(robot_heading + self._angle, robot_heading, now)
        if score < here + cfg.survey_margin:
            # Hold -- but still inside the current limit. A head parked at 78
            # degrees while the robot was stopped must pull back the moment it
            # starts moving, or "hold" quietly becomes the way the safety rule
            # gets bypassed.
            held = max(-limit, min(limit, self._angle))
            self.mode = HeadMode.SURVEY
            self._angle = held
            return HeadCommand(
                held, HeadMode.SURVEY,
                f"holding {math.degrees(held):+.0f} deg "
                f"(best elsewhere {score:.2f} vs {here:.2f} here)",
                robot_heading + held,
            )

        self._survey_pick = pick
        self._survey_until = now + cfg.survey_dwell
        self.surveys += 1
        wanted = max(-limit, min(limit, _wrap(pick - robot_heading)))
        self.mode = HeadMode.SURVEY
        self._angle = wanted
        return HeadCommand(
            wanted, HeadMode.SURVEY,
            f"looking {math.degrees(wanted):+.0f} deg, score {score:.2f}",
            pick,
        )

    def _scores(self, robot_heading: float, now: float) -> np.ndarray:
        cfg = self.cfg
        staleness = np.clip(
            (now - self._seen_at) / max(1e-6, cfg.staleness_horizon), 0.0, 1.0
        )
        return staleness * (1.0 + cfg.interest_weight * self._interest)

    def _score_at(self, world_bearing: float, robot_heading: float, now: float) -> float:
        if not self._started:
            return 0.0
        return float(self._scores(robot_heading, now)[self._bin_of(world_bearing)])

    def _best_bearing(
        self, robot_heading: float, now: float, limit: Optional[float] = None
    ) -> Tuple[Optional[float], float]:
        """The world bearing most worth a look, within reach of the head."""
        cfg = self.cfg
        if not self._started:
            return None, 0.0

        limit = cfg.max_angle if limit is None else limit
        centres = -math.pi + (np.arange(self._bins) + 0.5) * self._bin_width
        reachable = np.abs(
            (centres - robot_heading + math.pi) % (2.0 * math.pi) - math.pi
        ) <= limit
        if not reachable.any():
            return None, 0.0

        score = self._scores(robot_heading, now)
        score[~reachable] = -1.0

        best = int(np.argmax(score))
        if score[best] <= 0.02:
            return None, 0.0
        return float(centres[best]), float(score[best])

    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        if not self.cfg.enabled:
            return "head: fixed (servo disabled)"
        return (
            f"head: +/-{math.degrees(self.cfg.max_angle):.0f} deg, "
            f"{'survey on' if self.cfg.survey else 'survey off'}, "
            f"mode {self.mode.value}, {self.surveys} surveys, "
            f"{self.searches} searches"
        )


__all__ = ["HeadCommand", "HeadConfig", "HeadController", "HeadMode"]
