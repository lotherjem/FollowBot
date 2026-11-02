from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .backend import GpioBackend
from .config import DriveConfig, Gains
from .encoder import EncoderPair
from .motor import MotorPair
from .odometry import DifferentialOdometry, Pose


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


class VelocityLoop:
    """Feedforward plus PI on one wheel, with back-calculated anti-windup."""

    def __init__(self, gains: Gains, config: DriveConfig):
        self.g = gains
        self.cfg = config
        self._integral = 0.0
        self._prev_error = 0.0
        self._output = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._output = 0.0

    def update(self, target: float, measured: float, dt: float) -> float:
        """Target and measured in rad/s at the wheel. Returns duty percent.

        The loop owns rate limiting and the 100% ceiling, because it is the
        only thing here that knows ``dt`` and the only thing that needs to
        anti-wind against them. The motor owns the deadband, which is an
        electrical fact about the motor rather than a limit on the controller.

        That split is not cosmetic. Feeding the deadband back as if it were
        saturation is a lock-up: when the loop asks for a duty of 1 and the
        deadband delivers 12, the anti-windup sees an 11-point "shortfall" of
        the wrong sign and unwinds the integrator away from the target. It
        settles at a stable, wrong answer -- wheels creeping backwards against
        a large forward command -- and no amount of error makes it recover.
        """
        if dt <= 0.0:
            return self._output

        error = target - measured

        feedforward = 0.0
        if target != 0.0:
            feedforward = self.g.kv * target + math.copysign(self.g.ks, target)

        proportional = self.g.kp * error
        derivative = self.g.kd * (error - self._prev_error) / dt if self.g.kd else 0.0
        self._prev_error = error

        requested = feedforward + proportional + derivative + self._integral

        limited = _clamp(requested, -self.cfg.max_duty, self.cfg.max_duty)
        step = self.cfg.max_duty_step
        if step > 0:
            delta = limited - self._output
            if abs(delta) > step:
                limited = self._output + math.copysign(step, delta)

        # Integrate, then unwind by however much the actuator could not take.
        self._integral += self.g.ki * error * dt
        self._integral += self.g.kt * (limited - requested) * dt
        self._integral = _clamp(
            self._integral, -self.g.integral_limit, self.g.integral_limit
        )

        self._output = limited
        return limited


#: Seconds over which the worst edge delivery delay is reported.
EDGE_LAG_WINDOW = 0.5


class WheelGuard:
    """Notices a speed loop that is fighting its own wheel.

    Two failures look identical from inside the loop and both end with the
    motor at full power: the encoder reports the wheel going the opposite way
    to the drive (reversed, or wired to the other wheel), and the encoder
    reports nothing at all (unplugged, or the wheel is jammed). The loop's
    only answer to either is more duty. This watches for the duty being high
    while the wheel does the wrong thing, for long enough that no honest
    transient explains it, and names the problem.
    """

    def __init__(self, side: str, cfg: DriveConfig):
        self.side = side
        self.cfg = cfg
        self._against = 0.0
        self._stalled = 0.0

    def reset(self) -> None:
        self._against = self._stalled = 0.0

    def check(self, duty: float, velocity: float, dt: float) -> Optional[str]:
        cfg = self.cfg
        moving = abs(velocity) >= cfg.stopped_speed
        against = (cfg.runaway_duty > 0 and abs(duty) >= cfg.runaway_duty
                   and moving and duty * velocity < 0)
        stalled = (cfg.stall_duty > 0 and abs(duty) >= cfg.stall_duty
                   and not moving)
        self._against = self._against + dt if against else 0.0
        self._stalled = self._stalled + dt if stalled else 0.0
        if self._against >= cfg.runaway_time:
            return (
                f"{self.side} wheel runaway: driven at {duty:+.0f}% duty while its "
                f"encoder read {velocity:+.1f} rad/s the other way for "
                f"{self._against:.1f} s. The encoder is reversed or wired to the "
                "other wheel, so the speed loop was pushing harder the faster it "
                "went. Run the drive console's m test to see which.")
        if self._stalled >= cfg.stall_time:
            return (
                f"{self.side} wheel stalled: {duty:+.0f}% duty for "
                f"{self._stalled:.1f} s and its encoder saw no movement. The wheel "
                "is jammed, or its encoder is unplugged.")
        return None


@dataclass(frozen=True)
class DriveState:
    """An immutable picture of the base. Cheap to take, safe to hold."""

    stamp: float = 0.0
    seq: int = 0

    pose: Pose = field(default_factory=Pose)
    left_counts: int = 0
    right_counts: int = 0

    #: Measured wheel speeds, rad/s, filtered.
    left_speed: float = 0.0
    right_speed: float = 0.0
    #: Commanded wheel speeds, rad/s.
    left_target: float = 0.0
    right_target: float = 0.0
    #: Signed duty actually written, percent.
    left_duty: float = 0.0
    right_duty: float = 0.0

    #: Body velocity implied by the measured wheel speeds.
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0

    left_errors: int = 0
    right_errors: int = 0
    #: Times the decoded phase disagreed with the pins while stopped -- the
    #: only reliable sign that edges are being lost.
    left_desyncs: int = 0
    right_desyncs: int = 0
    watchdog_tripped: bool = False
    #: Worst delay between an encoder edge happening and being decoded, over
    #: the last ``EDGE_LAG_WINDOW`` seconds, ms. Tens of ms is normal; hundreds
    #: means the Pi is too busy to hand the edges over.
    left_edge_lag_ms: float = 0.0
    right_edge_lag_ms: float = 0.0
    #: Why the controller stopped itself, when it did (see ``WheelGuard``).
    #: Latched with ``emergency_stopped`` until ``clear_emergency_stop``.
    fault: str = ""
    emergency_stopped: bool = False
    motor_mode: str = "coast"

    #: Measured loop period, and how many ticks ran late. A rising overrun
    #: count is the first sign the Pi is too loaded to close the loop on time.
    loop_period: float = 0.0
    loop_overruns: int = 0

    @property
    def is_moving(self) -> bool:
        return abs(self.left_speed) > 1e-3 or abs(self.right_speed) > 1e-3

    def summary(self) -> str:
        return (
            f"pose=({self.pose.x:+.3f}, {self.pose.y:+.3f}, "
            f"{math.degrees(self.pose.theta):+6.1f}deg)  "
            f"counts=({self.left_counts:+7d}, {self.right_counts:+7d})  "
            f"wheel rad/s=({self.left_speed:+6.2f}, {self.right_speed:+6.2f}) "
            f"target=({self.left_target:+6.2f}, {self.right_target:+6.2f})  "
            f"duty=({self.left_duty:+6.1f}, {self.right_duty:+6.1f})  "
            f"v={self.linear_velocity:+.3f}m/s w={self.angular_velocity:+.2f}rad/s  "
            f"err=({self.left_errors},{self.right_errors}) "
            f"desync=({self.left_desyncs},{self.right_desyncs})  "
            f"{self.motor_mode}"
            f"{'  WATCHDOG' if self.watchdog_tripped else ''}"
            f"{'  ESTOP' if self.emergency_stopped else ''}"
            f"{'  FAULT: ' + self.fault if self.fault else ''}"
        )


class DriveController:
    """Fixed-rate closed-loop control of the drive base."""

    def __init__(
        self,
        config: Optional[DriveConfig] = None,
        backend: Optional[GpioBackend] = None,
    ):
        self.cfg = config or DriveConfig()
        self.cfg.validate()

        if backend is None:
            from .backend import LgpioBackend

            backend = LgpioBackend(self.cfg.chip)
        self.backend = backend

        self.encoders = EncoderPair(self.cfg)
        self.motors = MotorPair(self.cfg, backend)
        self.odometry = DifferentialOdometry(self.cfg)

        self._left_loop = VelocityLoop(self.cfg.left_gains, self.cfg)
        self._right_loop = VelocityLoop(self.cfg.right_gains, self.cfg)
        self._left_guard = WheelGuard("left", self.cfg)
        self._right_guard = WheelGuard("right", self.cfg)
        self._fault = ""
        self._lags: List[Tuple[float, float, float]] = []      # (time, left, right)

        self._lock = threading.Lock()
        self._state = DriveState()
        self._seq = 0

        self._left_target = 0.0
        self._right_target = 0.0
        self._open_loop: Optional[Tuple[float, float]] = None
        self._hold = False
        self._sync_on = False
        #: Wanted minus actual (right - left) wheel travel, radians.
        self._sync_error = 0.0
        self._sync_positions: Optional[Tuple[int, int]] = None
        self._last_command = 0.0
        self._watchdog_enabled = True
        self._emergency = False

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._abort = threading.Event()
        self._overruns = 0
        self.last_error: Optional[str] = None

        self._started = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> "DriveController":
        if self._started:
            return self
        self.motors.setup()
        self.encoders.attach(self.backend)
        self.odometry.reset()

        self._last_command = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="carrybot-drive", daemon=True
        )
        self._thread.start()
        self._started = True
        return self

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.motors.stop()
            self.motors.disable()
        finally:
            self.encoders.detach()
            self.backend.close()
        self._started = False

    def __enter__(self) -> "DriveController":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def set_velocity(self, linear: float, angular: float, _sync: bool = True) -> None:
        """Body velocity in m/s and rad/s. Feeds the watchdog.

        The two wheels are kept in step with each other (``sync_gain``), so
        the robot turns the way it was asked from the start. The blocking
        moves below pass ``_sync=False``: they close the same loop themselves,
        on distance since the move began.
        """
        linear = _clamp(linear, -self.cfg.max_linear_speed, self.cfg.max_linear_speed)
        angular = _clamp(
            angular, -self.cfg.max_angular_speed, self.cfg.max_angular_speed
        )
        left, right = self.cfg.wheel_speeds_for(linear, angular)
        self.set_wheel_velocities(left, right, _sync=_sync)

    def set_wheel_velocities(self, left: float, right: float, _sync: bool = False) -> None:
        """Wheel speeds in rad/s. Feeds the watchdog."""
        with self._lock:
            if self._emergency:
                return
            self._left_target = left
            self._right_target = right
            self._open_loop = None
            self._hold = False
            self._sync_on = _sync
            self._last_command = time.monotonic()

    def set_open_loop_duty(self, left: float, right: float) -> None:
        """Write duty straight to the motors, bypassing the velocity loop.

        For bringing a new robot up: proving the wiring, finding the deadband,
        and measuring the duty-to-speed curve the feedforward gains come from.
        The watchdog still applies, so this cannot leave the robot running.
        """
        with self._lock:
            if self._emergency:
                return
            self._open_loop = (left, right)
            self._left_target = 0.0
            self._right_target = 0.0
            self._last_command = time.monotonic()

    def clear_open_loop(self) -> None:
        with self._lock:
            self._open_loop = None
        self.stop()

    def stop(self) -> None:
        """Command zero. The loop brakes once the wheels are actually stopped."""
        with self._lock:
            self._open_loop = None
        self.set_wheel_velocities(0.0, 0.0)

    def hold(self) -> None:
        """Stop now: short the motors rather than asking the loop to wind down.

        Commanding zero velocity leaves the loop to bring the wheels down at
        whatever rate its gains allow, which with feedforward uncalibrated can
        take the better part of a second -- and the robot travels during it.
        Shorting the terminals stops the wheels in a fraction of that, without
        the reverse-drive overshoot the original controller's braking loop
        could produce.

        Any velocity command releases the hold.
        """
        with self._lock:
            if self._emergency:
                return
            self._hold = True
            self._open_loop = None
            self._left_target = 0.0
            self._right_target = 0.0
            self._last_command = time.monotonic()

    def emergency_stop(self) -> None:
        """Latch stopped: coast the motors and refuse commands until cleared."""
        with self._lock:
            self._emergency = True
            self._left_target = 0.0
            self._right_target = 0.0
        self._abort.set()
        self.motors.coast()
        self.motors.disable()

    def _synchronise(self, on: bool, left: float, right: float,
                     left_pos: int, right_pos: int, dt: float) -> Tuple[float, float]:
        """Keep the wheels in step: see ``DriveConfig.sync_gain``.

        Tracks how far apart the two wheels *should* have turned -- the
        integral of the difference between their targets -- against how far
        apart they *have*, from the counts, and splits the correction between
        them. Off, stopped, or between commands, it starts again from zero.
        """
        cfg = self.cfg
        moving = abs(left) > 1e-6 or abs(right) > 1e-6
        if not on or not moving or cfg.sync_gain <= 0:
            self._sync_error = 0.0
            self._sync_positions = None
            return left, right
        if self._sync_positions is None:
            self._sync_positions = (left_pos, right_pos)
            return left, right
        last_left, last_right = self._sync_positions
        self._sync_positions = (left_pos, right_pos)
        rpc = cfg.radians_per_count
        done = ((right_pos - last_right) - (left_pos - last_left)) * rpc
        wanted = (right - left) * dt
        error = self._sync_error + wanted - done
        limit = cfg.sync_limit
        self._sync_error = max(-limit, min(limit, error))
        correction = 0.5 * cfg.sync_gain * self._sync_error
        return left - correction, right + correction

    def _floor(self, duty: float, velocity: float, gains: "Gains") -> float:
        """The deadband edge for this wheel this tick: see ``running_duty``.

        Only with feedforward calibrated (``kv`` > 0). Without it the PI terms
        do all the work, and given room below ``min_duty`` they hunt -- in
        simulation the wheel speed swung three times as much as with the
        deadband left where it was.
        """
        cfg = self.cfg
        if (gains.kv > 0.0 and duty * velocity > 0.0
                and abs(velocity) >= 2.0 * cfg.stopped_speed):
            return cfg.running_duty if cfg.running_duty > 0 else 0.5 * cfg.min_duty
        return cfg.min_duty

    def _loop_duty(self, loop: "VelocityLoop", guard: "WheelGuard",
                   target: float, velocity: float, dt: float) -> float:
        """One wheel's duty for this tick. Zero means brake (or coast).

        Two cases are taken away from the speed loop, because it handles both
        badly at low speed:

        * **Zero commanded.** Asking the loop to *regulate* a turning wheel to
          zero leaves it whatever integral it built up getting the wheel
          moving, and every small output it produces is then pushed out to
          ``min_duty`` so the motor does not hum. The wheel is kept turning at
          the deadband, or kicked back and forth across it, for seconds after
          the move ended. That was the end of every turn. Zero now means stop:
          the motor is shorted (``brake_on_stop``), which the TB6612 does
          better than any loop.
        * **A small push the wrong way.** A loop slowing a wheel that is a
          little too fast asks for a few percent of *reverse*; the deadband
          would turn that into a ``min_duty`` kick backwards, and the next tick
          a kick forwards. Braking for that tick does what was meant.
        """
        if abs(target) < self.cfg.stopped_speed:
            # Zero, or so close to it that the deadband push-out would turn
            # it into min_duty: a hum that moves nothing.
            loop.reset()
            guard.reset()
            return 0.0
        duty = loop.update(target, velocity, dt)
        if duty * target < 0 and abs(duty) < self.cfg.min_duty:
            loop._output = 0.0           # so the next tick's slew starts from here
            return 0.0
        return duty

    def _latch_fault(self, reason: str) -> None:
        """The controller's own emergency stop, with the reason kept."""
        with self._lock:
            self._emergency = True
            self._fault = reason
            self._left_target = 0.0
            self._right_target = 0.0
        self.last_error = reason
        self._abort.set()
        self.motors.coast()
        self.motors.disable()

    def clear_emergency_stop(self) -> None:
        with self._lock:
            self._emergency = False
            self._fault = ""
            self._last_command = time.monotonic()
        self._abort.clear()
        self.motors.enable()

    def abort(self) -> None:
        """Interrupt any blocking move. Does not latch."""
        self._abort.set()
        self.stop()

    def set_watchdog(self, enabled: bool) -> None:
        """Turn the command timeout off. For bench work only.

        With it off, the last command stands forever -- including if whatever
        was sending commands has died.
        """
        with self._lock:
            self._watchdog_enabled = enabled
            self._last_command = time.monotonic()

    def reset_odometry(self, pose: Optional[Pose] = None) -> None:
        self.odometry.reset(pose)
        self.encoders.reset()

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def state(self) -> DriveState:
        """The latest snapshot. Never blocks for long, never returns None."""
        with self._lock:
            return self._state

    def wait_for_state(self, timeout: float = 1.0) -> DriveState:
        deadline = time.monotonic() + timeout
        start_seq = self.state().seq
        while time.monotonic() < deadline:
            state = self.state()
            if state.seq > start_seq:
                return state
            time.sleep(0.002)
        return self.state()

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        period = self.cfg.control_period
        next_tick = time.monotonic()
        last_time = next_tick

        while not self._stop_event.is_set():
            next_tick += period
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            try:
                self._tick(now, dt)
            except Exception:
                self.last_error = traceback.format_exc()
                # A controller that dies leaves the motors powered. Stop first,
                # then keep the loop alive so the watchdog and e-stop still work.
                try:
                    self.motors.stop()
                except Exception:
                    pass

            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._overruns += 1
                next_tick = time.monotonic()

    def _tick(self, now: float, dt: float) -> None:
        left_sample, right_sample = self.encoders.sample(now)
        pose = self.odometry.update(left_sample.position, right_sample.position)

        with self._lock:
            emergency = self._emergency
            left_target = self._left_target
            right_target = self._right_target
            open_loop = self._open_loop
            hold = self._hold
            sync_on = self._sync_on
            since_command = now - self._last_command
            watchdog = self._watchdog_enabled
            fault = self._fault

        tripped = watchdog and since_command > self.cfg.command_timeout

        if emergency or hold or open_loop is not None:
            # The guards watch the speed loop; outside it they start over.
            self._left_guard.reset()
            self._right_guard.reset()

        if emergency:
            self.motors.coast()
            self._left_loop.reset()
            self._right_loop.reset()
            left_target = right_target = 0.0
            left_duty = right_duty = 0.0
        elif hold:
            self.motors.brake()
            self._left_loop.reset()
            self._right_loop.reset()
            left_target = right_target = 0.0
            left_duty = right_duty = 0.0
        elif open_loop is not None and not tripped:
            self._left_loop.reset()
            self._right_loop.reset()
            # The idle channel coasts rather than braking. Braking holds its
            # PWM pin at 100%, so with PWMA and PWMB cross-wired the channel
            # under test borrowed full power from it and the swap passed the
            # wiring test unseen. Coasting holds it at 0%: a swap then turns
            # nothing at all, which the test reports.
            left_duty, right_duty = self.motors.set_duty(
                open_loop[0], open_loop[1], deadband=False, coast=(True, True)
            )
        else:
            if tripped:
                left_target = right_target = 0.0
            left_target, right_target = self._synchronise(
                sync_on, left_target, right_target,
                left_sample.position, right_sample.position, dt)

            stationary = (
                abs(left_target) < 1e-6
                and abs(right_target) < 1e-6
                and abs(left_sample.velocity) < self.cfg.stopped_speed
                and abs(right_sample.velocity) < self.cfg.stopped_speed
            )
            if stationary:
                # Nothing commanded and nothing turning: hold it shorted rather
                # than letting the loop hunt around zero. A stopped wheel is
                # also the only time the encoder phase can be checked against
                # the pins without racing the rotation, so do it here.
                self.encoders.check_alignment(self.backend)
                self.motors.stop()
                self._left_loop.reset()
                self._right_loop.reset()
                self._left_guard.reset()
                self._right_guard.reset()
                left_duty = right_duty = 0.0
            else:
                left_duty = self._loop_duty(
                    self._left_loop, self._left_guard, left_target,
                    left_sample.velocity, dt)
                right_duty = self._loop_duty(
                    self._right_loop, self._right_guard, right_target,
                    right_sample.velocity, dt)
                # The loop has already rate-limited; the motor only applies the
                # deadband and writes the pins. A wheel already turning the
                # way it is driven gets the lower running deadband, and a
                # wheel still meant to move coasts through a zero rather
                # than braking: together they are the difference between a
                # slow, steady speed and alternate kicks and stops.
                left_duty, right_duty = self.motors.set_duty(
                    left_duty, right_duty, max_step=100.0,
                    floors=(self._floor(left_duty, left_sample.velocity,
                                        self.cfg.left_gains),
                            self._floor(right_duty, right_sample.velocity,
                                        self.cfg.right_gains)),
                    coast=(left_target != 0.0, right_target != 0.0),
                )
                problem = (
                    self._left_guard.check(left_duty, left_sample.velocity, dt)
                    or self._right_guard.check(right_duty, right_sample.velocity, dt)
                )
                if problem:
                    self._latch_fault(problem)
                    emergency, fault = True, problem
                    left_duty = right_duty = 0.0

        linear, angular = self.cfg.body_velocity_for(
            left_sample.velocity, right_sample.velocity
        )
        self._lags.append((now, self.encoders.left.take_lag_ms(),
                           self.encoders.right.take_lag_ms()))
        while self._lags and now - self._lags[0][0] > EDGE_LAG_WINDOW:
            self._lags.pop(0)
        left_lag = max(entry[1] for entry in self._lags)
        right_lag = max(entry[2] for entry in self._lags)

        with self._lock:
            self._seq += 1
            self._state = DriveState(
                stamp=now,
                seq=self._seq,
                pose=pose,
                left_counts=left_sample.position,
                right_counts=right_sample.position,
                left_speed=left_sample.velocity,
                right_speed=right_sample.velocity,
                left_target=left_target,
                right_target=right_target,
                left_duty=left_duty,
                right_duty=right_duty,
                linear_velocity=linear,
                angular_velocity=angular,
                left_errors=left_sample.errors,
                right_errors=right_sample.errors,
                left_desyncs=self.encoders.left.desyncs,
                right_desyncs=self.encoders.right.desyncs,
                watchdog_tripped=tripped,
                emergency_stopped=emergency,
                fault=fault,
                left_edge_lag_ms=left_lag,
                right_edge_lag_ms=right_lag,
                motor_mode=self.motors.left.mode.value,
                loop_period=dt,
                loop_overruns=self._overruns,
            )

    # ------------------------------------------------------------------ #
    # Blocking manoeuvres
    #
    # These run on the *calling* thread, feeding set_velocity and watching the
    # published state. The control loop keeps running underneath, so the
    # watchdog and emergency stop stay live throughout -- which would not be
    # true if a move blocked inside the control thread the way the original
    # controller's braking loop did.
    # ------------------------------------------------------------------ #

    def move_distance(
        self,
        metres: float,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Drive straight. Returns True if it arrived, False if aborted."""
        speed = abs(speed or self.cfg.default_linear_speed)
        start = self.state()
        base_left, base_right = start.left_counts, start.right_counts

        def wheel_distances() -> Tuple[float, float]:
            now = self.state()
            return (
                (now.left_counts - base_left) * self.cfg.left_metres_per_count,
                (now.right_counts - base_right) * self.cfg.right_metres_per_count,
            )

        def travelled() -> float:
            d_left, d_right = wheel_distances()
            return 0.5 * (d_left + d_right)

        def command(v: float) -> None:
            # Steer on the difference in distance the two wheels have actually
            # covered. Two velocity loops tracking the same target still end up
            # a few centimetres apart, because neither of them is being asked
            # about position -- and that difference is the heading error.
            d_left, d_right = wheel_distances()
            correction = -self.cfg.straightness_gain * (d_right - d_left)
            self.set_velocity(v, correction, _sync=False)

        return self._profile(
            total=metres,
            measure=travelled,
            max_speed=speed,
            accel=self.cfg.linear_accel,
            tolerance=self.cfg.position_tolerance_m,
            command=command,
            timeout=timeout,
            actual_speed=lambda: abs(self.state().linear_velocity),
        )

    def turn_in_place(
        self,
        radians: float,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Pivot. Positive is anticlockwise (left)."""
        speed = abs(speed or self.cfg.default_angular_speed)
        start = self.state()
        base_left, base_right = start.left_counts, start.right_counts

        def turned() -> float:
            now = self.state()
            d_left = (now.left_counts - base_left) * self.cfg.left_metres_per_count
            d_right = (now.right_counts - base_right) * self.cfg.right_metres_per_count
            # From the wheels rather than pose.theta: no wrapping to worry
            # about, so a turn past 180 degrees behaves like any other.
            return (d_right - d_left) / self.cfg.track_width_m

        return self._profile(
            total=radians,
            measure=turned,
            max_speed=speed,
            accel=self.cfg.angular_accel,
            tolerance=self.cfg.angle_tolerance_rad,
            command=lambda w: self.set_velocity(0.0, w, _sync=False),
            timeout=timeout,
            actual_speed=lambda: abs(self.state().angular_velocity),
        )

    def arc_about_wheel(
        self,
        side: str,
        wheel_metres: float,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Drive one wheel while the other holds still.

        The robot pivots about the stationary wheel, so a given wheel distance
        turns the body by ``distance / track_width`` radians -- half the rate
        of a spin in place, over twice the ground.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        speed = abs(speed or self.cfg.default_linear_speed)
        start = self.state()
        base = start.left_counts if side == "left" else start.right_counts
        sign = -1.0 if side == "left" else 1.0

        def travelled() -> float:
            now = self.state()
            counts = (now.left_counts if side == "left" else now.right_counts) - base
            return counts * self.cfg.metres_per_count

        def command(wheel_speed: float) -> None:
            # Half the wheel speed at the body centre, and the full difference
            # across the track as rotation.
            self.set_velocity(
                0.5 * wheel_speed, sign * wheel_speed / self.cfg.track_width_m,
                _sync=False,
            )

        def wheel_speed() -> float:
            state = self.state()
            speed_rad = state.left_speed if side == "left" else state.right_speed
            return abs(speed_rad) * self.cfg.wheel_radius_m

        return self._profile(
            total=wheel_metres,
            measure=travelled,
            max_speed=speed,
            accel=self.cfg.linear_accel,
            tolerance=self.cfg.position_tolerance_m,
            command=command,
            timeout=timeout,
            actual_speed=wheel_speed,
        )

    def _profile(
        self,
        total: float,
        measure: Callable[[], float],
        max_speed: float,
        accel: float,
        tolerance: float,
        command: Callable[[float], None],
        timeout: Optional[float],
        actual_speed: Optional[Callable[[], float]] = None,
    ) -> bool:
        """Trapezoidal profile driven by remaining distance.

        The deceleration limit falls out of ``v = sqrt(2 * a * remaining)``,
        which is just "how fast may I be going and still stop in the distance
        I have left". Because it is recomputed from the *measured* remaining
        distance every tick, it self-corrects: a wheel that slips, a loop that
        runs late or a motor that is slow to respond all just mean the profile
        holds speed a little longer. That is the whole reason no reverse-brake
        hack is needed to stop on target.
        """
        if abs(total) < tolerance:
            return True

        self._abort.clear()
        direction = math.copysign(1.0, total)
        target = abs(total)
        period = self.cfg.control_period
        speed = 0.0
        deadline = None if timeout is None else time.monotonic() + timeout
        settle_until: Optional[float] = None

        try:
            while not self._abort.is_set():
                if deadline is not None and time.monotonic() > deadline:
                    return False
                if self.state().emergency_stopped:
                    return False

                remaining = target - abs(measure())

                # Brake a stopping-distance early. The wheels lag the command,
                # so a move that starts braking exactly on target stops past it.
                lead = 0.0
                if actual_speed is not None and self.cfg.stop_lead_time > 0:
                    lead = actual_speed() * self.cfg.stop_lead_time

                if remaining <= tolerance + lead:
                    # Short the motors rather than asking the velocity loop to
                    # wind down. The distance covered between "arrived" and
                    # "stopped" is pure overshoot, and this is the shortest
                    # route through it that the hardware offers.
                    self.hold()
                    if settle_until is None:
                        settle_until = time.monotonic() + 0.75
                    state = self.state()
                    stopped = (
                        abs(state.left_speed) < self.cfg.stopped_speed
                        and abs(state.right_speed) < self.cfg.stopped_speed
                    )
                    if stopped or time.monotonic() > settle_until:
                        return True
                    time.sleep(period)
                    continue

                settle_until = None
                stopping_speed = math.sqrt(2.0 * accel * remaining)
                speed = min(max_speed, stopping_speed, speed + accel * period)
                command(direction * speed)
                time.sleep(period)

            return False
        finally:
            # Brake. Not command(0.0): for a straight move that still carries
            # the straightness correction, a small pivot request the deadband
            # turns into a min_duty hum until the watchdog ends it.
            self.hold()


__all__ = ["DriveController", "DriveState", "VelocityLoop"]
