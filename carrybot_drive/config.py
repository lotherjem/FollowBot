from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MotorChannel:
    """One side of a TB6612FNG.

    ``invert`` flips the direction bits. Use it when a motor is wired backwards
    rather than negating speeds at the call site -- otherwise the sign
    convention leaks into every caller and eventually disagrees with the
    encoder's.
    """

    pwm: int
    in1: int
    in2: int
    invert: bool = False


@dataclass
class EncoderChannel:
    """One quadrature encoder.

    ``invert`` flips the counting direction. It must be set so that counts rise
    when the wheel drives the robot *forward*; if it disagrees with the motor's
    ``invert``, the velocity loop has positive feedback and the wheel runs away.
    :meth:`DriveConfig.validate` cannot catch that -- the ``p`` command in the
    CLI will, in about two seconds.
    """

    a: int
    b: int
    invert: bool = False


@dataclass
class Gains:
    """Per-wheel velocity loop gains, in duty percent per rad/s.

    Feedforward does the work and the PI terms trim it. That ordering matters:
    a feedforward-dominant loop degrades into roughly-right open-loop control
    when the gains are untuned, where a PI-dominant one oscillates. Get ``kv``
    and ``ks`` from the ``f`` command in the CLI before touching ``kp``.
    """

    #: Duty percent per rad/s of wheel speed. The slope of the motor's
    #: duty-to-speed curve, measured with the ``f`` calibration command.
    kv: float = 0.0
    #: Static duty needed to break stiction, added with the sign of the target.
    ks: float = 0.0
    kp: float = 0.8
    ki: float = 8.0
    kd: float = 0.0
    #: Ceiling on the integral term's contribution, in duty percent. Generous
    #: on purpose: with feedforward uncalibrated the integrator is the only
    #: thing producing output, and a tight clamp there caps the top speed
    #: instead of preventing windup. Windup is handled by ``kt`` below.
    integral_limit: float = 100.0
    #: Back-calculation anti-windup gain. Whenever the duty the loop asked for
    #: is not the duty that reached the motor -- because it hit 100%, because
    #: the slew limiter held it back, or because the deadband pushed it up --
    #: the integrator is unwound by this much per second per point of
    #: shortfall. Without it the integrator charges all the way up during the
    #: acceleration ramp, when the actuator was never going to deliver what was
    #: asked, and the wheel overshoots badly when it finally arrives.
    kt: float = 12.0


@dataclass
class DriveConfig:
    # -- Wiring (BCM numbering) ------------------------------------------ #
    #: Left side drives AIN1/AIN2/PWMA, right side BIN1/BIN2/PWMB. This
    #: matches the mapping that was verified working on the bench, even though
    #: the silkscreen labels suggest otherwise.
    left_motor: MotorChannel = field(
        default_factory=lambda: MotorChannel(pwm=12, in1=20, in2=21)
    )
    right_motor: MotorChannel = field(
        default_factory=lambda: MotorChannel(pwm=13, in1=24, in2=23)
    )
    left_encoder: EncoderChannel = field(
        default_factory=lambda: EncoderChannel(a=6, b=19)
    )
    right_encoder: EncoderChannel = field(
        default_factory=lambda: EncoderChannel(a=17, b=27)
    )

    #: TB6612 standby. ``None`` means it is strapped to VCC and the driver can
    #: never be forced into a known-safe state from software -- see the note in
    #: DRIVE.md. Wire it to a GPIO and set this if you can.
    standby: Optional[int] = None

    chip: int = 0

    # -- Geometry -- MEASURE THESE ---------------------------------------- #
    #: Encoder cycles per revolution of the *motor* shaft, before quadrature.
    encoder_cpr: int = 12
    #: 4 for x4 decoding (every edge of both channels), which is what the
    #: decoder in this package does.
    quadrature_multiplier: int = 4
    #: Motor shaft revolutions per wheel revolution. PLACEHOLDER -- the value
    #: below is a common yellow-gearmotor ratio, not a measurement of yours.
    gear_ratio: float = 48.0
    #: PLACEHOLDER. Measure across the tyre under load, not the rim.
    wheel_diameter_m: float = 0.065
    #: True distance divided by the distance odometry reported, from a measured
    #: straight run -- the encoder-to-true-distance calibration. 1.0 trusts the
    #: numbers above exactly. It scales the effective rolling diameter, so the
    #: odometry *and* the speed loop both use the corrected figure. The ``c``
    #: command in ``python -m carrybot_drive.cli`` measures it: drive, measure
    #: with a tape, paste the number it prints.
    distance_scale: float = 1.0
    #: The right wheel's rolling diameter divided by the left's. Two wheels
    #: are never quite the same size, and a 1% difference makes a robot that
    #: drives dead straight report a turn of several degrees per metre --
    #: so forward-and-back runs down one line show on the map as a fan of
    #: different lines. Above 1 means the right wheel covers more ground per
    #: turn. The average of the two stays ``wheel_diameter_m``, so
    #: ``distance_scale`` is unaffected.
    wheel_balance: float = 1.0
    #: Distance between the wheel contact patches. PLACEHOLDER; it is also the
    #: hardest number to measure with a ruler, so calibrate it by spinning.
    track_width_m: float = 0.150

    # -- PWM --------------------------------------------------------------- #
    #: lgpio's tx_pwm is timer-driven, so this is a compromise: high enough to
    #: keep torque ripple down, low enough that software PWM can hit it without
    #: excessive jitter. GPIO12 and GPIO13 are the Pi's hardware PWM pins, so
    #: there is a real upgrade available here if the whine or jitter bothers
    #: you -- see DRIVE.md.
    pwm_frequency: int = 1000
    #: Duty below which the motor does not turn at all. Commands inside the
    #: deadband are pushed out to it so a small requested speed produces
    #: motion rather than a stall and a hum.
    min_duty: float = 12.0
    #: The same, for a wheel that is *already turning* the way it is driven.
    #: Keeping a geared motor going takes less than starting it, and a turning
    #: wheel may be driven down to this: without it, any speed slower than
    #: what ``min_duty`` produces can only be had by alternating ``min_duty``
    #: with stopping, which is the stutter. A wheel that stalls at it falls
    #: back to ``min_duty`` on the next tick. 0 means half of ``min_duty``.
    #: Used only once feedforward is calibrated (``kv`` > 0, the ``f``
    #: command): without it the speed loop hunts below ``min_duty``.
    running_duty: float = 0.0
    max_duty: float = 100.0
    #: Largest duty change per control tick. Limits current surge and wheel
    #: slip; slip is invisible to the encoders and corrupts odometry silently.
    max_duty_step: float = 12.0

    # -- Control loop ------------------------------------------------------ #
    control_rate_hz: float = 50.0
    left_gains: Gains = field(default_factory=Gains)
    right_gains: Gains = field(default_factory=Gains)
    #: Time constant of the wheel-velocity low-pass, in seconds. At 12 CPR the
    #: raw estimate is coarse (see counts_per_wheel_rev below), so some
    #: filtering is not optional -- but every millisecond here is delay in the
    #: feedback loop, so do not raise it casually.
    velocity_filter_tau: float = 0.06

    # -- Safety ------------------------------------------------------------ #
    #: Stop if no new velocity command arrives within this long. A planner that
    #: crashes, a thread that deadlocks, or a cable that falls out must all
    #: result in a stopped robot, not one that keeps its last command forever.
    command_timeout: float = 0.5
    #: Speed limits applied to every command that enters the controller.
    max_linear_speed: float = 0.45
    max_angular_speed: float = 2.5
    #: Brake rather than coast when a move ends or the watchdog trips. A wheel
    #: that is still meant to be moving, and only wants less power for a tick,
    #: always coasts: braking it is a jolt every time.
    brake_on_stop: bool = True
    #: Runaway guard. A wheel driven at this duty % or more while its encoder
    #: says it is turning the *other* way, for ``runaway_time`` seconds, stops
    #: the robot and latches it like an emergency stop. That is never a slow
    #: reversal -- the duty slew and the motor settle one well inside half a
    #: second -- it is an encoder that is reversed, or wired to the other
    #: wheel. Either way the speed loop is in positive feedback: it pushes
    #: harder the faster the wheel goes, and the robot spins at full power
    #: and keeps spinning with zero commanded. 0 turns the guard off.
    runaway_duty: float = 30.0
    #: Seconds of that before it stops. The worst honest reversal measured in
    #: the simulator, with a sluggish 0.5 s motor time constant, was 0.32 s.
    runaway_time: float = 0.5
    #: Stall guard. At this duty % or more with the wheel not turning at all
    #: for ``stall_time`` seconds -- jammed, or its encoder unplugged -- stop
    #: and latch. 0 turns it off.
    stall_duty: float = 50.0
    #: Seconds of no movement at that duty before it stops.
    stall_time: float = 1.0

    # -- Motion profiles (the blocking move/turn helpers) ------------------ #
    default_linear_speed: float = 0.25
    default_angular_speed: float = 1.5
    linear_accel: float = 0.5
    angular_accel: float = 4.0
    #: Heading correction for straight moves, in rad/s per metre of
    #: left/right wheel-travel difference. Two independent velocity loops both
    #: settle on their target but never undo the difference they built up
    #: getting there, so a "straight" move ends a few degrees off. This closes
    #: the loop on the *difference in distance travelled*, which is the thing
    #: actually being asked for. Set to 0 to see how far it drifts without it.
    straightness_gain: float = 20.0
    #: Wheel synchronisation for driving by velocity (the live interface,
    #: the planner bridge), in rad/s of wheel speed per radian of wheel travel
    #: the two sides are out of step. Two speed loops on their own let
    #: whichever wheel answers first lead -- the robot starts every turn by
    #: swinging the wrong way, then over-corrects into an S. This compares the
    #: *difference* in how far the two wheels have turned with the difference
    #: asked for, from the encoder counts (not the noisier speeds), and moves
    #: both wheels' targets to close it. 0 turns it off.
    sync_gain: float = 4.0
    #: Largest out-of-step error it will try to recover, radians of wheel
    #: travel. Beyond this the reference is dragged along: a wheel that cannot
    #: keep up should not bank a debt that is paid back later as a lurch.
    sync_limit: float = 1.5
    #: How far ahead of the target to start braking, in seconds of travel at
    #: the current speed. The wheels lag the command by roughly the velocity
    #: filter's time constant plus the motor's, so a move that brakes exactly
    #: on target always stops past it. Overshoot is not an odometry error --
    #: the pose reports where it really ended up -- but a bench command asking
    #: for 100 counts should get near 100 counts.
    stop_lead_time: float = 0.12
    #: A move finishes when it is within this of target and has stopped.
    position_tolerance_m: float = 0.005
    angle_tolerance_rad: float = 0.03
    #: Wheel speed below which the robot counts as stationary, in rad/s.
    stopped_speed: float = 0.15

    # -- Encoder input ----------------------------------------------------- #
    #: "up", "down" or "none". Most encoder breakouts drive push-pull and want
    #: "none"; open-collector outputs need "up".
    encoder_pull: str = "none"
    #: Glitch filter in microseconds, 0 to disable. This delays every edge, so
    #: it trades count accuracy for noise immunity. Only raise it if you are
    #: seeing decode errors climb while the wheels are still.
    encoder_debounce_us: int = 0

    # ------------------------------------------------------------------ #
    # Derived geometry
    # ------------------------------------------------------------------ #

    @property
    def counts_per_motor_rev(self) -> float:
        return self.encoder_cpr * self.quadrature_multiplier

    @property
    def counts_per_wheel_rev(self) -> float:
        return self.counts_per_motor_rev * self.gear_ratio

    @property
    def effective_wheel_diameter_m(self) -> float:
        """The rolling diameter the robot actually behaves as if it has."""
        return self.wheel_diameter_m * self.distance_scale

    @property
    def metres_per_count(self) -> float:
        """Average of the two wheels; see ``left/right_metres_per_count``."""
        return math.pi * self.effective_wheel_diameter_m / self.counts_per_wheel_rev

    @property
    def _balance_factors(self) -> tuple:
        b = self.wheel_balance
        return 2.0 / (1.0 + b), 2.0 * b / (1.0 + b)

    @property
    def left_metres_per_count(self) -> float:
        return self.metres_per_count * self._balance_factors[0]

    @property
    def right_metres_per_count(self) -> float:
        return self.metres_per_count * self._balance_factors[1]

    @property
    def left_wheel_radius_m(self) -> float:
        return self.wheel_radius_m * self._balance_factors[0]

    @property
    def right_wheel_radius_m(self) -> float:
        return self.wheel_radius_m * self._balance_factors[1]

    @property
    def radians_per_count(self) -> float:
        return 2.0 * math.pi / self.counts_per_wheel_rev

    @property
    def wheel_radius_m(self) -> float:
        # Effective, so wheel speeds and distances agree with the calibration.
        return 0.5 * self.effective_wheel_diameter_m

    @property
    def control_period(self) -> float:
        return 1.0 / self.control_rate_hz

    def counts_to_metres(self, counts: float) -> float:
        return counts * self.metres_per_count

    def metres_to_counts(self, metres: float) -> float:
        return metres / self.metres_per_count

    def wheel_speeds_for(self, linear: float, angular: float) -> tuple:
        """Body velocity (m/s, rad/s) -> (left, right) wheel speeds in rad/s."""
        half_track = 0.5 * self.track_width_m
        left = (linear - angular * half_track) / self.left_wheel_radius_m
        right = (linear + angular * half_track) / self.right_wheel_radius_m
        return left, right

    def body_velocity_for(self, left: float, right: float) -> tuple:
        """(left, right) wheel speeds in rad/s -> body velocity (m/s, rad/s)."""
        v_left = left * self.left_wheel_radius_m
        v_right = right * self.right_wheel_radius_m
        linear = 0.5 * (v_left + v_right)
        angular = (v_right - v_left) / self.track_width_m
        return linear, angular

    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        pins = [
            self.left_motor.pwm, self.left_motor.in1, self.left_motor.in2,
            self.right_motor.pwm, self.right_motor.in1, self.right_motor.in2,
            self.left_encoder.a, self.left_encoder.b,
            self.right_encoder.a, self.right_encoder.b,
        ]
        if self.standby is not None:
            pins.append(self.standby)
        duplicates = {p for p in pins if pins.count(p) > 1}
        if duplicates:
            raise ValueError(f"GPIO pins used more than once: {sorted(duplicates)}")

        if self.encoder_cpr <= 0 or self.gear_ratio <= 0:
            raise ValueError("encoder_cpr and gear_ratio must be positive")
        if self.wheel_diameter_m <= 0 or self.track_width_m <= 0:
            raise ValueError("wheel_diameter_m and track_width_m must be positive")
        if not 0.9 <= self.wheel_balance <= 1.1:
            raise ValueError(
                f"wheel_balance {self.wheel_balance} is outside 0.9-1.1: two wheels "
                "of the same model are within a few percent of each other, so "
                "this is a wiring, encoder or gear-ratio fault, not wheel size")
        if not 0.5 <= self.distance_scale <= 2.0:
            raise ValueError(
                f"distance_scale {self.distance_scale} is outside 0.5-2.0: a "
                "correction that large means gear_ratio, encoder_cpr or the "
                "wheel diameter is wrong, not that the wheels need a nudge")
        if not 0 <= self.min_duty < self.max_duty <= 100:
            raise ValueError("need 0 <= min_duty < max_duty <= 100")
        if not 0 <= self.running_duty <= self.min_duty:
            raise ValueError("need 0 <= running_duty <= min_duty")
        if self.control_rate_hz <= 0:
            raise ValueError("control_rate_hz must be positive")
        if self.command_timeout <= 0:
            raise ValueError("command_timeout must be positive; safety depends on it")
        if self.encoder_pull not in ("up", "down", "none"):
            raise ValueError("encoder_pull must be 'up', 'down' or 'none'")
        if self.velocity_filter_tau < 0:
            raise ValueError("velocity_filter_tau must not be negative")
        for name in ("runaway_duty", "stall_duty"):
            if not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"{name} must be 0-100 (0 turns the guard off)")
        if self.sync_gain < 0 or self.sync_limit <= 0:
            raise ValueError("sync_gain must be >= 0 and sync_limit positive")
        for name in ("runaway_time", "stall_time"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def describe(self) -> str:
        return (
            f"counts/wheel rev : {self.counts_per_wheel_rev:.0f}\n"
            f"metres/count     : {self.metres_per_count * 1000:.4f} mm\n"
            f"counts/metre     : {1.0 / self.metres_per_count:.1f}\n"
            f"wheel diameter   : {self.wheel_diameter_m * 1000:.1f} mm"
            f"{f' x {self.distance_scale:.4f} scale' if self.distance_scale != 1.0 else ''}\n"
            f"track width      : {self.track_width_m * 1000:.1f} mm\n"
            f"max linear       : {self.max_linear_speed:.2f} m/s\n"
            f"control rate     : {self.control_rate_hz:.0f} Hz"
        )


__all__ = ["DriveConfig", "EncoderChannel", "Gains", "MotorChannel"]
