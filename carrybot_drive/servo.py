from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from .backend import GpioBackend


@dataclass
class ServoConfig:
    """Geometry and timing for one hobby servo.

    The defaults describe a common 9 g analogue servo. Measure yours: the
    pulse-width endpoints in particular vary by ten percent between units that
    claim the same spec, and getting them wrong shows up as a head that cannot
    quite reach one side.
    """

    #: GPIO pin carrying the control pulse.
    pin: int = 18
    #: Servo frame rate. 50 Hz is the near-universal analogue standard; digital
    #: servos often accept 200-333 Hz and respond proportionally faster.
    frequency: int = 50
    #: Pulse width at the two mechanical extremes, microseconds.
    min_pulse_us: float = 500.0
    max_pulse_us: float = 2500.0
    #: The angles those pulse widths correspond to, radians. Positive is a
    #: turn to the robot's **left**, matching the navigation frame -- if your
    #: linkage runs the other way, swap these two rather than negating angles
    #: at the call sites.
    min_angle: float = -math.pi / 2
    max_angle: float = math.pi / 2
    #: Mechanical travel limit actually used, radians from centre. Kept
    #: separate from the endpoints above so the head can be restricted without
    #: relying on the calibration being right.
    limit: float = math.radians(75.0)
    #: How fast the horn moves, radians per second, unloaded. A 0.12 s/60 deg
    #: servo is about 8.7 rad/s; halve it for the load of a camera.
    max_rate: float = 4.0
    #: Command changes smaller than this are not sent. Analogue servos hunt and
    #: buzz around a setpoint, and a camera that jitters ruins the disparity.
    deadband: float = math.radians(0.8)
    #: Stop sending pulses once the horn has been still this long. An idle
    #: servo that is still being driven holds torque, draws current and hums;
    #: 0 disables the release.
    idle_release: float = 0.5
    #: Trim added to every command, radians. The cheap way to square up a head
    #: that is mechanically a couple of degrees off centre.
    offset: float = 0.0

    def validate(self) -> None:
        if self.max_angle <= self.min_angle:
            raise ValueError("max_angle must exceed min_angle")
        if self.max_pulse_us <= self.min_pulse_us:
            raise ValueError("max_pulse_us must exceed min_pulse_us")
        if self.limit <= 0.0:
            raise ValueError("limit must be positive")
        if self.max_rate <= 0.0:
            raise ValueError("max_rate must be positive")

    def pulse_for(self, angle: float) -> float:
        """Angle in radians -> pulse width in microseconds, clamped."""
        lo = max(self.min_angle, -self.limit)
        hi = min(self.max_angle, self.limit)
        angle = max(lo, min(hi, angle))
        t = (angle - self.min_angle) / (self.max_angle - self.min_angle)
        return self.min_pulse_us + t * (self.max_pulse_us - self.min_pulse_us)

    def duty_for(self, angle: float) -> float:
        """The same, as a percentage duty cycle at ``frequency``."""
        period_us = 1_000_000.0 / self.frequency
        return 100.0 * self.pulse_for(angle) / period_us

    @property
    def travel(self) -> float:
        """Usable half-travel, radians."""
        return min(self.limit, min(abs(self.min_angle), abs(self.max_angle)))


class PanServo:
    """One servo, driven by angle, with a model of where it has got to."""

    def __init__(
        self,
        backend: GpioBackend,
        config: Optional[ServoConfig] = None,
        initial: float = 0.0,
    ):
        self.cfg = config or ServoConfig()
        self.cfg.validate()
        self.backend = backend
        self.backend.claim_output(self.cfg.pin, 0)

        self._command = self._clamp(initial)
        self._angle = self._command
        self._sent: Optional[float] = None
        self._last_update = time.monotonic()
        self._moving_until = 0.0
        self._driving = False
        self.commands = 0
        self.writes = 0

        self._write(self._command)

    # ------------------------------------------------------------------ #

    def _clamp(self, angle: float) -> float:
        travel = self.cfg.travel
        return max(-travel, min(travel, angle))

    @property
    def angle(self) -> float:
        """Modelled position of the horn, radians. What the maths should use."""
        return self._angle

    @property
    def target(self) -> float:
        """Where it was last told to go."""
        return self._command

    @property
    def settled(self) -> bool:
        return abs(self._command - self._angle) <= self.cfg.deadband

    # ------------------------------------------------------------------ #

    def command(self, angle: float, now: Optional[float] = None) -> float:
        """Ask for an angle. Returns the clamped command actually adopted.

        Safe to call every frame with a noisy number: anything inside the
        deadband of the current command is ignored outright, so a tracker that
        jitters by half a degree does not turn into a servo that buzzes.
        """
        now = time.monotonic() if now is None else now
        self.update(now)
        wanted = self._clamp(angle)
        if abs(wanted - self._command) < self.cfg.deadband:
            return self._command
        self._command = wanted
        self.commands += 1
        self._write(wanted)
        return wanted

    def update(self, now: Optional[float] = None) -> float:
        """Advance the model to ``now``. Returns the modelled angle.

        Call this every control cycle even when not commanding, so the angle
        used by the projection keeps up with a horn that is still travelling.
        """
        now = time.monotonic() if now is None else now
        dt = max(0.0, now - self._last_update)
        self._last_update = now

        error = self._command - self._angle
        step = self.cfg.max_rate * dt
        if abs(error) <= step:
            self._angle = self._command
        else:
            self._angle += math.copysign(step, error)

        if not self.settled:
            self._moving_until = now + self.cfg.idle_release
        elif (
            self._driving
            and self.cfg.idle_release > 0.0
            and now > self._moving_until
        ):
            # Let go. A servo held at a setpoint it has already reached is just
            # a heater that hums.
            self.backend.set_pwm(self.cfg.pin, self.cfg.frequency, 0.0)
            self._driving = False
        return self._angle

    def note_measured(self, angle: float) -> None:
        """Override the model with a real reading, if you have one."""
        self._angle = self._clamp(angle)

    def centre(self) -> None:
        self.command(0.0)

    def release(self) -> None:
        """Stop driving, leaving the horn wherever it is."""
        self.backend.set_pwm(self.cfg.pin, self.cfg.frequency, 0.0)
        self._driving = False

    def close(self) -> None:
        self.release()

    # ------------------------------------------------------------------ #

    def _write(self, angle: float) -> None:
        duty = self.cfg.duty_for(angle + self.cfg.offset)
        self.backend.set_pwm(self.cfg.pin, self.cfg.frequency, duty)
        self._sent = angle
        self._driving = True
        self.writes += 1
        self._moving_until = self._last_update + self.cfg.idle_release

    def describe(self) -> str:
        return (
            f"pan servo on pin {self.cfg.pin}: "
            f"+/-{math.degrees(self.cfg.travel):.0f} deg travel, "
            f"{math.degrees(self.cfg.max_rate):.0f} deg/s, "
            f"{self.cfg.min_pulse_us:.0f}-{self.cfg.max_pulse_us:.0f} us "
            f"at {self.cfg.frequency} Hz, "
            f"deadband {math.degrees(self.cfg.deadband):.1f} deg"
        )


class SimulatedServo(PanServo):
    """A servo with no GPIO behind it at all, for sims and tests."""

    def __init__(self, config: Optional[ServoConfig] = None, initial: float = 0.0):
        super().__init__(_NullBackend(), config, initial)


class _NullBackend:
    def claim_output(self, pin: int, level: int = 0) -> None: ...
    def claim_alert(self, pin: int, pull: str = "none", debounce_us: int = 0) -> None: ...
    def write(self, pin: int, level: int) -> None: ...
    def read(self, pin: int) -> int: return 0
    def set_pwm(self, pin: int, frequency: int, duty_percent: float) -> None: ...
    def add_callback(self, pin: int, func): raise NotImplementedError
    def close(self) -> None: ...


__all__ = ["PanServo", "ServoConfig", "SimulatedServo"]
