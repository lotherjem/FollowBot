from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Tuple

from .backend import GpioBackend
from .config import DriveConfig, MotorChannel


class MotorMode(Enum):
    DRIVE = "drive"
    #: Terminals shorted. Stops hard, holds against a push, draws no supply
    #: current. This is what you want when a move ends.
    BRAKE = "brake"
    #: Outputs high-impedance. The wheel free-wheels. Use it when something has
    #: gone wrong and you would rather the robot be pushable than locked.
    COAST = "coast"


class Motor:
    """One TB6612 channel."""

    def __init__(self, channel: MotorChannel, config: DriveConfig, backend: GpioBackend):
        self.channel = channel
        self.cfg = config
        self.backend = backend
        self._duty = 0.0
        self._mode = MotorMode.COAST

    def setup(self) -> None:
        self.backend.claim_output(self.channel.in1, 0)
        self.backend.claim_output(self.channel.in2, 0)
        self.backend.claim_output(self.channel.pwm, 0)
        self.coast()

    # -- state ---------------------------------------------------------- #

    @property
    def duty(self) -> float:
        """Signed duty currently applied, in percent."""
        return self._duty

    @property
    def mode(self) -> MotorMode:
        return self._mode

    # -- output --------------------------------------------------------- #

    def set_duty(
        self,
        target: float,
        max_step: Optional[float] = None,
        deadband: bool = True,
        floor: Optional[float] = None,
        coast_on_zero: bool = False,
    ) -> float:
        """Drive at a signed duty percent. Returns what was actually applied.

        ``max_step`` limits how far the duty may move in one call, which is how
        the current surge on a direction reversal is kept in check. Slipping a
        wheel is worse than being slow to accelerate: slip is invisible to the
        encoders, so it corrupts odometry without any signal that it happened.

        ``floor`` replaces ``min_duty`` as the deadband edge (a wheel already
        turning needs less to keep turning). ``coast_on_zero`` releases the
        motor on a zero duty instead of following ``brake_on_stop``: for a
        wheel that is still meant to be moving and wants less power for a
        moment, not a stop.
        """
        target = _clamp(target, -self.cfg.max_duty, self.cfg.max_duty)

        step = self.cfg.max_duty_step if max_step is None else max_step
        if step > 0:
            delta = target - self._duty
            if abs(delta) > step:
                target = self._duty + math.copysign(step, delta)

        # Below the deadband the motor hums and heats without turning. Push
        # small non-zero commands out to where they actually produce motion.
        # Calibration passes deadband=False, because measuring where the
        # deadband *is* requires being allowed to command duties inside it.
        edge = self.cfg.min_duty if floor is None else floor
        if deadband and 0.0 < abs(target) < edge:
            target = math.copysign(edge, target)

        self._apply(target, coast_on_zero)
        return target

    def _apply(self, signed_duty: float, coast_on_zero: bool = False) -> None:
        self._duty = signed_duty

        if signed_duty == 0.0:
            if coast_on_zero or not self.cfg.brake_on_stop:
                self.coast()
            else:
                self.brake()
            return

        forward = signed_duty > 0
        if self.channel.invert:
            forward = not forward

        self.backend.write(self.channel.in1, 1 if forward else 0)
        self.backend.write(self.channel.in2, 0 if forward else 1)
        self.backend.set_pwm(self.channel.pwm, self.cfg.pwm_frequency, abs(signed_duty))
        self._mode = MotorMode.DRIVE

    def brake(self) -> None:
        """Short the motor terminals. Hard stop, no supply current."""
        self._duty = 0.0
        self._mode = MotorMode.BRAKE
        self.backend.write(self.channel.in1, 1)
        self.backend.write(self.channel.in2, 1)
        self.backend.set_pwm(self.channel.pwm, self.cfg.pwm_frequency, 100.0)

    def coast(self) -> None:
        """Release the motor. The wheel free-wheels."""
        self._duty = 0.0
        self._mode = MotorMode.COAST
        self.backend.set_pwm(self.channel.pwm, self.cfg.pwm_frequency, 0.0)
        self.backend.write(self.channel.in1, 0)
        self.backend.write(self.channel.in2, 0)


class MotorPair:
    """Both motors, plus the standby line if it is under software control."""

    def __init__(self, config: DriveConfig, backend: GpioBackend):
        self.cfg = config
        self.backend = backend
        self.left = Motor(config.left_motor, config, backend)
        self.right = Motor(config.right_motor, config, backend)

    def setup(self) -> None:
        if self.cfg.standby is not None:
            self.backend.claim_output(self.cfg.standby, 0)
        self.left.setup()
        self.right.setup()
        self.enable()

    def enable(self) -> None:
        if self.cfg.standby is not None:
            self.backend.write(self.cfg.standby, 1)

    def disable(self) -> None:
        """Put the driver in standby: outputs off whatever the IN pins say.

        This is the only stop that survives a hung control loop, because it
        does not depend on anything else being written afterwards. It needs
        STBY on a GPIO -- with the pin strapped to VCC there is no such stop.
        """
        if self.cfg.standby is not None:
            self.backend.write(self.cfg.standby, 0)

    def set_duty(
        self,
        left: float,
        right: float,
        max_step: Optional[float] = None,
        deadband: bool = True,
        floors: Tuple[Optional[float], Optional[float]] = (None, None),
        coast: Tuple[bool, bool] = (False, False),
    ):
        return (
            self.left.set_duty(left, max_step, deadband, floors[0], coast[0]),
            self.right.set_duty(right, max_step, deadband, floors[1], coast[1]),
        )

    def brake(self) -> None:
        """Both wheels, one after the other with no polling in between."""
        self.left.brake()
        self.right.brake()

    def coast(self) -> None:
        self.left.coast()
        self.right.coast()

    def stop(self) -> None:
        self.brake() if self.cfg.brake_on_stop else self.coast()


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


__all__ = ["Motor", "MotorMode", "MotorPair"]
