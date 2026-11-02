from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .config import DriveConfig

#: Called as ``func(pin, level, timestamp_ns)``.
EdgeCallback = Callable[[int, int, int], None]

#: Gray-code sequence for one full quadrature cycle in the forward direction.
#: Index n and n+1 differ in exactly one bit, which is what makes each step a
#: single edge on a single channel.
QUAD_SEQUENCE = (0b00, 0b01, 0b11, 0b10)


class CallbackHandle(Protocol):
    def cancel(self) -> None: ...


class GpioBackend(Protocol):
    """The whole hardware surface the drive layer needs."""

    def claim_output(self, pin: int, level: int = 0) -> None: ...
    def claim_alert(self, pin: int, pull: str = "none", debounce_us: int = 0) -> None: ...
    def write(self, pin: int, level: int) -> None: ...
    def read(self, pin: int) -> int: ...
    def set_pwm(self, pin: int, frequency: int, duty_percent: float) -> None: ...
    def add_callback(self, pin: int, func: EdgeCallback) -> CallbackHandle: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Real hardware
# --------------------------------------------------------------------------- #


class LgpioBackend:
    """lgpio on a Raspberry Pi."""

    def __init__(self, chip: int = 0):
        import lgpio  # imported here so the module loads off-robot

        self._lgpio = lgpio
        self._handle = lgpio.gpiochip_open(chip)
        self._claimed: List[int] = []
        # lgpio's callback objects deregister themselves when collected, so
        # dropping the return value can silently stop the encoders. Hold them.
        self._callbacks: List[object] = []
        self._closed = False

    def claim_output(self, pin: int, level: int = 0) -> None:
        self._lgpio.gpio_claim_output(self._handle, pin, level)
        self._claimed.append(pin)

    def claim_alert(self, pin: int, pull: str = "none", debounce_us: int = 0) -> None:
        lg = self._lgpio
        flags = {
            "up": lg.SET_PULL_UP,
            "down": lg.SET_PULL_DOWN,
            "none": lg.SET_PULL_NONE,
        }[pull]
        lg.gpio_claim_alert(self._handle, pin, lg.BOTH_EDGES, flags)
        if debounce_us > 0:
            lg.gpio_set_debounce_micros(self._handle, pin, debounce_us)
        self._claimed.append(pin)

    def write(self, pin: int, level: int) -> None:
        self._lgpio.gpio_write(self._handle, pin, level)

    def read(self, pin: int) -> int:
        return self._lgpio.gpio_read(self._handle, pin)

    def set_pwm(self, pin: int, frequency: int, duty_percent: float) -> None:
        duty = max(0.0, min(100.0, duty_percent))
        self._lgpio.tx_pwm(self._handle, pin, frequency, duty)

    def add_callback(self, pin: int, func: EdgeCallback) -> CallbackHandle:
        lg = self._lgpio

        def shim(_chip, gpio, level, timestamp):
            # level 2 is lgpio's watchdog tick, not an edge.
            if level < 2:
                func(gpio, level, timestamp)

        cb = lg.callback(self._handle, pin, lg.BOTH_EDGES, shim)
        self._callbacks.append(cb)
        return cb

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cb in self._callbacks:
            try:
                cb.cancel()
            except Exception:
                pass
        self._callbacks.clear()
        for pin in self._claimed:
            try:
                self._lgpio.gpio_free(self._handle, pin)
            except Exception:
                pass
        self._claimed.clear()
        self._lgpio.gpiochip_close(self._handle)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


class _SimMotor:
    """First-order motor with stiction, driven by signed duty percent.

    Unpowered, a shorted (braked) motor stops much faster than a coasting
    one: the short turns the motor into a generator into a dead load. The
    difference is what makes braking for one control tick a jolt and coasting
    for it a dip, so it is modelled.
    """

    def __init__(self, no_load_speed: float, deadband: float, tau: float,
                 brake_tau: float = 0.04, coast_tau: float = 0.25):
        self.no_load_speed = no_load_speed  # rad/s at the wheel, 100% duty
        self.deadband = deadband
        self.tau = tau
        self.brake_tau = brake_tau
        self.coast_tau = coast_tau
        self.braking = False
        self.duty = 0.0
        self.speed = 0.0  # rad/s at the wheel
        self.angle = 0.0  # rad
        self.scale = 1.0  # per-wheel mismatch: real robots are never symmetric

    def step(self, dt: float) -> None:
        magnitude = abs(self.duty)
        if magnitude <= self.deadband:
            target = 0.0
        else:
            span = 100.0 - self.deadband
            target = math.copysign(
                (magnitude - self.deadband) / span * self.no_load_speed, self.duty
            )
        target *= self.scale

        tau = self.tau
        if self.duty == 0.0:
            tau = self.brake_tau if self.braking else self.coast_tau
        alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
        self.speed += (target - self.speed) * alpha
        self.angle += self.speed * dt


class SimBackend:
    """A robot on the bench, in software.

    Runs its own thread integrating two motors and emitting quadrature edges to
    whatever callbacks the decoder registered. Wheel mismatch is on by default
    (``skew``) because two motors at the same duty never produce the same speed,
    and a controller that only works on perfectly matched wheels is a
    controller that has not been tested.
    """

    def __init__(
        self,
        config: DriveConfig,
        tick_hz: float = 400.0,
        no_load_speed: float = 18.0,
        deadband: float = 10.0,
        tau: float = 0.12,
        skew: float = 0.12,
        real_time: bool = True,
        swap_encoders: bool = False,
        reversed_encoders: Sequence[str] = (),
        dead_encoders: Sequence[str] = (),
        delivery_lag: float = 0.0,
        seed: int = 0,
        deadbands: Optional[Tuple[float, float]] = None,
        swap_pwm: bool = False,
    ):
        """``swap_encoders`` and ``reversed_encoders`` are wiring faults the
        config does not know about: each wheel's encoder plugged into the other
        wheel's pins, or an encoder whose A and B are the other way round.
        ``dead_encoders`` never produce an edge: unplugged.

        ``deadbands`` gives each wheel its own (left, right) duty below which
        it will not turn -- one wheel stiffer than the other, as on the floor.

        ``delivery_lag`` holds edge callbacks back and releases them in
        bursts, up to that many seconds late, with their true timestamps --
        what a Pi does when the vision work keeps the callback thread waiting.
        ``swap_pwm`` crosses PWMA and PWMB: each motor takes its direction from
        its own IN pins and its power from the other motor's PWM pin.

        They exist so the controller's fault detection can be tested against
        the mistakes that actually get made."""
        self.swap_pwm = swap_pwm
        self.cfg = config
        self.tick = 1.0 / tick_hz
        self.real_time = real_time
        self.swap_encoders = swap_encoders
        self.reversed_encoders = frozenset(reversed_encoders)
        self.dead_encoders = frozenset(dead_encoders)
        self.delivery_lag = float(delivery_lag)
        self._held: List[tuple] = []
        self._next_flush = 0.0
        import random as _random

        self._rng = _random.Random(seed)

        self._levels: Dict[int, int] = {}
        self._pwm: Dict[int, float] = {}
        self._callbacks: Dict[int, List[EdgeCallback]] = {}
        self._lock = threading.Lock()

        self.left = _SimMotor(no_load_speed, deadband, tau)
        self.right = _SimMotor(no_load_speed, deadband, tau)
        self.left.scale = 1.0 + skew * 0.5
        self.right.scale = 1.0 - skew * 0.5
        if deadbands is not None:
            self.left.deadband, self.right.deadband = deadbands

        self._phase = {"left": 0, "right": 0}
        self._counts = {"left": 0, "right": 0}
        self._sim_time = 0.0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._closed = False

        for pin in (
            config.left_encoder.a, config.left_encoder.b,
            config.right_encoder.a, config.right_encoder.b,
        ):
            self._levels[pin] = 0

    # -- GpioBackend ---------------------------------------------------- #

    def claim_output(self, pin: int, level: int = 0) -> None:
        self._levels[pin] = level

    def claim_alert(self, pin: int, pull: str = "none", debounce_us: int = 0) -> None:
        self._levels.setdefault(pin, 0)

    def write(self, pin: int, level: int) -> None:
        self._levels[pin] = 1 if level else 0

    def read(self, pin: int) -> int:
        return self._levels.get(pin, 0)

    def set_pwm(self, pin: int, frequency: int, duty_percent: float) -> None:
        self._pwm[pin] = max(0.0, min(100.0, duty_percent))

    def add_callback(self, pin: int, func: EdgeCallback) -> CallbackHandle:
        self._callbacks.setdefault(pin, []).append(func)
        backend = self

        class _Handle:
            def cancel(self) -> None:
                try:
                    backend._callbacks[pin].remove(func)
                except (KeyError, ValueError):
                    pass

        return _Handle()

    def close(self) -> None:
        self._closed = True
        self.stop_thread()

    # -- Simulation ----------------------------------------------------- #

    def start_thread(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="carrybot-sim", daemon=True
        )
        self._thread.start()

    def stop_thread(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            self.advance(self.tick)
            if self.real_time:
                next_tick += self.tick
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()

    def advance(self, dt: float) -> None:
        """Step the simulation. Called by the thread, or directly in tests."""
        self._sim_time += dt
        cfg = self.cfg

        self.left.duty = self._signed_duty(cfg.left_motor)
        self.right.duty = self._signed_duty(cfg.right_motor)
        self.left.braking = self._braking(cfg.left_motor)
        self.right.braking = self._braking(cfg.right_motor)

        self.left.step(dt)
        self.right.step(dt)

        # Keyed by the physical wheel; the channel is wherever it is wired.
        left_channel, right_channel = cfg.left_encoder, cfg.right_encoder
        if self.swap_encoders:
            left_channel, right_channel = right_channel, left_channel
        if "left" not in self.dead_encoders:
            self._emit("left", left_channel, self._wired("left", self.left.angle))
        if "right" not in self.dead_encoders:
            self._emit("right", right_channel, self._wired("right", self.right.angle))
        if self._held and self._sim_time >= self._next_flush:
            held, self._held = self._held, []
            for func, pin, level, stamp in held:
                func(pin, level, stamp)
            self._next_flush = self._sim_time + self._rng.uniform(0.0, self.delivery_lag)

    def _wired(self, side: str, angle: float) -> float:
        return -angle if side in self.reversed_encoders else angle

    def _braking(self, motor) -> bool:
        """Both direction pins high: the TB6612's short brake."""
        return (self._levels.get(motor.in1, 0) == 1
                and self._levels.get(motor.in2, 0) == 1)

    def _signed_duty(self, motor) -> float:
        """Decode the TB6612 truth table back into a signed duty."""
        in1 = self._levels.get(motor.in1, 0)
        in2 = self._levels.get(motor.in2, 0)
        pwm_pin = motor.pwm
        if self.swap_pwm:
            left, right = self.cfg.left_motor, self.cfg.right_motor
            pwm_pin = right.pwm if motor is left else left.pwm
        duty = self._pwm.get(pwm_pin, 0.0)

        if in1 == in2:
            # Both high is short brake, both low is coast. Neither drives, and
            # the difference between them only shows up in how fast the wheel
            # slows -- modelled here by braking hard toward zero.
            return 0.0
        signed = duty if in1 else -duty
        return -signed if motor.invert else signed

    def _emit(self, side: str, channel, angle: float) -> None:
        """Walk the encoder forward or back one edge at a time."""
        counts_per_rev = self.cfg.counts_per_wheel_rev
        target = int(angle / (2.0 * math.pi) * counts_per_rev)
        if channel.invert:
            target = -target

        current = self._counts[side]
        if target == current:
            return

        step = 1 if target > current else -1
        pins = (channel.a, channel.b)
        # Stamped like the kernel stamps a real edge: on the monotonic clock,
        # when it happened -- which a held-back delivery does not change.
        stamp = time.monotonic_ns() if self.real_time else int(self._sim_time * 1e9)

        # Cap the work per tick. Overrunning this means the simulated wheel is
        # spinning faster than the sim can emit edges for, which is exactly the
        # regime where real hardware starts dropping counts too.
        for _ in range(min(abs(target - current), 4096)):
            self._phase[side] = (self._phase[side] + step) % 4
            state = QUAD_SEQUENCE[self._phase[side]]
            new_a = (state >> 1) & 1
            new_b = state & 1
            for pin, level in zip(pins, (new_a, new_b)):
                if self._levels.get(pin, 0) != level:
                    self._levels[pin] = level
                    for func in self._callbacks.get(pin, ()):
                        if self.delivery_lag > 0:
                            self._held.append((func, pin, level, stamp))
                        else:
                            func(pin, level, stamp)
            current += step
        self._counts[side] = current

    # -- Introspection for tests ---------------------------------------- #

    @property
    def truth_counts(self) -> Dict[str, int]:
        """What the encoders *should* read, straight from the simulation."""
        return dict(self._counts)

    @property
    def wheel_speeds(self) -> Dict[str, float]:
        return {"left": self.left.speed, "right": self.right.speed}


__all__ = ["EdgeCallback", "GpioBackend", "LgpioBackend", "QUAD_SEQUENCE", "SimBackend"]
