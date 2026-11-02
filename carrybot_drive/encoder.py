from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .backend import QUAD_SEQUENCE, GpioBackend
from .config import DriveConfig, EncoderChannel


def _build_transition_table() -> Dict[int, int]:
    """Map ``(old_state << 2) | new_state`` to -1, 0 or +1.

    Derived from the Gray-code sequence rather than typed out, so it cannot
    disagree with the phase order the rest of the package assumes.
    """
    table: Dict[int, int] = {}
    for i, state in enumerate(QUAD_SEQUENCE):
        forward = QUAD_SEQUENCE[(i + 1) % 4]
        backward = QUAD_SEQUENCE[(i - 1) % 4]
        table[(state << 2) | forward] = 1
        table[(state << 2) | backward] = -1
    return table


TRANSITIONS = _build_transition_table()


@dataclass(frozen=True)
class EncoderSample:
    position: int
    delta: int
    #: Seconds actually elapsed since the previous sample, measured -- not the
    #: nominal loop period. A sleep of 20 ms is never 20 ms, and dividing by
    #: the nominal value bakes that error straight into the velocity.
    dt: float
    #: Filtered wheel speed in rad/s.
    velocity: float
    #: Unfiltered wheel speed in rad/s, for diagnostics.
    raw_velocity: float
    errors: int
    edges: int


class QuadratureDecoder:
    """One encoder. Thread-safe; every public method takes the lock."""

    def __init__(self, channel: EncoderChannel, config: DriveConfig):
        self.channel = channel
        self.cfg = config
        self._lock = threading.Lock()

        self._a = 0
        self._b = 0
        self._state = 0
        self._position = 0
        self._errors = 0
        self._desyncs = 0
        self._edges = 0
        self._last_edge_ns = 0
        #: Which clock the edge stamps are on, found from the first edge, so
        #: delivery lag can be measured; None until known or if neither fits.
        self._clock = None
        self._clock_checked = False
        self._max_lag_ns = 0

        self._sign = -1 if channel.invert else 1

    # -- wiring --------------------------------------------------------- #

    def attach(self, backend: GpioBackend) -> List[object]:
        backend.claim_alert(
            self.channel.a, self.cfg.encoder_pull, self.cfg.encoder_debounce_us
        )
        backend.claim_alert(
            self.channel.b, self.cfg.encoder_pull, self.cfg.encoder_debounce_us
        )
        with self._lock:
            self._a = backend.read(self.channel.a)
            self._b = backend.read(self.channel.b)
            self._state = (self._a << 1) | self._b
        return [
            backend.add_callback(self.channel.a, self._on_edge),
            backend.add_callback(self.channel.b, self._on_edge),
        ]

    # -- the hot path --------------------------------------------------- #

    def _on_edge(self, pin: int, level: int, timestamp_ns: int) -> None:
        # How late this edge is being handled. lgpio's stamps are the
        # kernel's; which clock they are on is not documented the same way
        # everywhere, so take whichever clock puts the first edge within a few
        # seconds of now. Costs one clock read per edge.
        if not self._clock_checked:
            self._clock_checked = True
            for clock in (time.monotonic_ns, time.time_ns):
                if 0 <= clock() - timestamp_ns < 5_000_000_000:
                    self._clock = clock
                    break
        if self._clock is not None:
            lag = self._clock() - timestamp_ns
            if lag > self._max_lag_ns:
                self._max_lag_ns = lag
        with self._lock:
            if pin == self.channel.a:
                self._a = level
            else:
                self._b = level

            new_state = (self._a << 1) | self._b
            if new_state == self._state:
                # Same level reported twice: a glitch, or a debounced edge that
                # bounced back. Nothing moved.
                return

            movement = TRANSITIONS.get((self._state << 2) | new_state)
            self._state = new_state
            self._edges += 1
            self._last_edge_ns = timestamp_ns

            if movement is None:
                # A two-bit jump: both channels changed between one callback and
                # the next. Direction is genuinely unknowable, so take the count
                # loss and carry on from the new phase rather than decoding
                # every later edge from a stale one.
                self._errors += 1
                return

            self._position += movement * self._sign

    # -- readers -------------------------------------------------------- #

    @property
    def position(self) -> int:
        with self._lock:
            return self._position

    @property
    def errors(self) -> int:
        """Undecodable transitions seen.

        Expected to stay at zero: every callback carries one channel's level,
        so every transition it can produce is a legal single-bit move. A
        non-zero count means the backend coalesced or reordered levels, which
        is a fault in the backend rather than a fact about the wheel. For lost
        edges, watch :attr:`desyncs` instead.
        """
        with self._lock:
            return self._errors

    @property
    def desyncs(self) -> int:
        """Times the cached phase disagreed with the pins. See the module docs."""
        with self._lock:
            return self._desyncs

    def check_alignment(self, backend: GpioBackend) -> bool:
        """Compare the cached phase against the actual pins, and resync.

        Returns True if they already agreed. Only meaningful while the wheel is
        stopped: mid-rotation the pins change between the two reads and a
        disagreement means nothing.

        A rising desync count on a stationary wheel is the signature of edges
        being lost -- the count is not merely short, it has been walking in the
        wrong direction. Raise ``encoder_debounce_us`` if it correlates with
        electrical noise; lower the load on the Pi if it correlates with the
        loop overrun count.
        """
        a = backend.read(self.channel.a)
        b = backend.read(self.channel.b)
        state = (a << 1) | b
        with self._lock:
            if state == self._state:
                return True
            self._desyncs += 1
            self._a, self._b, self._state = a, b, state
            return False

    @property
    def edges(self) -> int:
        with self._lock:
            return self._edges

    def snapshot(self) -> Tuple[int, int, int, int]:
        """Position, errors, edges and last-edge timestamp in one lock take."""
        with self._lock:
            return self._position, self._errors, self._edges, self._last_edge_ns

    def take_lag_ms(self) -> float:
        """Worst delay between an edge happening and being decoded, since the
        last call, in milliseconds. A Pi too busy to hand edges to Python
        promptly shows here first."""
        lag, self._max_lag_ns = self._max_lag_ns, 0
        return lag / 1e6

    def reset(self) -> None:
        with self._lock:
            self._position = 0
            self._errors = 0
            self._desyncs = 0
            self._edges = 0


class VelocityEstimator:
    """Counts to filtered wheel speed, timed by the edges themselves.

    The speed is the distance between the newest edge this sample and the
    newest edge last sample, divided by the time *between those edges* as the
    kernel stamped them -- not by the time between two control ticks.

    That difference is the whole point. Edges reach Python through a callback
    thread, and on a Pi busy with stereo and the person detector that thread
    runs late and then catches up: a tick sees no edges, the next sees two
    ticks' worth. Counting edges per tick turned that into a wheel apparently
    stopping and then doubling its speed -- noise of +/-20 rad/s on a wheel
    turning at 5, which the speed loop chased into a stutter and the path
    follower's turn damping into a turn the wrong way. The edges' own
    timestamps do not care when they were delivered.

    A wheel counts as stopped once no edge has arrived for three expected edge
    intervals at the current speed, and never sooner than ``STALE_S``: long
    enough to ride out a late callback, short enough that a stopped wheel is
    not reported as moving for long.
    """

    #: Shortest silence that counts as stopped, seconds.
    STALE_S = 0.12
    #: A gap between edges longer than this is not used as a period.
    MAX_PERIOD_S = 0.5

    def __init__(self, config: DriveConfig):
        self.cfg = config
        self.reset()

    def reset(self, position: int = 0) -> None:
        self._last_position = position
        self._last_time: Optional[float] = None
        self._velocity = 0.0
        self._edge_ns = 0
        self._edge_position = position
        self._last_change: Optional[float] = None

    def update(self, decoder: QuadratureDecoder, now: float) -> EncoderSample:
        position, errors, edges, edge_ns = decoder.snapshot()

        if self._last_time is None:
            self._last_time = now
            self._last_position = position
            self._edge_ns, self._edge_position = edge_ns, position
            self._last_change = now
            return EncoderSample(position, 0, 0.0, 0.0, 0.0, errors, edges)

        dt = now - self._last_time
        if dt <= 0:
            return EncoderSample(position, 0, 0.0, self._velocity, 0.0, errors, edges)
        delta = position - self._last_position
        self._last_position = position
        self._last_time = now
        rpc = self.cfg.radians_per_count

        covered = dt
        if edge_ns != self._edge_ns:
            span = (edge_ns - self._edge_ns) * 1e-9
            moved = position - self._edge_position
            if self._edge_ns and 0.0 < span <= self.MAX_PERIOD_S:
                raw = moved * rpc / span
                # Filter by the time the measurement covers: a slowing wheel's
                # edges spread out, each one says more, and the estimate
                # follows it down instead of lagging a filter behind.
                covered = max(dt, span)
            else:
                raw = delta * rpc / dt           # first edge after a stop
            self._edge_ns, self._edge_position = edge_ns, position
            self._last_change = now
        else:
            # No edge this sample. Late delivery or a stopped wheel: wait the
            # longer of STALE_S and three edge intervals before deciding.
            interval = rpc / abs(self._velocity) if self._velocity else 0.0
            if now - self._last_change >= max(self.STALE_S, 3.0 * interval):
                self._velocity = 0.0
                return EncoderSample(position, 0, dt, 0.0, 0.0, errors, edges)
            return EncoderSample(position, delta, dt, self._velocity,
                                 self._velocity, errors, edges)

        tau = self.cfg.velocity_filter_tau
        alpha = 1.0 - math.exp(-covered / tau) if tau > 0 else 1.0
        self._velocity += (raw - self._velocity) * alpha
        return EncoderSample(position, delta, dt, self._velocity, raw, errors, edges)


class EncoderPair:
    """Both wheels, wired up together."""

    def __init__(self, config: DriveConfig):
        self.cfg = config
        self.left = QuadratureDecoder(config.left_encoder, config)
        self.right = QuadratureDecoder(config.right_encoder, config)
        self.left_velocity = VelocityEstimator(config)
        self.right_velocity = VelocityEstimator(config)
        self._handles: List[object] = []

    def attach(self, backend: GpioBackend) -> None:
        self._handles = self.left.attach(backend) + self.right.attach(backend)

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.cancel()
            except Exception:
                pass
        self._handles.clear()

    def sample(self, now: Optional[float] = None) -> Tuple[EncoderSample, EncoderSample]:
        now = time.monotonic() if now is None else now
        return (
            self.left_velocity.update(self.left, now),
            self.right_velocity.update(self.right, now),
        )

    def positions(self) -> Tuple[int, int]:
        return self.left.position, self.right.position

    def check_alignment(self, backend: GpioBackend) -> None:
        """Verify both decoders against the pins. Call only when stopped."""
        self.left.check_alignment(backend)
        self.right.check_alignment(backend)

    @property
    def desyncs(self) -> Tuple[int, int]:
        return self.left.desyncs, self.right.desyncs

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()
        self.left_velocity.reset()
        self.right_velocity.reset()


__all__ = [
    "TRANSITIONS",
    "EncoderPair",
    "EncoderSample",
    "QuadratureDecoder",
    "VelocityEstimator",
]
