"""How long the planner takes to turn what the camera saw into a path.

Two numbers, because they answer different questions:

``search``
    Wall-clock time inside the A* search, fresh searches only. This is the
    planner's own cost, and what ``max_expansions`` and ``search_radius_m``
    trade against path quality. Reused plans are left out: they cost a
    line-of-sight sweep, and averaging them in would make a slow search look
    fast.

``total`` (frame to path)
    From the capture of the camera frame behind the scan -- or, for a plan
    made only because the goal moved, the moment the goal was set -- to the
    moment the path built from it was published. It includes everything the robot actually waits for:
    queueing behind the previous cycle, the ``max_plan_rate_hz`` limit, fusing
    the scan into the map, and the search. Reused plans count here, because a
    re-referenced path is still the robot's answer to that frame.

Both are kept over a sliding window of recent plans, so a single slow search
shows up as a raised maximum without dragging the typical figure with it.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .types import PathSnapshot

#: How far back the summary looks, in seconds.
WINDOW_S = 10.0
#: Points of history handed out for a sparkline.
HISTORY = 120


def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank p95: with few samples it is the maximum, which is the
    # honest answer -- there is not enough data to say anything gentler.
    rank = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "last": values[-1],
        "mean": sum(values) / len(values),
        "p95": ordered[rank],
        "max": ordered[-1],
    }


class PlanLatency:
    """Thread-safe record of recent plans. ``record`` from the planner thread,
    ``summary`` from anywhere."""

    def __init__(self, window_s: float = WINDOW_S, history: int = HISTORY,
                 clock=time.monotonic):
        self.window_s = window_s
        self.clock = clock
        self._lock = threading.Lock()
        # (published at, search ms or None, total ms or None, reused)
        self._samples: Deque[Tuple[float, Optional[float], Optional[float], bool]] = deque()
        self._history: Deque[Tuple[Optional[float], Optional[float]]] = deque(maxlen=history)

    def record(self, snapshot: PathSnapshot) -> None:
        if snapshot.seq <= 0 or not snapshot.poses:
            return                       # idle / at-goal / no-path: no path to time
        search = None if snapshot.reused else snapshot.plan_time_ms
        total = None
        if snapshot.source_stamp > 0.0 and snapshot.stamp >= snapshot.source_stamp:
            total = (snapshot.stamp - snapshot.source_stamp) * 1e3
        with self._lock:
            self._samples.append((snapshot.stamp, search, total, snapshot.reused))
            self._history.append((search, total))
            self._trim(self.clock())

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._history.clear()

    def _trim(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.window_s:
            self._samples.popleft()

    def summary(self) -> Dict[str, Any]:
        """Everything a status panel needs, as plain JSON-able values."""
        now = self.clock()
        with self._lock:
            self._trim(now)
            samples = list(self._samples)
            history = list(self._history)
        searches = [s for _t, s, _x, _r in samples if s is not None]
        totals = [x for _t, _s, x, _r in samples if x is not None]
        span = (now - samples[0][0]) if len(samples) > 1 else 0.0
        rate = (len(samples) - 1) / span if span > 0 else 0.0
        return {
            "search_ms": _stats(searches),
            "total_ms": _stats(totals),
            "plans": len(samples),
            "rate_hz": rate,
            "reused_fraction": (sum(1 for *_x, r in samples if r) / len(samples)
                                if samples else 0.0),
            "window_s": self.window_s,
            "history": [[s, x] for s, x in history],
        }
