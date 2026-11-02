from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from carrybot_drive import DriveConfig, DriveController, Gains
from carrybot_drive.backend import SimBackend
from carrybot_drive.follower import FollowerConfig, PathFollower
from carrybot_drive.odometry import Pose as DrivePose
from carrybot_drive.odometry import wrap_angle
from carrybot_nav import LETHAL, FREE, PathPlanner, PlannerConfig
from carrybot_nav import Pose as NavPose
from tools.simulate import RayScanner, _footprint_hits, free_cell, make_world
from carrybot_drive.servo import ServoConfig, SimulatedServo
from carrybot_vision.follow import FollowConfig, FollowController, FollowState
from carrybot_vision.head import HeadConfig, HeadController
from carrybot_vision.types import CameraModel, TargetObservation, VisionConfig

MAX_TRAIL = 4000


def apply_params(target: Any, raw: Dict[str, Any]) -> None:
    """Copy form values onto a params object, coercing to each field's type."""
    for key, value in raw.items():
        if key.startswith("_") or not hasattr(target, key):
            continue
        current = getattr(target, key)
        if callable(current):
            continue
        try:
            if isinstance(current, bool):
                setattr(target, key, value if isinstance(value, bool)
                        else str(value).lower() in ("1", "true", "on", "yes"))
            elif isinstance(current, int):
                setattr(target, key, int(float(value)))
            elif isinstance(current, float):
                setattr(target, key, float(value))
            else:
                setattr(target, key, str(value))
        except (TypeError, ValueError):
            pass  # a half-typed number in a text box is not an error


@dataclass
class Params:
    """Everything the browser can change. Names match the form fields."""

    # world
    scenario: str = "open"
    seed: int = 3
    size: int = 160
    sensor_range: float = 2.0
    #: Sensor field of view in degrees. 360 is a lidar; 74.5 is the camera.
    sensor_fov: float = 360.0

    # head (pan servo carrying the sensor)
    head_enabled: bool = False
    #: How far the head may turn either side of straight ahead, degrees.
    head_max_angle: float = 60.0
    #: Look around when there is nothing to track.
    head_survey: bool = True
    #: Let the robot turn on the spot to keep the target in view when the head
    #: alone cannot reach -- the "they walked past me" case.
    head_body_assist: bool = True

    # moving target
    #: Metres per second the drawn target walks its path.
    target_speed: float = 0.8
    #: Radius of the person, metres. They are a real obstacle at this size.
    target_radius: float = 0.22
    #: What happens at the end of the drawn path.
    target_loop: str = "loop"      # loop | pingpong | once
    #: How far behind the target the robot tries to sit, metres.
    target_standoff: float = 1.20
    #: Seconds to carry the target along its last measured velocity after the
    #: tracker drops it. 0 freezes them where they were last seen.
    target_predict: float = 1.20
    #: How much of the way to the last estimated position the robot drives
    #: after properly losing the target, as a fraction. 0 stops it dead.
    target_pursue: float = 0.75
    #: Whether the target blocks the sensor and the robot. Off makes them a
    #: bare goal marker, which is useful for telling "the follow logic is
    #: wrong" apart from "the person is in the way".
    target_solid: bool = True

    # motion
    mode: str = "planner"          # "planner" | "fullstack"
    speed: float = 1.0             # sim-time multiplier, planner mode only
    #: What to do on arrival. The system is meant to run until told to stop,
    #: so "hold" is the only option that parks it.
    on_arrival: str = "hold"       # hold | queue | random | patrol

    # drive realism
    calibrated: bool = True
    wheel_error: float = 1.0
    track_error: float = 1.0
    skew: float = 0.12
    motor_tau: float = 0.12
    motor_deadband: float = 10.0
    no_load_speed: float = 18.0

    # planner tuning
    heuristic_weight: float = 2.0
    unknown_cost: float = 80.0
    turn_cost: float = 1.2
    safety_margin: float = 0.05
    search_radius: float = 4.0
    max_expansions: int = 40000
    allow_unknown: bool = True
    directional_states: bool = False

    # follower
    lookahead: float = 0.35
    max_linear: float = 0.45
    max_angular: float = 2.5

    def apply(self, raw: Dict[str, Any]) -> None:
        apply_params(self, raw)


@dataclass
class Telemetry:
    run_id: int = 0
    seq: int = 0
    running: bool = False
    sim_time: float = 0.0
    wall_time: float = 0.0

    true_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    odom_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal: Tuple[float, float] = (0.0, 0.0)
    start: Tuple[float, float] = (0.0, 0.0)

    path: List[Tuple[float, float]] = field(default_factory=list)
    plan_status: str = "IDLE"
    plan_ms: float = 0.0
    expansions: int = 0
    plan_seq: int = 0
    reused: bool = False

    linear: float = 0.0
    angular: float = 0.0
    left_speed: float = 0.0
    right_speed: float = 0.0
    left_duty: float = 0.0
    right_duty: float = 0.0
    left_counts: int = 0
    right_counts: int = 0

    drift: float = 0.0
    heading_drift: float = 0.0
    distance: float = 0.0
    collisions: int = 0
    encoder_errors: int = 0
    desyncs: int = 0
    overruns: int = 0

    reached: bool = False
    arrivals: int = 0
    queued: int = 0
    queue: List[Tuple[float, float]] = field(default_factory=list)
    known_fraction: float = 0.0

    #: Where the drawn target is now, and the path it walks. None when no
    #: target has been drawn.
    target: Optional[Tuple[float, float]] = None
    target_path: List[Tuple[float, float]] = field(default_factory=list)
    target_range: float = 0.0
    target_bearing: float = 0.0
    follow_state: str = "SEARCHING"
    follow_reason: str = ""
    #: Head angle, radians, positive to the robot's left.
    head_yaw: float = 0.0
    head_mode: str = "CENTRED"
    head_reason: str = ""
    #: Rotation the head is asking the body for, radians.
    body_hint: float = 0.0
    #: ...and how fast to make it, rad/s.
    body_rate: float = 0.0
    #: The target's measured speed over the ground, m/s.
    target_speed: float = 0.0
    #: True when the target's position is extrapolated, not seen.
    predicted: bool = False
    costmap_version: int = 0
    world_version: int = 0
    map_origin: Tuple[float, float] = (0.0, 0.0)
    map_shape: Tuple[int, int] = (0, 0)
    error: Optional[str] = None


class SimSession:
    """Owns the world, the planner, the drive, and the thread stepping them."""

    def __init__(self) -> None:
        self.params = Params()
        self._lock = threading.Lock()
        self._telemetry = Telemetry()
        self._trail_true: List[Tuple[float, float]] = []
        self._trail_odom: List[Tuple[float, float]] = []

        self._world = np.zeros((10, 10), dtype=np.uint8)
        self._world_version = 0
        self._costmap = np.zeros((10, 10), dtype=np.uint8)
        self._costmap_version = 0
        self._map_origin = (0.0, 0.0)

        self._running = False
        self._step_once = False
        self._run_id = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._planner: Optional[PathPlanner] = None
        self._drive: Optional[DriveController] = None
        self._sim: Optional[SimBackend] = None
        self._follower = PathFollower()
        self._scanner: Optional[RayScanner] = None

        self._true = [0.0, 0.0, 0.0]
        self._goal = (0.0, 0.0)
        self._start = (0.0, 0.0)
        self._sim_time = 0.0
        self._wall_start = 0.0
        self._distance = 0.0
        self._collisions = 0
        self._reached = False
        self._arrivals = 0
        self._queue: List[Tuple[float, float]] = []
        self._patrol: List[Tuple[float, float]] = []
        self._patrol_index = 0
        self._last_angles = (0.0, 0.0)
        self._commanded = (0.0, 0.0)

        # The drawn target, and the machinery that follows it. These are the
        # real FollowController and HeadController, not stand-ins: the point of
        # driving them from the UI is to exercise the code that will run on the
        # robot, including its hysteresis and its loss states.
        self._target_path: List[Tuple[float, float]] = []
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_s = 0.0
        self._target_dir = 1
        self._target_cells: Optional[Tuple[int, int, int, int]] = None
        self._target_saved: Optional[np.ndarray] = None
        self._follow = FollowController(FollowConfig())
        self._head = HeadController(HeadConfig(), VisionConfig())
        self._servo = SimulatedServo(ServoConfig())
        self._head_yaw = 0.0
        self._head_reason = ""
        self._head_mode = "CENTRED"
        self._body_hint = 0.0
        self._body_rate = 0.0
        self._goal_from_target = False
        self._follow_reason = ""
        self._target_range = 0.0
        self._target_bearing = 0.0
        self._target_speed = 0.0
        self._predicted = False

        self.build()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def build(self, keep_world: bool = False) -> None:
        """Tear down whatever is running and construct a fresh scenario."""
        self._teardown()

        p = self.params
        rng = np.random.default_rng(p.seed)

        if not keep_world or self._world.shape[0] != p.size:
            if p.scenario == "blank":
                world = np.zeros((p.size, p.size), dtype=np.uint8)
                world[0, :] = world[-1, :] = LETHAL
                world[:, 0] = world[:, -1] = LETHAL
            else:
                world = make_world(p.scenario, p.size, rng)
            self._world = world
            self._world_version += 1
        else:
            rng = np.random.default_rng(p.seed)
            if p.scenario != "blank":
                make_world(p.scenario, p.size, rng)  # keep rng in step

        nav_cfg = PlannerConfig()
        nav_cfg.heuristic_weight = p.heuristic_weight
        nav_cfg.unknown_cost = p.unknown_cost
        nav_cfg.turn_cost = p.turn_cost
        nav_cfg.safety_margin = p.safety_margin
        nav_cfg.search_radius_m = p.search_radius
        nav_cfg.max_expansions = p.max_expansions
        nav_cfg.allow_unknown = p.allow_unknown
        nav_cfg.directional_states = p.directional_states
        self._nav_cfg = nav_cfg

        res = nav_cfg.resolution
        if p.scenario == "trap":
            mid = p.size // 2
            start_rc, goal_rc = (mid, mid - 40), (mid, mid + 40)
        elif p.scenario == "blank":
            start_rc = (p.size // 2, p.size // 4)
            goal_rc = (p.size // 2, 3 * p.size // 4)
        else:
            start_rc = free_cell(self._world, rng)
            goal_rc = free_cell(self._world, rng)
            attempts = 0
            while (abs(goal_rc[0] - start_rc[0]) + abs(goal_rc[1] - start_rc[1])
                   < p.size and attempts < 50):
                goal_rc = free_cell(self._world, rng)
                attempts += 1

        self._start = (start_rc[1] * res, start_rc[0] * res)
        self._goal = (goal_rc[1] * res, goal_rc[0] * res)
        self._true = [self._start[0], self._start[1], 0.0]

        self._scanner = RayScanner(p.sensor_range, res,
                                   hfov=math.radians(p.sensor_fov))
        self._follower = PathFollower(FollowerConfig(
            lookahead=p.lookahead,
            max_linear=p.max_linear,
            max_angular=p.max_angular,
        ))

        self._planner = PathPlanner(nav_cfg).start()

        if p.mode == "fullstack":
            drive_cfg = DriveConfig()
            drive_cfg.max_linear_speed = p.max_linear
            drive_cfg.max_angular_speed = p.max_angular
            if p.calibrated:
                gains = Gains(kv=100.0 / max(1e-6, p.no_load_speed), ks=p.motor_deadband)
                drive_cfg.left_gains = gains
                drive_cfg.right_gains = Gains(**vars(gains))
            self._drive_cfg = drive_cfg
            self._sim = SimBackend(
                drive_cfg,
                no_load_speed=p.no_load_speed,
                deadband=p.motor_deadband,
                tau=p.motor_tau,
                skew=p.skew,
            )
            self._drive = DriveController(drive_cfg, self._sim).start()
            self._sim.start_thread()
            self._last_angles = (self._sim.left.angle, self._sim.right.angle)

        # The head, follower and servo are configured from the same Params as
        # everything else. Building without this leaves them on their
        # constructor defaults, which for the head means "disabled" -- the
        # slider then does nothing and the head never moves, with no error to
        # say why.
        self._apply_head_and_follow()

        self._set_goal_absolute(self._goal)

        self._trail_true = [(self._true[0], self._true[1])]
        self._trail_odom = [(self._true[0], self._true[1])]
        self._sim_time = 0.0
        self._wall_start = time.monotonic()
        self._distance = 0.0
        self._collisions = 0
        self._reached = False
        self._arrivals = 0
        self._queue = []
        self._patrol = [self._start, self._goal]
        self._patrol_index = 1
        self._run_id += 1

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="simui-session")
        self._thread.start()
        self._publish()

    def _teardown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._drive is not None:
            try:
                self._drive.close()
            except Exception:
                pass
            self._drive = None
        if self._sim is not None:
            try:
                self._sim.stop_thread()
            except Exception:
                pass
            self._sim = None
        if self._planner is not None:
            try:
                self._planner.stop()
            except Exception:
                pass
            self._planner = None

    def close(self) -> None:
        self._teardown()

    # ------------------------------------------------------------------ #
    # Commands from the browser
    # ------------------------------------------------------------------ #

    def command(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if action == "run":
            self._running = True
        elif action == "pause":
            self._running = False
            if self._drive is not None:
                self._drive.stop()
        elif action == "step":
            self._step_once = True
        elif action == "reset":
            self.build()
        elif action == "configure":
            self.params.apply(payload.get("params", {}))
            self.build(keep_world=bool(payload.get("keep_world")))
        elif action == "tune":
            # Planner and follower knobs that can change without a rebuild.
            self.params.apply(payload.get("params", {}))
            self._retune()
        elif action == "goal":
            self._goal = (float(payload["x"]), float(payload["y"]))
            self._reached = False
            self._patrol = [self._patrol[0] if self._patrol else self._start, self._goal]
            self._patrol_index = 1
            self._set_goal_absolute(self._goal)
        elif action == "queue":
            self._queue.append((float(payload["x"]), float(payload["y"])))
            if self._reached:
                self._advance_goal()
        elif action == "clear_queue":
            self._queue.clear()
        elif action == "target":
            self._set_target_path(payload.get("points", []))
        elif action == "clear_target":
            self._set_target_path([])
        elif action == "paint":
            self._paint(payload)
        elif action == "randomise":
            self.params.seed = int(np.random.default_rng().integers(0, 10_000))
            self.build()
        else:
            return {"ok": False, "error": f"unknown action {action!r}"}
        return {"ok": True, "params": vars(self.params)}

    def _retune(self) -> None:
        """Apply the knobs that do not need the world rebuilt."""
        p = self.params
        cfg = self._nav_cfg
        cfg.heuristic_weight = p.heuristic_weight
        cfg.unknown_cost = p.unknown_cost
        cfg.turn_cost = p.turn_cost
        cfg.search_radius_m = p.search_radius
        cfg.max_expansions = p.max_expansions
        cfg.allow_unknown = p.allow_unknown
        cfg.directional_states = p.directional_states
        self._follower = PathFollower(FollowerConfig(
            lookahead=p.lookahead,
            max_linear=p.max_linear,
            max_angular=p.max_angular,
        ))
        if self._scanner is not None and (
            abs(self._scanner.range_m - p.sensor_range) > 1e-9
            or abs(math.degrees(self._scanner.hfov) - p.sensor_fov) > 1e-6
        ):
            self._scanner = RayScanner(p.sensor_range, cfg.resolution,
                                       hfov=math.radians(p.sensor_fov))
        self._apply_head_and_follow()

    def _paint(self, payload: Dict[str, Any]) -> None:
        """Draw or erase obstacles in the ground-truth world, live.

        This is the most useful control here: put a wall in front of a moving
        robot and watch the costmap update and the path re-route, which is
        exactly the case that is tedious to construct any other way.
        """
        res = self._nav_cfg.resolution
        cx = int(round(float(payload["x"]) / res))
        cy = int(round(float(payload["y"]) / res))
        radius = max(1, int(round(float(payload.get("radius", 0.15)) / res)))
        value = LETHAL if payload.get("value", "wall") == "wall" else FREE

        h, w = self._world.shape
        r0, r1 = max(0, cy - radius), min(h, cy + radius + 1)
        c0, c1 = max(0, cx - radius), min(w, cx + radius + 1)
        if r1 <= r0 or c1 <= c0:
            return

        rows = np.arange(r0, r1)[:, None]
        cols = np.arange(c0, c1)[None, :]
        mask = (rows - cy) ** 2 + (cols - cx) ** 2 <= radius * radius
        block = self._world[r0:r1, c0:c1]
        block[mask] = value
        self._world_version += 1

    # ------------------------------------------------------------------ #
    # The drawn target
    # ------------------------------------------------------------------ #

    def _apply_head_and_follow(self) -> None:
        """Push the current Params into the head, servo and follower."""
        p = self.params
        self._follow.cfg = FollowConfig(
            standoff=p.target_standoff,
            max_range=max(p.target_standoff + 0.5, p.sensor_range),
            predict_horizon=p.target_predict,
            pursue_fraction=p.target_pursue,
        )
        self._head.cfg = HeadConfig(
            enabled=p.head_enabled,
            max_angle=math.radians(p.head_max_angle),
            survey=p.head_survey,
            body_assist=p.head_body_assist,
        )
        self._head.vision = self._vision_config()
        self._servo.cfg.limit = math.radians(max(1.0, p.head_max_angle))
        if not p.head_enabled:
            self._head_yaw = 0.0
            self._servo.command(0.0)

    def _vision_config(self) -> VisionConfig:
        """A camera model whose cone matches the simulator's sensor."""
        hfov = math.radians(min(179.0, max(1.0, self.params.sensor_fov)))
        width = 1280
        return VisionConfig(
            camera=CameraModel(
                width=width, height=720,
                focal_px=(width / 2.0) / math.tan(hfov / 2.0),
            ),
            grid_resolution=self._nav_cfg.resolution,
            grid_size=161,
        )

    def _set_target_path(self, points) -> None:
        """Adopt a polyline for the target to walk, in world metres.

        Clamped to the world, with a cell of margin. A path drawn past the edge
        puts the target outside the grid, where it gets stamped against the
        boundary wall and every line-of-sight test to it fails -- the robot
        then looks straight at someone it is told it cannot see.
        """
        self._unstamp_target()
        res = self._nav_cfg.resolution
        h, w = self._world.shape
        margin = max(self.params.target_radius, 2 * res)
        lo_x, hi_x = margin, (w - 1) * res - margin
        lo_y, hi_y = margin, (h - 1) * res - margin
        path = [
            (min(hi_x, max(lo_x, float(x))), min(hi_y, max(lo_y, float(y))))
            for x, y in points
        ]
        self._target_path = path
        self._target_s = 0.0
        self._target_dir = 1
        self._target_pos = path[0] if path else None
        self._follow.reset()
        self._head.reset()
        if self._planner is not None:
            # Drawing a path replaces whatever the robot was doing. Leaving the
            # previous goal live sends it off across the room to a waypoint
            # nobody asked about any more, which reads as the follow logic
            # being broken when it has simply not been given the job yet.
            self._planner.clear_goal()
            self._goal_from_target = False
        if path:
            self._goal = path[0]

    def _walk_path(self) -> List[Tuple[float, float]]:
        """The polyline the target actually walks.

        In ``loop`` mode the drawn line is closed back to its first point, so
        the target walks the return leg instead of teleporting across the room
        at the end of every circuit. A teleport looks like a tracking failure,
        and the whole value of this tool is being able to trust what you see.
        """
        path = self._target_path
        if self.params.target_loop == "loop" and len(path) > 2:
            if math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1]) > 1e-6:
                return path + [path[0]]
        return path

    def _path_length(self) -> float:
        walk = self._walk_path()
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(walk, walk[1:])
        )

    def _point_at(self, s: float) -> Tuple[float, float]:
        """Position ``s`` metres along the walked polyline."""
        path = self._walk_path()
        if len(path) == 1:
            return path[0]
        travelled = 0.0
        for a, b in zip(path, path[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg < 1e-9:
                continue
            if travelled + seg >= s:
                t = (s - travelled) / seg
                return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            travelled += seg
        return path[-1]

    def _advance_target(self, dt: float) -> None:
        if len(self._target_path) < 2:
            self._target_pos = self._target_path[0] if self._target_path else None
            return

        total = self._path_length()
        if total < 1e-6:
            self._target_pos = self._target_path[0]
            return

        self._target_s += self._target_dir * self.params.target_speed * dt
        mode = self.params.target_loop
        if mode == "pingpong":
            if self._target_s > total:
                self._target_s = 2.0 * total - self._target_s
                self._target_dir = -1
            elif self._target_s < 0.0:
                self._target_s = -self._target_s
                self._target_dir = 1
        elif mode == "once":
            self._target_s = max(0.0, min(total, self._target_s))
        else:
            self._target_s %= total
        self._target_pos = self._point_at(self._target_s)

    def _unstamp_target(self) -> None:
        """Put back whatever the target was standing on."""
        if self._target_cells is None or self._target_saved is None:
            return
        r0, c0, r1, c1 = self._target_cells
        self._world[r0:r1, c0:c1] = self._target_saved
        self._target_cells = None
        self._target_saved = None

    def _stamp_target(self) -> None:
        """Write the target into the ground-truth world as a real obstacle.

        Into the *world*, not the costmap: the point is that the sensor has to
        discover them like anything else, complete with the shadow they cast
        and the way they occlude what is behind them.
        """
        self._unstamp_target()
        if self._target_pos is None or not self.params.target_solid:
            return

        res = self._nav_cfg.resolution
        radius = max(1, int(round(self.params.target_radius / res)))
        cx = int(round(self._target_pos[0] / res))
        cy = int(round(self._target_pos[1] / res))
        h, w = self._world.shape
        r0, r1 = max(0, cy - radius), min(h, cy + radius + 1)
        c0, c1 = max(0, cx - radius), min(w, cx + radius + 1)
        if r1 <= r0 or c1 <= c0:
            return

        self._target_saved = self._world[r0:r1, c0:c1].copy()
        self._target_cells = (r0, c0, r1, c1)
        rows = np.arange(r0, r1)[:, None]
        cols = np.arange(c0, c1)[None, :]
        mask = (rows - cy) ** 2 + (cols - cx) ** 2 <= radius * radius
        self._world[r0:r1, c0:c1][mask] = LETHAL
        self._world_version += 1

    def _sight_target(self, now: float) -> Optional[TargetObservation]:
        """Can the robot see the target from where it is, with the head where
        it is? Range, cone and line of sight, in that order."""
        if self._target_pos is None:
            return None

        tx, ty = self._target_pos
        dx, dy = tx - self._true[0], ty - self._true[1]
        distance = math.hypot(dx, dy)
        if distance > self.params.sensor_range:
            return None

        # Bearing in the robot frame, positive to the RIGHT to match what the
        # real tracker reports.
        world_bearing = math.atan2(dy, dx)
        bearing = -wrap_angle(world_bearing - self._true[2])

        half_fov = math.radians(self.params.sensor_fov) / 2.0
        if abs(wrap_angle(-bearing - self._head_yaw)) > half_fov:
            return None

        if not self._line_of_sight(self._true[0], self._true[1], tx, ty):
            return None

        return TargetObservation(
            track_id=1, distance=distance, bearing=bearing,
            confidence=0.95, stamp=now, valid=True, depth_samples=200,
            head_yaw=self._head_yaw,
        )

    def _line_of_sight(self, x0, y0, x1, y1) -> bool:
        """Walk the world grid; stop at the first wall that is not the target."""
        res = self._nav_cfg.resolution
        steps = max(2, int(math.hypot(x1 - x0, y1 - y0) / (res * 0.5)))
        h, w = self._world.shape
        skip = self._target_cells
        for i in range(1, steps):
            t = i / steps
            c = int(round((x0 + (x1 - x0) * t) / res))
            r = int(round((y0 + (y1 - y0) * t) / res))
            if not (0 <= r < h and 0 <= c < w):
                return False
            if skip is not None and skip[0] <= r < skip[2] and skip[1] <= c < skip[3]:
                continue           # that is the target itself, not an occluder
            if self._world[r, c] == LETHAL:
                return False
        return True

    def _aim_head(self, observation, state, now: float, command=None) -> None:
        p = self.params
        if not p.head_enabled:
            self._head_yaw = 0.0
            self._head_mode = "CENTRED"
            self._head_reason = "head disabled"
            self._body_hint = 0.0
            self._body_rate = 0.0
            return
        predicted = None
        if command is not None and command.predicted and command.state in (
            FollowState.COASTING, FollowState.PURSUING, FollowState.LOST
        ):
            predicted = command.target_bearing
        aim = self._head.update(
            observation, state, current=self._head_yaw,
            robot_heading=self._true[2], now=now,
            predicted_bearing=predicted,
        )
        self._servo.command(aim.angle, now=now)
        self._head_yaw = self._servo.update(now)
        self._head_mode = aim.mode.value
        self._head_reason = aim.reason
        self._body_hint = aim.body_hint
        self._body_rate = aim.body_rate

    def _apply_follow_goal(self, command) -> None:
        """Turn a FollowCommand into a planner goal, in world terms."""
        if self._planner is None:
            return
        if command.goal is None or command.stop:
            if self._goal_from_target:
                self._planner.clear_goal()
                self._goal_from_target = False
            return
        gx, gy = command.goal
        cos_t, sin_t = math.cos(self._true[2]), math.sin(self._true[2])
        self._goal = (
            self._true[0] + gx * cos_t - gy * sin_t,
            self._true[1] + gx * sin_t + gy * cos_t,
        )
        # Straight through to the planner in robot-relative terms: converting
        # to world and back would round-trip through the odometry error the
        # full-stack mode deliberately introduces.
        self._planner.set_goal(gx, gy)
        self._goal_from_target = True

    def _set_goal_absolute(self, goal: Tuple[float, float]) -> None:
        """The planner takes goals relative to the robot; convert."""
        if self._planner is None:
            return
        # The planner's frame is anchored where the robot started.
        gx = goal[0] - self._start[0]
        gy = goal[1] - self._start[1]
        robot = self._planner.robot_pose()
        dx, dy = gx - robot.x, gy - robot.y
        cos_t, sin_t = math.cos(-robot.theta), math.sin(-robot.theta)
        self._planner.set_goal(dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        dt = 0.05
        next_tick = time.monotonic()
        while not self._stop.is_set():
            stepping = self._step_once
            if self._running or stepping:
                self._step_once = False
                try:
                    self._tick(dt)
                except Exception:
                    with self._lock:
                        self._telemetry.error = traceback.format_exc()
                    self._running = False

            # Planner mode can outrun the clock; full stack cannot, because the
            # drive and motor threads are scheduled against it.
            factor = self.params.speed if self.params.mode == "planner" else 1.0
            next_tick += dt / max(0.1, factor)
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
            self._publish()

    def _tick(self, dt: float) -> None:
        planner = self._planner
        if planner is None or self._scanner is None:
            return

        self._sim_time += dt
        now = self._sim_time

        # 0. walk the drawn target and put them in the world, so the sensor has
        #    to find them the way it will on the robot.
        follow_state = FollowState.SEARCHING
        if self._target_path:
            self._advance_target(dt)
            self._stamp_target()
            self._follow.note_pose(now, self._true[0], self._true[1], self._true[2])
            sighting = self._sight_target(now)
            command = self._follow.update(sighting, now=now)
            follow_state = command.state
            self._follow_reason = command.reason
            self._target_range = command.target_range
            self._target_bearing = command.target_bearing
            self._target_speed = command.target_speed
            self._predicted = command.predicted
            self._head.note_heading(self._true[2], now=now)
            self._head.note_motion(self._commanded[0])
            self._aim_head(sighting, follow_state, now, command)
            self._apply_follow_goal(command)
        else:
            self._head.note_heading(self._true[2], now=now)
            self._head.note_motion(self._commanded[0])
            self._aim_head(None, follow_state, now)

        # 1. sense from where the robot really is, through a head that may be
        #    pointing somewhere other than straight ahead
        pose = NavPose(self._true[0], self._true[1], self._true[2])
        scan = self._scanner.scan(self._world, pose, (0.0, 0.0), head_yaw=self._head_yaw)
        planner.submit_scan(scan)
        if self.params.head_enabled:
            self._head.note_scan(scan, self._true[2], self._head_yaw, now=now)

        # 2. tell the planner where it believes it is
        if self.params.mode == "fullstack" and self._drive is not None:
            state = self._drive.state()
            believed = NavPose(state.pose.x, state.pose.y, state.pose.theta)
        else:
            believed = NavPose(
                self._true[0] - self._start[0],
                self._true[1] - self._start[1],
                self._true[2],
            )
        planner.update_odometry_absolute(believed)

        # 3. follow the plan
        snapshot = planner.latest()
        linear = angular = 0.0
        if snapshot.is_drivable and not self._reached:
            relative = [DrivePose(p.x, p.y, p.theta)
                        for p in snapshot.in_frame_of(believed)]
            measured = (self._drive.state().angular_velocity
                        if self.params.mode == "fullstack" and self._drive else 0.0)
            result = self._follower.compute(relative, measured)
            linear, angular = result.linear, result.angular

        # Nothing to drive, but something to watch. Holding station at the
        # standoff leaves the planner with no goal and the robot with no reason
        # to move -- so when someone walks past, the body never turns and they
        # are simply lost. The head says how much rotation it needs; apply it
        # here, where we know no path is being followed and so nothing is being
        # fought.
        if abs(linear) < 1e-6 and abs(angular) < 1e-6 and self._body_rate:
            # The head already worked out how fast to turn, knowing whether the
            # robot is moving and what the sensor can stand. All that is left
            # here is the chassis limit.
            limit = self.params.max_angular
            angular = max(-limit, min(limit, self._body_rate))
        self._commanded = (linear, angular)

        # 4. move
        if self.params.mode == "fullstack" and self._drive is not None:
            self._drive.set_velocity(linear, angular)
            self._advance_from_wheels()
        else:
            self._advance_ideal(linear, angular, dt)

        # 5. arrived? Arriving is not the end of the run -- the robot is
        # supposed to be ready for the next goal immediately, so the only
        # policy that stops it is "hold".
        at_goal = math.hypot(
            self._goal[0] - self._true[0], self._goal[1] - self._true[1]
        ) <= 0.15
        if self._target_path:
            # Following a person has its own idea of "arrived" -- the standoff,
            # with hysteresis -- and the waypoint policies would fight it.
            self._reached = at_goal
        elif at_goal and not self._reached:
            self._reached = True
            self._arrivals += 1
            if self._drive is not None:
                self._drive.stop()
            self._advance_goal()
        elif not at_goal:
            self._reached = False

        self._costmap = planner.costmap.cost.copy()
        self._map_origin = planner.costmap.origin
        self._costmap_version += 1

    def _advance_goal(self) -> None:
        """Pick the next goal according to the arrival policy."""
        policy = self.params.on_arrival

        if policy == "queue":
            if not self._queue:
                return
            target = self._queue.pop(0)
        elif policy == "random":
            target = self._random_free_point()
            if target is None:
                return
        elif policy == "patrol":
            if len(self._patrol) < 2:
                return
            self._patrol_index = (self._patrol_index + 1) % len(self._patrol)
            target = self._patrol[self._patrol_index]
        else:
            return  # "hold": stay put, still planning, still ready

        self._goal = target
        self._reached = False
        self._set_goal_absolute(target)

    def _random_free_point(self, tries: int = 400) -> Optional[Tuple[float, float]]:
        """Somewhere clear and far enough away to be worth driving to."""
        res = self._nav_cfg.resolution
        margin = max(3, self._nav_cfg.cells(self._nav_cfg.inflation_radius) + 2)
        h, w = self._world.shape
        rng = np.random.default_rng()
        for _ in range(tries):
            row = int(rng.integers(margin, h - margin))
            col = int(rng.integers(margin, w - margin))
            if self._world[row - margin:row + margin, col - margin:col + margin].max() != FREE:
                continue
            point = (col * res, row * res)
            if math.hypot(point[0] - self._true[0], point[1] - self._true[1]) > 1.5:
                return point
        return None

    def _advance_ideal(self, linear: float, angular: float, dt: float) -> None:
        theta = self._true[2]
        candidate = [
            self._true[0] + linear * dt * math.cos(theta),
            self._true[1] + linear * dt * math.sin(theta),
            wrap_angle(theta + angular * dt),
        ]
        self._commit(candidate, abs(linear) * dt)

    def _advance_from_wheels(self) -> None:
        assert self._sim is not None
        p = self.params
        radius = self._drive_cfg.wheel_radius_m * p.wheel_error
        track = self._drive_cfg.track_width_m * p.track_error

        left, right = self._sim.left.angle, self._sim.right.angle
        d_left = (left - self._last_angles[0]) * radius
        d_right = (right - self._last_angles[1]) * radius
        self._last_angles = (left, right)

        centre = 0.5 * (d_left + d_right)
        d_theta = (d_right - d_left) / track
        mid = self._true[2] + 0.5 * d_theta
        candidate = [
            self._true[0] + centre * math.cos(mid),
            self._true[1] + centre * math.sin(mid),
            wrap_angle(self._true[2] + d_theta),
        ]
        self._commit(candidate, abs(centre))

    def _commit(self, candidate: List[float], travelled: float) -> None:
        """Apply the motion unless a wall is in the way.

        Walls stop the body but not the wheels, so nosing into something makes
        the encoders keep counting while the robot goes nowhere -- which is how
        odometry runs away on a real robot.
        """
        hit = _footprint_hits(
            self._world, NavPose(candidate[0], candidate[1], candidate[2]), self._nav_cfg
        )
        if hit:
            self._collisions += 1
            self._true[2] = candidate[2]
        else:
            self._distance += travelled
            self._true = candidate

        self._trail_true.append((self._true[0], self._true[1]))
        if len(self._trail_true) > MAX_TRAIL:
            del self._trail_true[: len(self._trail_true) - MAX_TRAIL]

    # ------------------------------------------------------------------ #
    # Reading out
    # ------------------------------------------------------------------ #

    def _publish(self) -> None:
        planner = self._planner
        if planner is None:
            return
        snapshot = planner.latest()

        if self.params.mode == "fullstack" and self._drive is not None:
            state = self._drive.state()
            odom = (state.pose.x + self._start[0],
                    state.pose.y + self._start[1],
                    state.pose.theta)
            drive_bits = dict(
                left_speed=state.left_speed, right_speed=state.right_speed,
                left_duty=state.left_duty, right_duty=state.right_duty,
                left_counts=state.left_counts, right_counts=state.right_counts,
                linear=state.linear_velocity, angular=state.angular_velocity,
                encoder_errors=state.left_errors + state.right_errors,
                desyncs=state.left_desyncs + state.right_desyncs,
                overruns=state.loop_overruns,
            )
        else:
            odom = (self._true[0], self._true[1], self._true[2])
            drive_bits = {}

        self._trail_odom.append((odom[0], odom[1]))
        if len(self._trail_odom) > MAX_TRAIL:
            del self._trail_odom[: len(self._trail_odom) - MAX_TRAIL]

        path: List[Tuple[float, float]] = []
        if snapshot.is_drivable:
            ref = snapshot.reference_pose
            cos_t, sin_t = math.cos(ref.theta), math.sin(ref.theta)
            for p in snapshot.poses:
                path.append((
                    self._start[0] + ref.x + p.x * cos_t - p.y * sin_t,
                    self._start[1] + ref.y + p.x * sin_t + p.y * cos_t,
                ))

        known = float((planner.costmap.raw != 254).mean()) if planner.costmap.raw.size else 0.0

        with self._lock:
            t = self._telemetry
            t.run_id = self._run_id
            t.seq += 1
            t.running = self._running
            t.sim_time = self._sim_time
            t.wall_time = time.monotonic() - self._wall_start
            t.true_pose = (self._true[0], self._true[1], self._true[2])
            t.odom_pose = odom
            t.goal = self._goal
            t.start = self._start
            t.path = path
            t.plan_status = snapshot.status.value
            t.plan_ms = snapshot.plan_time_ms
            t.expansions = snapshot.expansions
            t.plan_seq = snapshot.seq
            t.reused = snapshot.reused
            t.drift = math.hypot(odom[0] - self._true[0], odom[1] - self._true[1])
            t.heading_drift = abs(wrap_angle(odom[2] - self._true[2]))
            t.distance = self._distance
            t.collisions = self._collisions
            t.reached = self._reached
            t.arrivals = self._arrivals
            t.queued = len(self._queue)
            t.queue = list(self._queue)
            t.known_fraction = known
            t.target = self._target_pos
            t.target_path = list(self._target_path)
            t.follow_state = self._follow.state.value if self._target_path else "SEARCHING"
            t.follow_reason = self._follow_reason
            t.target_range = self._target_range
            t.target_bearing = self._target_bearing
            t.head_yaw = self._head_yaw
            t.head_mode = self._head_mode
            t.head_reason = self._head_reason
            t.body_hint = self._body_hint
            t.body_rate = self._body_rate
            t.target_speed = self._target_speed
            t.predicted = self._predicted
            t.costmap_version = self._costmap_version
            t.world_version = self._world_version
            t.map_origin = self._map_origin
            t.map_shape = tuple(self._costmap.shape)
            for key, value in drive_bits.items():
                setattr(t, key, value)
            if not drive_bits:
                # Planner mode has no drive to measure, so report what the
                # follower asked for -- otherwise the trace is flat and looks
                # like a bug rather than an absence.
                t.linear, t.angular = self._commanded

    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            data = dict(vars(self._telemetry))
        data["trail_true"] = self._trail_true[::2][-1200:]
        data["trail_odom"] = self._trail_odom[::2][-1200:]
        data["params"] = vars(self.params)
        data["mode"] = self.params.mode
        return data

    def world_rgb(self) -> np.ndarray:
        return WORLD_LUT[self._world]

    def costmap_rgb(self) -> np.ndarray:
        return COST_LUT[self._costmap]

    @property
    def resolution(self) -> float:
        return self._nav_cfg.resolution


# --------------------------------------------------------------------------- #
# Palettes, as 256-entry lookup tables so colouring a map is one numpy index.
# --------------------------------------------------------------------------- #

def _build_cost_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for value in range(254):
        # free -> deep slate, rising cost -> teal -> amber
        t = value / 253.0
        if t < 0.25:
            lut[value] = (0x16 + int(t * 40), 0x20 + int(t * 60), 0x2b + int(t * 70))
        elif t < 0.55:
            lut[value] = (0x2b + int((t - 0.25) * 180), 0x4a + int((t - 0.25) * 60), 0x55)
        else:
            lut[value] = (0xa8, 0x63 - int((t - 0.55) * 60), 0x2f)
    lut[254] = (0x0b, 0x0e, 0x12)   # unknown
    lut[255] = (0xd9, 0x48, 0x4b)   # lethal
    return lut


def _build_world_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:] = (0x1b, 0x22, 0x2b)
    lut[255] = (0x5a, 0x65, 0x72)
    return lut


COST_LUT = _build_cost_lut()
WORLD_LUT = _build_world_lut()

__all__ = ["Params", "SimSession", "Telemetry"]
