from __future__ import annotations

import importlib.util
import math
import os
import platform
import threading
import time
import traceback
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from carrybot_drive.controller import DriveState
from carrybot_drive.follower import PathFollower
from carrybot_drive.odometry import Pose as DrivePose
from carrybot_nav import PathPlanner, PlanStatus
from carrybot_nav import Pose as NavPose
from carrybot_vision.follow import FollowState
from carrybot_vision.pipeline import VisionPipeline
from carrybot_vision.types import TargetObservation, VisionConfig
from tools.simui.session import COST_LUT, apply_params

MODES = ("manual", "goto", "follow")
MAX_TRAIL = 3000


# --------------------------------------------------------------------------- #
# Settings the interface can change
# --------------------------------------------------------------------------- #


@dataclass
class LiveParams:
    #: Speed caps for every mode, including the autonomous ones. Deliberately
    #: low by default: the first time a robot moves under a new interface is
    #: not the time to find out what 0.45 m/s feels like against a shin.
    max_linear: float = 0.25
    max_angular: float = 1.2
    lookahead: float = 0.35

    # follow-me
    standoff: float = 1.20
    predict: float = 1.20
    pursue: float = 0.75
    #: Keep following if the interface goes quiet. Off by default; see the
    #: module docstring for why it exists at all.
    follow_offline: bool = False

    # head
    head_enabled: bool = True
    head_max_angle: float = 60.0
    head_survey: bool = True
    head_body_assist: bool = True

    # planner
    heuristic_weight: float = 2.0
    unknown_cost: float = 80.0
    allow_unknown: bool = True

    # safety timing, seconds; not exposed as sliders on purpose
    client_timeout: float = 1.0
    teleop_timeout: float = 0.15
    vision_timeout: float = 1.0

    @classmethod
    def from_config(cls, robot) -> "LiveParams":
        """The interface's starting values, taken from ``carrybot.toml``.

        The sliders adjust these for the session; restarting goes back to the
        file. Keeping separate defaults here, as this class once did, meant a
        standoff set in the file was quietly replaced by the interface's own.
        """
        return cls(
            max_linear=robot.live.max_linear,
            max_angular=robot.live.max_angular,
            lookahead=robot.follower.lookahead,
            standoff=robot.follow.standoff,
            predict=robot.follow.predict_horizon,
            pursue=robot.follow.pursue_fraction,
            follow_offline=robot.live.follow_offline,
            head_enabled=robot.head.enabled,
            head_max_angle=math.degrees(robot.head.max_angle),
            head_survey=robot.head.survey,
            head_body_assist=robot.head.body_assist,
            heuristic_weight=robot.planner.heuristic_weight,
            unknown_cost=robot.planner.unknown_cost,
            allow_unknown=robot.planner.allow_unknown,
            client_timeout=robot.live.client_timeout,
            teleop_timeout=robot.live.teleop_timeout,
            vision_timeout=robot.live.vision_timeout,
        )

    def apply(self, raw: Dict[str, Any]) -> None:
        safe = {k: v for k, v in raw.items()
                if k not in ("client_timeout", "teleop_timeout", "vision_timeout")}
        apply_params(self, safe)
        self.max_linear = min(max(0.0, self.max_linear), 1.0)
        self.max_angular = min(max(0.0, self.max_angular), 4.0)


# --------------------------------------------------------------------------- #
# Stand-ins for missing hardware
# --------------------------------------------------------------------------- #


class DryDrive:
    """A drive that goes nowhere.

    Records what it was asked to do, so the interface can show the command the
    robot *would* have sent, and reports a pose that never changes -- which is
    the truth for a camera sitting on a desk.
    """

    def __init__(self) -> None:
        self.commanded: Tuple[float, float] = (0.0, 0.0)
        self.estopped = False
        self.commands = 0
        self._pose = DrivePose()

    def start(self) -> "DryDrive":
        return self

    def set_velocity(self, linear: float, angular: float) -> None:
        self.commands += 1
        if not self.estopped:
            self.commanded = (float(linear), float(angular))

    def stop(self) -> None:
        self.commanded = (0.0, 0.0)

    def hold(self) -> None:
        self.commanded = (0.0, 0.0)

    def emergency_stop(self) -> None:
        self.estopped = True
        self.commanded = (0.0, 0.0)

    def clear_emergency_stop(self) -> None:
        self.estopped = False

    def reset_odometry(self, pose=None) -> None:
        self._pose = pose or DrivePose()

    def state(self) -> DriveState:
        return DriveState(stamp=time.monotonic(), pose=self._pose,
                          emergency_stopped=self.estopped)

    def close(self) -> None:
        self.commanded = (0.0, 0.0)


class NullTracker:
    """Stands in for the person tracker when YOLO is not installed.

    The rest of the pipeline -- depth, obstacle mapping, the planner -- is
    worth having without it, so its absence disables follow-me rather than
    the whole interface.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.frames = 0
        self.sightings = 0
        self.last_reason = reason

    def update(self, image, depth, camera, stamp=None, detections=None) -> TargetObservation:
        self.frames += 1
        return TargetObservation(stamp=stamp or time.monotonic(), valid=False)

    def reset_lock(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# The real motors, and whether they can be reached
# --------------------------------------------------------------------------- #


class HardwareError(RuntimeError):
    """The motors could not be brought up."""


class HardwareLink:
    """How to reach the robot's motors and servo, and whether it looks possible.

    ``preflight`` is cheap and side-effect free, so the interface can show its
    answer continuously. It cannot prove the pins are free -- only claiming
    them does that -- so ``open`` can still fail, and says why when it does.
    """

    def __init__(self, config, servo: bool = False):
        self.config = config
        self.servo_wanted = bool(servo)

    @staticmethod
    def board_model() -> str:
        try:
            with open("/proc/device-tree/model", "rb") as handle:
                return handle.read().decode("ascii", "replace").strip("\x00 \n")
        except OSError:
            return ""

    def describe(self) -> str:
        model = self.board_model()
        node = f"/dev/gpiochip{self.config.drive.chip}"
        where = model or platform.system()
        return f"{where}, {node}" + (", pan servo" if self.servo_wanted else ", no servo")

    def preflight(self) -> Tuple[List[str], List[str]]:
        """``(blockers, warnings)``. Any blocker keeps the motors off."""
        blockers: List[str] = []
        warnings: List[str] = []
        d = self.config.drive
        node = f"/dev/gpiochip{d.chip}"
        system = platform.system()
        if system != "Linux":
            blockers.append(
                f"No motor hardware on this machine: it is running {system}. The "
                "motors are driven from the robot's Raspberry Pi -- run the "
                "interface there.")
        else:
            model = self.board_model()
            if not os.path.exists(node):
                blockers.append(
                    f"GPIO hardware not detected: there is no {node}. "
                    + ("Check [drive] chip in carrybot.toml." if model
                       else "This does not look like a Raspberry Pi."))
            elif not os.access(node, os.R_OK | os.W_OK):
                blockers.append(
                    f"No permission to use {node}. Add this user to the gpio "
                    "group (sudo usermod -aG gpio $USER), then log in again.")
            if model and "raspberry pi" not in model.lower():
                warnings.append(f"This board calls itself {model!r}, not a Raspberry Pi.")
        if importlib.util.find_spec("lgpio") is None:
            blockers.append("The lgpio library is not installed "
                            "(on the Pi: sudo apt install python3-lgpio).")

        from carrybot_drive.config import DriveConfig

        # Each one on its own: measuring the wheel does not make the gear ratio
        # right, and a check that only fired when all of them were untouched
        # went quiet the moment any one was set.
        stock = DriveConfig()
        unmeasured = [name for name in ("gear_ratio", "wheel_diameter_m", "track_width_m")
                      if getattr(d, name) == getattr(stock, name)]
        if unmeasured:
            warnings.append(
                f"Still at the placeholder default: {', '.join(unmeasured)} -- so "
                "distances and turns will be off. Set them in [drive], then run "
                "the drive console's c command.")
        if d.left_gains.kv == 0.0 or d.right_gains.kv == 0.0:
            warnings.append("Motor feedforward is uncalibrated (kv = 0), so speed "
                            "control will be sluggish (drive console, f).")
        return blockers, warnings

    def open(self):
        """Claim the GPIO and start the drive. Returns ``(drive, servo or None)``."""
        try:
            from carrybot_drive.backend import LgpioBackend

            backend = LgpioBackend(self.config.drive.chip)
        except Exception as exc:
            raise HardwareError(f"could not open the GPIO: {exc}") from exc
        try:
            from carrybot_drive import DriveController

            drive = DriveController(self.config.drive, backend).start()
        except Exception as exc:
            try:
                backend.close()
            except Exception:
                pass
            raise HardwareError(f"the drive would not start (pins in use by "
                                f"another program?): {exc}") from exc
        servo = None
        if self.servo_wanted:
            try:
                from carrybot_drive.servo import PanServo

                servo = PanServo(backend, self.config.servo)
            except Exception as exc:
                drive.close()
                raise HardwareError(f"the servo would not start: {exc}") from exc
        return drive, servo


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class LiveSession:
    """Owns the camera pipeline, the planner, the drive and the control loop.

    Exposes the same surface as :class:`~tools.simui.session.SimSession` --
    ``command``, ``telemetry``, ``costmap_rgb``, ``close`` -- plus the camera
    images, so the same server can host either.
    """

    live = True

    def __init__(
        self,
        source,
        drive,
        *,
        drive_kind: str = "dry",
        servo=None,
        tracker=None,
        tracker_note: str = "",
        vision_config: Optional[VisionConfig] = None,
        config=None,
        camera_label: str = "",
        sim_backend=None,
        hardware: Optional[HardwareLink] = None,
        hardware_note: str = "",
        clock: Callable[[], float] = time.monotonic,
        start: bool = True,
        control_rate: float = 20.0,
        log_dir: Optional[str] = None,
    ) -> None:
        from carrybot_config import RobotConfig

        self.config = config if config is not None else RobotConfig()
        if vision_config is not None:
            self.config.vision = vision_config
            self.config.camera = vision_config.camera
        self.params = LiveParams.from_config(self.config)
        self.clock = clock
        self.drive = drive
        self.drive_kind = drive_kind
        self.servo = servo
        self.camera_label = camera_label
        self._sim_backend = sim_backend
        self.control_dt = 1.0 / control_rate

        self.vision_cfg = self.config.vision
        self.nav_cfg = replace(self.config.planner,
                               resolution=self.vision_cfg.grid_resolution)
        self.planner = PathPlanner(self.nav_cfg).start()
        self.follower = PathFollower(self.config.follower)
        self._servo_limit = servo.cfg.limit if servo is not None else None
        #: How to reach the motors, if this session may. None when it cannot
        #: (a recording, simulated motors), with the reason in hardware_note.
        self.hardware = hardware
        self.hardware_note = hardware_note or (
            "" if hardware is not None else "motor control is not available in this session")
        self._hardware_failure = ""
        self._hardware_status: Dict[str, Any] = {}
        self._hardware_checked = -math.inf

        self.follow_available = tracker is not None
        self.follow_note = "" if tracker is not None else (
            tracker_note or "no person tracker")
        self.head_available = servo is not None

        self.pipeline = VisionPipeline(
            source,
            config=self.vision_cfg,
            planner=self.planner,
            tracker=tracker if tracker is not None else NullTracker(self.follow_note),
            pose_provider=self._pose_xyz,
            servo=servo,
        )
        self.pipeline.drive_goals = False

        self._lock = threading.Lock()
        self._cmd_lock = threading.RLock()
        self.armed = False
        self.estopped = False
        self.mode = "manual"
        self.disarm_reason = "starts disarmed"
        self._teleop = (0.0, 0.0, -math.inf)
        #: The last autonomous command after the acceleration limit.
        self._ramped: Tuple[float, float] = (0.0, 0.0)
        #: Turning to face the person while not driving; see _face.
        self._facing = False
        self._face_seq = -1
        self._face_heading = 0.0
        self._face_cmd = 0.0
        #: Odometry heading of the direction the person was last seen in, and
        #: which way to keep turning once that is reached and they are still
        #: not in view (+1 left, -1 right). None: nobody seen in this follow.
        self._last_seen_dir: Optional[float] = None
        self._backing = False
        #: The follow goal in the odometry frame, for driving straight at it
        #: when there is no planned path; and the frame it came from.
        self._direct_goal: Optional[Tuple[float, float]] = None
        self._direct_seq = -1
        #: Range calibration against the person: (known metres, raw full-
        #: resolution disparity) pairs, and what the last one concluded.
        self._range_samples: List[Tuple[float, float]] = []
        self._range_note = ""
        #: The floor as recent frames saw it, (offset m, slope rad), for the
        #: mounting check; and the status seq last added.
        from collections import deque as _deque
        self._floor_fits = _deque(maxlen=self.FLOOR_FRAMES)
        self._floor_seq = -1
        self._search_turn = 1.0
        self._searching = False
        #: (clock time, odometry heading) for the last couple of seconds.
        from collections import deque

        self._headings = deque(maxlen=60)
        #: Where armed driving is logged, one row per control tick.
        self._log = None
        self._log_dir = log_dir
        #: Newest teleop sequence number seen from each page. A command that
        #: arrives after a newer one is dropped: requests overtake each other
        #: on the way in, and a stale "go" landing after the "stop" it
        #: preceded would drive the robot on for a whole teleop_timeout.
        self._teleop_seq: Dict[Optional[str], int] = {}
        #: When each page was last heard from, by the id it sends. None is
        #: "no id" -- a caller that does not identify itself, like the tests.
        self._seen: Dict[Optional[str], float] = {None: clock()}
        #: The page that armed the robot. Only its contact keeps it armed and
        #: only it may drive; every other page can watch, STOP and disarm.
        self._controller: Optional[str] = None
        self._goal: Optional[Tuple[float, float]] = None
        self._commanded = (0.0, 0.0)
        self._wanted = (0.0, 0.0)
        self._note = ""
        self._trail: List[Tuple[float, float]] = []
        self._costmap_version = 0
        self._ticks = 0
        self._fps = 0.0
        self._fps_frames = 0
        self._fps_since = clock()
        #: Per-stage vision timings, ms, smoothed; see VisionStatus.timings.
        self._stage_ms: Dict[str, float] = {}
        self._stage_seq = 0
        self._telemetry: Dict[str, Any] = {}
        self._camera_cache: Tuple[int, Optional[bytes]] = (-1, None)
        self._depth_cache: Tuple[int, Optional[bytes]] = (-1, None)
        self.error: Optional[str] = None

        self._apply_params()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if start:
            self.pipeline.start()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="carrybot-live")
            self._thread.start()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Stop everything, motors first."""
        self._stop.set()
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None
        try:
            self.drive.stop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.pipeline.stop()
        except Exception:
            pass
        try:
            self.planner.stop()
        except Exception:
            pass
        # The servo shares the drive's GPIO handle on hardware, and closing the
        # drive closes that handle -- so the servo lets go first.
        if self.servo is not None:
            try:
                self.servo.close()
            except Exception:
                pass
        try:
            self.drive.close()
        except Exception:
            pass
        if self._sim_backend is not None:
            try:
                self._sim_backend.stop_thread()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Commands from the interface
    # ------------------------------------------------------------------ #

    #: What a page that is not driving may still do while another page is:
    #: everything that makes the robot safer, nothing that makes it move.
    VIEWER_ACTIONS = frozenset({"heartbeat", "estop", "disarm", "clear_estop"})

    def touch(self, client: Optional[str] = None) -> None:
        """Record that a page is still there."""
        now = self.clock()
        self._seen[client] = now
        if len(self._seen) > 64:            # pages come and go; forget old ones
            for key, when in list(self._seen.items()):
                if key is not None and key != self._controller and now - when > 60:
                    del self._seen[key]

    @property
    def _last_client(self) -> float:
        """When the page in control -- the one whose silence disarms -- was last heard.

        Every open page used to count. A tab left open on a PC then kept the
        robot armed after the phone driving it had dropped off the Wi-Fi,
        which is exactly the case the timeout exists for.
        """
        return self._seen.get(self._controller, -math.inf)

    def command(self, action: str, payload: Dict[str, Any],
                client: Optional[str] = None) -> Dict[str, Any]:
        self.touch(client)
        with self._cmd_lock:
            viewer_ok = action in self.VIEWER_ACTIONS or (
                action == "hardware" and not payload.get("enable"))
            if action == "teleop" and self._stale_teleop(client, payload):
                result = None             # superseded in flight; not an error
            elif self.armed and client != self._controller and not viewer_ok:
                result = ("the robot is being driven from another page. You can "
                          "press STOP or Disarm here; to drive from this page, "
                          "disarm it and arm it again here.")
            else:
                try:
                    result = self._command(action, payload)
                    if action == "arm" and result is None:
                        self._controller = client
                except (KeyError, TypeError, ValueError) as exc:
                    result = f"bad {action!r} request: {exc}"
        if isinstance(result, str):
            return {"ok": False, "error": result}
        return {"ok": True, "params": vars(self.params)}

    def _stale_teleop(self, client: Optional[str], payload: Dict[str, Any]) -> bool:
        seq = payload.get("seq")
        if not isinstance(seq, (int, float)) or isinstance(seq, bool):
            return False                 # no number, no ordering to enforce
        last = self._teleop_seq.get(client)
        if last is not None and seq <= last:
            return True
        self._teleop_seq[client] = int(seq)
        if len(self._teleop_seq) > 64:
            for key in list(self._teleop_seq):
                if key not in self._seen:
                    del self._teleop_seq[key]
        return False

    def _halt(self) -> None:
        """Stop now. Brake rather than ask the loop to wind down.

        ``set_velocity(0, 0)`` leaves the velocity loop to bring the wheels
        down at whatever rate its gains and duty slew allow -- a good part of a
        second with feedforward uncalibrated, travelling all the while. Manual
        driving wants the robot to stop where it was let go.
        """
        hold = getattr(self.drive, "hold", None)
        (hold or self.drive.stop)()

    def _command(self, action: str, payload: Dict[str, Any]) -> Optional[str]:
        now = self.clock()
        if action == "heartbeat":
            return None
        if action == "arm":
            if self.estopped:
                return "clear the emergency stop first"
            if self.mode == "follow" and self._follow_block():
                return self._follow_block()
            self.armed = True
            self.disarm_reason = ""
            return None
        if action == "disarm":
            self._disarm("disarmed from the interface")
            return None
        if action == "estop":
            self.estopped = True
            self.armed = False
            self.disarm_reason = "emergency stop"
            self._teleop = (0.0, 0.0, -math.inf)
            self.drive.emergency_stop()
            return None
        if action == "clear_estop":
            self.drive.clear_emergency_stop()
            self.estopped = False
            self.disarm_reason = "emergency stop cleared; arm again to move"
            return None
        if action == "mode":
            return self._set_mode(str(payload["mode"]))
        if action == "perception":
            return self._perception(payload)
        if action == "range_cal":
            return self._range_cal(payload)
        if action == "floor_apply":
            return self._floor_apply()
        if action == "camera_mount":
            return self._camera_mount(payload)
        if action == "teleop":
            lin = max(-1.0, min(1.0, float(payload.get("linear", 0.0))))
            ang = max(-1.0, min(1.0, float(payload.get("angular", 0.0))))
            stop = abs(lin) < 1e-3 and abs(ang) < 1e-3
            self._teleop = (0.0, 0.0, -math.inf) if stop else (lin, ang, now)
            # Act on it here rather than at the next control tick, up to 50 ms
            # away. Under the command lock, like the tick, so the two cannot
            # interleave.
            if self.mode == "manual" and self.armed and not self.estopped:
                if stop:
                    self._halt()
                    self._commanded = (0.0, 0.0)
                else:
                    p = self.params
                    linear, angular = lin * p.max_linear, ang * p.max_angular
                    self.drive.set_velocity(linear, angular)
                    self._commanded = (linear, angular)
                self._wanted = self._commanded
            return None
        if action == "goal":
            x, y = float(payload["x"]), float(payload["y"])
            if self.mode != "goto":
                err = self._set_mode("goto")
                if err:
                    return err
            self._goal = (x, y)
            self._send_goal(x, y)
            return None
        if action == "clear_goal":
            self._goal = None
            self.planner.clear_goal()
            return None
        if action == "head":
            if not self.head_available:
                return "no pan servo fitted (start with --servo on the robot)"
            angle = payload.get("angle")
            self.pipeline.head_override = (
                None if angle is None else math.radians(float(angle)))
            return None
        if action == "tune":
            self.params.apply(payload.get("params", {}))
            self._apply_params()
            return None
        if action == "hardware":
            if payload.get("enable"):
                return self._enable_hardware(bool(payload.get("confirm")))
            self._disable_hardware("motors switched off from the interface")
            return None
        if action == "reset":
            if self.armed:
                return "disarm before resetting the map and odometry"
            self._reset_map()
            return None
        return f"unknown action {action!r}"

    # ------------------------------------------------------------------ #
    # Switching the real motors on and off
    # ------------------------------------------------------------------ #

    @property
    def hardware_active(self) -> bool:
        return self.drive_kind == "hardware"

    def hardware_check(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Everything standing between this session and the real motors."""
        now = self.clock() if now is None else now
        blockers: List[str] = []
        warnings: List[str] = []
        detail = ""
        if self.hardware is None:
            blockers.append(self.hardware_note)
        else:
            detail = self.hardware.describe()
            b, w = self.hardware.preflight()
            blockers += b
            warnings += w
        if self.estopped:
            blockers.append("The emergency stop is latched: clear it first.")
        status = self.pipeline.status
        if status.seq == 0:
            blockers.append("The camera has not delivered a frame yet: you would "
                            "be driving blind.")
        elif now - status.stamp > 2.0:
            blockers.append(f"No camera frame for {now - status.stamp:.0f} s: you "
                            "would be driving blind.")
        if self._hardware_failure:
            warnings.append(f"Last attempt failed: {self._hardware_failure}")
        cam = self.vision_cfg.camera
        if not (cam.disparity_offset or cam.row_offset):
            warnings.append("Depth is uncalibrated, so the map's distances are "
                            "provisional. Manual driving is fine; trust the "
                            "autonomous modes less.")
        if self.pipeline.eye_test_running:
            warnings.append("The lenses have moved: the eye test is re-measuring "
                            "them, and Follow me and Go to hold still until it "
                            "finishes.")
        elif not self.pipeline.stereo.geometry_trusted:
            warnings.append("The lenses have moved since calibration: follow-me "
                            "will hold still.")
        warnings += list(getattr(self.config, "warnings", []))
        return dict(active=self.hardware_active, available=self.hardware is not None,
                    blockers=blockers, warnings=warnings, detail=detail)

    def _enable_hardware(self, confirmed: bool) -> Optional[str]:
        if self.hardware_active:
            return None
        if not confirmed:
            return "switching the motors on has to be confirmed"
        check = self.hardware_check()
        if check["blockers"]:
            # Checked again here rather than trusting the page: the page's copy
            # can be a second old, and a second is long enough to unplug a cable.
            return "motors stay off: " + " ".join(check["blockers"])
        try:
            drive, servo = self.hardware.open()
        except HardwareError as exc:
            self._hardware_failure = str(exc)
            return f"motors stay off: {exc}"
        self._hardware_failure = ""
        self._disarm("motors switched on; arm to move")
        self.disarm_reason = "motors switched on; arm to move"
        old = self.drive
        self.drive = drive
        self.drive_kind = "hardware"
        self._attach_servo(servo)
        try:
            old.close()
        except Exception:
            pass
        self._hardware_checked = -math.inf
        return None

    def _disable_hardware(self, reason: str) -> None:
        if not self.hardware_active:
            return
        self._disarm(reason)
        self.disarm_reason = reason
        servo, drive = self.servo, self.drive
        self._attach_servo(None)
        # The servo shares the drive's GPIO handle, so it lets go first.
        if servo is not None:
            try:
                servo.close()
            except Exception:
                pass
        self.drive = DryDrive()
        self.drive_kind = "dry"
        try:
            drive.close()
        except Exception as exc:
            self.error = f"closing the drive: {exc!r}"
        self._hardware_checked = -math.inf

    def _attach_servo(self, servo) -> None:
        self.servo = servo
        self.pipeline.servo = servo
        self.pipeline.head_override = None
        self.head_available = servo is not None
        self._servo_limit = servo.cfg.limit if servo is not None else None
        self._apply_params()

    def _disarm(self, reason: str) -> None:
        if self.armed:
            self.disarm_reason = reason
        self.armed = False
        self._controller = None
        self._teleop = (0.0, 0.0, -math.inf)
        try:
            self.drive.stop()
        except Exception:
            pass

    #: Facing the person while not driving: start turning beyond FACE_START
    #: off-centre, stop inside FACE_STOP. The gap keeps it from hunting.
    FACE_START = math.radians(5.0)
    FACE_STOP = math.radians(2.0)
    FACE_GAIN = 4.0                       # rad/s per rad of bearing
    FACE_MAX = 2.5                        # rad/s, and never above the page's limit
    #: Never turn faster than can be stopped in the angle left, decelerating
    #: at this fraction of the drive's angular_accel -- the ramp every
    #: autonomous command goes through. Asking for more just overshoots: the
    #: ramp cannot reverse a 1.2 rad/s turn in under 0.6 s, and at that
    #: speed that is 20 degrees past the person, then back again.
    FACE_DECEL_FRACTION = 0.7
    #: The wheels and the ramp answer this late; the error is projected
    #: forward over it at the turn rate already measured.
    FACE_LAG = 0.15
    #: Too close: back away when nearer than standoff minus BACK_START, stop
    #: at standoff minus BACK_STOP, at BACK_GAIN m/s per metre, at most
    #: BACK_MAX m/s. Nothing watches behind the robot, so slowly.
    BACK_START = 0.30
    BACK_STOP = 0.10
    BACK_GAIN = 0.8
    BACK_MAX = 0.15
    FACE_MAX_AGE = 1.0                    # s: older than this, the person is out of sight
    #: Out of sight: turn back to where they were last seen at up to
    #: FACE_MAX, then keep turning the way they went at SEARCH_RATE until the
    #: camera finds them again. Slow enough that a person crosses the view
    #: in two or three frames at 3 frames a second.
    SEARCH_RATE = 1.0

    def _face(self, status, heading: float, p, omega: float = 0.0) -> float:
        """Turn on the spot towards the person when there is nothing to drive.

        At the standoff the follower stops driving, and without this the robot
        simply watched a person walk round it until they left the frame. The
        sighting is half a second old at 2 frames a second, so the bearing is
        brought up to date with the turn made since: a person who has not moved
        moves the other way across the view by exactly what the robot turns.
        """
        cmd = status.command
        if (cmd.state is FollowState.SEARCHING
                or cmd.age > self.FACE_MAX_AGE or status.seq == 0):
            self._facing = False
            return self._search(heading, p)
        self._searching = False
        if status.seq != self._face_seq:
            # The heading the bearing is relative to. The follower has already
            # moved the sighting from the capture pose into the robot's frame
            # at the moment it worked the command out, so the turn to add is
            # the one made since *then*. Counting from the capture instead
            # added the turn made during processing a second time: the robot
            # thought a person it was turning towards had swung across to the
            # other side, stopped short, and -- once they left the view --
            # searched for them the wrong way.
            self._face_seq = status.seq
            anchor = getattr(status, "command_stamp", 0.0) or status.stamp
            self._face_heading = self._heading_at(anchor, heading)
        # Bearing is positive to the right; turning left (positive) by d moves
        # a still person d further right.
        turned = math.atan2(math.sin(heading - self._face_heading),
                            math.cos(heading - self._face_heading))
        bearing = cmd.target_bearing + turned
        # Remember where they are, in odometry terms, and which way they are
        # heading out of view: the side of centre they are on.
        self._last_seen_dir = heading - bearing
        if abs(bearing) > math.radians(2.0):
            self._search_turn = -1.0 if bearing > 0 else 1.0
        # Turning left is positive, so the error is minus the bearing; and
        # by the time a new command takes hold, the turn already under way
        # will have eaten into it.
        error = -bearing - omega * self.FACE_LAG
        if self._facing and abs(error) < self.FACE_STOP:
            self._facing = False
        elif not self._facing and abs(error) > self.FACE_START:
            self._facing = True
        if not self._facing:
            return 0.0
        decel = self.FACE_DECEL_FRACTION * self.config.drive.angular_accel
        speed = min(self.FACE_MAX, p.max_angular, self.FACE_GAIN * abs(error),
                    math.sqrt(2.0 * decel * abs(error)))
        return math.copysign(speed, error)

    def _back_off(self, status, p) -> float:
        """Reverse, slowly, when the person has come inside the standoff."""
        cmd = status.command
        standoff = self.pipeline.follower.cfg.standoff
        fresh = (cmd.state is FollowState.HOLDING and cmd.age <= 0.8
                 and cmd.target_range > 0.0)
        if not fresh:
            self._backing = False
            return 0.0
        short = standoff - cmd.target_range
        if self._backing and short < self.BACK_STOP:
            self._backing = False
        elif not self._backing and short > self.BACK_START:
            self._backing = True
        if not self._backing:
            return 0.0
        return -min(self.BACK_MAX, p.max_linear, self.BACK_GAIN * short)

    def _note_direct_goal(self, status, pose) -> None:
        """Pin each new follow goal in the odometry frame, for _direct."""
        if status.seq == self._direct_seq:
            return
        self._direct_seq = status.seq
        cmd = status.command
        if cmd.goal is None or cmd.stop:
            self._direct_goal = None
            return
        gx, gy = cmd.goal
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        self._direct_goal = (pose.x + gx * c - gy * s, pose.y + gx * s + gy * c)

    def _direct(self, pose, state) -> Tuple[float, float, bool]:
        """Drive straight at the follow goal when the planner has no path.

        The planner says NO_PATH for reasons that do not stop a person
        walking there -- most often obstacles mapped earlier that are no
        longer being refreshed with terrain mapping off -- and the robot
        stood watching them leave. With terrain mapping on, the map is live
        and a refusal is believed; with it off there is nothing better to
        go on than the straight line the person just walked.
        """
        if self.pipeline.map_terrain or self._direct_goal is None:
            return 0.0, 0.0, False
        wx, wy = self._direct_goal
        dx, dy = wx - pose.x, wy - pose.y
        c, s = math.cos(-pose.theta), math.sin(-pose.theta)
        gx, gy = dx * c - dy * s, dx * s + dy * c
        path = [DrivePose(0.0, 0.0, 0.0), DrivePose(gx, gy, math.atan2(gy, gx))]
        result = self.follower.compute(path, state.angular_velocity)
        if result.remaining <= self.follower.cfg.arrival_tolerance:
            return 0.0, 0.0, False
        return result.linear, result.angular, True

    def _search(self, heading: float, p) -> float:
        """Out of sight: back to where they were last seen, then keep looking.

        Never gives up while Follow me is on -- the person asked for that. It
        turns at the search rate, round and round if need be, the way they
        were last heading, until the camera finds them again; switching mode
        or disarming stops it, as always.
        """
        if self._last_seen_dir is None:
            return 0.0                    # nobody seen yet: nothing to look for
        limit = min(self.FACE_MAX, p.max_angular)
        search = min(self.SEARCH_RATE, p.max_angular)
        error = math.atan2(math.sin(self._last_seen_dir - heading),
                           math.cos(self._last_seen_dir - heading))
        if not self._searching and error * self._search_turn > self.FACE_STOP:
            # Where they went is still ahead in the direction they were
            # going: get there fast, and carry on the same way after. Once
            # it is reached -- or already behind, because the robot was
            # turning after them -- it is a steady search that way.
            return max(-limit, min(limit, self.FACE_GAIN * error
                                   + math.copysign(search, error)))
        self._searching = True
        return self._search_turn * search

    def _heading_at(self, when: float, fallback: float) -> float:
        """Odometry heading at an earlier moment, from the tick history."""
        best = fallback
        for stamp, theta in self._headings:
            if stamp > when:
                break
            best = theta
        return best

    def _ramp(self, linear: float, angular: float) -> Tuple[float, float]:
        """Limit how fast the autonomous modes change speed.

        Go to and Follow me recompute their command from every new plan and
        frame, and those arrive in jumps: a fresh path, a person found or lost,
        a goal reached. Passed straight through, each jump is a lurch -- and a
        jump to zero is a full brake, then a start from standstill against the
        motor's deadband. That reads as stuttering. Limited to the drive's
        own acceleration figures, a brief zero is a dip, not a stop. Manual
        driving is not ramped: letting go stops at once.
        """
        d = self.config.drive
        dt = self.control_dt
        last_lin, last_ang = self._ramped

        def toward(old, new, step):
            return old + max(-step, min(step, new - old))

        lin = toward(last_lin, linear, d.linear_accel * dt)
        ang = toward(last_ang, angular, d.angular_accel * dt)
        self._ramped = (lin, ang)
        return lin, ang

    LOG_COLUMNS = (
        "t", "mode", "armed", "follow_state", "follow_reason", "seen",
        "range_m", "bearing_deg", "vision_fps", "vision_age_ms", "plan",
        "path_m", "plan_age_ms", "wanted_lin", "wanted_ang", "cmd_lin", "cmd_ang",
        "meas_lin", "meas_ang", "l_target", "r_target", "l_speed", "r_speed",
        "l_duty", "r_duty", "motor_mode", "fault", "note",
        "l_enc_err", "r_enc_err", "l_desync", "r_desync", "loop_overruns",
        "terrain", "people", "face_ang", "l_edge_lag_ms", "r_edge_lag_ms",
        "head_yaw_deg", "cam_bearing_deg", "odom_theta_deg",
        "obstacle_cells", "floor_cm", "floor_pitch_deg",
    )

    def _log_tick(self, now: float, state, status, snapshot_status: str) -> None:
        """One CSV row per control tick while armed. Never stops the robot."""
        if not self._log_dir:
            return
        try:
            if not self.armed:
                if self._log is not None:
                    self._log.close()
                    self._log = None
                return
            if self._log is None:
                import csv
                import datetime
                from pathlib import Path

                folder = Path(self._log_dir)
                folder.mkdir(parents=True, exist_ok=True)
                name = datetime.datetime.now().strftime("drive-%Y%m%d-%H%M%S.csv")
                handle = open(folder / name, "w", newline="")
                self._log = handle
                self._log_writer = csv.writer(handle)
                self._log_writer.writerow(self.LOG_COLUMNS)
                self._log_started = now
                self.log_path = str(folder / name)
            obs = status.observation
            cmd = status.command
            snap = self.planner.latest()
            seen = bool(obs is not None and obs.valid)
            row = (
                f"{now - self._log_started:.3f}", self.mode, int(self.armed),
                cmd.state.value, cmd.reason, int(seen),
                f"{obs.distance:.3f}" if seen else "",
                f"{math.degrees(obs.bearing):.1f}" if seen else "",
                f"{self._fps:.1f}",
                f"{(now - status.stamp) * 1e3:.0f}" if status.seq else "",
                snapshot_status, f"{snap.length_m:.2f}",
                f"{(time.monotonic() - snap.stamp) * 1e3:.0f}" if snap.seq else "",
                f"{self._wanted[0]:.3f}", f"{self._wanted[1]:.3f}",
                f"{self._commanded[0]:.3f}", f"{self._commanded[1]:.3f}",
                f"{state.linear_velocity:.3f}", f"{state.angular_velocity:.3f}",
                f"{state.left_target:.2f}", f"{state.right_target:.2f}",
                f"{state.left_speed:.2f}", f"{state.right_speed:.2f}",
                f"{state.left_duty:.1f}", f"{state.right_duty:.1f}",
                state.motor_mode, getattr(state, "fault", ""), self._note,
                state.left_errors, state.right_errors,
                state.left_desyncs, state.right_desyncs, state.loop_overruns,
                int(self.pipeline.map_terrain), int(self.pipeline.detect_people),
                f"{self._face_cmd:.3f}",
                f"{getattr(state, 'left_edge_lag_ms', 0.0):.1f}",
                f"{getattr(state, 'right_edge_lag_ms', 0.0):.1f}",
                f"{math.degrees(status.head_yaw):.1f}",
                # The bearing the lens itself saw, before the head angle was
                # taken out: separates the robot turning from the head turning.
                f"{math.degrees(obs.bearing + status.head_yaw):.1f}" if seen else "",
                f"{math.degrees(state.pose.theta):.2f}",
                *self._floor_log(status),
            )
            self._log_writer.writerow(row)
            if self._ticks % 20 == 0:
                self._log.flush()
        except Exception as exc:          # a full disk must not stop the robot
            self.error = f"driving log stopped: {exc}"
            self._log_dir = None

    def _plan_status(self) -> str:
        try:
            return self.planner.latest().status.value
        except Exception:
            return ""

    def _follow_block(self) -> str:
        """Why follow-me cannot run right now, or "" if it can."""
        if not self.follow_available:
            return self.follow_note
        if not self.pipeline.detect_people:
            return "person recognition is switched off"
        return ""

    def _perception(self, payload: Dict[str, Any]) -> Optional[str]:
        """The two vision toggles. Terrain off keeps follow-me working: stereo
        then runs on the people only, to range them."""
        if "people" in payload:
            on = bool(payload["people"])
            if not on and self.mode == "follow":
                return ("Follow me needs person recognition: switch to another "
                        "mode first")
            self.pipeline.detect_people = on
            if not on:
                reset = getattr(self.pipeline.tracker, "reset_lock", None)
                if reset is not None:
                    reset()
        if "terrain" in payload:
            on = bool(payload["terrain"])
            self.pipeline.map_terrain = on
            # What is already mapped stops being kept up to date, so it stops
            # blocking: costly to cross, but crossable.
            self.planner.set_obstacles_soft(not on)
        return None

    def _range_cal(self, payload: Dict[str, Any]) -> Optional[str]:
        """Calibrate ranging with the person standing at a known distance.

        One distance fixes ``disparity_offset`` with the configured focal
        length. Two or more at least half a metre apart fix both. The values
        are used at once and reported for copying into ``[camera]``.
        """
        cam = self.vision_cfg.camera
        if payload.get("clear"):
            self._range_samples = []
            self._range_note = ""
            return None
        z = float(payload["distance"])
        if not 0.3 <= z <= 8.0:
            return "give the distance in metres, between 0.3 and 8"
        disparity = self.pipeline.measure_target_disparity()
        if disparity is None:
            return ("could not measure: stand still, facing the camera, in view "
                    "and well lit, until the page shows you tracked -- then try again")
        self._range_samples.append((z, disparity))
        samples = self._range_samples
        zs = [q for q, _d in samples]
        if max(zs) - min(zs) >= 0.5:
            inv = np.array([1.0 / q for q in zs])
            obs = np.array([d for _q, d in samples])
            A = np.stack([inv, np.ones_like(inv)], axis=1)
            (slope, offset), *_ = np.linalg.lstsq(A, obs, rcond=None)
            if slope <= 0:
                return ("those readings say nearer is further -- clear them and "
                        "measure the distances again")
            cam.focal_px = float(slope / cam.baseline)
            cam.disparity_offset = float(offset)
            how = f"from {len(samples)} distances"
        else:
            cam.disparity_offset = float(disparity - cam.focal_px * cam.baseline / z)
            how = "from one distance (add a second, 1 m or more away, to fit the focal length too)"
        reset = getattr(self.pipeline.tracker, "reset_lock", None)
        if reset is not None:
            reset()                       # the old ranges are in the old units
        self._range_note = (
            f"In use now, {how}: focal_px = {cam.focal_px:.1f}, "
            f"disparity_offset = {cam.disparity_offset:.1f}. Copy both into "
            "[camera] in carrybot.toml to keep them.")
        return None

    #: Frames the floor check takes its median over, and how far off the
    #: floor may read before it is reported: past these, bare floor starts
    #: turning into obstacles (ground_height is 6 cm) or holes.
    FLOOR_FRAMES = 30
    FLOOR_HEIGHT_TOLERANCE = 0.025
    FLOOR_PITCH_TOLERANCE = math.radians(1.5)

    def _note_floor(self, status) -> None:
        if status.seq == self._floor_seq:
            return
        self._floor_seq = status.seq
        stats = status.stats
        if stats is None or not math.isfinite(getattr(stats, "floor_offset", math.nan)):
            return
        self._floor_fits.append((stats.floor_offset, stats.floor_slope))

    def _floor_estimate(self):
        """(height error m, pitch error rad, frames), or None without enough."""
        if len(self._floor_fits) < self.FLOOR_FRAMES // 2:
            return None
        a = float(np.median([f[0] for f in self._floor_fits]))
        b = float(np.median([f[1] for f in self._floor_fits]))
        return a, b, len(self._floor_fits)

    def _floor_check(self) -> Tuple[str, bool]:
        """What the floor says about the camera's mounting, and whether it is off."""
        est = self._floor_estimate()
        cam = self.vision_cfg.camera
        if est is None:
            return ("Floor check: needs terrain mapping on and bare floor in "
                    "view for a few seconds.", False)
        a, b, n = est
        height = cam.mount_height - a
        pitch = cam.mount_pitch + b
        off = (abs(a) > self.FLOOR_HEIGHT_TOLERANCE
               or abs(b) > self.FLOOR_PITCH_TOLERANCE)
        at_two = a + 2.0 * b
        text = (f"Floor check ({n} frames): the floor reads {a * 100:+.1f} cm at the "
                f"camera and {at_two * 100:+.1f} cm 2 m out, where it should read 0. ")
        if off:
            text += (f"The mounting is off, and floor further than "
                     f"{self._floor_reach(a, b)} is being mapped as obstacles or holes. "
                     f"It fits mount_height = {height:.3f} and mount_pitch_deg = "
                     f"{math.degrees(pitch):.1f} (now {cam.mount_height:.3f} and "
                     f"{math.degrees(cam.mount_pitch):.1f}).")
        else:
            text += "The mounting is right."
        return text, off

    def _floor_reach(self, a: float, b: float) -> str:
        """Where the misread floor crosses ground_height, as words."""
        g = self.vision_cfg.ground_height
        if a >= g:
            return "anywhere"
        if b <= 1e-6:
            return "nowhere ahead" if a > -g else "anywhere"
        return f"{(g - a) / b:.1f} m"

    def _floor_apply(self) -> Optional[str]:
        est = self._floor_estimate()
        if est is None:
            return "no floor measured yet: terrain mapping on, bare floor in view, a few seconds"
        a, b, _n = est
        cam = self.vision_cfg.camera
        cam.mount_height = float(cam.mount_height - a)
        cam.mount_pitch = float(cam.mount_pitch + b)
        self._floor_fits.clear()
        self._range_note = (
            f"Floor applied: mount_height = {cam.mount_height:.3f}, mount_pitch_deg = "
            f"{math.degrees(cam.mount_pitch):.1f}. Copy both into [camera] in "
            "carrybot.toml to keep them.")
        return None

    def _camera_mount(self, payload: Dict[str, Any]) -> Optional[str]:
        """Set the camera's pitch and height by hand, from the page."""
        cam = self.vision_cfg.camera
        if "pitch_deg" in payload:
            pitch = float(payload["pitch_deg"])
            if not -45.0 <= pitch <= 45.0:
                return "the camera angle must be between -45 and 45 degrees"
            cam.mount_pitch = math.radians(pitch)
        if "height" in payload:
            height = float(payload["height"])
            if not 0.02 <= height <= 2.0:
                return "the camera height must be between 0.02 and 2 m"
            cam.mount_height = height
        self._floor_fits.clear()          # measured against the old values
        self._range_note = (
            f"Camera set to {math.degrees(cam.mount_pitch):.1f} deg "
            f"({'down' if cam.mount_pitch > 0 else 'up' if cam.mount_pitch < 0 else 'level'}) "
            f"at {cam.mount_height:.3f} m. Copy mount_pitch_deg and mount_height "
            "into [camera] in carrybot.toml to keep them. Obstacles already "
            "mapped with the old angle fade within the obstacle lifetime, or "
            "press Reset map.")
        return None

    def _floor_log(self, status) -> Tuple[str, str, str]:
        stats = status.stats
        cells = str(stats.occupied_cells) if stats is not None else ""
        est = self._floor_estimate()
        if est is None:
            return cells, "", ""
        return cells, f"{est[0] * 100:.1f}", f"{math.degrees(est[1]):.2f}"

    def _range_pinned(self, status) -> str:
        """A range stuck at the closest the search can see: offset wrong."""
        obs = status.observation
        if obs is None or not obs.valid or not math.isfinite(obs.distance):
            return ""
        cam = self.pipeline.stereo.scaled_camera
        near = cam.focal_px * cam.baseline / max(1, self.vision_cfg.num_disparities - 1)
        if abs(obs.distance - near) < 0.02:
            return (f"The person reads {obs.distance:.2f} m, the closest this "
                    "disparity offset can measure: their real disparity is outside "
                    "the search, so every range is wrong. Calibrate below.")
        return ""

    def _set_mode(self, mode: str) -> Optional[str]:
        if mode not in MODES:
            return f"unknown mode {mode!r}"
        if mode == "follow" and self._follow_block():
            return self._follow_block()
        if mode == self.mode:
            return None
        # Leaving a mode takes its goal with it. A goal left behind is a robot
        # driving somewhere nobody asked it to go any more.
        if self.mode == "follow":
            self.pipeline.drive_goals = False
            self.pipeline.release_goal()
        if self.mode == "goto":
            self._goal = None
            self.planner.clear_goal()
        self._teleop = (0.0, 0.0, -math.inf)
        self.mode = mode
        if mode == "follow":
            self.pipeline.follower.reset()
            self.pipeline.drive_goals = True
            self._last_seen_dir = None
            self._searching = False
        return None

    def _send_goal(self, x: float, y: float) -> None:
        """Planner goals are robot-relative; the map is in the odometry frame."""
        pose = self.drive.state().pose
        dx, dy = x - pose.x, y - pose.y
        c, s = math.cos(-pose.theta), math.sin(-pose.theta)
        self.planner.set_goal(dx * c - dy * s, dx * s + dy * c)

    def _reset_map(self) -> None:
        self.drive.reset_odometry()
        old = self.planner
        self.planner = PathPlanner(self.nav_cfg).start()
        if not self.pipeline.map_terrain:
            self.planner.set_obstacles_soft(True)
        self.pipeline.planner = self.planner
        self.pipeline.release_goal()
        try:
            old.stop()
        except Exception:
            pass
        self._goal = None
        self._trail.clear()

    def _apply_params(self) -> None:
        # Everything the interface does not expose comes from the config file;
        # the sliders only replace the fields they are named after.
        p = self.params
        base = self.config
        self.follower = PathFollower(replace(
            base.follower, lookahead=p.lookahead,
            max_linear=min(p.max_linear, base.follower.max_linear),
            max_angular=min(p.max_angular, base.follower.max_angular),
        ))
        self.pipeline.follower.cfg = replace(
            base.follow,
            standoff=p.standoff,
            max_range=max(p.standoff + 0.5, base.follow.max_range),
            predict_horizon=p.predict,
            pursue_fraction=p.pursue,
        )
        self.pipeline.head.cfg = replace(
            base.head,
            enabled=p.head_enabled and self.head_available,
            max_angle=math.radians(p.head_max_angle),
            survey=p.head_survey,
            body_assist=p.head_body_assist,
        )
        if self.servo is not None:
            # The slider can narrow the servo's travel, never widen it past
            # the mechanical limit in [servo].
            self.servo.cfg.limit = min(self._servo_limit,
                                       math.radians(max(1.0, p.head_max_angle)))
        cfg = self.nav_cfg
        cfg.heuristic_weight = p.heuristic_weight
        cfg.unknown_cost = p.unknown_cost
        cfg.allow_unknown = p.allow_unknown

    # ------------------------------------------------------------------ #
    # The control loop
    # ------------------------------------------------------------------ #

    def _pose_xyz(self) -> Tuple[float, float, float]:
        pose = self.drive.state().pose
        return pose.x, pose.y, pose.theta

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                self.error = traceback.format_exc()
                try:
                    self.drive.stop()
                except Exception:
                    pass
            next_tick += self.control_dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def tick(self, now: Optional[float] = None) -> None:
        """One control cycle. Public so tests can drive it without threads."""
        now = self.clock() if now is None else now
        p = self.params
        status = self.pipeline.status
        self._note_floor(status)

        # Under the command lock from the first read of the drive to the last
        # write, because the motors can be switched on or off between ticks and
        # a tick must not read one drive and command another.
        with self._cmd_lock:
            state = self.drive.state()
            self._headings.append((now, state.pose.theta))
            if state.emergency_stopped and not self.estopped:
                # The drive latched itself -- its runaway or stall guard. Show
                # it as the STOP it is, with the drive's reason, so it needs
                # the same deliberate Clear and re-arm.
                self.estopped = True
                self.armed = False
                self._teleop = (0.0, 0.0, -math.inf)
                self.disarm_reason = getattr(state, "fault", "") or "the drive stopped itself"
            pose = state.pose
            believed = NavPose(pose.x, pose.y, pose.theta)
            self.planner.update_odometry_absolute(believed)
            contact = now - self._last_client
            lenient = self.mode == "follow" and p.follow_offline
            if self.armed and contact > p.client_timeout and not lenient:
                self._disarm(f"lost contact with the interface for {contact:.1f} s")

            linear = angular = 0.0
            note = ""
            halt = False
            if self.mode == "manual":
                lin, ang, stamp = self._teleop
                if now - stamp <= p.teleop_timeout:
                    linear, angular = lin * p.max_linear, ang * p.max_angular
                else:
                    halt = True          # let go, or the stop never came: brake
            else:
                snapshot = self.planner.latest()
                planned = snapshot.is_drivable and not (
                    # Exploring towards a frontier is for reaching a clicked
                    # point through unknown ground. Following a person, it
                    # drove off to the edge of the map instead of to them.
                    self.mode == "follow" and snapshot.status is PlanStatus.FRONTIER)
                if planned:
                    relative = [DrivePose(q.x, q.y, q.theta)
                                for q in snapshot.in_frame_of(believed)]
                    result = self.follower.compute(relative, state.angular_velocity)
                    linear, angular = result.linear, result.angular
                    goal = getattr(snapshot, "goal", None)
                    if (result.remaining <= self.follower.cfg.arrival_tolerance
                            and (goal is None or math.isnan(goal.theta))):
                        # At the end of a path whose goal asked for no
                        # heading. Turning to the last segment's direction
                        # means nothing -- and in Follow me it fought the
                        # turn to face the person: one turned right to line
                        # up with a stale path, the other left to face them,
                        # each tick, and the robot jittered on the spot.
                        linear = angular = 0.0
                self._face_cmd = 0.0
                if self.pipeline.eye_test_running:
                    # Re-measuring the lenses: stand still, no facing turns
                    # either, until it has a value it trusts.
                    linear = angular = 0.0
                    note = "the eyes have moved -- running the eye test, holding still"
                    self._facing = False
                elif self.mode == "follow":
                    stalled = status.seq == 0 or (now - status.stamp) > p.vision_timeout
                    self._note_direct_goal(status, pose)
                    if stalled:
                        linear = angular = 0.0
                        note = "vision has stalled -- holding"
                    else:
                        if not planned:
                            linear, angular, direct = self._direct(pose, state)
                            if direct:
                                note = "no usable path -- driving straight at them"
                        if not self.pipeline.stereo.geometry_trusted:
                            # Ranges are untrustworthy, so do not drive -- but
                            # the bearing does not depend on them, so still
                            # turn to keep the person in view.
                            linear = angular = 0.0
                            note = "the camera's lenses have moved -- recalibrate"
                        if abs(linear) < 1e-6 and abs(angular) < 1e-6:
                            self._face_cmd = self._face(status, pose.theta, p,
                                                        state.angular_velocity)
                            angular = self._face_cmd or status.body_rate
                            if self._face_cmd and status.command.age > self.FACE_MAX_AGE:
                                note = note or "out of sight -- turning to find them"
                            if self.pipeline.stereo.geometry_trusted:
                                linear = self._back_off(status, p)
                                if linear:
                                    note = note or "too close -- backing away"
                else:
                    self._facing = False

            linear = max(-p.max_linear, min(p.max_linear, linear))
            angular = max(-p.max_angular, min(p.max_angular, angular))
            self._wanted = (linear, angular)
            self._note = note

            if self.estopped:
                self._commanded = (0.0, 0.0)
            elif self.armed and halt:
                self._halt()
                self._commanded = (0.0, 0.0)
            elif self.armed:
                if self.mode != "manual":
                    linear, angular = self._ramp(linear, angular)
                self.drive.set_velocity(linear, angular)
                self._commanded = (linear, angular)
            else:
                self.drive.stop()
                self._commanded = (0.0, 0.0)
            if not self.armed or self.estopped or self.mode == "manual":
                self._ramped = self._commanded
            self._log_tick(now, state, status, snapshot_status=self._plan_status())

        self._ticks += 1
        if not self._trail or math.hypot(pose.x - self._trail[-1][0],
                                         pose.y - self._trail[-1][1]) > 0.02:
            self._trail.append((pose.x, pose.y))
            if len(self._trail) > MAX_TRAIL:
                del self._trail[: len(self._trail) - MAX_TRAIL]
        if self._ticks % 4 == 0:
            self._costmap_version += 1
        self._update_fps(now)
        if now - self._hardware_checked >= 1.0:
            self._hardware_checked = now
            self._hardware_status = self.hardware_check(now)
        self._note_stages(status)
        self._publish(now, state, status)

    def _note_stages(self, status) -> None:
        """Smooth each new frame's stage timings for display."""
        if not status.timings or status.seq == self._stage_seq:
            return
        self._stage_seq = status.seq
        for name, ms in status.timings:
            old = self._stage_ms.get(name)
            self._stage_ms[name] = ms if old is None else old + 0.2 * (ms - old)

    def _update_fps(self, now: float) -> None:
        span = now - self._fps_since
        if span >= 1.0:
            frames = self.pipeline.frames
            self._fps = (frames - self._fps_frames) / span
            self._fps_frames = frames
            self._fps_since = now

    # ------------------------------------------------------------------ #
    # Reading out
    # ------------------------------------------------------------------ #

    def _publish(self, now: float, state: DriveState, status) -> None:
        pose = state.pose
        snapshot = self.planner.latest()
        path: List[Tuple[float, float]] = []
        if snapshot.is_drivable:
            ref = snapshot.reference_pose
            c, s = math.cos(ref.theta), math.sin(ref.theta)
            path = [(ref.x + q.x * c - q.y * s, ref.y + q.x * s + q.y * c)
                    for q in snapshot.poses]

        command = status.command
        target = None
        if command.state is not FollowState.SEARCHING and command.target_range > 0:
            # target_bearing is positive to the right; the map is anticlockwise.
            angle = pose.theta - command.target_bearing
            target = (pose.x + command.target_range * math.cos(angle),
                      pose.y + command.target_range * math.sin(angle))
        goal = self._goal
        if self.mode == "follow" and command.goal is not None:
            gx, gy = command.goal
            c, s = math.cos(pose.theta), math.sin(pose.theta)
            goal = (pose.x + gx * c - gy * s, pose.y + gx * s + gy * c)

        obs = status.observation
        cam = self.vision_cfg.camera
        costmap = self.planner.costmap
        known = float((costmap.raw != 254).mean()) if costmap.raw.size else 0.0
        stats = status.stats
        registration = (self.pipeline.stereo.monitor.describe()
                        if self.pipeline.stereo.monitor is not None else "not monitored")
        eye_test = self.pipeline.describe_eye_test()
        if eye_test:
            registration = f"{registration}; {eye_test}"
        vision_error = self.pipeline.last_error
        data = dict(
            live=True,
            seq=self._ticks,
            drive_kind=self.drive_kind,
            hardware=self._hardware_status,
            camera=self.camera_label,
            armed=self.armed,
            estopped=self.estopped,
            disarm_reason=self.disarm_reason,
            mode=self.mode,
            note=self._note,
            range_cal=self._range_note,
            floor_check=self._floor_check()[0],
            mount_pitch_deg=round(math.degrees(self.vision_cfg.camera.mount_pitch), 2),
            mount_height=round(self.vision_cfg.camera.mount_height, 3),
            floor_off=self._floor_check()[1],
            range_pinned=self._range_pinned(status),
            range_samples=[[round(z, 2), round(d, 1)] for z, d in self._range_samples],
            follow_available=not self._follow_block(),
            follow_note=self._follow_block(),
            terrain_on=self.pipeline.map_terrain,
            log_path=getattr(self, "log_path", None),
            people_on=self.pipeline.detect_people,
            tracker_installed=self.follow_available,
            parallel=self.vision_cfg.parallel_detect,
            head_available=self.head_available,
            head_manual=(None if self.pipeline.head_override is None
                         else math.degrees(self.pipeline.head_override)),
            client_age=now - self._last_client,
            pose=(pose.x, pose.y, pose.theta),
            trail=self._trail[-600:],
            goal=goal,
            path=path,
            plan_status=snapshot.status.value,
            plan_ms=snapshot.plan_time_ms,
            planning=self._planning(snapshot),
            wanted=self._wanted,
            commanded=self._commanded,
            measured=(state.linear_velocity, state.angular_velocity),
            wheels=(state.left_speed, state.right_speed),
            duty=(state.left_duty, state.right_duty),
            counts=(state.left_counts, state.right_counts),
            encoder_errors=state.left_errors + state.right_errors,
            desyncs=state.left_desyncs + state.right_desyncs,
            overruns=state.loop_overruns,
            edge_lag=(getattr(state, "left_edge_lag_ms", 0.0),
                      getattr(state, "right_edge_lag_ms", 0.0)),
            watchdog=state.watchdog_tripped,
            motor_mode=state.motor_mode,
            follow_state=command.state.value,
            follow_reason=command.reason,
            target=target,
            target_range=command.target_range,
            target_bearing=command.target_bearing,
            target_speed=command.target_speed,
            predicted=command.predicted,
            seen=bool(obs is not None and obs.valid),
            head_yaw=status.head_yaw,
            head_mode=status.head.mode.value,
            head_reason=status.head.reason,
            vision_seq=status.seq,
            vision_fps=self._fps,
            vision_latency=status.latency,
            vision_stages=[[name, round(ms, 1)] for name, ms in self._stage_ms.items()],
            capture_scale=self.vision_cfg.capture_scale,
            detect_size=self.vision_cfg.detect_size,
            vision_age=(now - status.stamp) if status.seq else None,
            vision_frames=self.pipeline.frames,
            vision_dropped=self.pipeline.dropped,
            obstacles=stats.occupied_cells if stats else 0,
            vision_error=(vision_error.strip().splitlines()[-1]
                          if vision_error else None),
            depth_calibrated=bool(cam.disparity_offset or cam.row_offset),
            geometry_trusted=self.pipeline.stereo.geometry_trusted,
            registration=registration,
            hfov=cam.hfov,
            sensor_range=self.vision_cfg.max_range,
            costmap_version=self._costmap_version,
            map_origin=costmap.origin,
            map_shape=tuple(costmap.cost.shape),
            known_fraction=known,
            resolution=self.nav_cfg.resolution,
            error=self.error,
        )
        with self._lock:
            self._telemetry = data

    def _planning(self, snapshot) -> Dict[str, Any]:
        """Pathfinding latency: the planner's window, plus how old the path
        being followed right now is.

        Both on the planner's own clock (``time.monotonic``), not the
        session's: plans are stamped there, and a test's virtual clock would
        make every age nonsense.
        """
        summary = self.planner.latency.summary()
        now = time.monotonic()
        current = snapshot.seq > 0 and bool(snapshot.poses)
        summary["path_age_ms"] = (
            (now - snapshot.source_stamp) * 1e3
            if current and snapshot.source_stamp > 0 else None)
        summary["idle"] = not current
        return summary

    def telemetry(self, client: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            data = dict(self._telemetry)
        data["params"] = vars(self.params)
        now = self.clock()
        with self._cmd_lock:
            armed, controller = self.armed, self._controller
            pages = sum(1 for key, when in self._seen.items()
                        if key is not None and now - when < 3.0)
        data["control"] = {
            "you": armed and client == controller,
            "other": armed and client != controller,
            "pages": pages,
        }
        return data

    def costmap_rgb(self) -> np.ndarray:
        return COST_LUT[self.planner.costmap.cost]

    world_rgb = costmap_rgb

    # -- camera images -------------------------------------------------- #

    def camera_jpeg(self) -> Optional[bytes]:
        """The left eye with what the robot thinks it sees drawn on it."""
        left = self.pipeline.last_left
        if left is None:
            return None
        seq = self.pipeline.frames
        if self._camera_cache[0] == seq:
            return self._camera_cache[1]
        import cv2

        scale = 640.0 / left.shape[1]
        image = cv2.resize(left, (640, int(round(left.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        status = self.pipeline.status
        obs = status.observation
        if obs is not None and obs.bbox is not None:
            x1, y1, x2, y2 = (int(round(v * scale)) for v in obs.bbox)
            colour = (255, 170, 60) if obs.valid else (120, 120, 120)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            label = f"{obs.distance:.2f} m" if obs.valid else "?"
            cv2.putText(image, label, (x1 + 3, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        data = buf.tobytes() if ok else None
        self._camera_cache = (seq, data)
        return data

    def depth_jpeg(self) -> Optional[bytes]:
        """Depth as colour: warm is near, cool is far, black is no match."""
        depth = self.pipeline.last_depth
        if depth is None:
            return None
        seq = self.pipeline.frames
        if self._depth_cache[0] == seq:
            return self._depth_cache[1]
        import cv2

        valid = np.isfinite(depth) & (depth > 0)
        near, far = 0.3, max(1.0, self.vision_cfg.max_range)
        closeness = np.zeros(depth.shape, np.float32)
        closeness[valid] = np.clip((far - depth[valid]) / (far - near), 0.0, 1.0)
        image = cv2.applyColorMap((closeness * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[~valid] = 0
        if image.shape[1] != 640:
            image = cv2.resize(image, (640, int(round(image.shape[0] * 640 / image.shape[1]))),
                               interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        data = buf.tobytes() if ok else None
        self._depth_cache = (seq, data)
        return data


# --------------------------------------------------------------------------- #
# Building one from the command line
# --------------------------------------------------------------------------- #


class PacedReplay:
    """Recorded frames, looped, delivered at the rate they were captured.

    Lets the live interface be tried with no camera attached. Without the
    pacing the vision thread would spin through the recording as fast as the
    CPU allows, which looks nothing like the robot.
    """

    def __init__(self, frames, fps: float = 15.0):
        if not frames:
            raise ValueError("the recording has no frames")
        self.frames = frames
        self.period = 1.0 / max(0.5, fps)
        self._i = 0
        self._next = time.monotonic()

    def read(self):
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self.period, time.monotonic() - self.period)
        frame = self.frames[self._i % len(self.frames)]
        self._i += 1
        return frame

    def close(self) -> None:
        pass


def make_tracker(cfg: VisionConfig, weights: str) -> Tuple[Optional[object], str]:
    """The YOLO tracker if it can run here, or None and the reason it cannot.

    The reason names the Python that looked. "I installed it and it is not
    found" is nearly always ultralytics in a venv and the server started by a
    different interpreter -- /usr/bin/python3, sudo, or the boot service -- and
    the path is what tells those apart. A package that is present but will not
    import (usually torch) is reported as that, not as missing.
    """
    import sys

    if importlib.util.find_spec("ultralytics") is None:
        return None, (f"follow-me needs the person tracker, and ultralytics is not "
                      f"installed for {sys.executable}. Install it into that Python, "
                      "or start the server with the venv's python "
                      "(mapping, click-to-go and manual driving work without it)")
    try:
        import ultralytics  # noqa: F401  -- pulls in torch, which is what breaks
    except Exception as exc:              # pragma: no cover - depends on the install
        return None, (f"ultralytics is installed for {sys.executable} but will not "
                      f"import: {type(exc).__name__}: {exc}")
    from carrybot_vision.tracker import PersonTracker

    return PersonTracker(cfg, weights=weights), ""


def build_live_session(
    drive: str = "dry",
    servo: Optional[bool] = None,
    device: Optional[str] = None,
    width: int = 0,
    height: int = 0,
    weights: Optional[str] = None,
    no_tracker: bool = False,
    replay: Optional[str] = None,
    config=None,
) -> LiveSession:
    """Open the camera and the chosen drive, and wire them together.

    ``config`` is a loaded :class:`~carrybot_config.RobotConfig`; left out,
    ``carrybot.toml`` is found and loaded as usual. Arguments given here win
    over the file. Whether to drive real motors is only ever an argument.
    """
    if drive not in ("dry", "sim", "hardware"):
        raise ValueError(f"--drive must be dry, sim or hardware, not {drive!r}")
    if replay and drive == "hardware":
        raise ValueError("a recording cannot drive the real robot: the pictures "
                         "would not change as it moved")
    if config is None:
        from carrybot_config import load_config

        config = load_config()
    if servo is None:
        servo = config.live.servo_fitted
    # A fitted servo is driven whenever the motors are switched on, from the
    # command line or from the interface; never with a dry or simulated drive,
    # where a modelled servo would report angles the camera never turned to.
    servo = bool(servo)
    device = device or config.live.camera or None
    weights = weights or config.live.weights

    vision_cfg = config.vision
    if replay:
        from carrybot_vision.recording import describe, load_recording

        from carrybot_vision.pipeline import shrink_frames

        frames, meta = load_recording(replay)
        # Shrunk once, up front, to the size the camera would hand over, so a
        # replay costs what the live camera would.
        frames = shrink_frames(frames, vision_cfg.capture_scale)
        source = PacedReplay(frames, fps=float(meta.get("fps") or 15.0))
        label = f"replaying {replay}: {describe(meta)}"
        if vision_cfg.capture_scale < 1.0:
            label += f", at {vision_cfg.capture_scale:g}x"
    else:
        from carrybot_vision.pipeline import CameraSource

        source = CameraSource(device, width, height, scale=vision_cfg.capture_scale)
        if vision_cfg.threaded_capture:
            from carrybot_vision.pipeline import LatestFrameSource

            source = LatestFrameSource(source, nice=vision_cfg.nice)
        label = source.describe()

    if no_tracker:
        tracker, note = None, "person tracker turned off with --no-tracker"
    else:
        tracker, note = make_tracker(vision_cfg, weights)

    sim_backend = None
    link: Optional[HardwareLink] = None
    note_hw = ""
    if replay:
        note_hw = ("The camera is a recording (--replay): its pictures would not "
                   "change as the robot moved.")
    elif drive == "sim":
        note_hw = "Started with simulated motors (--drive sim)."
    else:
        link = HardwareLink(config, servo=servo)

    body = DryDrive()
    try:
        if drive == "sim":
            from carrybot_drive import DriveController
            from carrybot_drive.backend import SimBackend

            sim_backend = SimBackend(config.drive)
            body = DriveController(config.drive, sim_backend).start()
            sim_backend.start_thread()
    except Exception:
        source.close()
        raise

    session = LiveSession(
        source, body, drive_kind="sim" if drive == "sim" else "dry",
        tracker=tracker, tracker_note=note, camera_label=label,
        sim_backend=sim_backend, config=config,
        hardware=link, hardware_note=note_hw,
        log_dir=config.live.log_dir or None,
    )
    if drive == "hardware":
        # The command line asked for the motors outright. Same path as the
        # interface's switch, minus the camera check (the pipeline has not had
        # a chance to deliver a frame yet), and it still starts disarmed.
        blockers, _warnings = link.preflight()
        if blockers:
            session.close()
            raise HardwareError("motors unavailable: " + " ".join(blockers))
        try:
            drive_obj, pan = link.open()
        except HardwareError:
            session.close()
            raise
        with session._cmd_lock:
            session.drive, session.drive_kind = drive_obj, "hardware"
            session._attach_servo(pan)
    return session


__all__ = ["DryDrive", "HardwareError", "HardwareLink", "LiveParams", "LiveSession",
           "NullTracker", "PacedReplay",
           "build_live_session", "make_tracker"]
