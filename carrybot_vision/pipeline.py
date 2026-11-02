from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Tuple

import numpy as np

from .camera import CameraDevice
from .costmap import DepthProjector, ProjectionStats, box_mask
from .follow import FollowCommand, FollowConfig, FollowController, FollowState
from .head import HeadCommand, HeadConfig, HeadController, HeadMode
from .stereo import EyeTest, StereoDepth, split_side_by_side
from .tracker import PersonTracker
from .types import TargetObservation, VisionConfig


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #


def yield_to_drive(nice: int, threads: int = 0) -> None:
    """Make the calling thread defer to the motors, and cap the libraries.

    Per-thread on Linux: ``setpriority`` on the thread's own id lowers only
    it, and threads it starts afterwards -- OpenCV's and PyTorch's worker
    pools, created on first use -- inherit the lower priority. Raising a
    priority needs root; lowering one never does. Anything that fails here
    is ignored: a robot that runs a little less smoothly beats one that does
    not start.
    """
    import os

    if nice > 0 and hasattr(os, "setpriority") and hasattr(threading, "get_native_id"):
        try:
            tid = threading.get_native_id()
            current = os.getpriority(os.PRIO_PROCESS, tid)
            os.setpriority(os.PRIO_PROCESS, tid, max(current, nice))
        except (OSError, AttributeError):
            pass
    if threads > 0:
        try:
            import cv2

            cv2.setNumThreads(threads)
        except Exception:
            pass


class FrameSource:
    """Anything that can hand over a side-by-side stereo frame."""

    def read(self) -> Optional[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class CameraSource(FrameSource):
    """The stereo camera, found by what it is rather than by index.

    ``device=None`` runs the identification in :mod:`carrybot_vision.camera`:
    an explicit override, then the camera pinned by ``--probe --pin``, then
    whichever capture device advertises a side-by-side mode. It raises rather
    than falling back to index 0, because a robot navigating on the wrong
    camera is worse than one that refuses to start.

    The delivered mode is read back and checked. A camera that silently gives
    1280x480 when asked for 2560x720 produces depth wrong by exactly the ratio
    I didn't notice, which is a very annoying way to lose an afternoon.
    """

    def __init__(self, device=None, width: int = 0, height: int = 0,
                 fourcc: str = "MJPG", scale: float = 1.0):
        from .camera import VideoMode, open_camera, select_camera

        if isinstance(device, CameraDevice):
            chosen, self.why = device, "supplied directly"
        else:
            chosen, self.why = select_camera(
                override=None if device is None else str(device)
            )
        self.device = chosen
        mode = VideoMode(width, height) if width and height else None
        self.cap, self.mode, self.info = open_camera(
            chosen, mode=mode, fourcc=fourcc)

        #: Full frame size, both eyes together.
        self.width, self.height = self.mode.width, self.mode.height
        #: Frames are handed over at this fraction of the full size.
        self.scale = float(scale)
        #: How they get there: "full", "jpeg" (decoded small) or "resize".
        self.decode = "full"
        if self.scale < 1.0:
            self.decode = "jpeg" if self._raw_jpeg() else "resize"

    _REDUCED = {0.5: "IMREAD_REDUCED_COLOR_2", 0.25: "IMREAD_REDUCED_COLOR_4",
                0.125: "IMREAD_REDUCED_COLOR_8"}

    def _raw_jpeg(self) -> bool:
        """Ask OpenCV for the camera's JPEG bytes instead of a decoded image.

        OpenCV's V4L2 capture decodes MJPG with imdecode at full size; with
        RGB conversion off it hands over the compressed frame, which can be
        decoded straight to half or quarter size for less work. Whether a
        backend honours that is checked by looking at what comes back, not
        trusted from ``set``'s return value. Anything other than a JPEG puts
        conversion back on and leaves the ordinary path in charge.
        """
        import cv2

        if self.scale not in self._REDUCED:
            return False
        try:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            for _ in range(3):           # the first read after a change can be stale
                ok, buf = self.cap.read()
                if ok and _is_jpeg(buf):
                    self._flag = getattr(cv2, self._REDUCED[self.scale])
                    return True
        except cv2.error:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        except cv2.error:
            pass
        return False

    def read(self) -> Optional[np.ndarray]:
        import cv2

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.decode == "jpeg":
            if not _is_jpeg(frame):
                return None
            return cv2.imdecode(frame.reshape(-1), self._flag)   # None if corrupt
        if self.decode == "resize":
            return cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self.cap.release()

    def describe(self) -> str:
        warn = ""
        if not self.info.fourcc_honoured:
            warn = (f"  !! delivering {self.info.fourcc or 'an unknown format'}, "
                    f"not {self.info.requested_fourcc} -- expect a low frame rate")
        size = ("" if self.decode == "full" else
                f", handed over at {self.scale:g}x "
                f"({'decoded small from the JPEG' if self.decode == 'jpeg' else 'decoded in full, then resized'})")
        return (f"{self.device.path} {self.info.describe()}{size} ({self.why})"
                + (f" [{self.device.usb_id}]" if self.device.usb_id else "")
                + warn)


class LatestFrameSource(FrameSource):
    """Read another source on a thread of its own, keeping only the newest frame.

    The camera delivers at its own pace whatever the pipeline does. Read
    inline, every frame pays for waiting on the camera and decoding before any
    processing starts; read here, that happens while the previous frame is
    being processed, and a slow pipeline gets the newest frame rather than
    the oldest one still queued in the driver. ``last_stamp`` is when the
    frame handed out arrived, for the pipeline to date it by.
    """

    def __init__(self, inner: FrameSource, timeout: float = 1.0, nice: int = 0):
        self.inner = inner
        self.nice = nice
        self.timeout = timeout
        self.last_stamp: Optional[float] = None
        self.skipped = 0                 # frames replaced before anyone read them
        self._frame: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._fresh = False
        self._ended = False
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="carrybot-capture",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        yield_to_drive(self.nice)
        misses = 0
        while self._running:
            try:
                frame = self.inner.read()
            except Exception:
                frame = None
            if frame is None:
                misses += 1
                if misses > 30:          # the source has ended or died
                    with self._cond:
                        self._ended = True
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            misses = 0
            with self._cond:
                if self._fresh:
                    self.skipped += 1
                self._frame, self._stamp, self._fresh = frame, time.monotonic(), True
                self._cond.notify_all()

    def read(self) -> Optional[np.ndarray]:
        with self._cond:
            if not self._cond.wait_for(lambda: self._fresh or self._ended,
                                       timeout=self.timeout) or not self._fresh:
                return None
            self._fresh = False
            self.last_stamp = self._stamp
            return self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.inner.close()

    def describe(self) -> str:
        inner = getattr(self.inner, "describe", None)
        return (inner() if inner else "camera") + ", read on its own thread"

    def __getattr__(self, name):
        # Everything else -- mode, info, decode, scale -- is the camera's.
        return getattr(self.inner, name)


def shrink_frames(frames, scale: float) -> list:
    """Recorded frames at the size a camera running at ``scale`` would give."""
    if scale >= 1.0:
        return list(frames)
    import cv2

    return [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            for f in frames]


def _is_jpeg(buf) -> bool:
    """A compressed frame, as OpenCV returns one with RGB conversion off."""
    if buf is None or buf.dtype != np.uint8 or buf.size < 512:
        return False
    if buf.ndim > 2 or (buf.ndim == 2 and min(buf.shape) != 1):
        return False
    flat = buf.reshape(-1)
    return int(flat[0]) == 0xFF and int(flat[1]) == 0xD8


class ListSource(FrameSource):
    """Replay a list of frames. Used by the tests and the offline viewer."""

    def __init__(self, frames, loop: bool = False):
        self.frames = list(frames)
        self.loop = loop
        self.i = 0

    def read(self) -> Optional[np.ndarray]:
        if self.i >= len(self.frames):
            if not self.loop or not self.frames:
                return None
            self.i = 0
        frame = self.frames[self.i]
        self.i += 1
        return frame


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VisionStatus:
    """A consistent read of what the last frame produced."""

    seq: int = 0
    stamp: float = 0.0
    #: Wall time the frame took end to end, seconds.
    latency: float = 0.0
    observation: Optional[TargetObservation] = None
    command: FollowCommand = field(default_factory=FollowCommand)
    stats: Optional[ProjectionStats] = None
    head: HeadCommand = field(default_factory=HeadCommand)
    #: The head angle the frame was actually taken at, radians.
    head_yaw: float = 0.0
    dropped: int = 0
    note: str = ""
    #: When the follow command was worked out, ``time.monotonic()``. Its goal
    #: and ``target_bearing`` are in the robot's frame *at this moment* -- the
    #: follower has already moved the sighting on from the capture pose -- so
    #: anything correcting them for a later turn counts the turn from here,
    #: not from ``stamp``. 0 when not recorded; then ``stamp`` is the anchor.
    command_stamp: float = 0.0
    #: Where the frame's time went, in milliseconds, in pipeline order:
    #: capture (waiting for and decoding the frame), stereo, detect (the
    #: person tracker), map (projection, follower, head). Empty for a frame
    #: that never arrived.
    timings: Tuple[Tuple[str, float], ...] = ()

    @property
    def state(self) -> FollowState:
        return self.command.state

    @property
    def head_mode(self) -> HeadMode:
        return self.head.mode

    @property
    def body_rate(self) -> float:
        """Angular velocity to turn the body at, rad/s, anticlockwise positive.

        Already limited to a rate the sensor survives, so a caller can apply it
        after clamping to its own chassis limit. Act on it when no path is
        being driven; ignore it when one is, or the robot fights its planner.
        """
        return self.head.body_rate

    @property
    def body_turn(self) -> float:
        """Rotation the body should make to keep the target coverable, radians.

        Zero unless the head has run out of travel. Positive is anticlockwise,
        matching the navigation frame. It is advice, not a command: act on it
        when there is no path being driven, and ignore it when there is, or the
        robot will fight its own planner.
        """
        return self.head.body_hint

    def summary(self) -> str:
        obs = self.observation
        where = (
            f"{obs.distance:5.2f} m @ {np.degrees(obs.bearing):+6.1f} deg"
            if obs is not None and obs.valid
            else "     --            "
        )
        cells = f"{self.stats.occupied_cells:4d} obs" if self.stats else "   - obs"
        head = (
            f" head {np.degrees(self.head_yaw):+5.0f} {self.head.mode.value:<7}"
            if self.head.mode is not HeadMode.CENTRED or self.head_yaw
            else ""
        )
        return (
            f"[{self.seq:5d}] {self.state.value:<9} {where} {cells} "
            f"{self.latency * 1000:5.1f} ms{head}  {self.command.reason}"
        )


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


class VisionPipeline:
    """Runs the camera, and feeds a planner if it is given one.

    ``planner`` is optional on purpose: the same object drives the real robot
    and the bench viewer, and the bench viewer has no planner. When one is
    present this calls ``submit_scan`` and ``set_goal`` on it, and nothing
    else -- the planner keeps owning its own thread, its own map and its own
    cadence.
    """

    def __init__(
        self,
        source: FrameSource,
        config: Optional[VisionConfig] = None,
        follow_config: Optional[FollowConfig] = None,
        planner=None,
        tracker: Optional[PersonTracker] = None,
        stereo: Optional[StereoDepth] = None,
        pose_provider: Optional[Callable[[], Tuple[float, float, float]]] = None,
        publish_costmap: bool = True,
        head_config: Optional[HeadConfig] = None,
        servo=None,
    ):
        self.cfg = config or VisionConfig()
        self.cfg.validate()
        self.source = source
        self.planner = planner
        self.pose_provider = pose_provider
        self.publish_costmap = publish_costmap
        #: Build the obstacle map from stereo. Off, stereo runs only on the
        #: people the detector found, to range them -- follow-me still works,
        #: and nothing is added to the map.
        self.map_terrain = True
        #: Run the person detector. Off, nothing is tracked or followed.
        self.detect_people = True
        self._pool = None                # the detector's thread, made on first use
        self._overlapped = False

        self.stereo = stereo or StereoDepth(self.cfg)
        self.tracker = tracker or PersonTracker(self.cfg)
        self.projector = DepthProjector(self.cfg)
        self.follower = FollowController(follow_config or FollowConfig())

        # The head is optional in both directions: no controller means a fixed
        # camera, and a controller with no servo still computes and reports the
        # angle it *would* use, which is how the simulator drives a modelled
        # head without any GPIO in the process.
        self.head = HeadController(head_config or HeadConfig(), self.cfg)
        self.servo = servo
        self._head_yaw = 0.0
        self._last_head = HeadCommand()

        self._lock = threading.Lock()
        self._status = VisionStatus()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._seq = 0
        self.frames = 0
        self.dropped = 0
        self.last_error: Optional[str] = None
        self._on_status: Optional[Callable[[VisionStatus], None]] = None
        #: Kept for the viewer; None until the first successful frame.
        self.last_depth: Optional[np.ndarray] = None
        self.last_scan: Optional[np.ndarray] = None
        self._goal_live = False
        self._heading = 0.0

        #: Whether follow commands become planner goals. The live interface
        #: turns this off outside follow-me mode, so the robot still maps and
        #: tracks while it is being driven by hand or sent to a clicked point,
        #: without the follower quietly overwriting that goal every frame.
        self.drive_goals = True
        #: A fixed head angle, radians, positive left -- or None to let the
        #: head controller aim. Set from the live interface for checking the
        #: servo and the camera mounting.
        self.head_override: Optional[float] = None
        #: The left image of the last frame, for display. Not a copy.
        self.last_left: Optional[np.ndarray] = None
        #: And the right, for measuring against a known distance.
        self.last_right: Optional[np.ndarray] = None
        #: The eye test in progress, if any. See :meth:`start_eye_test`.
        self.eye_test: Optional[EyeTest] = None
        #: The last finished eye test: ``stamp``, ``ok``, ``old``, ``new``,
        #: ``readings``, ``reason``. None until one has run.
        self.last_eye_test: Optional[dict] = None
        self._eye_test_after = 0.0

    # -- lifecycle ------------------------------------------------------- #

    def set_status_callback(self, fn: Optional[Callable[[VisionStatus], None]]) -> None:
        self._on_status = fn

    def start(self) -> "VisionPipeline":
        if self._thread is not None:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="carrybot-vision", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.source.close()

    def __enter__(self) -> "VisionPipeline":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def status(self) -> VisionStatus:
        with self._lock:
            return self._status

    # -- the frame ------------------------------------------------------- #

    def step(self, frame: Optional[np.ndarray] = None) -> VisionStatus:
        """Process exactly one frame. Returns the status it produced.

        Separate from the thread so the whole pipeline can be driven
        synchronously -- by a test, by a replay, or by the bench viewer, which
        wants the frames in order and does not want a second thread.
        """
        began = time.monotonic()
        if frame is None:
            frame = self.source.read()
        if frame is None:
            self.dropped += 1
            return self._publish(began, None, None, note="no frame")
        marks = [("capture", time.monotonic())]

        # The capture stamp is taken here, before any of the work below. Every
        # millisecond spent in SGBM and YOLO is a millisecond of robot motion
        # that the follower has to undo, and it can only do that if the number
        # it is given is the capture time and not the completion time.
        # A threaded source says when the frame actually arrived, which can be
        # a little before this read returned it.
        arrived = getattr(self.source, "last_stamp", None)
        stamp = arrived if arrived is not None and arrived <= marks[0][1] else began
        self.frames += 1

        # Both ends of the processing interval, sampled at the moments they
        # describe. The follower places the sighting in the world using the
        # pose at capture and re-expresses it from the pose now; sampling the
        # first one *after* SGBM and YOLO have run would hand it the same pose
        # twice and quietly turn motion compensation off while still reporting
        # it as on.
        capture_pose = self._note_pose(stamp)

        # Freeze the head angle for this frame *before* anything reads the
        # camera. The servo may still be travelling; every number derived from
        # this frame -- obstacle positions, the target's bearing -- belongs to
        # the angle the lens was at when the shutter opened, not the angle it
        # has reached by the time the maths runs.
        head_yaw = self._sample_head()

        left, right = split_side_by_side(frame)
        depth, camera, observation, stereo_ms, detect_ms = self._see(
            left, right, stamp)
        # After stereo, so a lens check that trips on this frame starts the
        # eye test on it rather than one frame later.
        self._eye_check(left, right, stamp)
        self.last_depth = depth
        self.last_left = left
        self.last_right = right
        marks.append(("stereo", marks[-1][1] + stereo_ms / 1e3))
        marks.append(("detect", marks[-1][1] + detect_ms / 1e3))
        # The tracker measures bearing relative to the lens, positive to the
        # *right*. Head yaw is positive to the *left*, matching the servo, the
        # projector and the navigation frame. Mixing the two silently rotates
        # the target the wrong way by twice the head angle -- the robot drives
        # at a person who is not there, and the obstacle map disagrees with the
        # goal by 2 x yaw, which reads like odometry drift and is not.
        #
        #     robot bearing (right +) = camera bearing (right +) - head yaw
        #
        # Converted once, here, rather than left for each consumer to remember.
        observation = replace(
            observation,
            bearing=observation.bearing - head_yaw,
            head_yaw=head_yaw,
        )

        now = time.monotonic()
        self._note_pose(now)
        command = self.follower.update(observation, now=now)

        stats = None
        if depth is not None and self.publish_costmap and self.map_terrain:
            exclude = self._exclude_mask(depth, left, observation)
            scan = self.projector.project(
                depth, camera, exclude=exclude, head_yaw=head_yaw
            )
            stats = self.projector.last_stats
            self.last_scan = scan
            if self.planner is not None:
                # Hand over the pose the frame was taken from, so the planner
                # puts the obstacles where they were seen rather than rotating
                # them by however far the robot has turned since.
                if capture_pose is None:
                    self.planner.submit_scan(scan, copy=False, stamp=stamp)
                else:
                    self.planner.submit_scan(scan, copy=False, stamp=stamp,
                                             odometry=capture_pose)
            self.head.note_scan(scan, self._heading, head_yaw, now=now)

        head = self._aim(observation, command, now)

        if self.planner is not None and self.drive_goals:
            self._drive(command)

        done = time.monotonic()
        # Stereo and detect are measured on their own; when they ran side by
        # side their sum is more than the time they took, so "map" is what is
        # left of the frame after the slower of the two, not after both.
        overlapped = self._overlapped
        work = max(stereo_ms, detect_ms) if overlapped else stereo_ms + detect_ms
        timings = [
            ("capture", (marks[0][1] - began) * 1e3),
            ("stereo", stereo_ms),
            ("detect", detect_ms),
            ("map", max(0.0, (done - marks[0][1]) * 1e3 - work)),
        ]
        return self._publish(stamp, observation, command, stats, head=head,
                             head_yaw=head_yaw, timings=tuple(timings),
                             command_stamp=now)

    def _see(self, left, right, stamp):
        """Stereo and the person detector, as the toggles and threads allow.

        Returns ``(depth, camera, observation, stereo_ms, detect_ms)``.
        """
        self._overlapped = False
        people = self.detect_people
        terrain = self.map_terrain
        splittable = hasattr(self.tracker, "detect")
        no_one = TargetObservation(stamp=stamp, valid=False)

        if not people:
            t0 = time.monotonic()
            if terrain:
                depth, camera = self.stereo.compute(left, right)
            else:
                depth, camera = None, self.stereo.scaled_camera
            return depth, camera, no_one, (time.monotonic() - t0) * 1e3, 0.0

        if terrain and self.cfg.parallel_detect and splittable:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor

                # One worker, always the same thread: the tracker keeps its
                # identities between calls and is not built for two at once.
                self._pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="carrybot-detect",
                    initializer=yield_to_drive, initargs=(self.cfg.nice,))

            def timed():
                t = time.monotonic()
                found = self.tracker.detect(left)
                return found, (time.monotonic() - t) * 1e3

            pending = self._pool.submit(timed)
            t0 = time.monotonic()
            try:
                depth, camera = self.stereo.compute(left, right)
            finally:
                stereo_ms = (time.monotonic() - t0) * 1e3
                detections, detect_ms = pending.result()
            self._overlapped = True
            observation = self.tracker.update(left, depth, camera, stamp=stamp,
                                              detections=detections)
            return depth, camera, observation, stereo_ms, detect_ms

        if terrain or not splittable:
            t0 = time.monotonic()
            if terrain:
                depth, camera = self.stereo.compute(left, right)
            else:
                depth, camera = None, self.stereo.scaled_camera
            stereo_ms = (time.monotonic() - t0) * 1e3
            t1 = time.monotonic()
            observation = self.tracker.update(left, depth, camera, stamp=stamp)
            return depth, camera, observation, stereo_ms, (time.monotonic() - t1) * 1e3

        # Terrain off, people on: find them first, then range only them.
        t1 = time.monotonic()
        detections = self.tracker.detect(left)
        detect_ms = (time.monotonic() - t1) * 1e3
        t0 = time.monotonic()
        roi = self._people_box(detections, left.shape)
        if roi is not None:
            depth, camera = self.stereo.compute(left, right, roi=roi)
        else:
            depth, camera = None, self.stereo.scaled_camera
        stereo_ms = (time.monotonic() - t0) * 1e3
        observation = self.tracker.update(left, depth, camera, stamp=stamp,
                                          detections=detections)
        return depth, camera, observation, stereo_ms, detect_ms

    def _people_box(self, detections, shape) -> Optional[Tuple[float, float, float, float]]:
        """One box around everyone worth ranging, with room for the depth
        patch. A single box, because the tracker decides which of them is the
        target only after it has their ranges."""
        usable = [d for d in detections if d.confidence >= self.cfg.min_confidence]
        if not usable:
            return None
        pad = self.cfg.depth_patch * 2
        x1 = max(0.0, min(d.bbox[0] for d in usable) - pad)
        y1 = max(0.0, min(d.bbox[1] for d in usable) - pad)
        x2 = min(float(shape[1]), max(d.bbox[2] for d in usable) + pad)
        y2 = min(float(shape[0]), max(d.bbox[3] for d in usable) + pad)
        return (x1, y1, x2, y2)

    def release_goal(self) -> None:
        """Clear the goal this pipeline set, if it set one.

        For handing the planner to someone else: without it, switching out of
        follow-me leaves the last standoff goal live and the robot keeps
        driving to it.
        """
        if self._goal_live and self.planner is not None:
            self.planner.clear_goal()
        self._goal_live = False

    def _sample_head(self) -> float:
        """The head angle to attribute this frame to."""
        if self.servo is not None:
            self._head_yaw = self.servo.update()
        return self._head_yaw

    def _aim(
        self, observation: TargetObservation, command: FollowCommand, now: float
    ) -> HeadCommand:
        """Run the head controller and pass its answer to the servo."""
        # Hand the head the follower's prediction when the follower is running
        # on one. Without it the head stares at the last sighting while the
        # follower drives towards where the person actually is, and the two
        # halves of the robot disagree about where their target is.
        predicted = (
            command.target_bearing
            if command.predicted
            and command.state in (
                FollowState.COASTING, FollowState.PURSUING, FollowState.LOST
            )
            else None
        )
        if self.head_override is not None:
            limit = self.head.cfg.max_angle
            aim = HeadCommand(
                angle=max(-limit, min(limit, float(self.head_override))),
                mode=HeadMode.MANUAL, reason="pointed by hand",
            )
        else:
            aim = self.head.update(
                observation,
                command.state,
                current=self._head_yaw,
                robot_heading=self._heading,
                now=now,
                predicted_bearing=predicted,
            )
        self._last_head = aim
        if self.servo is not None:
            self.servo.command(aim.angle, now=now)
            self._head_yaw = self.servo.update(now)
        else:
            # No servo: model it as instantaneous. Fine for a bench run, and
            # wrong enough during fast sweeps that the simulator supplies a
            # real slew-limited model instead of relying on this.
            self._head_yaw = aim.angle
        return aim

    # -- range calibration --------------------------------------------- #

    def measure_target_disparity(self, search: int = 256) -> Optional[float]:
        """Raw disparity at the tracked person's torso, full-resolution pixels.

        Matched with a search far wider than the running one, from zero, so
        it finds the person's disparity even when ``disparity_offset`` is so
        wrong that the running search window does not contain it -- which is
        exactly when every range reads as the window's near limit. With the
        person at a known distance this is one point on ``d = f*B/z + d0``.
        None if there is no recent sighting or too little matched.
        """
        import cv2

        from .stereo import _shift_rows

        left, right = self.last_left, self.last_right
        obs = self.status.observation
        if left is None or right is None or obs is None or not obs.valid or not obs.bbox:
            return None
        scale = left.shape[1] / self.cfg.camera.width
        x1, y1, x2, y2 = (int(v) for v in obs.bbox)
        h, w = left.shape[:2]
        # The torso: the middle of the box, away from the edges and the floor.
        bw, bh = x2 - x1, y2 - y1
        tx1, tx2 = max(0, x1 + bw // 4), min(w, x2 - bw // 4)
        ty1, ty2 = max(0, y1 + bh // 4), min(h, y1 + 3 * bh // 4)
        if tx2 - tx1 < 8 or ty2 - ty1 < 8:
            return None

        def grey(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        a, b = grey(left), grey(right)
        if self.cfg.camera.row_offset:
            b = _shift_rows(cv2, b, self.cfg.camera.row_offset * scale)
        block = self.cfg.block_size
        r1, r2 = max(0, ty1 - block), min(h, ty2 + block)
        span = 16 * (min(int(search * scale), w - 32) // 16)
        if span < 16:
            return None
        matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=span, blockSize=block,
            P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
            uniquenessRatio=self.cfg.uniqueness_ratio,
            speckleWindowSize=self.cfg.speckle_window,
            speckleRange=self.cfg.speckle_range,
            disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disparity = matcher.compute(np.ascontiguousarray(a[r1:r2]),
                                    np.ascontiguousarray(b[r1:r2])).astype(np.float32) / 16.0
        patch = disparity[ty1 - r1: ty2 - r1, tx1:tx2]
        good = patch[patch > 0.5]
        if good.size < 50:
            return None
        return float(np.median(good)) / scale

    # -- the eye test ---------------------------------------------------- #

    @property
    def eye_test_running(self) -> bool:
        return self.eye_test is not None

    def start_eye_test(self) -> bool:
        """Re-measure ``row_offset`` over the next few frames, then use it.

        Returns False if one is already running. Started automatically when
        the lens check says the eyes have moved (``auto_eye_test``).
        """
        if self.eye_test is not None:
            return False
        self.eye_test = EyeTest(readings=self.cfg.eye_test_readings)
        return True

    def _eye_check(self, left, right, stamp: float) -> None:
        """Start the eye test when the eyes have moved; feed it while it runs."""
        stereo = self.stereo
        if (self.eye_test is None and self.cfg.auto_eye_test
                and not getattr(stereo, "geometry_trusted", True)
                and stamp >= self._eye_test_after):
            self.start_eye_test()
        test = self.eye_test
        if test is None:
            return
        scale = left.shape[1] / self.cfg.camera.width
        try:
            finished = test.feed(left, right, scale=scale)
        except Exception as exc:          # a diagnostic must never stop the robot
            test.done, test.ok, test.reason = True, False, f"failed: {exc!r}"
            finished = True
        if not finished:
            return
        old = self.cfg.camera.row_offset
        if test.ok:
            stereo.set_row_offset(test.value)
            self._eye_test_after = 0.0
        else:
            self._eye_test_after = stamp + self.cfg.eye_test_retry
        self.last_eye_test = dict(
            stamp=stamp, ok=test.ok, old=old,
            new=test.value if test.ok else None,
            readings=list(test.values), reason=test.reason)
        self.eye_test = None

    def describe_eye_test(self) -> str:
        """One line for the page: running, or how the last one went."""
        if self.eye_test is not None:
            t = self.eye_test
            return (f"eye test running ({len(t.values)}/{t.readings} readings)")
        last = self.last_eye_test
        if last is None:
            return ""
        if last["ok"]:
            return (f"eye test: row_offset {last['old']:+.1f} -> {last['new']:+.1f} px"
                    " (in use now; copy it to [camera] row_offset to keep it)")
        return f"eye test failed: {last['reason']}"

    def _note_pose(self, stamp: float) -> Optional[Tuple[float, float, float]]:
        """Record the odometry pose, if the caller gave us a way to read one.

        Returns the ``(x, y, heading)`` read, or None when there is none.

        A pose provider that throws is a broken sensor, not a reason to stop
        driving: the follower falls back to using the sighting uncorrected,
        which is what it would have done without a provider at all.
        """
        if self.pose_provider is None:
            return None
        try:
            x, y, heading = self.pose_provider()
        except Exception as exc:
            self.last_error = f"pose_provider: {exc!r}"
            return None
        self.follower.note_pose(stamp, x, y, heading)
        self._heading = heading
        self.head.note_heading(heading, stamp)
        return (float(x), float(y), float(heading))

    def _exclude_mask(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        observation: TargetObservation,
    ) -> Optional[np.ndarray]:
        """The tracked person's box, in the depth map's own pixel grid.

        The detection ran on the full-resolution left image and the depth map
        is at ``disparity_scale`` of it, so the box has to be scaled. Getting
        this wrong masks a quarter of the frame or none of the person; both
        look like the projector misbehaving.
        """
        if observation.bbox is None:
            return None
        sx = depth.shape[1] / image.shape[1]
        sy = depth.shape[0] / image.shape[0]
        x1, y1, x2, y2 = observation.bbox
        scaled = (
            int(round(x1 * sx)), int(round(y1 * sy)),
            int(round(x2 * sx)), int(round(y2 * sy)),
        )
        padding = max(1, int(round(self.cfg.target_mask_padding * sx)))
        return box_mask(depth.shape, scaled, padding=padding)

    def _drive(self, command: FollowCommand) -> None:
        """Hand the follow command to the planner as a goal, or clear it."""
        if command.goal is not None and not command.stop:
            gx, gy = command.goal
            self.planner.set_goal(gx, gy)
            self._goal_live = True
        elif self._goal_live:
            # HOLDING and LOST both mean "do not drive". Clearing the goal is
            # how you say that to a planner whose whole job is to keep finding
            # a way to the last one it was given. Once, on the transition:
            # clearing an already-cleared goal still counts as a change, and
            # re-sending it every frame would wake the planner at camera rate
            # to publish IDLE over and over.
            self.planner.clear_goal()
            self._goal_live = False

    def _publish(
        self,
        began: float,
        observation: Optional[TargetObservation],
        command: Optional[FollowCommand],
        stats: Optional[ProjectionStats] = None,
        note: str = "",
        head: Optional[HeadCommand] = None,
        head_yaw: float = 0.0,
        timings: Tuple[Tuple[str, float], ...] = (),
        command_stamp: float = 0.0,
    ) -> VisionStatus:
        self._seq += 1
        status = VisionStatus(
            seq=self._seq,
            stamp=began,
            latency=time.monotonic() - began,
            observation=observation,
            command=command or FollowCommand(),
            stats=stats,
            head=head or self._last_head,
            head_yaw=head_yaw,
            dropped=self.dropped,
            note=note,
            timings=timings,
            command_stamp=command_stamp,
        )
        with self._lock:
            self._status = status
        if self._on_status is not None:
            try:
                self._on_status(status)
            except Exception as exc:  # a noisy consumer must not stop the robot
                self.last_error = f"status callback: {exc!r}"
        return status

    # -- thread ---------------------------------------------------------- #

    def _run(self) -> None:
        import traceback

        yield_to_drive(self.cfg.nice, self.cfg.cpu_threads)
        misses = 0
        while self._running:
            try:
                status = self.step()
            except Exception:
                self.last_error = traceback.format_exc()
                time.sleep(0.05)
                continue

            if status.note == "no frame":
                misses += 1
                if misses > 30:
                    self.last_error = "camera delivered 30 consecutive empty frames"
                    self._running = False
                time.sleep(0.02)
            else:
                misses = 0

    # -- reporting ------------------------------------------------------- #

    def health(self) -> str:
        lines = [
            self.cfg.describe(),
            f"  stereo: {self.stereo.health()}",
            f"  tracker: {self.tracker.frames} frames, {self.tracker.sightings} "
            f"sightings, last: {self.tracker.last_reason}",
            f"  pipeline: {self.frames} frames, {self.dropped} dropped",
            f"  {self.head.describe()}",
        ]
        if self.last_error:
            lines.append(f"  last error: {self.last_error.strip().splitlines()[-1]}")
        return "\n".join(lines)


__all__ = [
    "CameraSource",
    "FrameSource",
    "ListSource",
    "VisionPipeline",
    "VisionStatus",
]
