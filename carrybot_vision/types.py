from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


def bearing_to_robot_frame(distance: float, bearing_rad: float) -> Tuple[float, float]:
    """Range and image-frame bearing to a point in the robot frame.

    ``bearing_rad`` is positive when the target is to the **right** of the
    image centre, as ``atan2(u - cx, fx)`` gives it. The robot frame has ``+y``
    to the **left**, hence the negation.
    """
    return distance * math.cos(bearing_rad), -distance * math.sin(bearing_rad)


@dataclass
class CameraModel:
    """Intrinsics for one eye, plus where that eye sits on the robot.

    ``focal_px``, ``cx`` and ``cy`` must all refer to the **same resolution**
    as the images actually being processed. That is not a pedantic note: the
    original tracker computed disparity at half scale and divided by the
    full-resolution focal length, which reports every distance twice as far
    away as it is. :class:`~carrybot_vision.stereo.StereoDepth` scales the
    intrinsics with the images so the mistake cannot be made by accident.
    """

    width: int = 1280
    height: int = 720
    focal_px: float = 842.0
    #: Principal point. Defaults to the image centre when left as None.
    cx: Optional[float] = None
    cy: Optional[float] = None
    #: Stereo baseline in metres. Measured lens centre to lens centre on the
    #: fitted camera, not taken from a datasheet.
    baseline: float = 0.085

    #: Disparity, in pixels **at this scale**, that a point at infinity
    #: produces. Zero for a textbook parallel pair. Real dual-lens modules are
    #: rarely parallel and rarely pack the two halves in perfect register, so
    #: this is usually tens of pixels, and ignoring it is catastrophic rather
    #: than merely inaccurate: it is a floor under every disparity, so the
    #: whole scene collapses towards the distance that offset alone implies.
    #: Measure it with ``probe --focal`` at two or more known distances;
    #: never guess it.
    disparity_offset: float = 0.0
    #: Rows the right eye must be shifted **down** by to line up with the
    #: left, in pixels at this scale. Block matching searches along rows, so a
    #: vertical error does not bias the result, it destroys matches outright.
    #: The sign is easy to get backwards and the wrong sign is worse than no
    #: correction, so it is settled by measurement, not by reasoning: the
    #: right value is whichever one raises the matched-pixel count.
    #: ``probe --offsets`` reports it already signed for this field.
    row_offset: float = 0.0

    # -- where the camera is bolted ------------------------------------- #
    #: Height of the optical centre above the floor, metres.
    mount_height: float = 0.25
    #: Downward tilt in radians, positive looking down. A few degrees here
    #: moves the floor a long way at range: 3 degrees puts the floor 21 cm up
    #: in the image at 4 m, and an uncorrected floor reads as a wall.
    mount_pitch: float = 0.0
    #: Forward offset of the camera from the **pan axis**, metres. With a
    #: fixed head (``pan_offset = 0``) this is simply its offset from the
    #: wheel axis, which is what it has always meant.
    mount_forward: float = 0.10
    #: Forward offset of the pan axis itself from the wheel axis, metres. Only
    #: meaningful on a robot whose head turns: the camera then swings around
    #: this point, so an obstacle seen at 40 degrees of head yaw is not where
    #: it would be if the head merely rotated on the spot. Leave it at 0 and
    #: the geometry collapses to the fixed-head case exactly.
    pan_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0

    def scaled(self, factor: float) -> "CameraModel":
        """The same camera described at a different image scale."""
        return CameraModel(
            width=max(1, int(round(self.width * factor))),
            height=max(1, int(round(self.height * factor))),
            focal_px=self.focal_px * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            baseline=self.baseline,
            disparity_offset=self.disparity_offset * factor,
            row_offset=self.row_offset * factor,
            mount_height=self.mount_height,
            mount_pitch=self.mount_pitch,
            mount_forward=self.mount_forward,
            pan_offset=self.pan_offset,
        )

    # -- derived -------------------------------------------------------- #

    @property
    def hfov(self) -> float:
        return 2.0 * math.atan(self.width / (2.0 * self.focal_px))

    @property
    def vfov(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.focal_px))

    def depth_at(self, disparity_px: float) -> float:
        """Depth in metres for a disparity measured at *this* scale.

        Only the part of the disparity above :attr:`disparity_offset` carries
        depth. Anything at or below it is at or beyond the range where the two
        eyes stop disagreeing, which reads as infinity.
        """
        relative = disparity_px - self.disparity_offset
        if relative <= 0.0:
            return float("inf")
        return self.focal_px * self.baseline / relative

    def disparity_for(self, depth_m: float) -> float:
        """The disparity a depth produces, including the offset."""
        return (self.focal_px * self.baseline / max(1e-9, depth_m)
                + self.disparity_offset)

    def bearing_of(self, u: float) -> float:
        """Bearing of an image column, positive to the right."""
        return math.atan2(u - self.cx, self.focal_px)

    def range_resolution(self, depth_m: float, disparity_error_px: float = 0.5) -> float:
        """Depth uncertainty at a range, for a given matching error.

        Grows with the square of distance. At this baseline it is a few
        centimetres at 1 m and most of a metre at 5 m, which is why a follow
        standoff belongs in the near half of the range.
        """
        d = self.disparity_for(depth_m)
        near = self.depth_at(d + disparity_error_px)
        far = self.depth_at(max(1e-6, d - disparity_error_px))
        return abs(far - near) / 2.0

    def describe(self) -> str:
        return (
            f"{self.width}x{self.height}  f={self.focal_px:.0f}px  "
            f"HFOV {math.degrees(self.hfov):.1f} deg  "
            f"VFOV {math.degrees(self.vfov):.1f} deg  "
            f"baseline {self.baseline * 1000:.0f} mm\n"
            f"  disparity offset {self.disparity_offset:+.1f} px, "
            f"row offset {self.row_offset:+.1f} px"
            f"{'  (UNCALIBRATED - assumes a perfect parallel pair)' if self.disparity_offset == 0.0 and self.row_offset == 0.0 else ''}\n"
            f"  mounted {self.mount_height * 100:.0f} cm up, "
            f"{math.degrees(self.mount_pitch):.1f} deg down, "
            f"{self.mount_forward * 100:.0f} cm forward of the axis"
        )


@dataclass(frozen=True)
class TargetObservation:
    """One sighting of the person being followed.

    ``stamp`` is when the *frame was captured*, not when this was computed.
    Everything downstream needs it: at 5 Hz with 200 ms of processing latency
    and a robot turning at 2.5 rad/s, the bearing is a quarter of a radian out
    of date by the time it is acted on.
    """

    #: Persistent tracker id. None means the tracker could not keep identity,
    #: which for a follow robot is the same as having lost the target.
    track_id: Optional[int] = None
    #: Range to the target in metres.
    distance: float = 0.0
    #: Bearing in radians, positive when the target is to the **right**.
    bearing: float = 0.0
    #: Detection confidence, 0..1.
    confidence: float = 0.0
    #: Capture time on the same clock as the odometry (``time.monotonic()``).
    stamp: float = 0.0
    #: Pixel box, for masking the target out of the obstacle map.
    bbox: Optional[Tuple[int, int, int, int]] = None
    #: How many depth samples backed the range estimate. One pixel is not an
    #: estimate; single-digit counts mean the stereo match was poor.
    depth_samples: int = 0
    valid: bool = False
    #: Where the pan head was pointing when this frame was captured, radians.
    #: ``bearing`` above is already in the **robot** frame, this having been
    #: added to it; the raw camera-relative angle is ``bearing - head_yaw``.
    #: Recorded because the head is part of the extrinsics: a sighting taken at
    #: 40 degrees of yaw and interpreted as if the head were straight ahead
    #: points the robot 40 degrees away from the person.
    head_yaw: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Target position in the robot frame, metres."""
        return bearing_to_robot_frame(self.distance, self.bearing)


@dataclass
class VisionConfig:
    camera: CameraModel = field(default_factory=CameraModel)

    # -- capture --------------------------------------------------------- #
    #: Decode camera frames at this fraction of full resolution: 1, 0.5, 0.25
    #: or 0.125. The camera's MJPG is decoded straight to that size -- JPEG
    #: can skip the fine detail at decode time, which is the same 2x2 average
    #: a resize would give, without first paying for the full-size image.
    #: Nothing downstream loses anything down to ``disparity_scale``: stereo
    #: works at that scale anyway, and the person detector shrinks its input
    #: to ``detect_size``. Must not be below ``disparity_scale``.
    capture_scale: float = 0.5
    #: The person detector's input size, in pixels along the long side; a
    #: multiple of 32. Its cost goes with the square: 320 is about a quarter
    #: of 640. A person 1.7 m tall at 6 m is still ~60 px high at 320.
    detect_size: int = 320
    #: Read the camera on its own thread, always keeping the newest frame, so
    #: waiting for and decoding the next frame overlaps the work on this one
    #: instead of adding to it. The frame processed is also the freshest.
    threaded_capture: bool = True
    #: Run the person detector alongside stereo, on another thread, instead
    #: of after it. They need nothing from each other until the person's range
    #: is read off the depth map, and both spend their time in native code
    #: that releases Python's lock, so they really do run at once.
    parallel_detect: bool = True
    #: Threads each of OpenCV and PyTorch may use. Both default to one per
    #: core, and together with the drive that is more threads than a Pi has
    #: cores: the encoder callbacks and the 50 Hz speed loop then wait their
    #: turn, wheel speeds arrive in bursts, and the robot stutters. 2 leaves
    #: room on a 4-core Pi. 0 leaves the libraries' own defaults.
    cpu_threads: int = 2
    #: How much lower than the drive the vision threads run (Linux nice, 0-19).
    #: When the Pi is short of time, the motors get it first.
    nice: int = 5

    # -- stereo ---------------------------------------------------------- #
    #: Disparity is computed at this fraction of full resolution. Intrinsics
    #: are scaled to match automatically. SGBM's cost goes with the pixel
    #: count times ``num_disparities``; range error grows as this shrinks.
    disparity_scale: float = 0.5
    #: Search range, in disparities at the *scaled* resolution. This sets the
    #: near limit: nothing closer than focal*baseline/num_disparities can be
    #: measured at all.
    num_disparities: int = 64
    block_size: int = 5
    #: Reject matches whose uniqueness margin is below this percentage.
    uniqueness_ratio: int = 10
    speckle_window: int = 64
    speckle_range: int = 2
    #: Ignore anything beyond this; the range error is larger than the reading.
    max_range: float = 6.0
    #: Re-measure the vertical registration between the eyes every this many
    #: frames and flag depth as untrustworthy if it has drifted from
    #: ``camera.row_offset``. 0 turns it off. See RegistrationMonitor.
    registration_check_every: int = 150
    #: When that check says the eyes have moved, re-measure ``row_offset``
    #: from a few frames (the eye test) and use the new value. Follow me and
    #: Go to hold still while it runs. Off, the check only warns and holds.
    auto_eye_test: bool = True
    #: Frames the eye test measures; they must agree to within 1.5 px.
    eye_test_readings: int = 5
    #: Seconds before an eye test that could not get a clean reading tries
    #: again.
    eye_test_retry: float = 20.0

    # -- ground plane ---------------------------------------------------- #
    #: Points below this height above the floor are floor, not obstacles.
    ground_height: float = 0.06
    #: Points above this are overhead and cannot be hit.
    ceiling_height: float = 0.90
    #: Trim this many pixels from the image border before using depth; stereo
    #: is unreliable at the edges where one eye has no overlap.
    border_trim: int = 8

    # -- occupancy output ------------------------------------------------ #
    #: Must match ``PlannerConfig.resolution``.
    grid_resolution: float = 0.05
    #: Side length of the robot-centric array handed to the planner, in cells.
    #: Odd, so the robot sits on an exact centre cell.
    grid_size: int = 161
    #: Obstacle points per cell before the cell counts as occupied. Raising it
    #: rejects speckle at the cost of missing thin obstacles.
    min_points_per_cell: int = 2
    #: Width of the bearing bins used to carve free space, in radians.
    bearing_bin: float = math.radians(0.5)

    # -- target ---------------------------------------------------------- #
    #: Pixels of padding around the tracked person's box when masking them out
    #: of the obstacle map.
    target_mask_padding: int = 12
    #: Minimum detection confidence to accept a sighting.
    min_confidence: float = 0.4
    #: Minimum keypoint confidence before a joint is used for the range patch.
    min_keypoint_confidence: float = 0.3
    #: Half-size of the depth patch sampled around the torso, in pixels.
    depth_patch: int = 9
    #: Fewest valid depth samples for a range estimate to be trusted.
    min_depth_samples: int = 8

    def validate(self) -> None:
        cam = self.camera
        if cam.focal_px <= 0 or cam.baseline <= 0:
            raise ValueError("focal length and baseline must be positive")
        if not 0.05 <= self.disparity_scale <= 1.0:
            raise ValueError("disparity_scale must be between 0.05 and 1.0")
        if self.capture_scale not in (1.0, 0.5, 0.25, 0.125):
            raise ValueError("capture_scale must be 1, 0.5, 0.25 or 0.125: the "
                             "sizes a JPEG can be decoded to directly")
        if self.capture_scale < self.disparity_scale:
            raise ValueError(
                f"capture_scale {self.capture_scale:g} is below disparity_scale "
                f"{self.disparity_scale:g}: stereo cannot work at a finer scale "
                "than the frames it is given. Lower disparity_scale too.")
        if self.cpu_threads < 0:
            raise ValueError("cpu_threads must be 0 (library default) or more")
        if not 0 <= self.nice <= 19:
            raise ValueError("nice must be 0-19")
        if self.detect_size < 64 or self.detect_size % 32:
            raise ValueError("detect_size must be a multiple of 32, at least 64")
        if self.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be a multiple of 16 for SGBM")
        if self.block_size % 2 == 0:
            raise ValueError("block_size must be odd")
        if self.grid_size % 2 == 0:
            raise ValueError("grid_size must be odd so the robot has a centre cell")
        if self.ground_height >= self.ceiling_height:
            raise ValueError("ground_height must be below ceiling_height")

    @property
    def near_limit(self) -> float:
        """Closest measurable range, metres. Inside this the robot is blind.

        The search window runs from the disparity offset upwards, not from
        zero, so the near limit is set by the *relative* disparity the window
        spans. An offset does not move the near limit, it moves the window.
        """
        scaled = self.camera.scaled(self.disparity_scale)
        return scaled.depth_at(scaled.disparity_offset + self.num_disparities)

    @property
    def grid_half(self) -> int:
        return self.grid_size // 2

    @property
    def grid_reach(self) -> float:
        """Half-width of the output grid in metres."""
        return self.grid_half * self.grid_resolution

    def describe(self) -> str:
        scaled = self.camera.scaled(self.disparity_scale)
        return (
            f"{self.camera.describe()}\n"
            f"  disparity at {self.disparity_scale:g}x -> f={scaled.focal_px:.0f}px, "
            f"{self.num_disparities} disparities\n"
            f"  usable range {self.near_limit:.2f} - {self.max_range:.1f} m\n"
            # Against the *scaled* camera: the error follows the resolution
            # the disparity was actually measured at, not the full frame.
            f"  range error ~{scaled.range_resolution(1.5) * 100:.0f} cm at 1.5 m, "
            f"~{scaled.range_resolution(3.0) * 100:.0f} cm at 3.0 m\n"
            f"  output grid {self.grid_size}x{self.grid_size} "
            f"@ {self.grid_resolution * 100:.0f} cm = "
            f"+/-{self.grid_reach:.1f} m"
        )


__all__ = [
    "CameraModel",
    "TargetObservation",
    "VisionConfig",
    "bearing_to_robot_frame",
]
