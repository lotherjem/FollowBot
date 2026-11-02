from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .types import CameraModel, VisionConfig


@dataclass
class Rectification:
    """Precomputed remap tables from ``cv2.stereoRectify``."""

    left_map: Tuple[np.ndarray, np.ndarray]
    right_map: Tuple[np.ndarray, np.ndarray]
    #: Focal length and principal point *after* rectification.
    focal_px: float
    cx: float
    cy: float
    baseline: float

    def apply(self, left: np.ndarray, right: np.ndarray):
        import cv2

        return (
            cv2.remap(left, *self.left_map, interpolation=cv2.INTER_LINEAR),
            cv2.remap(right, *self.right_map, interpolation=cv2.INTER_LINEAR),
        )


def load_calibration(path: str, image_size: Tuple[int, int]) -> Rectification:
    """Build remap tables from a `.npz` of stereo calibration parameters.

    Expects ``K1 D1 K2 D2 R T``, as ``cv2.stereoCalibrate`` produces them.
    """
    import cv2

    data = np.load(path)
    K1, D1, K2, D2 = data["K1"], data["D1"], data["K2"], data["D2"]
    R, T = data["R"], data["T"]

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0
    )
    left_map = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    right_map = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    return Rectification(
        left_map=left_map,
        right_map=right_map,
        focal_px=float(P1[0, 0]),
        cx=float(P1[0, 2]),
        cy=float(P1[1, 2]),
        baseline=float(abs(T[0])) if np.ndim(T) else float(abs(T)),
    )


def _shift_rows(cv2, image: np.ndarray, dy: float) -> np.ndarray:
    """Translate an image vertically by a possibly fractional number of rows."""
    matrix = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, float(dy)]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def measure_registration(left_half: np.ndarray, right_half: np.ndarray,
                         tile: int = 256, stride: int = 64) -> Optional[dict]:
    """Measure how the two halves sit relative to each other, in pixels.

    Returns the vertical offset (how far the right eye sits below the left),
    the horizontal offset a distant surface produces, and how much each varies
    across the frame.

    Both numbers come from phase correlation on overlapping tiles. The tile has
    to be wider than twice the shift being measured or the correlation wraps
    and reports a confident wrong answer, which is why ``tile`` defaults well
    above the few tens of pixels these offsets usually reach.

    The horizontal figure is an **upper bound** on the disparity offset, not
    the offset itself, and indoors it is a badly loose one. It is the smallest
    disparity anything in view produces, which equals the offset only if
    something is effectively at infinity. In a room the most distant thing is a
    wall three metres away, which still produces plenty of disparity, so this
    number comes back far too high --- and subtracting too much pushes the
    whole scene out to absurd ranges, the mirror image of the error it is meant
    to fix. On a real recording it read 63 px where a known 2.20 m distance
    later showed the offset to be nearer 38.

    Treat it only as "the offset is no larger than this". The offset is
    measured properly by ``--focal`` at two known distances, which solves for
    it instead of inferring it from whatever happened to be in shot.
    """
    import cv2

    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return np.ascontiguousarray(np.float32(g) / 255.0)

    a, b = prep(left_half), prep(right_half)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    tile = int(min(tile, h, w))
    try:
        window = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    except cv2.error:
        return None

    dxs, dys, skipped = [], [], 0
    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            pa = np.ascontiguousarray(a[y:y + tile, x:x + tile])
            pb = np.ascontiguousarray(b[y:y + tile, x:x + tile])
            if float(np.std(pa)) < 0.04:
                skipped += 1
                continue                      # featureless: correlation is noise
            try:
                (dx, dy), response = cv2.phaseCorrelate(pa, pb, window)
            except cv2.error:
                skipped += 1
                continue
            if response < 0.10 or abs(dx) > tile / 2.5 or abs(dy) > tile / 2.5:
                skipped += 1
                continue
            dxs.append(dx)
            dys.append(dy)
    if len(dxs) < 4:
        return None
    dxs = np.array(dxs)
    dys = np.array(dys)
    return {
        "row_offset": float(-np.median(dys)),
        "row_spread": float(np.std(dys)),
        "disparity_ceiling": float(-np.percentile(dxs, 98)),
        "disparity_median": float(-np.median(dxs)),
        "disparity_span": float(np.percentile(dxs, 98) - np.percentile(dxs, 2)),
        "tiles": int(len(dxs)),
        "skipped": int(skipped),
    }


class RegistrationMonitor:
    """Watches whether the two eyes have moved relative to each other.

    Everything depth depends on -- ``row_offset``, ``disparity_offset``, the
    focal length fit -- is only valid while the lenses stay exactly where they
    were when it was measured. On the real camera they did not: between two
    sessions the right eye's image moved 7.3 px sideways and 14.4 px vertically
    relative to the left, with no relative rotation and no change of scale, the
    signature of a lens turned in its thread. The sideways part moved the
    disparity offset by the same 7 px, and nothing downstream could have
    noticed.

    The vertical registration can be read off any textured frame with no known
    distances, so it is watched continuously. It cannot see a purely sideways
    shift -- that would need a known range -- but a lens turning in its thread
    moves the image round a small circle, so a real disturbance almost always
    shows vertically too. When the measured value strays from the configured
    one on several checks in a row, depth is flagged untrustworthy.
    """

    def __init__(self, expected_row_offset: float, every: int = 150,
                 tolerance: float = 2.5, confirm: int = 3):
        self.expected = float(expected_row_offset)
        self.every = max(1, int(every))
        self.tolerance = float(tolerance)
        self.confirm = max(1, int(confirm))
        self.frames = 0
        self.history: List[float] = []
        self.strikes = 0
        self.disturbed = False

    def observe(self, left: np.ndarray, right: np.ndarray,
                scale: float = 1.0) -> Optional[float]:
        """Call every frame; measures only every ``every`` frames.

        ``scale`` is the size of the images relative to the calibrated camera,
        so a half-size frame's measurement is compared at full scale.
        """
        self.frames += 1
        if self.frames % self.every:
            return None
        # The tile stays 256 px whatever the scale. Shrinking it with the image
        # halves the largest horizontal shift phase correlation can unwrap,
        # and near objects at half scale already come close to that limit; a
        # wrapped tile reads as a vertical error, and enough of those flag the
        # lenses as moved when they have not. Finer stride, same coverage.
        stride = max(48, int(round(128 * scale)))
        reg = measure_registration(left, right, tile=256, stride=stride)
        if reg is None:
            return None                  # featureless view: no evidence either way
        measured = reg["row_offset"] / scale
        self.history = (self.history + [measured])[-20:]
        if abs(measured - self.expected) > self.tolerance:
            self.strikes += 1
        else:
            self.strikes = 0
        self.disturbed = self.strikes >= self.confirm
        return measured

    @property
    def latest(self) -> Optional[float]:
        return float(np.median(self.history)) if self.history else None

    def rebase(self, row_offset: float) -> None:
        """Start watching against a newly measured ``row_offset``."""
        self.expected = float(row_offset)
        self.history = []
        self.strikes = 0
        self.disturbed = False

    def describe(self) -> str:
        if not self.history:
            return "registration not yet checked"
        text = (f"row offset {self.latest:+.1f} px measured, "
                f"{self.expected:+.1f} configured")
        if self.disturbed:
            text += (" -- THE EYES HAVE MOVED: recalibrate before trusting "
                     "any range (probe --offsets, then --focal)")
        return text


class EyeTest:
    """Re-measure the vertical registration between the eyes, a few frames' worth.

    The same measurement as ``probe --offsets``, done in place on the frames
    the robot is already capturing: phase correlation on tiles of both halves,
    ``readings`` times, on frames with enough texture to say anything. It
    passes only if the readings agree with each other to ``max_spread`` px,
    so one frame of a passing person or a wrapped tile cannot set the value.

    The first ``settle`` frames are skipped: the test starts the moment the
    lens check trips, and the robot needs a moment to stop.

    This measures the vertical part only. A lens turned in its thread moves
    the image sideways as well, and that changes ``disparity_offset``, which
    needs targets at known distances (``probe --focal``) to measure.
    """

    def __init__(self, readings: int = 5, max_frames: int = 40,
                 max_spread: float = 1.5, settle: int = 2):
        self.readings = max(1, int(readings))
        self.max_frames = max(self.readings, int(max_frames))
        self.max_spread = float(max_spread)
        self.settle = max(0, int(settle))
        self.frames = 0
        self.values: List[float] = []
        self.done = False
        self.ok = False
        self.value: Optional[float] = None
        self.reason = ""

    def feed(self, left: np.ndarray, right: np.ndarray,
             scale: float = 1.0) -> bool:
        """One frame. Returns True once the test has finished, pass or fail."""
        if self.done:
            return True
        self.frames += 1
        if self.frames > self.settle:
            stride = max(48, int(round(128 * scale)))
            reg = measure_registration(left, right, tile=256, stride=stride)
            if reg is not None:
                self.values.append(reg["row_offset"] / scale)
        if len(self.values) >= self.readings:
            spread = float(np.max(self.values) - np.min(self.values))
            self.value = float(np.median(self.values))
            self.ok = spread <= self.max_spread
            self.reason = "" if self.ok else (
                f"readings disagree by {spread:.1f} px "
                f"(allowed {self.max_spread:.1f})")
            self.done = True
        elif self.frames >= self.max_frames + self.settle:
            self.reason = (f"only {len(self.values)} of {self.readings} frames "
                           "had enough texture to measure -- point the camera "
                           "at something detailed")
            self.done = True
        return self.done


def split_side_by_side(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side stereo frame into left and right eyes."""
    mid = frame.shape[1] // 2
    return frame[:, :mid], frame[:, mid:]


class StereoDepth:
    """SGBM disparity and metric depth, at a declared resolution."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        rectification: Optional[Rectification] = None,
    ):
        self.cfg = config or VisionConfig()
        self.cfg.validate()
        self.rectification = rectification
        self._matcher = None
        self._matcher_floor: Optional[int] = None
        #: Why the last frame was refused, when it was not an SGBM error.
        self.last_failure = ""
        self.failures = 0
        self.frames = 0
        self.monitor: Optional[RegistrationMonitor] = None
        if rectification is None and self.cfg.registration_check_every > 0:
            self.monitor = RegistrationMonitor(self.cfg.camera.row_offset,
                                               every=self.cfg.registration_check_every)

    # ------------------------------------------------------------------ #

    @property
    def scaled_camera(self) -> CameraModel:
        """The camera model describing the depth maps this object returns
        (for frames at or above ``disparity_scale``, which is all of them in
        normal use; see :meth:`compute`)."""
        return self._full_camera.scaled(self.cfg.disparity_scale)

    @property
    def _full_camera(self) -> CameraModel:
        camera = self.cfg.camera
        if self.rectification is not None:
            camera = CameraModel(
                width=camera.width,
                height=camera.height,
                focal_px=self.rectification.focal_px,
                cx=self.rectification.cx,
                cy=self.rectification.cy,
                baseline=self.rectification.baseline,
                mount_height=camera.mount_height,
                mount_pitch=camera.mount_pitch,
                mount_forward=camera.mount_forward,
                pan_offset=camera.pan_offset,
                # Deliberately not carried over: a rectified pair is parallel
                # and row-aligned by construction, so both offsets are zero.
                # Keeping the raw camera's values here would subtract the
                # error twice.
            )
        return camera

    def _build_matcher(self, floor_disparity: int):
        import cv2

        cfg = self.cfg
        block = cfg.block_size
        # Start the search where depth actually begins. With a large offset a
        # window starting at zero spends most of its range on disparities no
        # surface in front of the camera can produce, and the near limit
        # silently moves in by the same amount.
        return cv2.StereoSGBM_create(
            minDisparity=floor_disparity,
            numDisparities=cfg.num_disparities,
            blockSize=block,
            P1=8 * 3 * block**2,
            P2=32 * 3 * block**2,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=cfg.speckle_window,
            speckleRange=cfg.speckle_range,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    # ------------------------------------------------------------------ #

    def compute(
        self, left: np.ndarray, right: np.ndarray,
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Optional[np.ndarray], CameraModel]:
        """Return ``(depth_metres, camera_for_that_depth_map)``.

        ``roi`` -- ``(x1, y1, x2, y2)`` in the left eye's pixels as given --
        matches only that box and leaves the rest of the map ``inf``. For
        ranging one person with terrain mapping off: the cost goes with the
        box, not the frame. The search margin to the box's left, which the
        right eye needs to find anything, is added here.

        Depth is at ``disparity_scale`` of the *camera's* full resolution,
        whatever size the eyes arrive at, and the returned camera model says
        so. Frames decoded small to save time (``capture_scale``) are
        recognised by their width against ``camera.width`` and everything --
        the row shift, the resize, the intrinsics -- follows from that. Eyes
        smaller than ``disparity_scale`` are matched as they are, at their own
        scale, never upsampled. Invalid matches are ``inf``, never a
        plausible-looking number and never NaN: ``inf`` propagates through
        comparisons the way a missing reading should, where NaN poisons every
        mean it touches.
        """
        import cv2

        self.frames += 1
        full = self.cfg.camera
        arrived = left.shape[1] / full.width
        if abs(left.shape[0] / full.height - arrived) > 0.02:
            # Not a scaled copy of the calibrated view: a different camera
            # mode. Scaling the intrinsics would produce confident nonsense.
            self.failures += 1
            self.last_failure = (
                f"eyes are {left.shape[1]}x{left.shape[0]}, not a scaled "
                f"{full.width}x{full.height}: the camera is in a different mode")
            return None, self.scaled_camera
        if self.monitor is not None:
            try:
                self.monitor.observe(left, right, scale=arrived)
            except Exception:
                pass                     # a diagnostic must never stop the robot

        try:
            if self.rectification is not None:
                if abs(arrived - 1.0) > 1e-6:
                    # The maps are built for the full image.
                    size = (full.width, full.height)
                    left = cv2.resize(left, size, interpolation=cv2.INTER_LINEAR)
                    right = cv2.resize(right, size, interpolation=cv2.INTER_LINEAR)
                    arrived = 1.0
                left, right = self.rectification.apply(left, right)
            elif full.row_offset:
                # Block matching only searches along rows, so a vertical error
                # does not bias the result, it destroys matches outright.
                # Sub-pixel, because the measured value rarely lands on an
                # integer and rounding half a pixel away is a real cost.
                right = _shift_rows(cv2, right, full.row_offset * arrived)

            effective = min(self.cfg.disparity_scale, arrived)
            resize = effective / arrived
            if abs(resize - 1.0) > 1e-6:
                left = cv2.resize(left, None, fx=resize, fy=resize,
                                  interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, None, fx=resize, fy=resize,
                                   interpolation=cv2.INTER_AREA)
            camera = self._full_camera.scaled(effective)
            floor = int(round(camera.disparity_offset))
            if self._matcher is None or floor != self._matcher_floor:
                self._matcher = self._build_matcher(floor)
                self._matcher_floor = floor

            if left.ndim == 3:
                left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            if right.ndim == 3:
                right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            if roi is None:
                raw = self._matcher.compute(left, right).astype(np.float32) / 16.0
            else:
                raw = self._match_roi(left, right, roi, resize, floor)
        except cv2.error:
            # Counted, not swallowed. A silent failure here leaves the caller
            # using the previous depth map with no sign that it is stale.
            self.failures += 1
            return None, self.scaled_camera

        # Only disparity above the offset carries depth; see
        # CameraModel.disparity_offset. Subtracting it here rather than
        # folding it into the focal length keeps the near/far behaviour right:
        # the offset is an additive floor, not a scale factor.
        relative = raw - camera.disparity_offset
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = camera.focal_px * camera.baseline / relative
        depth[relative <= 0.0] = np.inf
        depth[~np.isfinite(depth)] = np.inf
        return depth, camera

    def _match_roi(self, left, right, roi, resize: float, floor: int) -> np.ndarray:
        """SGBM on a box only, embedded in an otherwise empty disparity map."""
        h, w = left.shape[:2]
        block = self.cfg.block_size
        x1, y1, x2, y2 = (float(v) * resize for v in roi)
        # Rows: the box plus the block's reach. Columns: the right eye sees a
        # point at xL - d, so the crop has to start a whole search range to
        # the left of the box, or the box's left edge has nothing to match.
        ry1 = max(0, int(y1) - block)
        ry2 = min(h, int(np.ceil(y2)) + block)
        cx2 = min(w, int(np.ceil(x2)) + block)
        cx1 = max(0, int(x1) - (floor + self.cfg.num_disparities + block))
        raw = np.zeros((h, w), np.float32)           # 0 is below every floor: invalid
        if ry2 - ry1 <= 2 * block or cx2 - cx1 <= self.cfg.num_disparities + floor + block:
            return raw
        crop = self._matcher.compute(
            np.ascontiguousarray(left[ry1:ry2, cx1:cx2]),
            np.ascontiguousarray(right[ry1:ry2, cx1:cx2]),
        ).astype(np.float32) / 16.0
        raw[ry1:ry2, cx1:cx2] = crop
        return raw

    def set_row_offset(self, row_offset: float) -> None:
        """Use a newly measured ``row_offset`` from the next frame on.

        Changes the shared camera model in place, so everything holding this
        config -- the page, the warnings -- sees the value actually in use.
        The lens check is rebased to watch for movement from here.
        """
        self.cfg.camera.row_offset = float(row_offset)
        if self.monitor is not None:
            self.monitor.rebase(row_offset)

    @property
    def geometry_trusted(self) -> bool:
        """False once the eyes are seen to have moved since calibration."""
        return not (self.monitor is not None and self.monitor.disturbed)

    def health(self) -> str:
        rate = self.failures / max(1, self.frames)
        return (
            f"{self.frames} frames, {self.failures} failures ({rate * 100:.1f}%), "
            f"{'rectified' if self.rectification else 'UNRECTIFIED - depth is provisional'}"
            + (f"; {self.monitor.describe()}" if self.monitor is not None else "")
            + (f"; last refusal: {self.last_failure}" if self.last_failure else "")
        )


def sample_depth(
    depth: np.ndarray,
    u: float,
    v: float,
    half_patch: int = 9,
    min_samples: int = 8,
) -> Tuple[float, int]:
    """Median depth over a patch. Returns ``(metres, samples_used)``.

    A patch median rather than one pixel, for two reasons. Stereo depth has
    impulsive outliers, so the median is the right estimator and a mean is not.
    And a single pixel is frequently an invalid match, which in the original
    code became NaN and then poisoned the smoothing buffer for five frames.
    """
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(0, ui - half_patch), min(w, ui + half_patch + 1)
    v0, v1 = max(0, vi - half_patch), min(h, vi + half_patch + 1)
    if u1 <= u0 or v1 <= v0:
        return float("inf"), 0

    patch = depth[v0:v1, u0:u1]
    good = patch[np.isfinite(patch)]
    if good.size < min_samples:
        return float("inf"), int(good.size)
    return float(np.median(good)), int(good.size)


__all__ = [
    "Rectification",
    "StereoDepth",
    "load_calibration",
    "sample_depth",
    "split_side_by_side",
]
