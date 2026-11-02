from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from typing import Sequence, List, Optional, Tuple

import numpy as np

from pathlib import Path

from .camera import (
    CameraDevice,
    CameraNotFound,
    ENV_OVERRIDE,
    VideoMode,
    enumerate_cameras,
    fourcc_of,
    measure_fps,
    open_camera,
    open_strategies,
    quiet_opencv,
    save_pin,
    select_camera,
)
from .camera import KNOWN_COMPRESSED, wire_compressed
from .recording import save_recording
from .stereo import measure_registration  # noqa: F401  (re-exported)
from .types import VisionConfig

OK, WARN, BAD, INFO = "ok", "warn", "BAD", "--"


class Report:
    """Accumulates findings so the verdict can be printed at the end."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, status: str, what: str, detail: str = "") -> None:
        self.rows.append((status, what, detail))
        mark = {OK: "  ok  ", WARN: " warn ", BAD: " FAIL ", INFO: "      "}[status]
        print(f"[{mark}] {what}" + (f"\n         {detail}" if detail else ""))

    @property
    def failures(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == BAD)

    @property
    def warnings(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == WARN)


def heading(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


# --------------------------------------------------------------------------- #


def list_devices(devices: List[CameraDevice]) -> None:
    heading(f"Capture devices ({platform.system()})")
    if not devices:
        print("  none found")
        return
    for device in devices:
        print(device.describe())
        print()


def grab(cap, count: int, warmup: int = 5) -> Tuple[List[np.ndarray], float, int]:
    """Grab frames, discarding a few so auto-exposure settles first.

    Returns ``(frames, seconds, failed reads)``. The failure count is reported
    rather than printed per-read: a backend that has stopped delivering emits one
    warning line per attempt, and forty identical lines bury the finding they
    are trying to convey.
    """
    import cv2

    frames: List[np.ndarray] = []
    failed = 0
    with quiet_opencv(cv2):
        for _ in range(warmup):
            cap.read()
        began = time.monotonic()
        for _ in range(count):
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
            else:
                failed += 1
        elapsed = time.monotonic() - began
    return frames, elapsed, failed


def benchmark_strategies(device: CameraDevice, mode: VideoMode,
                         fourcc: str, frames: int = 25):
    """Open the camera every way available, measure each, print a table.

    Returns the best working strategy, or None. Empirical on purpose: which
    backend and which pixel format a given camera actually honours is not
    predictable from reputation, and guessing it cost this project a working
    camera once already.
    """
    import cv2

    from .camera import _try_open

    rows, working = [], []
    with quiet_opencv(cv2):
        for name, api, fmt, first in open_strategies(cv2, fourcc):
            order = "fmt,size" if first else "size,fmt"
            cap, result = _try_open(cv2, device.open_target, api, mode, fmt,
                                    first, 1, verify=6)
            if cap is None:
                time.sleep(0.5)          # see survey_strategies: DSHOW lets go slowly
                cap, result = _try_open(cv2, device.open_target, api, mode, fmt,
                                        first, 1, verify=6)
            if cap is None:
                rows.append((name, fmt or "any", order, "-", "-", str(result)))
                continue
            try:
                fps = measure_fps(cap, frames=frames)
                got = fourcc_of(cap)
                shown = got if got and got.isprintable() and got.strip() else "?"
                packed = wire_compressed(got, mode.width, mode.height, fps)
                note = ("compressed -- inferred from the frame rate"
                        if packed and got.upper() not in KNOWN_COMPRESSED else "")
                rows.append((name, fmt or "any", order, shown, f"{fps:.1f}", note))
                working.append((fps, packed, name, fmt, first, shown))
            finally:
                cap.release()

    wide = max((len(r[0]) for r in rows), default=8)
    print(f"  {'backend':<{wide}}  {'asked':<6}  {'order':<8}  {'got':<5}  "
          f"{'fps':>5}  why not")
    for name, asked, order, got, fps, note in rows:
        print(f"  {name:<{wide}}  {asked:<6}  {order:<8}  {got:<5}  {fps:>5}  {note}")

    if not working:
        print()
        print("  Nothing worked. That is the finding.")
        return None

    # Compressed first, then frame rate. A working uncompressed stream beats a
    # compressed one that cannot sustain frames, which is why liveness is
    # already filtered out above rather than traded off here.
    working.sort(key=lambda r: (r[1], r[0]), reverse=True)
    fps, compressed, name, fmt, first, got = working[0]
    print()
    print(f"  Best: {name} asking for {fmt or 'anything'} "
          f"({'format first' if first else 'size first'}) -> {got} at {fps:.1f} fps")
    if not compressed:
        print(f"  Nothing here negotiated a compressed format. At "
              f"{mode.width}x{mode.height} that caps the frame rate at roughly "
              f"{24e6 / (mode.width * mode.height * 2):.0f} fps over USB 2.0.")
    return {"backend": name, "fourcc": fmt, "fourcc_first": first,
            "fps": fps, "got": got}


def eye_order(left_half: np.ndarray, right_half: np.ndarray,
              cfg: VisionConfig) -> Tuple[str, float, float, float, float]:
    """Which half is the left eye? Returns (verdict, as-is %, swapped %, d1, d2).

    SGBM searches disparities from 0 upward only. With the halves in the right
    order a real stereo pair produces valid matches over most of the overlap;
    with them swapped the true disparities are negative and it finds almost
    nothing. The asymmetry is large and unambiguous.
    """
    import cv2

    def grey(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    scale = cfg.disparity_scale
    a = cv2.resize(grey(left_half), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(grey(right_half), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = _align_rows(cv2, b, cfg.camera.row_offset * scale)

    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=cfg.num_disparities,
        blockSize=cfg.block_size,
        P1=8 * 3 * cfg.block_size ** 2, P2=32 * 3 * cfg.block_size ** 2,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window, speckleRange=cfg.speckle_range,
        disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    def score(first, second) -> Tuple[float, float]:
        disp = matcher.compute(first, second).astype(np.float32) / 16.0
        valid = disp > 0.5
        frac = float(valid.mean())
        median = float(np.median(disp[valid])) if valid.any() else 0.0
        return frac, median

    as_is, d_as_is = score(a, b)
    swapped, d_swapped = score(b, a)

    if max(as_is, swapped) < 0.02:
        verdict = "no correspondence either way"
    elif as_is > swapped * 1.5:
        verdict = "left half is the LEFT eye (as the code assumes)"
    elif swapped > as_is * 1.5:
        verdict = "left half is the RIGHT eye -- the halves are SWAPPED"
    else:
        verdict = "ambiguous"
    return verdict, as_is, swapped, d_as_is, d_swapped


def _align_rows(cv2, right: np.ndarray, dy: float) -> np.ndarray:
    """Shift the right eye the way :class:`StereoDepth` does, by ``row_offset``.

    It has to be the same shift, in the same direction. This used to move the
    image the opposite way, so with any ``row_offset`` set the calibration
    matched rows twice the offset apart -- mostly garbage -- and ``--focal``
    solved a focal length and disparity offset from it that put everything
    at a fraction of its real distance.
    """
    if not dy:
        return right
    from .stereo import _shift_rows

    return _shift_rows(cv2, right, dy)


def measure_disparity(left_half: np.ndarray, right_half: np.ndarray,
                      cfg: VisionConfig,
                      patch: float = 0.35,
                      search: int = 256) -> Optional[Tuple[float, int]]:
    """Median disparity over the middle of the frame, at ``disparity_scale``.

    Returns ``(median_disparity, samples)``, or None if too little matched.

    The search window is deliberately much wider than the running config's:
    this runs before the disparity offset is known, and a window sized for the
    depth range alone cannot see past an offset of a few tens of pixels. It
    just returns nothing, which reads as "no texture" rather than "look
    further". Calibration happens once, so the extra time is free.

    Only the middle of the frame is used, on the assumption that is where the
    target is, and the median is taken because stereo disparity has impulsive
    outliers that would wreck a mean.
    """
    import cv2

    def grey(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    scale = cfg.disparity_scale
    a = cv2.resize(grey(left_half), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(grey(right_half), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = _align_rows(cv2, b, cfg.camera.row_offset * scale)

    span = max(cfg.num_disparities, 16 * (int(search) // 16))
    span = min(span, 16 * ((a.shape[1] - 1) // 16))
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=span,
        blockSize=cfg.block_size,
        P1=8 * 3 * cfg.block_size ** 2, P2=32 * 3 * cfg.block_size ** 2,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window, speckleRange=cfg.speckle_range,
        disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(a, b).astype(np.float32) / 16.0

    h, w = disparity.shape
    ch, cw = int(h * patch / 2), int(w * patch / 2)
    centre = disparity[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
    good = centre[centre > 0.5]
    if good.size < 200:
        return None
    return float(np.median(good)), int(good.size)


def constrain_geometry(distance_m: float, disparity: float, cfg: VisionConfig,
                       span: Optional[Tuple[float, float]] = None,
                       focals: Sequence[float] = (842, 700, 600, 500, 420, 350, 300),
                       ) -> List[dict]:
    """What a *single* known distance does tell you.

    One range cannot separate ``f`` from ``d0``, but it does tie them together:
    ``d0 = d - f*B/z``. Walking a few candidate focal lengths along that line
    shows which pairs are physically possible, and a wrong one usually gives
    itself away --- a negative ``d0`` means the eyes diverge, and a focal
    length that puts the far wall of a room at twenty metres is not right
    either.

    ``span`` is the ``(min, max)`` raw disparity seen across the frame, used to
    report what each candidate implies about the nearest and farthest thing in
    shot. ``disparity`` and ``span`` are at full resolution.
    """
    B = cfg.camera.baseline
    out = []
    for f in focals:
        d0 = disparity - f * B / distance_m
        row = {"focal_px": float(f),
               "hfov_deg": math.degrees(2 * math.atan(cfg.camera.width / (2 * f))),
               "disparity_offset": float(d0),
               "physical": d0 > -2.0}
        if span is not None:
            lo, hi = span
            row["nearest"] = f * B / (hi - d0) if hi > d0 else float("inf")
            row["farthest"] = f * B / (lo - d0) if lo > d0 else float("inf")
        out.append(row)
    return out


def solve_geometry(samples: Sequence[Tuple[float, float]],
                   cfg: VisionConfig) -> Optional[dict]:
    """Solve for focal length *and* disparity offset from known distances.

    ``samples`` is ``[(distance_m, median_disparity_at_scale), ...]``.

    The model is ``d = f*B/z + d0``, which is linear in ``1/z``. Fitting both
    terms is not optional refinement: a dual-lens module whose eyes are toed in
    or whose halves are packed out of register has a ``d0`` of tens of pixels,
    and solving ``f = z*d/B`` from a single distance folds all of it into the
    focal length. That gives a focal length wrong by whatever ``d0`` happened
    to be and an answer that looks perfectly self-consistent, because one point
    can always be fitted exactly by one parameter.

    Two distances is the minimum and fits exactly, so it cannot be checked;
    three or more give a residual worth reading. Spread them widely --- the
    slope is set by the difference in ``1/z``, so 1 m and 1.2 m tell you almost
    nothing while 1 m and 4 m tell you a lot.
    """
    if len(samples) < 2:
        return None
    scale = cfg.disparity_scale
    inv = np.array([1.0 / z for z, _d in samples], dtype=float)
    obs = np.array([d for _z, d in samples], dtype=float)
    A = np.stack([inv, np.ones_like(inv)], axis=1)
    (slope, offset), *_ = np.linalg.lstsq(A, obs, rcond=None)
    if slope <= 0.0:
        return None                      # disparity must fall with distance
    predicted = A @ [slope, offset]
    residual = float(np.sqrt(np.mean((obs - predicted) ** 2)))
    spread = float(np.var(obs))
    r2 = 1.0 - float(np.var(obs - predicted)) / spread if spread > 1e-12 else 1.0
    focal_scaled = slope / cfg.camera.baseline
    return {
        "focal_px": focal_scaled / scale,          # full resolution
        "disparity_offset": offset / scale,        # full resolution
        "residual_px": residual,                   # at disparity_scale
        "r2": r2,
        "samples": len(samples),
        "predicted": [float(v) for v in predicted],
    }


def check_alignment(left_half: np.ndarray, right_half: np.ndarray) -> Optional[float]:
    """Rough vertical offset between the eyes, in pixels, by phase correlation.

    Unrectified disparity is only as good as this number is small: a pair out by
    more than a couple of pixels vertically will not match on the epipolar line
    SGBM searches along.
    """
    import cv2

    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return np.float32(g) / 255.0

    a, b = prep(left_half), prep(right_half)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    try:
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (_dx, dy), _response = cv2.phaseCorrelate(a[:h, :w], b[:h, :w], window)
        return float(dy)
    except cv2.error:
        return None


# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    report = Report()
    from carrybot_config import ConfigError, load_config

    try:
        robot = load_config(getattr(args, "config", None))
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    cfg = robot.vision
    print(robot.describe_source())

    heading("Environment")
    print(f"  python   {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    try:
        import cv2
        print(f"  opencv   {cv2.__version__}")
    except ImportError:
        print("  opencv   NOT INSTALLED -- pip install opencv-python")
        return 2

    devices = enumerate_cameras(max_index=args.max_index)
    list_devices(devices)

    if args.list:
        return 0

    heading("Which camera")
    try:
        device, why = select_camera(devices, override=args.device or robot.live.camera or None,
                                   require_stereo=not args.any_camera)
    except CameraNotFound as exc:
        print(exc)
        report.add(BAD, "could not identify a stereo camera")
        _verdict(report)
        return 1
    report.add(OK, f"selected {device.path}", f"because: {why}")
    if device.usb_id:
        report.add(INFO, f"usb {device.usb_id}"
                         + (f" serial {device.serial}" if device.serial else ""),
                   device.name)
    if device.stable_path:
        report.add(OK, "has a stable path", device.stable_path)
    elif platform.system() == "Linux":
        report.add(WARN, "no /dev/v4l/by-id symlink",
                   "device numbering can move between boots; see the udev rule "
                   "in tools/install/")

    heading("Opening it")
    mode = VideoMode(args.width, args.height) if args.width and args.height else None
    try:
        cap, got, info = open_camera(device, mode=mode, fourcc=args.fourcc,
                                     backend=args.backend)
    except CameraNotFound as exc:
        print(exc)
        report.add(BAD, "could not open the camera at a stereo mode")
        _verdict(report)
        return 1
    report.add(OK, f"delivered {got}", f"via {info.backend}; frames verified at "
                                       f"this shape, not just the property")
    if info.fourcc_honoured:
        report.add(OK, f"pixel format {info.fourcc}", "as requested")
    elif info.compression_inferred:
        rate = info.measured_fps or info.reported_fps
        raw = rate * got.width * got.height * 2 / 1e6
        report.add(OK, "compressed on the wire (the backend does not say which format)",
                   f"{rate:.0f} fps at {got.width}x{got.height} would be {raw:.0f} MB/s "
                   f"raw; USB 2.0 carries about 24, so it cannot be uncompressed")
    elif info.compressed:
        report.add(WARN, f"pixel format is {info.fourcc or 'unknown'}, "
                         f"not the requested {info.requested_fourcc}",
                   "it is still compressed, so bandwidth is probably fine")
    else:
        report.add(BAD, f"pixel format is {info.fourcc or 'unknown'} -- uncompressed",
                   f"an uncompressed {got.width}x{got.height} frame is "
                   f"{got.width * got.height * 2 / 1e6:.1f} MB, which caps USB 2.0 "
                   f"at about {24e6 / (got.width * got.height * 2):.0f} fps. "
                   f"The same camera in MJPG typically does 30. See the backend "
                   f"table below.")
    if info.reported_fps:
        report.add(INFO, f"camera reports {info.reported_fps:.0f} fps for this mode")

    try:
        frames, elapsed, failed = grab(cap, args.frames)
    finally:
        cap.release()

    heading("Frames")
    if not frames:
        report.add(BAD, f"captured no frames ({failed} reads failed)",
                   "the device opened, verified, and then stopped delivering. "
                   "If another program has the camera, close it and try again.")
        _verdict(report)
        return 1
    if failed:
        report.add(WARN, f"{failed} of {args.frames} reads failed",
                   "an intermittent stream; the follow loop will see gaps")
    fps = len(frames) / max(1e-9, elapsed)
    report.add(OK, f"captured {len(frames)}/{args.frames} frames",
               f"{fps:.1f} fps at {got}")
    wire_ok = wire_compressed(info.fourcc, got.width, got.height, fps)
    if fps < 10:
        report.add(WARN, f"{fps:.1f} fps is low",
                   "the follow loop assumes roughly 5-8 Hz after YOLO and SGBM; "
                   "below that the motion compensation is doing a lot of work")

    winner = None
    if args.benchmark or not wire_ok:
        heading("Every way of opening it, measured")
        winner = benchmark_strategies(device, got, args.fourcc,
                                      frames=max(20, args.frames // 2))
        if winner and winner["fps"] > fps * 1.5:
            report.add(WARN, f"a better strategy exists: {winner['backend']} at "
                             f"{winner['got']}, {winner['fps']:.1f} fps",
                       f"against {fps:.1f} fps for the one chosen. "
                       f"Run with --pin to record it.")

    frame = frames[-1]
    h, w = frame.shape[:2]
    report.add(INFO, f"frame is {w}x{h}, {frame.dtype}, "
                     f"{'colour' if frame.ndim == 3 else 'mono'}")
    if (w, h) != (got.width, got.height):
        report.add(BAD, f"frame shape {w}x{h} disagrees with the reported mode {got}",
                   "trust the frame, not the property")

    heading("Is it really side by side?")
    if w % 2:
        report.add(WARN, f"width {w} is odd", "splitting in half loses a column")
    left, right = frame[:, : w // 2], frame[:, w // 2:]
    diff = float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))
    if diff < 1.0:
        report.add(BAD, "the two halves are identical",
                   "this is one image doubled, not a stereo pair")
    else:
        report.add(OK, "the halves differ", f"mean absolute difference {diff:.1f} levels")
    for name, half in (("left", left), ("right", right)):
        spread = float(np.std(half))
        if spread < 4.0:
            report.add(BAD, f"{name} half is nearly blank (std {spread:.1f})",
                       "lens cap on, or that eye is not streaming")
        else:
            report.add(OK, f"{name} half has content", f"std {spread:.1f} levels")

    heading("Which half is the left eye?")
    verdict, as_is, swapped, d_a, d_b = eye_order(left, right, cfg)
    print(f"  as written : {as_is:6.1%} of pixels matched, median disparity {d_a:5.1f} px")
    print(f"  swapped    : {swapped:6.1%} of pixels matched, median disparity {d_b:5.1f} px")
    print()
    if verdict.startswith("left half is the LEFT"):
        report.add(OK, verdict, "split_side_by_side() is correct as it stands")
    elif verdict.startswith("left half is the RIGHT"):
        report.add(BAD, verdict,
                   "swap the halves in split_side_by_side(), or every depth "
                   "reading will be garbage")
    elif verdict == "no correspondence either way":
        report.add(BAD, verdict,
                   "either this is not a stereo camera, or it is in a single-eye "
                   "mode, or the scene is too featureless to match -- point it at "
                   "something textured and try again")
    else:
        report.add(WARN, verdict, "too close to call; try a more textured scene")

    dy: Optional[float] = None           # right eye relative to left, rows
    if args.check_alignment:
        heading("Vertical alignment")
        dy = check_alignment(left, right)
        if dy is None:
            report.add(WARN, "could not estimate vertical offset")
        elif abs(dy) <= 2.0:
            report.add(OK, f"eyes are aligned to {dy:+.1f} px vertically")
        else:
            report.add(WARN, f"eyes are {dy:+.1f} px out vertically",
                       "SGBM matches along rows, so this caps how well it can "
                       "ever do; it wants stereoRectify with a real calibration")

    reg = measure_registration(left, right) if args.offsets else None
    if reg is not None and dy is None:
        dy = -reg["row_offset"]

    heading("What the code assumes about this camera")
    eye_w, eye_h = got.width // 2, got.height
    assumed = cfg.camera
    rows = [
        ("frame is side by side, split in half",
         f"{got.width}x{got.height} -> two {eye_w}x{eye_h} eyes",
         OK if diff >= 1.0 else BAD),
        ("per-eye resolution",
         f"config says {assumed.width}x{assumed.height}, camera gives {eye_w}x{eye_h}",
         OK if (assumed.width, assumed.height) == (eye_w, eye_h) else BAD),
        ("left half is the left eye", verdict,
         OK if verdict.startswith("left half is the LEFT") else BAD),
        (f"focal length {assumed.focal_px:.0f} px",
         f"implies {np.degrees(assumed.hfov):.1f} deg horizontal field of view "
         f"at {eye_w} px wide -- NOT MEASURED HERE",
         WARN),
        (f"baseline {assumed.baseline * 1000:.0f} mm",
         "CameraModel.baseline -- scales every range, so it should be a "
         "measurement, lens centre to lens centre",
         INFO),
        ("images are row-aligned (no rectification)",
         "not measured -- add --offsets" if dy is None else
         f"measured {abs(dy):.1f} px out; configured row_offset "
         f"{assumed.row_offset:+.1f}",
         INFO if dy is None else
         (OK if abs(dy + assumed.row_offset) <= 2.0 or abs(dy) <= 2.0 else WARN)),
    ]
    for what, detail, status in rows:
        report.add(status, what, detail)

    if d_a > 0.5 and verdict.startswith("left half is the LEFT"):
        scaled = assumed.scaled(cfg.disparity_scale)
        heading("Sanity check on distance")
        full = d_a / cfg.disparity_scale
        print(f"  median raw disparity {full:.1f} px at full resolution")
        if assumed.disparity_offset == 0.0:
            print("  The disparity offset is not calibrated, so no distance can be")
            print("  given. Raw disparity is not depth on this kind of camera: a")
            print("  point at infinity reads tens of pixels, not zero. Run --focal at")
            print("  two or more measured distances to solve for it.")
        else:
            print(f"  offset {assumed.disparity_offset:.1f} px, focal "
                  f"{assumed.focal_px:.0f} px -> median scene depth "
                  f"{scaled.depth_at(d_a):.2f} m")
            print("  Check that against a tape measure before trusting it.")

    if args.offsets:
        heading("Measuring the registration between the two halves")
        print("  Point the camera at a textured scene with something distant in")
        print("  view -- down a corridor, or out of a window. Both numbers are")
        print("  read off the scene, so a blank wall gives nothing.")
        print()
        if reg is None:
            report.add(WARN, "could not measure the registration",
                       "too little texture, or the halves are not a stereo pair")
        else:
            print(f"  {reg['tiles']} usable tiles ({reg['skipped']} skipped)")
            print(f"  vertical   right eye needs shifting {reg['row_offset']:+.2f} px "
                  f"(spread {reg['row_spread']:.2f})")
            print(f"  horizontal disparity spans about {reg['disparity_span']:.1f} px, "
                  f"smallest seen {reg['disparity_ceiling']:.1f} px")
            print()
            print("  Put this in carrybot.toml, under [camera]:")
            print(f"      row_offset = {reg['row_offset']:.1f}")
            print()
            print(f"  disparity_offset is NOT given here. The smallest disparity in")
            print(f"  shot ({reg['disparity_ceiling']:.1f} px) is only the offset if something is")
            print("  effectively at infinity; indoors the farthest thing is a wall a few")
            print("  metres off, so this is an upper bound and usually a loose one.")
            print("  Use --focal at two known distances, which solves for it.")
            print()
            if abs(reg['row_offset']) > 2.0:
                report.add(WARN,
                           f"the eyes are {abs(reg['row_offset']):.1f} px out vertically",
                           "block matching searches along rows, so this costs "
                           "matches outright -- set row_offset and it comes back")
            else:
                report.add(OK, f"eyes row-aligned to {abs(reg['row_offset']):.1f} px")
            if reg['row_spread'] > 2.0:
                report.add(WARN,
                           f"the vertical error varies by {reg['row_spread']:.1f} px "
                           f"across the frame",
                           "a single shift cannot fix a varying offset; this pair "
                           "needs a real stereoRectify calibration to do better")
            if abs(reg['disparity_ceiling']) > 4.0:
                report.add(WARN,
                           f"disparity offset is somewhere below "
                           f"{reg['disparity_ceiling']:.0f} px, and is not zero",
                           "a parallel, correctly-split pair reads ~0 at infinity. "
                           "Treating this floor as depth puts the whole scene at "
                           "one distance. Confirm it with --focal at two ranges")

    if args.focal:
        heading("Solving for focal length and disparity offset")
        print(f"  baseline taken as {cfg.camera.baseline * 1000:.0f} mm "
              f"(measured lens centre to lens centre)")
        print("  using the middle third of the frame, so fill it with the target")
        print()
        if len(args.focal) < 2:
            distance = args.focal[0]
            input(f"  put a flat textured target at {distance:.2f} m, "
                  f"fill the centre of the frame, then press Enter... ")
            cap2, _mode2, _info2 = open_camera(device, mode=got, fourcc=args.fourcc,
                                   backend=info.backend)
            try:
                shots, _elapsed, _failed = grab(cap2, 10)
            finally:
                cap2.release()
            measured = measure_disparity(shots[-1][:, :shots[-1].shape[1] // 2],
                                         shots[-1][:, shots[-1].shape[1] // 2:],
                                         cfg) if shots else None
            if measured is None:
                report.add(BAD, f"no usable measurement at {distance:.2f} m")
            else:
                full = measured[0] / cfg.disparity_scale
                print(f"\n  {distance:.2f} m -> raw disparity {full:.1f} px "
                      f"at full resolution\n")
                print("  One range cannot separate focal length from offset, but it does")
                print("  tie them together. These are the pairs consistent with it:\n")
                print(f"  {'focal':>7} {'HFOV':>7} {'offset':>8}   "
                      f"{'nearest':>8} {'farthest':>9}")
                for row in constrain_geometry(distance, full, cfg,
                                              span=(full - 10.0, full + 10.0)):
                    flag = "" if row["physical"] else "  <- offset negative, unphysical"
                    print(f"  {row['focal_px']:7.0f} {row['hfov_deg']:6.1f}d "
                          f"{row['disparity_offset']:8.1f}   "
                          f"{row.get('nearest', 0):7.2f}m {row.get('farthest', 0):8.2f}m{flag}")
                print()
            report.add(BAD, "one distance is not enough to calibrate",
                       "disparity is f*B/z + d0, which has two unknowns. A single "
                       "range fits it exactly however wrong d0 is, and quietly "
                       "folds d0 into the focal length. Pass --focal twice, far "
                       "apart -- 1 m and 4 m, not 1 m and 1.2 m")
        else:
            samples = []
            for distance in args.focal:
                input(f"  put a flat textured target at {distance:.2f} m, "
                      f"fill the centre of the frame, then press Enter... ")
                cap2, _mode2, _info2 = open_camera(device, mode=got, fourcc=args.fourcc,
                                   backend=info.backend)
                try:
                    shots, _elapsed, _failed = grab(cap2, 10)
                finally:
                    cap2.release()
                if not shots:
                    report.add(BAD, f"no frames at {distance:.2f} m")
                    continue
                shot = shots[-1]
                half = shot.shape[1] // 2
                measured = measure_disparity(shot[:, :half], shot[:, half:], cfg)
                if measured is None:
                    report.add(WARN, f"not enough matched pixels at {distance:.2f} m",
                               "the target needs visible texture -- a newspaper or a "
                               "patterned cloth works, a blank wall does not")
                    continue
                median, samples_n = measured
                samples.append((distance, median))
                print(f"    {distance:5.2f} m -> median disparity {median:5.1f} px "
                      f"({samples_n} samples)")

            solved = solve_geometry(samples, cfg)
            if solved is None:
                report.add(BAD, "could not solve the geometry",
                           "need at least two distances that actually differ in "
                           "disparity, with the nearer one reading higher")
            else:
                print()
                focal = solved["focal_px"]
                offset = solved["disparity_offset"]
                assumed = cfg.camera.focal_px
                print(f"  focal length     {focal:7.1f} px at full resolution "
                      f"(config says {assumed:.0f})")
                print(f"  disparity offset {offset:7.1f} px at full resolution "
                      f"(config says {cfg.camera.disparity_offset:.0f})")
                print(f"  implied HFOV     {math.degrees(2 * math.atan(cfg.camera.width / (2 * focal))):7.1f} deg")
                if solved["samples"] > 2:
                    print(f"  fit residual     {solved['residual_px']:7.2f} px  "
                          f"(r2 {solved['r2']:.4f})")
                print()
                print("  Put these in carrybot.toml, under [camera]:")
                print(f"      focal_px = {focal:.1f}")
                print(f"      disparity_offset = {offset:.1f}")
                print()
                if solved["samples"] == 2:
                    report.add(WARN, "two distances fit exactly, so nothing checks them",
                               "add a third range to get a residual worth reading")
                elif solved["r2"] < 0.98:
                    report.add(WARN, f"the fit is poor (r2 {solved['r2']:.3f})",
                               "one of the distances or the target was off, or the "
                               "target did not fill the centre of the frame")
                else:
                    report.add(OK, f"clean fit across {solved['samples']} ranges "
                                   f"(r2 {solved['r2']:.4f})")
                ratio = focal / max(1e-9, assumed)
                if not 0.85 <= ratio <= 1.15:
                    report.add(WARN, f"measured focal is {ratio:.2f}x the configured one",
                               "every distance the robot reports scales with this, "
                               "so update the config before trusting any range")

    if args.record:
        heading("Recording")
        extra: List[np.ndarray] = list(frames)
        if len(extra) < args.record_frames:
            cap3, _m3, _i3 = open_camera(device, mode=got, fourcc=args.fourcc,
                             backend=info.backend)
            try:
                more, _e, _f = grab(cap3, args.record_frames - len(extra))
            finally:
                cap3.release()
            extra.extend(more)
        if extra:
            meta = {
                "backend": info.backend, "fourcc": info.fourcc,
                "requested_fourcc": info.requested_fourcc,
                "mode": str(got), "fps": round(fps, 2),
                "device": str(device.path), "device_name": device.name,
                "usb_id": device.usb_id,
                "camera": {"focal_px": assumed.focal_px, "baseline": assumed.baseline,
                           "disparity_offset": assumed.disparity_offset,
                           "row_offset": assumed.row_offset},
            }
            if reg is not None:
                meta["registration"] = reg
            if args.note:
                meta["note"] = args.note
            save_recording(args.record, extra, meta)
            size = Path(args.record).stat().st_size / 1e6
            report.add(OK, f"wrote {len(extra)} frames to {args.record}",
                       f"{size:.1f} MB -- replay it with "
                       f"`python -m carrybot_vision.cli --replay {args.record} --view`")
        else:
            report.add(BAD, "nothing to record")

    if args.pin:
        heading("Pinning")
        if not device.usb_id and not device.stable_path:
            report.add(WARN, "nothing stable to pin",
                       "no USB id and no by-id symlink on this platform; the "
                       "index is all there is")
        else:
            path = save_pin(device, strategy=winner)
            detail = str(path)
            if winner:
                detail += (f"  (also remembered {winner['backend']}/"
                           f"{winner['fourcc'] or 'any'} so startup skips the search)")
            report.add(OK, f"pinned to {device.usb_id or device.stable_path}", detail)

    return _verdict(report)


def _verdict(report: Report) -> int:
    heading("Verdict")
    if report.failures:
        print(f"  {report.failures} failure(s), {report.warnings} warning(s). "
              "Fix the failures before trusting any depth from this camera.")
        return 1
    if report.warnings:
        print(f"  Usable. {report.warnings} thing(s) worth attention above -- "
              "the focal length and baseline are the ones that set your accuracy.")
        return 0
    print("  All checks passed.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Identify and test the CarryBot stereo camera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--list", action="store_true",
                   help="enumerate capture devices and stop")
    p.add_argument("--device", default=None,
                   help=f"force one: a path, an index, or vid:pid (same as {ENV_OVERRIDE})")
    p.add_argument("--pin", action="store_true",
                   help="remember this camera by USB id, so it is found next time")
    p.add_argument("--frames", type=int, default=40, help="frames to capture")
    p.add_argument("--width", type=int, default=0, help="force a frame width")
    p.add_argument("--height", type=int, default=0, help="force a frame height")
    p.add_argument("--fourcc", default="MJPG", help="pixel format to request")
    p.add_argument("--max-index", type=int, default=8,
                   help="how many indices to probe where enumeration is unavailable")
    p.add_argument("--any-camera", action="store_true",
                   help="accept a device with no side-by-side mode")
    p.add_argument("--check-alignment", action="store_true",
                   help="also estimate the vertical offset between the eyes")
    p.add_argument("--config", metavar="PATH", default=None,
                   help="robot config file (default: carrybot.toml)")
    p.add_argument("--backend", default=None, metavar="NAME",
                   help="use only this capture backend (DSHOW, MSMF, V4L2) and "
                        "fail rather than fall back -- for comparing backends")
    p.add_argument("--note", default="", metavar="TEXT",
                   help="free text stored in a --record file, e.g. what is in "
                        "shot and how far away it is")
    p.add_argument("--offsets", action="store_true",
                   help="measure the row and disparity offsets between the two "
                        "halves and print the CameraModel values to paste in")
    p.add_argument("--record", metavar="FILE.npz", default=None,
                   help="save raw frames for replay, so the rest of the "
                        "pipeline can be exercised away from the camera")
    p.add_argument("--record-frames", type=int, default=120,
                   help="how many frames to record")
    p.add_argument("--benchmark", action="store_true",
                   help="time every capture backend (automatic if the requested "
                        "pixel format was not honoured)")
    p.add_argument("--focal", type=float, action="append", metavar="METRES",
                   default=None,
                   help="solve for focal length AND disparity offset using a "
                        "flat textured target at this distance. REPEAT at a "
                        "second, well-separated range: two unknowns need two "
                        "measurements, and a single range folds the offset "
                        "into the focal length without complaining.")
    args = p.parse_args(argv)

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
