
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

from .pipeline import CameraSource, ListSource, VisionPipeline, VisionStatus
from .types import VisionConfig


# --------------------------------------------------------------------------- #
# Configuration from the command line
# --------------------------------------------------------------------------- #


def build_config(args: argparse.Namespace,
                 base: Optional[VisionConfig] = None) -> VisionConfig:
    """The config file's vision settings, with any flags given on top.

    Flags default to None and only override what they name. They used to carry
    their own defaults (``--focal 842``, ``--baseline 0.085``...), which meant
    a calibrated camera model was silently replaced by the uncalibrated one
    every time this tool ran.
    """
    from dataclasses import replace

    base = base or VisionConfig()
    cam = base.camera
    camera_changes = {}
    if args.width is not None:
        camera_changes["width"] = args.width // 2
    if args.height is not None:
        camera_changes["height"] = args.height
    for flag, name in (("focal", "focal_px"), ("baseline", "baseline"),
                       ("mount_height", "mount_height"), ("mount_forward", "mount_forward")):
        if getattr(args, flag) is not None:
            camera_changes[name] = getattr(args, flag)
    if args.mount_pitch is not None:
        camera_changes["mount_pitch"] = float(np.radians(args.mount_pitch))
    if camera_changes:
        if "width" in camera_changes or "height" in camera_changes:
            # Let the principal point follow a new size unless it was set.
            camera_changes.setdefault("cx", None)
            camera_changes.setdefault("cy", None)
        cam = replace(cam, **camera_changes)

    changes = {"camera": cam}
    for flag, name in (("scale", "disparity_scale"), ("disparities", "num_disparities"),
                       ("max_range", "max_range"), ("resolution", "grid_resolution")):
        if getattr(args, flag) is not None:
            changes[name] = getattr(args, flag)
    cfg = replace(base, **changes)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def draw(pipeline: VisionPipeline, status: VisionStatus) -> bool:
    """One OpenCV window per useful signal. Returns False when asked to quit."""
    import cv2

    depth = pipeline.last_depth
    if depth is not None:
        finite = np.isfinite(depth)
        shown = np.zeros(depth.shape, np.uint8)
        if finite.any():
            near, far = 0.4, pipeline.cfg.max_range
            norm = np.clip((depth - near) / (far - near), 0, 1)
            shown[finite] = (255 * (1.0 - norm[finite])).astype(np.uint8)
        cv2.imshow("depth", cv2.applyColorMap(shown, cv2.COLORMAP_TURBO))

    scan = pipeline.last_scan
    if scan is not None:
        # Row 0 is the robot's *right*, and +x (forward) runs along the
        # columns. Transposing and flipping puts forward up the screen, which
        # is the only orientation anyone can read at a glance.
        rgb = np.zeros((*scan.shape, 3), np.uint8)
        rgb[scan == 254] = (60, 60, 60)        # unknown
        rgb[scan == 0] = (30, 110, 30)         # free
        rgb[scan == 255] = (40, 40, 220)       # obstacle
        view = np.flipud(np.transpose(rgb, (1, 0, 2)))
        half = pipeline.cfg.grid_half
        cv2.circle(view, (half, half), 3, (255, 255, 255), -1)
        cmd = status.command
        if cmd.goal is not None:
            gx = int(half + cmd.goal[0] / pipeline.cfg.grid_resolution)
            gy = int(half - cmd.goal[1] / pipeline.cfg.grid_resolution)
            if 0 <= gx < scan.shape[0] and 0 <= gy < scan.shape[1]:
                cv2.circle(view, (gy, scan.shape[0] - 1 - gx), 4, (0, 255, 255), -1)
        cv2.imshow("occupancy", cv2.resize(
            view, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def run_check(cfg: VisionConfig) -> int:
    print(cfg.describe())
    scaled = cfg.camera.scaled(cfg.disparity_scale)
    print()
    print(f"{'range':>8} {'disparity':>10} {'error':>8}  {'note':<40}")
    for r in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        d = scaled.disparity_for(r)
        err = scaled.range_resolution(r)
        note = ""
        if d - scaled.disparity_offset > cfg.num_disparities:
            note = "TOO CLOSE - outside the disparity search"
        elif r > cfg.max_range:
            note = "beyond max_range, discarded"
        elif err > 0.35:
            note = "error exceeds the standoff band; expect hunting"
        print(f"{r:7.2f}m {d:9.1f}px {err * 100:6.1f}cm  {note:<40}")
    print()
    print(f"The camera sees {np.degrees(scaled.hfov):.0f} degrees of the "
          f"{cfg.grid_size}x{cfg.grid_size} cell array it fills, so most of that "
          f"array stays UNKNOWN.\nThat is correct and the planner is built for "
          f"it -- but do not set allow_unknown=False unless the robot has "
          f"another sensor.")
    return 0


def run_loop(
    pipeline: VisionPipeline,
    limit: Optional[int],
    view: bool,
    quiet: bool,
    record: Optional[str],
) -> int:
    frames: List[np.ndarray] = []
    seen = 0
    began = time.monotonic()
    try:
        while limit is None or seen < limit:
            frame = pipeline.source.read()
            if frame is None:
                break
            if record is not None:
                frames.append(frame)
            status = pipeline.step(frame)
            seen += 1
            if not quiet:
                line = status.summary()
                if pipeline.planner is not None:
                    snap = pipeline.planner.latest()
                    line = f"{line}  | plan {snap.status.name}"
                print(line, flush=True)
            if view and not draw(pipeline, status):
                break
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)

    elapsed = time.monotonic() - began
    print(file=sys.stderr)
    print(pipeline.health(), file=sys.stderr)
    if seen:
        print(f"  {seen} frames in {elapsed:.1f} s = {seen / max(1e-9, elapsed):.1f} fps",
              file=sys.stderr)

    if record is not None and frames:
        from .recording import save_recording

        meta = {"fps": round(seen / max(1e-9, elapsed), 2)}
        src = pipeline.source
        info = getattr(src, "info", None)
        if info is not None:
            meta.update(backend=info.backend, fourcc=info.fourcc,
                        requested_fourcc=info.requested_fourcc, mode=str(src.mode),
                        device=str(src.device.path), usb_id=src.device.usb_id)
        cam = pipeline.cfg.camera
        meta["camera"] = {"focal_px": cam.focal_px, "baseline": cam.baseline,
                          "disparity_offset": cam.disparity_offset,
                          "row_offset": cam.row_offset}
        save_recording(record, frames, meta)
        print(f"  wrote {len(frames)} frames to {record}", file=sys.stderr)
    if view:
        import cv2

        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="CarryBot stereo person-follower bench tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--check", action="store_true",
                   help="print the geometry the calibration implies, then exit")
    p.add_argument("--view", action="store_true", help="show depth and occupancy windows")
    p.add_argument("--plan", action="store_true",
                   help="run a PathPlanner on the output and report its status")
    p.add_argument("--quiet", action="store_true", help="suppress the per-frame line")
    p.add_argument("-n", "--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--record", metavar="FILE.npz", help="save raw frames for replay")
    p.add_argument("--replay", metavar="FILE.npz", help="run over recorded frames")

    cam = p.add_argument_group("camera")
    cam.add_argument("--device", default=None,
                     help="force a camera: a path, an index, or vid:pid. "
                          "Left unset, the stereo camera is identified "
                          "automatically -- see `python -m carrybot_vision.probe`")
    cam.add_argument("--width", type=int, default=None, help="combined frame width")
    cam.add_argument("--height", type=int, default=None)
    cam.add_argument("--focal", type=float, default=None,
                     help="focal length in pixels, at full per-eye resolution")
    cam.add_argument("--baseline", type=float, default=None, help="metres")
    cam.add_argument("--mount-height", type=float, default=None, help="metres above floor")
    cam.add_argument("--mount-pitch", type=float, default=None, help="degrees, down positive")
    cam.add_argument("--mount-forward", type=float, default=None,
                     help="metres ahead of the wheel axis")

    tune = p.add_argument_group("processing")
    tune.add_argument("--scale", type=float, default=None, help="disparity resolution")
    tune.add_argument("--disparities", type=int, default=None, help="multiple of 16")
    tune.add_argument("--max-range", type=float, default=None, help="metres")
    tune.add_argument("--resolution", type=float, default=None,
                      help="occupancy cell size; must match PlannerConfig")
    tune.add_argument("--weights", default=None)
    tune.add_argument("--standoff", type=float, default=None,
                      help="metres to keep behind the person")

    p.add_argument("--config", metavar="PATH", default=None,
                   help="robot config file; flags given here override it")
    args = p.parse_args(argv)

    from carrybot_config import ConfigError, load_config

    try:
        robot = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(robot.describe_source(), file=sys.stderr)
    try:
        cfg = build_config(args, robot.vision)
    except ValueError as exc:
        print(f"bad configuration: {exc}", file=sys.stderr)
        return 2

    if args.check:
        return run_check(cfg)

    if args.replay:
        from .recording import describe, load_recording

        from .pipeline import shrink_frames

        frames, meta = load_recording(args.replay)
        source = ListSource(shrink_frames(frames, cfg.capture_scale))
        print(f"replaying {len(source.frames)} frames from {args.replay}",
              file=sys.stderr)
        print(f"  recorded: {describe(meta)}", file=sys.stderr)
        if "note" in meta:
            print(f"  note:     {meta['note']}", file=sys.stderr)
    else:
        try:
            source = CameraSource(args.device or robot.live.camera or None,
                                  args.width or 0, args.height or 0,
                                  scale=cfg.capture_scale)
            if cfg.threaded_capture:
                from .pipeline import LatestFrameSource

                source = LatestFrameSource(source, nice=cfg.nice)
            print(f"camera: {source.describe()}", file=sys.stderr)
        except (RuntimeError, ImportError) as exc:
            print(f"camera unavailable: {exc}", file=sys.stderr)
            print("try --check for geometry, or --replay to work from a recording",
                  file=sys.stderr)
            return 1

    planner = None
    if args.plan:
        from dataclasses import replace as _replace

        from carrybot_nav import PathPlanner

        planner = PathPlanner(
            _replace(robot.planner, resolution=cfg.grid_resolution)).start()

    from dataclasses import replace

    from .tracker import PersonTracker

    follow = robot.follow
    if args.standoff is not None:
        follow = replace(follow, standoff=args.standoff)
    pipeline = VisionPipeline(
        source, cfg,
        follow_config=follow,
        planner=planner,
        tracker=PersonTracker(cfg, weights=args.weights or robot.live.weights),
        head_config=robot.head if robot.live.servo_fitted else None,
    )
    print(cfg.describe(), file=sys.stderr)
    print(file=sys.stderr)

    try:
        return run_loop(pipeline, args.frames, args.view, args.quiet, args.record)
    finally:
        if planner is not None:
            snap = planner.latest()
            print(f"  planner: {snap.status.name}, {len(snap.poses)} poses, "
                  f"{planner.plan_count} plans, {planner.reuse_count} reuses, "
                  f"{planner.costmap.stats()['lethal_cells']} obstacle cells",
                  file=sys.stderr)
            planner.stop()
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
