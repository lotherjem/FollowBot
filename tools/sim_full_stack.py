from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carrybot_drive import DriveConfig, DriveController, Gains
from carrybot_drive.backend import SimBackend
from carrybot_drive.bridge import NavBridge
from carrybot_drive.odometry import wrap_angle
from carrybot_nav import PathPlanner, PlannerConfig
from carrybot_nav import Pose as NavPose
from tools.simulate import RayScanner, _footprint_hits, free_cell, make_world


@dataclass
class TruePose:
    """Where the robot really is, as opposed to where it believes it is."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class Frame:
    true_pose: TruePose
    odom_pose: Tuple[float, float, float]
    known: np.ndarray
    origin: Tuple[float, float]
    path_world: List[Tuple[float, float]]
    plan_status: str
    drift: float
    linear: float
    angular: float


@dataclass
class Result:
    reached: bool = False
    arrivals: int = 0
    elapsed: float = 0.0
    distance: float = 0.0
    drift: float = 0.0
    heading_drift: float = 0.0
    collisions: int = 0
    encoder_errors: int = 0
    desyncs: int = 0
    loop_overruns: int = 0
    frames: List[Frame] = field(default_factory=list)


def simulate_full_stack(
    scenario: str = "open",
    seed: int = 3,
    size_cells: int = 160,
    sensor_range: float = 2.0,
    sensor_hfov: float = 360.0,
    duration: float = 45.0,
    goals: int = 1,
    wheel_error: float = 1.0,
    track_error: float = 1.0,
    calibrated: bool = True,
    skew: float = 0.12,
    capture: bool = True,
    capture_period: float = 0.25,
    verbose: bool = True,
) -> Result:
    """Run the whole stack against a synthetic building.

    ``wheel_error`` and ``track_error`` scale the *true* geometry away from what
    the config believes, which is what an uncalibrated robot actually has. They
    are the fastest way to see why the ``c`` calibration in the drive console
    is not optional.
    """
    rng = np.random.default_rng(seed)
    world = make_world(scenario, size_cells, rng)

    nav_cfg = PlannerConfig()
    drive_cfg = DriveConfig()
    if calibrated:
        # Matches the simulated motor: 18 rad/s at 100% duty, 10% deadband.
        drive_cfg.left_gains = Gains(kv=100.0 / 18.0, ks=10.0)
        drive_cfg.right_gains = Gains(kv=100.0 / 18.0, ks=10.0)

    # The true machine, which the config only approximates.
    true_wheel_radius = drive_cfg.wheel_radius_m * wheel_error
    true_track = drive_cfg.track_width_m * track_error

    if scenario == "trap":
        mid = size_cells // 2
        start_rc, goal_rc = (mid, mid - 40), (mid, mid + 40)
    else:
        start_rc = free_cell(world, rng)
        goal_rc = free_cell(world, rng)
        while abs(goal_rc[0] - start_rc[0]) + abs(goal_rc[1] - start_rc[1]) < size_cells:
            goal_rc = free_cell(world, rng)

    res = nav_cfg.resolution
    truth = TruePose(x=start_rc[1] * res, y=start_rc[0] * res, theta=0.0)
    goal_xy = (goal_rc[1] * res, goal_rc[0] * res)

    scanner = RayScanner(sensor_range, res, hfov=math.radians(sensor_hfov))
    sim = SimBackend(drive_cfg, skew=skew)
    drive = DriveController(drive_cfg, sim).start()
    sim.start_thread()
    planner = PathPlanner(nav_cfg).start()
    bridge = NavBridge(planner, drive, rate_hz=20.0).start()

    # The planner's frame is anchored on the drive's odometry, which starts at
    # the origin. So the goal goes in relative to the robot, exactly as the
    # real system will supply it.
    planner.set_goal(goal_xy[0] - truth.x, goal_xy[1] - truth.y)

    result = Result()
    remaining_goals = goals
    last_left = sim.left.angle
    last_right = sim.right.angle
    start_time = time.monotonic()
    next_capture = 0.0

    try:
        while True:
            now = time.monotonic() - start_time
            if now > duration:
                break

            # -- advance ground truth from the simulated wheels ------------ #
            left_angle, right_angle = sim.left.angle, sim.right.angle
            d_left = (left_angle - last_left) * true_wheel_radius
            d_right = (right_angle - last_right) * true_wheel_radius
            last_left, last_right = left_angle, right_angle

            d_centre = 0.5 * (d_left + d_right)
            d_theta = (d_right - d_left) / true_track
            mid_theta = truth.theta + 0.5 * d_theta
            candidate = TruePose(
                x=truth.x + d_centre * math.cos(mid_theta),
                y=truth.y + d_centre * math.sin(mid_theta),
                theta=wrap_angle(truth.theta + d_theta),
            )

            # Walls block the body but not the wheels: the encoders keep
            # counting while the robot goes nowhere, which is exactly how a
            # real robot's odometry runs away when it noses into something.
            if _footprint_hits(world, NavPose(candidate.x, candidate.y, candidate.theta), nav_cfg):
                result.collisions += 1
                truth.theta = candidate.theta
            else:
                result.distance += abs(d_centre)
                truth = candidate

            # -- sensor ---------------------------------------------------- #
            scan = scanner.scan(world, NavPose(truth.x, truth.y, truth.theta), (0.0, 0.0))
            planner.submit_scan(scan)

            # -- drift: believed pose vs true pose ------------------------- #
            state = drive.state()
            believed = (
                state.pose.x + start_rc[1] * res,
                state.pose.y + start_rc[0] * res,
                state.pose.theta,
            )
            drift = math.hypot(believed[0] - truth.x, believed[1] - truth.y)

            if math.hypot(goal_xy[0] - truth.x, goal_xy[1] - truth.y) <= 0.15:
                result.reached = True
                result.arrivals += 1
                remaining_goals -= 1
                if remaining_goals <= 0:
                    break
                # Nothing is restarted here: the planner is still running, the
                # drive is still closing its loop, and a new goal simply takes
                # effect on the next cycle.
                goal_rc = free_cell(world, rng)
                goal_xy = (goal_rc[1] * res, goal_rc[0] * res)
                _set_goal_from_world(planner, goal_xy, start_rc, res)
                if verbose:
                    print(f"  reached goal {result.arrivals}, "
                          f"next at ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")

            if capture and now >= next_capture:
                next_capture = now + capture_period
                result.frames.append(
                    _capture(planner, bridge, truth, believed, drift, start_rc, res)
                )

            time.sleep(0.05)

        result.elapsed = time.monotonic() - start_time
        result.drift = drift
        result.heading_drift = abs(wrap_angle(believed[2] - truth.theta))
        result.encoder_errors = state.left_errors + state.right_errors
        result.desyncs = state.left_desyncs + state.right_desyncs
        result.loop_overruns = state.loop_overruns
    finally:
        bridge.stop()
        planner.stop()
        drive.close()
        sim.stop_thread()

    if verbose:
        straight = math.hypot(goal_xy[0] - start_rc[1] * res, goal_xy[1] - start_rc[0] * res)
        print(f"scenario={scenario} seed={seed} "
              f"wheel_error={wheel_error} track_error={track_error} "
              f"calibrated={calibrated}")
        print(f"  goals reached    : {result.arrivals}")
        print(f"  wall clock       : {result.elapsed:.1f} s")
        print(f"  distance driven  : {result.distance:.2f} m "
              f"(straight line {straight:.2f} m)")
        print(f"  odometry drift   : {result.drift * 1000:.0f} mm, "
              f"{math.degrees(result.heading_drift):.1f} deg")
        print(f"  wall contacts    : {result.collisions}")
        print(f"  encoder err/desync: {result.encoder_errors} / {result.desyncs}")
        print(f"  control overruns : {result.loop_overruns}")
        if bridge.last_error:
            print("  BRIDGE ERROR:", bridge.last_error)
        if drive.last_error:
            print("  DRIVE ERROR:", drive.last_error)

    result._world = world  # type: ignore[attr-defined]
    result._goal = goal_xy  # type: ignore[attr-defined]
    result._start = (start_rc[1] * res, start_rc[0] * res)  # type: ignore[attr-defined]
    result._res = res  # type: ignore[attr-defined]
    return result


def _set_goal_from_world(planner, goal_xy, start_rc, res) -> None:
    """The planner takes goals relative to the robot; convert from the map."""
    gx = goal_xy[0] - start_rc[1] * res
    gy = goal_xy[1] - start_rc[0] * res
    robot = planner.robot_pose()
    dx, dy = gx - robot.x, gy - robot.y
    cos_t, sin_t = math.cos(-robot.theta), math.sin(-robot.theta)
    planner.set_goal(dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)


def _capture(planner, bridge, truth, believed, drift, start_rc, res) -> Frame:
    snapshot = planner.latest()
    status = bridge.status()

    path_world: List[Tuple[float, float]] = []
    if snapshot.is_drivable:
        # Poses are relative to the plan's reference pose, in the odometry
        # frame. Lift them into the map frame for drawing.
        ref = snapshot.reference_pose
        offset_x, offset_y = start_rc[1] * res, start_rc[0] * res
        cos_t, sin_t = math.cos(ref.theta), math.sin(ref.theta)
        for pose in snapshot.poses:
            path_world.append((
                offset_x + ref.x + pose.x * cos_t - pose.y * sin_t,
                offset_y + ref.y + pose.x * sin_t + pose.y * cos_t,
            ))

    return Frame(
        true_pose=TruePose(truth.x, truth.y, truth.theta),
        odom_pose=believed,
        known=planner.costmap.cost.copy(),
        origin=planner.costmap.origin,
        path_world=path_world,
        plan_status=snapshot.status.value,
        drift=drift,
        linear=status.linear,
        angular=status.angular,
    )


# --------------------------------------------------------------------------- #


def render(result: Result, gif_path: Optional[str], live: bool) -> None:
    import matplotlib

    if not live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import BoundaryNorm, ListedColormap

    frames = result.frames
    if not frames:
        print("no frames captured")
        return

    world = result._world  # type: ignore[attr-defined]
    goal = result._goal  # type: ignore[attr-defined]
    start = result._start  # type: ignore[attr-defined]
    res = result._res  # type: ignore[attr-defined]
    extent = [0, world.shape[1] * res, 0, world.shape[0] * res]

    fig, (ax_world, ax_map) = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.patch.set_facecolor("#101418")
    for ax in (ax_world, ax_map):
        ax.set_facecolor("#101418")
        ax.set_aspect("equal")
        ax.tick_params(colors="#8a94a0", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2a323c")

    ax_world.imshow(
        world, cmap=ListedColormap(["#1b222b", "#5a6572"]), origin="lower",
        extent=extent, interpolation="nearest",
    )
    ax_world.set_title("ground truth  ·  blue = real, orange = odometry",
                       color="#cdd6e0", fontsize=10)
    ax_map.set_title("what the planner knows", color="#cdd6e0", fontsize=10)

    levels = [0, 1, 60, 130, 200, 254, 255, 256]
    cmap = ListedColormap(
        ["#16202b", "#1d3040", "#2b4a55", "#6a5a34", "#a8632f", "#0b0e12", "#d9484b"]
    )
    norm = BoundaryNorm(levels, cmap.N)
    im = ax_map.imshow(frames[0].known, cmap=cmap, norm=norm, origin="lower",
                       interpolation="nearest")

    ax_world.plot([start[0]], [start[1]], "s", color="#63d297", ms=7)
    ax_world.plot([goal[0]], [goal[1]], "*", color="#ffd166", ms=15)
    (true_trail,) = ax_world.plot([], [], color="#4ea1ff", lw=1.6)
    (odom_trail,) = ax_world.plot([], [], color="#ff9f45", lw=1.2, alpha=0.85)
    (robot,) = ax_world.plot([], [], "o", color="#4ea1ff", ms=8)
    (heading,) = ax_world.plot([], [], color="#ffffff", lw=1.6)
    (path_line,) = ax_world.plot([], [], ".-", color="#ffd166", lw=1.4, ms=3, alpha=0.9)

    text = fig.text(0.5, 0.03, "", ha="center", color="#cdd6e0",
                    fontsize=9, family="monospace")

    tx: List[float] = []
    ty: List[float] = []
    ox: List[float] = []
    oy: List[float] = []

    def update(i: int):
        frame = frames[i]
        tx.append(frame.true_pose.x)
        ty.append(frame.true_pose.y)
        ox.append(frame.odom_pose[0])
        oy.append(frame.odom_pose[1])

        true_trail.set_data(tx, ty)
        odom_trail.set_data(ox, oy)
        robot.set_data([frame.true_pose.x], [frame.true_pose.y])
        heading.set_data(
            [frame.true_pose.x, frame.true_pose.x + 0.35 * math.cos(frame.true_pose.theta)],
            [frame.true_pose.y, frame.true_pose.y + 0.35 * math.sin(frame.true_pose.theta)],
        )
        if frame.path_world:
            path_line.set_data([p[0] for p in frame.path_world],
                               [p[1] for p in frame.path_world])
        else:
            path_line.set_data([], [])

        known = frame.known
        ox0, oy0 = frame.origin
        im.set_data(known)
        im.set_extent([ox0, ox0 + known.shape[1] * res,
                       oy0, oy0 + known.shape[0] * res])
        ax_map.set_xlim(ox0, ox0 + known.shape[1] * res)
        ax_map.set_ylim(oy0, oy0 + known.shape[0] * res)

        text.set_text(
            f"frame {i + 1}/{len(frames)}   plan={frame.plan_status:<9}"
            f"   v={frame.linear:+.2f} m/s  w={frame.angular:+.2f} rad/s"
            f"   odometry drift={frame.drift * 1000:5.0f} mm"
        )
        return true_trail, odom_trail, robot, heading, path_line, im

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    if gif_path:
        anim.save(gif_path, writer=animation.PillowWriter(fps=10), dpi=80)
        print(f"wrote {gif_path}")
    if live:
        plt.show()
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="open", choices=["open", "rooms", "maze", "trap"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--size", type=int, default=160, help="world size in cells")
    ap.add_argument("--sensor-range", type=float, default=2.0)
    ap.add_argument("--sensor-fov", type=float, default=360.0,
                    help="sensor field of view in degrees; 74.5 matches the "
                         "stereo camera the real robot actually has")
    ap.add_argument("--duration", type=float, default=45.0, help="seconds of real time")
    ap.add_argument("--goals", type=int, default=1,
                    help="keep going for this many goals; a new one is chosen "
                         "on each arrival without restarting anything")
    ap.add_argument("--wheel-error", type=float, default=1.0,
                    help="true wheel radius / configured (1.05 = 5%% larger than believed)")
    ap.add_argument("--track-error", type=float, default=1.0,
                    help="true track width / configured")
    ap.add_argument("--skew", type=float, default=0.12,
                    help="mismatch between the two motors, 0 for a perfect pair")
    ap.add_argument("--uncalibrated", action="store_true",
                    help="run with kv = ks = 0, as a fresh robot would")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    result = simulate_full_stack(
        scenario=args.scenario,
        seed=args.seed,
        size_cells=args.size,
        sensor_range=args.sensor_range,
        sensor_hfov=args.sensor_fov,
        duration=args.duration,
        goals=args.goals,
        wheel_error=args.wheel_error,
        track_error=args.track_error,
        skew=args.skew,
        calibrated=not args.uncalibrated,
        capture=not args.no_plot,
    )

    if not args.no_plot and (args.gif or args.live):
        render(result, args.gif, args.live)
    return 0 if result.reached else 1


if __name__ == "__main__":
    raise SystemExit(main())
