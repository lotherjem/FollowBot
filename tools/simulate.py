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

from carrybot_nav import LETHAL, FREE, UNKNOWN, PathPlanner, PlannerConfig, PlanStatus, Pose
from carrybot_nav.types import wrap_angle
from carrybot_drive.servo import ServoConfig, SimulatedServo
from carrybot_vision.follow import FollowState
from carrybot_vision.head import HeadConfig, HeadController

# --------------------------------------------------------------------------- #
# World generation
# --------------------------------------------------------------------------- #


def make_world(scenario: str, size_cells: int, rng: np.random.Generator) -> np.ndarray:
    world = np.full((size_cells, size_cells), FREE, dtype=np.uint8)
    world[0, :] = LETHAL
    world[-1, :] = LETHAL
    world[:, 0] = LETHAL
    world[:, -1] = LETHAL

    if scenario == "open":
        for _ in range(18):
            r = int(rng.integers(10, size_cells - 20))
            c = int(rng.integers(10, size_cells - 20))
            h = int(rng.integers(4, 16))
            w = int(rng.integers(4, 16))
            world[r : r + h, c : c + w] = LETHAL

    elif scenario == "rooms":
        # Two internal walls with a doorway in each, plus clutter.
        for frac in (0.35, 0.68):
            c = int(size_cells * frac)
            world[:, c : c + 2] = LETHAL
            door = int(rng.integers(int(size_cells * 0.15), int(size_cells * 0.8)))
            world[door : door + 14, c : c + 2] = FREE
        for frac in (0.5,):
            r = int(size_cells * frac)
            world[r : r + 2, :] = LETHAL
            door = int(rng.integers(int(size_cells * 0.1), int(size_cells * 0.85)))
            world[r : r + 2, door : door + 14] = FREE
        for _ in range(12):
            r = int(rng.integers(5, size_cells - 12))
            c = int(rng.integers(5, size_cells - 12))
            world[r : r + int(rng.integers(3, 9)), c : c + int(rng.integers(3, 9))] = LETHAL

    elif scenario == "maze":
        step = max(12, size_cells // 12)
        for r in range(step, size_cells - step, step):
            world[r : r + 2, :] = LETHAL
            gaps = rng.integers(2, 5)
            for _ in range(int(gaps)):
                c = int(rng.integers(2, size_cells - 20))
                world[r : r + 2, c : c + 18] = FREE
        for c in range(step, size_cells - step, step * 2):
            world[:, c : c + 2] = LETHAL
            for _ in range(3):
                r = int(rng.integers(2, size_cells - 20))
                world[r : r + 18, c : c + 2] = FREE

    elif scenario == "trap":
        # A cul-de-sac directly between robot and goal: the planner must give
        # up on the direct route and commit to going around.
        mid = size_cells // 2
        world[mid - 30 : mid + 30, mid : mid + 2] = LETHAL
        world[mid - 30 : mid - 28, mid - 25 : mid + 2] = LETHAL
        world[mid + 28 : mid + 30, mid - 25 : mid + 2] = LETHAL

    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    return world


def free_cell(world: np.ndarray, rng: np.random.Generator, margin: int = 8) -> Tuple[int, int]:
    """A cell with `margin` cells of clearance, relaxing the margin if the
    scenario is too tight to offer one."""
    h, w = world.shape
    while margin >= 2:
        for _ in range(5000):
            r = int(rng.integers(margin, h - margin))
            c = int(rng.integers(margin, w - margin))
            if world[r - margin : r + margin, c - margin : c + margin].max() == FREE:
                return r, c
        margin -= 2
    raise RuntimeError("could not find a free cell")


# --------------------------------------------------------------------------- #
# Sensor
# --------------------------------------------------------------------------- #


def _vision_for(cfg: PlannerConfig, hfov_deg: float):
    """A VisionConfig whose cone and grid match this simulator's sensor.

    The head controller reasons about what it can see, so handing it the real
    camera's 74.5 degrees while the simulator casts a full circle would have it
    survey for gaps that the sensor had already filled in.
    """
    from carrybot_vision.types import CameraModel, VisionConfig

    hfov = math.radians(min(179.0, max(1.0, hfov_deg)))
    # focal length that yields this field of view at the nominal width
    width = 1280
    focal = (width / 2.0) / math.tan(hfov / 2.0)
    return VisionConfig(
        camera=CameraModel(width=width, height=720, focal_px=focal),
        grid_resolution=cfg.resolution,
        grid_size=161,
    )


class RayScanner:
    """Ranging sensor with occlusion, rendered in the robot frame.

    ``hfov`` is the horizontal field of view in radians, centred on the
    robot's forward axis. The default of a full turn is the convenient lie --
    convenient because every direction is observed and the planner never has to
    reason about what is behind it, a lie because CarryBot's sensor is a stereo
    camera with a 75-degree cone and no rear view whatsoever.

    Simulate with the cone. A policy that only works with 360-degree vision
    looks perfect here and walks the real robot into the doorframe it passed
    thirty centimetres ago, because the planner was never asked what to do
    about a goal it cannot see.
    """

    def __init__(
        self,
        range_m: float,
        resolution: float,
        n_rays: int = 720,
        hfov: float = 2.0 * math.pi,
    ):
        self.range_m = range_m
        self.res = resolution
        self.radius = int(math.ceil(range_m / resolution))
        self.size = 2 * self.radius + 1
        self.hfov = max(1e-3, min(hfov, 2.0 * math.pi))
        if self.hfov >= 2.0 * math.pi - 1e-6:
            self.angles = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
        else:
            # Same angular density as the full-circle case, so narrowing the
            # cone costs proportionally less work instead of secretly
            # increasing the resolution inside it.
            rays = max(16, int(round(n_rays * self.hfov / (2.0 * math.pi))))
            self.angles = np.linspace(-self.hfov / 2.0, self.hfov / 2.0, rays)
        self._steps = np.arange(0.0, self.radius + 1.0, 0.5)

    def scan(
        self,
        world: np.ndarray,
        pose: Pose,
        world_origin: Tuple[float, float],
        head_yaw: float = 0.0,
    ) -> np.ndarray:
        """Cast every ray at once. Fully vectorised -- the sim is not the
        thing under test, and a Python ray loop dominates the runtime.

        ``head_yaw`` swings the cone relative to the body, positive to the
        robot's left, modelling a sensor on a pan servo. The array that comes
        back is always in the **robot** frame regardless of where the head is
        pointing, because that is the planner's contract: a scan that arrived
        in the head's frame would offset every obstacle by the head angle and
        look exactly like odometry drift.
        """
        ox, oy = world_origin
        h, w = world.shape
        centre = self.radius
        steps = self._steps  # (S,)

        angles = self.angles + head_yaw if head_yaw else self.angles

        # Robot-frame sample offsets, in cells.
        ca = np.cos(angles)[:, None]
        sa = np.sin(angles)[:, None]
        bx = steps[None, :] * ca
        by = steps[None, :] * sa

        # Same samples in world-grid indices.
        wa = angles[:, None] + pose.theta
        wx = pose.x + steps[None, :] * self.res * np.cos(wa)
        wy = pose.y + steps[None, :] * self.res * np.sin(wa)
        wc = np.rint((wx - ox) / self.res).astype(np.int32)
        wr = np.rint((wy - oy) / self.res).astype(np.int32)

        oob = (wr < 0) | (wr >= h) | (wc < 0) | (wc >= w)
        occ = world[np.clip(wr, 0, h - 1), np.clip(wc, 0, w - 1)]
        blocked = (occ == LETHAL) | oob

        n_steps = steps.shape[0]
        has_hit = blocked.any(axis=1)
        first = np.where(has_hit, blocked.argmax(axis=1), n_steps)

        order = np.arange(n_steps)[None, :]
        free_mask = order < first[:, None]
        hit_mask = (order == first[:, None]) & has_hit[:, None] & ~oob

        ix = np.rint(bx).astype(np.int32) + centre
        iy = np.rint(by).astype(np.int32) + centre
        inside = (ix >= 0) & (ix < self.size) & (iy >= 0) & (iy < self.size)

        out = np.full((self.size, self.size), UNKNOWN, dtype=np.uint8)
        sel = free_mask & inside
        out[iy[sel], ix[sel]] = FREE
        sel = hit_mask & inside
        out[iy[sel], ix[sel]] = LETHAL
        return out


# --------------------------------------------------------------------------- #
# Differential drive follower
# --------------------------------------------------------------------------- #


@dataclass
class DriveLimits:
    v_max: float = 0.45
    w_max: float = 2.2
    #: Above this bearing error the robot pivots in place instead of driving.
    pivot_threshold: float = 0.55
    #: Distance ahead along the path to aim at.
    lookahead: float = 0.45


def follow(poses: List[Pose], limits: DriveLimits) -> Tuple[float, float]:
    """Lookahead controller. Returns body velocities ``(v, omega)``.

    Deliberately simple -- it exists to prove the *plan* is followable, not to
    be a good controller. Two details matter and are worth keeping in whatever
    replaces it:

    * **Aim at a lookahead point, not the next waypoint.** Chasing a waypoint
      the robot is nearly standing on produces a violently swinging bearing.
    * **Use one threshold, not a deadband.** Pivot above it, drive below it,
      with continuous steering either side; a "rotate until aligned, then
      drive" pair of thresholds sets up a limit cycle.
    """
    if len(poses) < 2:
        return 0.0, 0.0

    # Walk the polyline and interpolate the point exactly `lookahead` metres
    # along it. Picking "the first waypoint further away than the lookahead"
    # instead would cut corners -- straight past the doorframe the path was
    # carefully routed through.
    remaining = limits.lookahead
    tx, ty = poses[-1].x, poses[-1].y
    for i in range(1, len(poses)):
        ax, ay = poses[i - 1].x, poses[i - 1].y
        bx, by = poses[i].x, poses[i].y
        seg = math.hypot(bx - ax, by - ay)
        if seg <= 1e-9:
            continue
        if seg >= remaining:
            t = remaining / seg
            tx, ty = ax + t * (bx - ax), ay + t * (by - ay)
            break
        remaining -= seg

    target = Pose(tx, ty, poses[-1].theta)
    dist = math.hypot(target.x, target.y)
    if dist < 1e-3:
        return 0.0, max(-limits.w_max, min(limits.w_max, 2.0 * wrap_angle(target.theta)))

    bearing = math.atan2(target.y, target.x)
    w = max(-limits.w_max, min(limits.w_max, 2.5 * bearing))
    if abs(bearing) > limits.pivot_threshold:
        return 0.0, w

    # Ease off the throttle in proportion to how hard we are turning.
    v = min(limits.v_max, 1.5 * dist) * max(0.15, 1.0 - abs(bearing) / limits.pivot_threshold)
    return v, w


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def _footprint_hits(world: np.ndarray, pose: Pose, cfg: PlannerConfig) -> bool:
    """True if the square chassis, at this pose, overlaps a wall."""
    half = 0.5 * max(cfg.robot_width, cfg.robot_length)
    res = cfg.resolution
    h, w = world.shape
    r0 = int((pose.y - half) / res)
    r1 = int((pose.y + half) / res)
    c0 = int((pose.x - half) / res)
    c1 = int((pose.x + half) / res)
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return True
    return bool((world[r0 : r1 + 1, c0 : c1 + 1] == LETHAL).any())


@dataclass
class Frame:
    pose: Pose
    known: np.ndarray
    #: World coordinates of the centre of costmap cell (0, 0) at capture time.
    origin: Tuple[float, float]
    path: List[Pose]
    status: PlanStatus
    plan_ms: float
    expansions: int


@dataclass
class RunResult:
    reached: bool
    steps: int
    distance: float
    frames: List[Frame] = field(default_factory=list)
    plan_times: List[float] = field(default_factory=list)
    statuses: List[PlanStatus] = field(default_factory=list)
    head_angles: List[float] = field(default_factory=list)
    replans: int = 0
    reuses: int = 0
    collisions: int = 0
    #: Cells of the world the robot had observed by the end. The point of a
    #: head that looks around is that this goes up without the distance doing.
    known_cells: int = 0


def simulate(
    scenario: str = "rooms",
    seed: int = 3,
    size_cells: int = 220,
    sensor_range: float = 2.0,
    sensor_hfov: float = 360.0,
    head: bool = False,
    head_max_angle: float = 60.0,
    dt: float = 0.1,
    max_steps: int = 1600,
    capture_every: int = 4,
    capture: bool = True,
    config: Optional[PlannerConfig] = None,
    verbose: bool = True,
) -> RunResult:
    rng = np.random.default_rng(seed)
    cfg = config or PlannerConfig()
    world = make_world(scenario, size_cells, rng)

    if scenario == "trap":
        mid = size_cells // 2
        start_rc = (mid, mid - 40)
        goal_rc = (mid, mid + 40)
    else:
        start_rc = free_cell(world, rng)
        goal_rc = free_cell(world, rng)
        while abs(goal_rc[0] - start_rc[0]) + abs(goal_rc[1] - start_rc[1]) < size_cells:
            goal_rc = free_cell(world, rng)

    origin = (0.0, 0.0)  # world cell (0,0) sits at world metres (0,0)
    start = Pose(start_rc[1] * cfg.resolution, start_rc[0] * cfg.resolution, 0.0)
    goal_xy = (goal_rc[1] * cfg.resolution, goal_rc[0] * cfg.resolution)

    scanner = RayScanner(sensor_range, cfg.resolution,
                         hfov=math.radians(sensor_hfov))

    # There is no person to follow in this simulator, so an enabled head spends
    # its whole time in SURVEY: pointing at whatever the map is least sure
    # about. That makes this a clean measurement of whether looking around
    # actually helps a robot explore, uncontaminated by tracking.
    head_cfg = HeadConfig(
        enabled=head, max_angle=math.radians(head_max_angle), survey=True
    )
    head_ctl = HeadController(head_cfg, _vision_for(cfg, sensor_hfov))
    servo = SimulatedServo(ServoConfig(limit=math.radians(max(1.0, head_max_angle))))
    head_yaw = 0.0
    planner = PathPlanner(cfg)

    # The planner anchors its world frame at the robot's first pose, so the
    # goal must be handed over in robot-relative terms -- exactly as the real
    # system will do it.
    dx = goal_xy[0] - start.x
    dy = goal_xy[1] - start.y
    planner.set_goal(dx, dy)

    pose = start
    limits = DriveLimits()
    result = RunResult(reached=False, steps=0, distance=0.0)
    travelled = 0.0

    for step in range(max_steps):
        sim_now = step * dt
        scan = scanner.scan(world, pose, origin, head_yaw=head_yaw)

        # Planner-frame pose: the planner anchored on `start`.
        planner_pose = pose.relative_to(start)
        # `dt` rather than wall time: this loop runs as fast as the CPU allows,
        # so anything the planner ages must age in simulated seconds or it will
        # behave differently here than on the robot.
        snap = planner.step_once(scan, planner_pose, elapsed=dt)

        result.statuses.append(snap.status)
        if snap.plan_time_ms:
            result.plan_times.append(snap.plan_time_ms)

        if snap.status is PlanStatus.AT_GOAL:
            result.reached = True
            result.steps = step
            break

        rel_path = snap.in_frame_of(planner_pose) if snap.is_drivable else []
        v, w = follow(rel_path, limits) if rel_path else (0.0, 0.0)

        # The head is aimed knowing how fast the body is about to move: a
        # survey may not turn so far that the direction of travel leaves the
        # camera's cone, and that rule needs the speed to apply it.
        if head:
            head_ctl.note_scan(scan, pose.theta, head_yaw, now=sim_now)
            head_ctl.note_heading(pose.theta, now=sim_now)
            head_ctl.note_motion(v)
            aim = head_ctl.update(
                None, FollowState.SEARCHING, current=head_yaw,
                robot_heading=pose.theta, now=sim_now,
            )
            servo.command(aim.angle, now=sim_now)
            head_yaw = servo.update(sim_now)
            result.head_angles.append(head_yaw)

        # Integrate the differential drive. Translation is blocked by walls so
        # that a bad plan shows up as a stuck robot rather than one that
        # cheerfully drives through a wall and arrives anyway.
        nxt = Pose(
            x=pose.x + v * dt * math.cos(pose.theta),
            y=pose.y + v * dt * math.sin(pose.theta),
            theta=wrap_angle(pose.theta + w * dt),
        )
        if _footprint_hits(world, nxt, cfg):
            result.collisions += 1
            nxt = Pose(pose.x, pose.y, nxt.theta)
        else:
            travelled += abs(v) * dt
        pose = nxt

        if capture and step % capture_every == 0:
            result.frames.append(
                Frame(
                    pose=pose,
                    known=planner.costmap.cost.copy(),
                    origin=planner.costmap.origin,
                    path=list(rel_path),
                    status=snap.status,
                    plan_ms=snap.plan_time_ms,
                    expansions=snap.expansions,
                )
            )

        if math.hypot(goal_xy[0] - pose.x, goal_xy[1] - pose.y) <= cfg.goal_tolerance:
            result.reached = True
            result.steps = step
            break
    else:
        result.steps = max_steps

    result.distance = travelled
    result.replans = planner.plan_count
    result.reuses = planner.reuse_count

    if verbose:
        straight = math.hypot(goal_xy[0] - start.x, goal_xy[1] - start.y)
        times = result.plan_times or [0.0]
        print(f"scenario={scenario} seed={seed}")
        print(f"  reached          : {result.reached}")
        print(f"  steps            : {result.steps} ({result.steps * dt:.1f}s sim)")
        print(f"  distance driven  : {result.distance:.2f} m (straight line {straight:.2f} m)")
        print(f"  replans / reuses : {result.replans} / {result.reuses}")
        print(f"  wall contacts    : {result.collisions}")
        print(
            f"  plan latency     : mean {np.mean(times):.1f} ms, "
            f"p95 {np.percentile(times, 95):.1f} ms, max {np.max(times):.1f} ms"
        )
        print(f"  costmap          : {planner.costmap.stats()}")
        if head and result.head_angles:
            swept = np.degrees(np.ptp(result.head_angles))
            print(
                f"  head             : swept {swept:.0f} deg, "
                f"{head_ctl.surveys} survey moves, "
                f"mean |angle| {np.degrees(np.mean(np.abs(result.head_angles))):.0f} deg"
            )
        if planner.last_error:
            print("  ERROR:", planner.last_error)

    result.known_cells = int(planner.costmap.stats()["known_cells"])
    result.frames = result.frames  # keep reference for plotting
    result._world = world  # type: ignore[attr-defined]
    result._start = start  # type: ignore[attr-defined]
    result._goal = goal_xy  # type: ignore[attr-defined]
    result._cfg = cfg  # type: ignore[attr-defined]
    return result


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def render(result: RunResult, gif_path: Optional[str], live: bool) -> None:
    import matplotlib

    if not live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import ListedColormap, BoundaryNorm

    world = result._world  # type: ignore[attr-defined]
    start = result._start  # type: ignore[attr-defined]
    goal = result._goal  # type: ignore[attr-defined]
    cfg = result._cfg  # type: ignore[attr-defined]
    res = cfg.resolution

    if not result.frames:
        print("no frames captured")
        return

    fig, (ax_truth, ax_known) = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.patch.set_facecolor("#101418")
    for ax in (ax_truth, ax_known):
        ax.set_facecolor("#101418")
        ax.set_aspect("equal")
        ax.tick_params(colors="#8a94a0", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2a323c")

    h, w = world.shape
    ax_truth.imshow(
        world, cmap=ListedColormap(["#1b222b", "#5a6572"]), origin="lower",
        extent=[0, w * res, 0, h * res], interpolation="nearest",
    )
    ax_truth.set_title("ground truth", color="#cdd6e0", fontsize=10)
    ax_known.set_title("planner's world (inflated cost)", color="#cdd6e0", fontsize=10)

    # Cost colouring: free -> dim, graded cost -> amber, lethal -> red,
    # unknown -> near-black.
    levels = [0, 1, 60, 130, 200, 254, 255, 256]
    cost_cmap = ListedColormap(
        ["#16202b", "#1d3040", "#2b4a55", "#6a5a34", "#a8632f", "#0b0e12", "#d9484b"]
    )
    norm = BoundaryNorm(levels, cost_cmap.N)

    known0 = result.frames[0].known
    im_known = ax_known.imshow(
        known0, cmap=cost_cmap, norm=norm, origin="lower",
        extent=[0, known0.shape[1] * res, 0, known0.shape[0] * res],
        interpolation="nearest",
    )

    (truth_trail,) = ax_truth.plot([], [], color="#4ea1ff", lw=1.4, alpha=0.9)
    (truth_robot,) = ax_truth.plot([], [], "o", color="#4ea1ff", ms=7)
    (truth_head,) = ax_truth.plot([], [], color="#ffffff", lw=1.6)
    ax_truth.plot([start.x], [start.y], "s", color="#63d297", ms=7, label="start")
    ax_truth.plot([goal[0]], [goal[1]], "*", color="#ffd166", ms=15, label="goal")
    ax_truth.legend(loc="upper right", fontsize=7, facecolor="#1b222b", labelcolor="#cdd6e0")

    (path_line,) = ax_known.plot([], [], color="#4ea1ff", lw=2.0)
    (path_pts,) = ax_known.plot([], [], "o", color="#ffd166", ms=3)
    (robot_known,) = ax_known.plot([], [], "o", color="#63d297", ms=7)

    status_text = fig.text(0.5, 0.03, "", ha="center", color="#cdd6e0", fontsize=9, family="monospace")

    trail_x: List[float] = []
    trail_y: List[float] = []

    def update(i: int):
        frame = result.frames[i]
        p = frame.pose
        trail_x.append(p.x)
        trail_y.append(p.y)
        truth_trail.set_data(trail_x, trail_y)
        truth_robot.set_data([p.x], [p.y])
        truth_head.set_data(
            [p.x, p.x + 0.4 * math.cos(p.theta)], [p.y, p.y + 0.4 * math.sin(p.theta)]
        )

        known = frame.known
        im_known.set_data(known)
        im_known.set_extent([0, known.shape[1] * res, 0, known.shape[0] * res])

        # Planner-frame pose, then path back into planner-world coordinates.
        pp = p.relative_to(start)
        # The costmap origin shifts as the grid grows, so plot relative to it.
        ox, oy = frame.origin
        px = [pp.x - ox]
        py = [pp.y - oy]
        for wp in frame.path:
            c, s = math.cos(pp.theta), math.sin(pp.theta)
            px.append(pp.x + wp.x * c - wp.y * s - ox)
            py.append(pp.y + wp.x * s + wp.y * c - oy)
        path_line.set_data(px, py)
        path_pts.set_data(px[1:], py[1:])
        robot_known.set_data([px[0]], [py[0]])

        status_text.set_text(
            f"frame {i+1}/{len(result.frames)}   status={frame.status.value:<9}"
            f"   plan={frame.plan_ms:6.1f} ms   expansions={frame.expansions:>6}"
        )
        return truth_trail, truth_robot, truth_head, im_known, path_line, path_pts, robot_known

    anim = animation.FuncAnimation(
        fig, update, frames=len(result.frames), interval=60, blit=False
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    if gif_path:
        anim.save(gif_path, writer=animation.PillowWriter(fps=14), dpi=80)
        print(f"wrote {gif_path}")
    if live:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="rooms", choices=["open", "rooms", "maze", "trap"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--size", type=int, default=220, help="world size in cells")
    ap.add_argument("--sensor-range", type=float, default=2.0)
    ap.add_argument("--head", action="store_true",
                    help="put the sensor on a pan servo that looks around")
    ap.add_argument("--head-max-angle", type=float, default=60.0,
                    help="how far the head may turn either side, degrees")
    ap.add_argument("--sensor-fov", type=float, default=360.0,
                    help="sensor field of view in degrees; use 74.5 to match "
                         "the stereo camera and see what the robot really has")
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--gif", default=None, help="write an animated GIF here")
    ap.add_argument("--live", action="store_true", help="show an interactive window")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    result = simulate(
        scenario=args.scenario,
        seed=args.seed,
        size_cells=args.size,
        sensor_range=args.sensor_range,
        sensor_hfov=args.sensor_fov,
        head=args.head,
        head_max_angle=args.head_max_angle,
        max_steps=args.max_steps,
        capture=not args.no_plot,
    )
    print(f"  wall clock       : {time.time() - t0:.1f}s")

    if not args.no_plot and (args.gif or args.live):
        render(result, args.gif, args.live)
    return 0 if result.reached else 1


if __name__ == "__main__":
    raise SystemExit(main())
