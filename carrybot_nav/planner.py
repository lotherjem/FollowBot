from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import numpy as np

from .costmap import WorldCostmap, bresenham
from .latency import PlanLatency
from .path import (
    assign_headings,
    cells_to_points,
    enforce_spacing,
    path_deviation,
    path_is_clear,
    path_length,
    plan_cost,
    shortcut,
    simplify,
)
from .search import AStarSearch, GridSearch
from .types import (
    EMPTY_SNAPSHOT,
    IDENTITY_POSE,
    PathSnapshot,
    PlannerConfig,
    PlanStatus,
    Pose,
    wrap_angle,
)


class PathPlanner:
    """Continuously replans a route to a goal while the robot drives.

    Typical wiring::

        planner = PathPlanner(PlannerConfig())
        planner.start()

        # sensor thread, every cycle
        planner.update_odometry_absolute(Pose(x, y, theta))
        planner.submit_scan(local_costmap)

        # behaviour thread, whenever the target changes
        planner.set_goal_cells(ix, iy)

        # control thread, every cycle
        snap = planner.latest()
        if snap.is_drivable:
            follow(snap.in_frame_of(current_pose))

        planner.stop()
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        search: Optional[GridSearch] = None,
    ):
        self.cfg = config or PlannerConfig()
        self.cfg.validate()

        self.costmap = WorldCostmap(self.cfg)
        self.search: GridSearch = search or AStarSearch(self.cfg)

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._published = threading.Condition(threading.Lock())

        self._thread: Optional[threading.Thread] = None
        self._running = False

        # -- inputs (guarded by _lock) ------------------------------------ #
        self._pending_scan: Optional[np.ndarray] = None
        self._robot_pose: Pose = IDENTITY_POSE
        self._odom_anchor: Optional[Pose] = None
        self._goal_world: Optional[Pose] = None
        #: The goal as it was when it last counted as new. Movement is judged
        #: against this, not against the previous call: a goal creeping a few
        #: centimetres per frame never moves ``goal_epsilon`` from one frame
        #: to the next, and was never replanned for however far it went.
        self._goal_anchor: Optional[Pose] = None
        self._goal_dirty = False
        self._scan_dirty = False
        self._soft_request: Optional[bool] = None
        #: When the pending scan's frame was captured, and when the goal last
        #: changed. Monotonic. They date the next plan; see ``_run``.
        self._pending_stamp = 0.0
        self._goal_stamp = 0.0
        #: Where the robot was, in the world frame, when the pending scan's
        #: frame was captured; None to use the pose at integration time.
        self._pending_scan_pose: Optional[Pose] = None

        # -- outputs ------------------------------------------------------ #
        self._snapshot: PathSnapshot = EMPTY_SNAPSHOT
        self._seq = 0
        self._last_world_poses: List[Pose] = []
        self._last_status: PlanStatus = PlanStatus.IDLE
        self._planned_goal: Optional[Pose] = None
        self._reuse_run = 0
        self._committed_frontier: Optional[Tuple[int, int]] = None
        self._frontier_age = 0
        self.last_error: Optional[str] = None
        self.plan_count = 0
        self.reuse_count = 0
        self._last_scan_at: Optional[float] = None
        #: Stamp of the input the cycle in progress is answering.
        self._source_stamp = 0.0
        #: Frame-to-path and search timing over recent plans.
        self.latency = PlanLatency()

        self._on_plan: Optional[Callable[[PathSnapshot], None]] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> "PathPlanner":
        if self._thread is not None:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="carrybot-planner", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        with self._wake:
            self._running = False
            self._wake.notify_all()
        self._thread.join(timeout=timeout)
        self._thread = None

    def __enter__(self) -> "PathPlanner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def at_goal(self) -> bool:
        """True when the current goal has been reached.

        The planner keeps running and keeps publishing; this is a state, not
        an ending. Hand it a new goal and it plans immediately -- there is
        nothing to reset and no need to stop and restart it.
        """
        return self.latest().status is PlanStatus.AT_GOAL

    def set_plan_callback(self, fn: Optional[Callable[[PathSnapshot], None]]) -> None:
        """Called on the worker thread each time a new plan is published."""
        self._on_plan = fn

    # ------------------------------------------------------------------ #
    # Inputs
    # ------------------------------------------------------------------ #

    def update_odometry(self, delta: Pose) -> None:
        """Integrate a body-frame movement since the previous call."""
        with self._wake:
            self._robot_pose = self._robot_pose.compose(delta)

    def update_odometry_absolute(self, odom: Pose) -> None:
        """Feed an absolute pose from the odometry/IMU filter.

        The first reading anchors the world frame, so the robot always starts
        at the world origin regardless of what the odometry filter reports.
        """
        with self._wake:
            if self._odom_anchor is None:
                self._odom_anchor = odom
                self._robot_pose = IDENTITY_POSE
                return
            self._robot_pose = self._world_from_odometry(odom)

    def _world_from_odometry(self, odom: Pose) -> Pose:
        """An odometry pose in the planner's world frame. Needs the anchor."""
        a = self._odom_anchor
        dx = odom.x - a.x
        dy = odom.y - a.y
        c = math.cos(-a.theta)
        s = math.sin(-a.theta)
        return Pose(
            x=dx * c - dy * s,
            y=dx * s + dy * c,
            theta=wrap_angle(odom.theta - a.theta),
        )

    def submit_scan(self, scan: np.ndarray, copy: bool = True,
                    stamp: Optional[float] = None,
                    odometry: Optional[Pose] = None) -> None:
        """Hand over the latest robot-centric terrain array.

        ``scan[iy, ix]`` with the robot at the exact centre cell, ``+ix``
        forward and ``+iy`` to the robot's left. Pass ``copy=False`` only if
        you will not touch the array again.

        ``stamp`` is when the camera frame behind the scan was captured
        (``time.monotonic()``). It is only used to measure latency; left out,
        the time of this call is used, which hides the vision pipeline's share.

        ``odometry`` is where the robot was when that frame was captured, in
        the same frame :meth:`update_odometry_absolute` is fed -- a ``Pose``
        or an ``(x, y, theta)`` tuple. The scan is
        placed in the map from there. Left out, it is placed from wherever the
        robot is when the planner gets round to it -- by then the frame can be
        half a second old, and a robot turning at 1 rad/s puts every obstacle
        in it 30 degrees round from where it really is.
        """
        arr = np.array(scan, dtype=np.uint8, copy=copy) if copy else np.asarray(scan)
        with self._wake:
            self._pending_scan = arr
            self._pending_stamp = time.monotonic() if stamp is None else float(stamp)
            if odometry is not None and not isinstance(odometry, Pose):
                x, y, theta = odometry
                odometry = Pose(float(x), float(y), float(theta))
            self._pending_scan_pose = (
                self._world_from_odometry(odometry)
                if odometry is not None and self._odom_anchor is not None else None)
            self._scan_dirty = True
            self._wake.notify()

    def set_goal(self, x: float, y: float, theta: Optional[float] = None) -> None:
        """Set the target in metres, relative to the robot's *current* pose.

        Safe to call every frame with a target that is moving. A goal within
        ``goal_epsilon`` of the current one updates the value but does not
        count as a *new* goal, so path reuse and the stickiness gate stay
        eligible. Without that distinction a follower re-issuing the goal at
        camera rate forces a fresh search every cycle and the published path
        shimmers -- even when the goal it keeps sending is identical.
        """
        with self._wake:
            pose = self._robot_pose
            c = math.cos(pose.theta)
            s = math.sin(pose.theta)
            world = Pose(
                x=pose.x + x * c - y * s,
                y=pose.y + x * s + y * c,
                theta=wrap_angle(theta + pose.theta) if theta is not None else math.nan,
            )

            previous = self._goal_anchor
            moved_far = (
                previous is None
                or self.cfg.goal_epsilon <= 0.0
                or math.hypot(world.x - previous.x, world.y - previous.y)
                > self.cfg.goal_epsilon
                or _heading_changed(previous.theta, world.theta)
            )

            self._goal_world = world
            if moved_far:
                self._goal_anchor = world
                self._goal_dirty = True
                self._goal_stamp = time.monotonic()
            self._wake.notify()

    def set_goal_cells(self, ix: int, iy: int, theta: Optional[float] = None) -> None:
        """Set the target as an offset in array cells from the array centre.

        This is the form the outside system speaks: the target's position in
        the terrain array, which is centred on the robot.
        """
        self.set_goal(ix * self.cfg.resolution, iy * self.cfg.resolution, theta)

    def set_obstacles_soft(self, soft: bool) -> None:
        """Make mapped obstacles costly instead of blocking, or back again.

        Applied on the worker thread, which owns the map, followed by a fresh
        search: a route planned round a wall that has just become passable --
        or through one that has just become solid again -- is out of date.
        """
        with self._wake:
            self._soft_request = bool(soft)
            self._wake.notify()

    def _apply_soft(self, soft: bool) -> None:
        if soft != self.costmap.soft:
            self.costmap.set_soft(soft)
            self._reuse_run = self.cfg.max_reuse_cycles
            # A frontier chosen because the way was blocked is no reason to
            # keep exploring once it is not -- and vice versa.
            self._committed_frontier = None
            self._frontier_age = 0

    def clear_goal(self) -> None:
        with self._wake:
            self._goal_world = None
            self._goal_anchor = None
            self._goal_dirty = True
            self._goal_stamp = time.monotonic()
            self._last_world_poses = []
            self._wake.notify()

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #

    def latest(self) -> PathSnapshot:
        """The most recent plan. Never blocks, never returns None."""
        with self._lock:
            return self._snapshot

    def robot_pose(self) -> Pose:
        """Current pose in the internal world frame."""
        with self._lock:
            return self._robot_pose

    def wait_for_plan(self, timeout: Optional[float] = None, after_seq: int = 0) -> PathSnapshot:
        """Block until a plan newer than ``after_seq`` is published.

        Sequence numbers start at 1, so the default of 0 means "wait for the
        first real plan" -- the initial IDLE snapshot does not count.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            snap = self.latest()
            if snap.seq > after_seq:
                return snap
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return snap
            with self._published:
                self._published.wait(timeout=remaining if remaining else 0.05)

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        min_period = 1.0 / max(1e-6, self.cfg.max_plan_rate_hz)
        next_allowed = 0.0

        while True:
            with self._wake:
                while self._running and not (self._scan_dirty or self._goal_dirty
                                             or self._soft_request is not None):
                    self._wake.wait(timeout=0.2)
                if not self._running:
                    return

                scan = self._pending_scan
                scan_pose = self._pending_scan_pose
                self._pending_scan_pose = None
                goal = self._goal_world
                goal_changed = self._goal_dirty
                # A new scan dates the plan by its camera frame, even when the
                # goal moved too: follow-me sets the goal from that same frame
                # *after* processing it, and dating by the goal would hide the
                # vision pipeline's share. A goal-only cycle (a click) is dated
                # by the goal.
                if scan is not None:
                    self._source_stamp = self._pending_stamp
                elif goal_changed:
                    self._source_stamp = self._goal_stamp
                self._pending_scan = None
                self._scan_dirty = False
                self._goal_dirty = False
                soft, self._soft_request = self._soft_request, None

            now = time.monotonic()
            if now < next_allowed:
                time.sleep(next_allowed - now)
            next_allowed = time.monotonic() + min_period
            # Plan from where the robot is now -- read after the rate limit's
            # sleep, not before it -- but place the scan where it was taken.
            with self._lock:
                pose = self._robot_pose

            if soft is not None:
                self._apply_soft(soft)
            try:
                self._cycle(scan, pose, goal, goal_changed, scan_pose=scan_pose)
            except Exception:  # keep the thread alive; a dead planner is worse
                self.last_error = traceback.format_exc()
                self._publish(
                    PathSnapshot(
                        seq=self._next_seq(),
                        status=PlanStatus.NO_PATH,
                        reference_pose=pose,
                        stamp=time.monotonic(),
                    )
                )

    def _cycle(
        self,
        scan: Optional[np.ndarray],
        pose: Pose,
        goal: Optional[Pose],
        goal_changed: bool,
        elapsed: Optional[float] = None,
        scan_pose: Optional[Pose] = None,
    ) -> None:
        if scan is not None:
            now = time.monotonic()
            if elapsed is None:
                elapsed = 0.0 if self._last_scan_at is None else now - self._last_scan_at
            self._last_scan_at = now
            if self.costmap.integrate(
                    scan, pose if scan_pose is None else scan_pose, elapsed):
                # An obstacle expiring can only ever open a route, never close
                # one, so there is no safety reason to force a search. But the
                # reuse check compares against the map, and a plan held over a
                # cell that has just become unknown deserves re-costing rather
                # than another free ride.
                self._reuse_run = self.cfg.max_reuse_cycles

        if goal is None:
            if goal_changed:
                self._publish(
                    PathSnapshot(
                        seq=self._next_seq(),
                        status=PlanStatus.IDLE,
                        reference_pose=pose,
                        stamp=time.monotonic(),
                    )
                )
            return

        if math.hypot(goal.x - pose.x, goal.y - pose.y) <= self.cfg.goal_tolerance:
            # Arriving is not the end of anything. The next goal may land in
            # the same cycle, so leave no state behind that would colour it:
            # no half-followed route, no frontier the robot was committed to
            # on the way here, no reuse counter part-way through its run.
            self._last_world_poses = []
            self._last_status = PlanStatus.AT_GOAL
            self._committed_frontier = None
            self._frontier_age = 0
            self._reuse_run = 0
            self._publish(
                PathSnapshot(
                    seq=self._next_seq(),
                    status=PlanStatus.AT_GOAL,
                    reference_pose=pose,
                    goal=goal.relative_to(pose),
                    stamp=time.monotonic(),
                )
            )
            return

        if (
            self.cfg.reuse_valid_path
            and not goal_changed
            and self._reuse_run < self.cfg.max_reuse_cycles
            and self._can_reuse(pose, goal)
        ):
            self.reuse_count += 1
            self._reuse_run += 1
            # Republish rather than going quiet: the consumer always wants the
            # path expressed from where the robot is *now*, and the trimmed
            # route is what it should actually be following.
            self._publish_path(
                self._last_world_poses, pose, goal, PlanStatus.PATH_OK, reused=True
            )
            return

        self._reuse_run = 0
        self._replan(pose, goal, goal_changed)

    # -- reuse ------------------------------------------------------------ #

    def _can_reuse(self, pose: Pose, goal: Pose) -> bool:
        """Is the previous plan still good enough to skip a search?

        This is the safety valve that lets a from-scratch A* keep up with a
        fast scan cadence: most cycles reveal nothing that invalidates the
        current route, and those cycles cost a line-of-sight sweep instead of
        a full search.
        """
        poses = self._last_world_poses
        if len(poses) < 2 or self._planned_goal is None:
            return False
        # A frontier or budget-truncated plan was provisional. Reusing one
        # would stop the robot re-evaluating as new terrain comes into view,
        # which is precisely when it most needs to.
        if self._last_status is not PlanStatus.PATH_OK:
            return False
        # Against the goal the current plan was built for, not the previous
        # frame's -- so a target drifting a centimetre per frame still forces a
        # replan once it has drifted a whole epsilon from the planned route.
        if (
            math.hypot(goal.x - self._planned_goal.x, goal.y - self._planned_goal.y)
            > max(1e-6, self.cfg.goal_epsilon)
        ):
            return False

        trimmed = _trim_to_robot(poses, pose)
        if len(trimmed) < 2:
            return False
        if not path_is_clear(trimmed, self.costmap):
            return False

        # Don't let the robot drift arbitrarily far from the route it is
        # supposedly following.
        if _cross_track_error(trimmed, pose) > 4.0 * self.cfg.inflation_radius:
            return False

        self._last_world_poses = trimmed
        return True

    # -- replan ----------------------------------------------------------- #

    def _replan(self, pose: Pose, goal: Pose, goal_changed: bool = True) -> None:
        t0 = time.perf_counter()
        cm = self.costmap
        cfg = self.cfg

        start_cell = cm.world_to_cell(pose.x, pose.y)
        start_free = cm.nearest_free_cell(
            *start_cell, radius_cells=max(2, cfg.cells(cfg.escape_radius))
        )
        # The robot can end up inside inflated space: odometry drifts, or a
        # fresh observation grows an obstacle over where it is standing. When
        # that happens there is no clear cell to start from, so the search is
        # allowed to cross lethal ground at a punitive cost -- just enough to
        # let it back out instead of reporting NO_PATH forever.
        escape = start_free is None
        if escape:
            start_free = start_cell

        goal_cell: Optional[Tuple[int, int]] = cm.world_to_cell(goal.x, goal.y)
        if cm.in_bounds(*goal_cell):
            snapped = cm.nearest_free_cell(
                *goal_cell, radius_cells=cfg.cells(cfg.goal_snap_radius)
            )
            goal_cell = snapped if snapped is not None else goal_cell
        # A goal outside the grid is kept as-is: it still steers the heuristic
        # and the frontier bias, it simply will not be reached this cycle.

        corridor = self._corridor_mask()

        # While exploring, keep driving to the frontier already chosen instead
        # of re-deciding every cycle. Re-deciding is what makes an explorer
        # dither: two doorways trade places as the better option and the robot
        # reaches neither. The commitment ends when the robot arrives (at which
        # point the cell is no longer a frontier), when it becomes unreachable,
        # or when the periodic re-evaluation comes round.
        result = None
        target = self._committed_frontier
        if (
            target is not None
            and self._frontier_age < cfg.frontier_commit_cycles
            and cm.is_frontier(*target)
        ):
            attempt = self.search.plan(
                cm, start_free, target, start_theta=pose.theta,
                escape_lethal=escape, corridor=corridor,
            )
            if attempt.status is PlanStatus.PATH_OK and len(attempt.cells) > 1:
                attempt.status = PlanStatus.FRONTIER
                result = attempt
                self._frontier_age += 1
            else:
                self._committed_frontier = None
        else:
            self._committed_frontier = None

        if result is None:
            result = self.search.plan(
                cm,
                start_free,
                goal_cell,
                start_theta=pose.theta,
                escape_lethal=escape,
                corridor=corridor,
            )
            if result.status is PlanStatus.FRONTIER:
                self._committed_frontier = result.target_cell
                self._frontier_age = 0
            else:
                self._committed_frontier = None

        # Boxed in: the robot is standing on clear ground, but the clear ground
        # is a pocket a few cells across with inflated obstacle on every side.
        # It happens in corridors barely wider than the footprint. Retry with
        # lethal cells traversable so it can squeeze out, rather than sitting
        # there reporting NO_PATH until someone picks it up.
        if result.status is PlanStatus.NO_PATH and not escape:
            result = self.search.plan(
                cm,
                start_free,
                goal_cell,
                start_theta=pose.theta,
                escape_lethal=True,
                corridor=corridor,
            )

        if not result.cells or len(result.cells) < 2:
            self._last_world_poses = []
            self._last_status = PlanStatus.NO_PATH
            self._publish(
                PathSnapshot(
                    seq=self._next_seq(),
                    status=PlanStatus.NO_PATH,
                    reference_pose=pose,
                    goal=goal.relative_to(pose),
                    plan_time_ms=(time.perf_counter() - t0) * 1e3,
                    expansions=result.expansions,
                    stamp=time.monotonic(),
                )
            )
            return

        # Begin at the robot's true position, not the centre of the cell the
        # search started from. Prepending rather than overwriting matters: if
        # the start had to be snapped to clear ground, the segment between the
        # two is not necessarily traversable, and the shortcut pass below will
        # only merge them once it has checked.
        points = [(pose.x, pose.y)] + cells_to_points(cm, result.cells)
        if cfg.smooth_path:
            points = shortcut(points, cm)
        points = simplify(points, cfg.simplify_tolerance, cm)
        points = enforce_spacing(points, cfg.min_waypoint_spacing, cm)

        goal_theta = None if math.isnan(goal.theta) else goal.theta
        if result.status is not PlanStatus.PATH_OK:
            goal_theta = None  # only honour final heading on a complete path

        world_poses = assign_headings(points, start_theta=pose.theta, goal_theta=goal_theta)

        # Stay with the route the robot is already committed to unless the new
        # one is genuinely different or genuinely better. Two things are going
        # on: two ways round an obstacle routinely cost within a percent of
        # each other, and even a settled route comes back slightly
        # re-discretised every cycle. Both make the published path shimmer, and
        # the follower chases the shimmer.
        if (
            not goal_changed
            and result.status is PlanStatus.PATH_OK
            and self._last_status is PlanStatus.PATH_OK
            and len(self._last_world_poses) >= 2
        ):
            incumbent = _trim_to_robot(self._last_world_poses, pose)
            if len(incumbent) >= 2 and path_is_clear(incumbent, cm):
                keep = False

                remaining = path_length(incumbent)
                if cfg.path_stickiness > 0.0 and remaining > cfg.stickiness_horizon:
                    # Only sticky in the far field. Once the end of the route is
                    # inside the comparison horizon the robot is arriving, the
                    # fresh plan is the accurate one, and clinging to a stale
                    # incumbent parks it just short of the goal forever.
                    deviation = path_deviation(
                        world_poses, incumbent, cfg.stickiness_horizon
                    )
                    shorter = path_length(world_poses) < 0.85 * remaining
                    keep = deviation <= cfg.path_stickiness and not shorter

                if not keep and cfg.switch_margin > 0.0:
                    old = plan_cost(incumbent, cm, cfg, pose.theta)
                    new = plan_cost(world_poses, cm, cfg, pose.theta)
                    keep = new >= old * (1.0 - cfg.switch_margin)

                if keep:
                    self._reuse_run = 0
                    self._publish_path(
                        incumbent,
                        pose,
                        goal,
                        PlanStatus.PATH_OK,
                        plan_time_ms=(time.perf_counter() - t0) * 1e3,
                        expansions=result.expansions,
                        reused=True,
                    )
                    return

        self._planned_goal = goal
        self.plan_count += 1

        self._publish_path(
            world_poses,
            pose,
            goal,
            result.status,
            plan_time_ms=(time.perf_counter() - t0) * 1e3,
            expansions=result.expansions,
        )

    def _corridor_mask(self) -> Optional[np.ndarray]:
        """Rasterise the previous route so the search can prefer staying on it.

        Two ways round an obstacle often cost within a percent of each other.
        Without a preference for the incumbent, a single cell of map change
        flips the answer, the path swings across the robot, and it spends its
        time pivoting instead of driving.
        """
        poses = self._last_world_poses
        if len(poses) < 2 or self.cfg.path_hysteresis <= 0.0:
            return None

        cm = self.costmap
        h, w = cm.shape
        mask = np.zeros((h, w), dtype=bool)
        r = max(1, self.cfg.cells(self.cfg.hysteresis_radius))

        for i in range(1, len(poses)):
            a = cm.world_to_cell(poses[i - 1].x, poses[i - 1].y)
            b = cm.world_to_cell(poses[i].x, poses[i].y)
            for row, col in bresenham(a[0], a[1], b[0], b[1]):
                if 0 <= row < h and 0 <= col < w:
                    mask[
                        max(0, row - r) : row + r + 1, max(0, col - r) : col + r + 1
                    ] = True
        return mask

    def _publish_path(
        self,
        world_poses: List[Pose],
        pose: Pose,
        goal: Pose,
        status: PlanStatus,
        plan_time_ms: float = 0.0,
        expansions: int = 0,
        reused: bool = False,
    ) -> None:
        self._last_world_poses = world_poses
        self._last_status = status
        rel = [p.relative_to(pose) for p in world_poses]
        self._publish(
            PathSnapshot(
                seq=self._next_seq(),
                status=status,
                poses=tuple(rel),
                reference_pose=pose,
                goal=goal.relative_to(pose),
                length_m=path_length(world_poses),
                plan_time_ms=plan_time_ms,
                expansions=expansions,
                stamp=time.monotonic(),
                reused=reused,
            )
        )

    # -- plumbing --------------------------------------------------------- #

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _publish(self, snapshot: PathSnapshot) -> None:
        if snapshot.source_stamp == 0.0 and self._source_stamp > 0.0:
            snapshot = replace(snapshot, source_stamp=self._source_stamp)
        self.latency.record(snapshot)
        with self._lock:
            self._snapshot = snapshot
        with self._published:
            self._published.notify_all()
        if self._on_plan is not None:
            try:
                self._on_plan(snapshot)
            except Exception:
                self.last_error = traceback.format_exc()

    # ------------------------------------------------------------------ #
    # Synchronous helper -- same pipeline, no thread. For tests and replay.
    # ------------------------------------------------------------------ #

    def step_once(
        self,
        scan: Optional[np.ndarray] = None,
        pose: Optional[Pose] = None,
        force_replan: bool = False,
        elapsed: Optional[float] = None,
        scan_pose: Optional[Pose] = None,
    ) -> PathSnapshot:
        """Run one planning cycle on the calling thread. For sims and tests.

        Pass ``elapsed`` when the caller's clock is not wall time. A simulator
        stepping at 10 sim-Hz as fast as the CPU allows would otherwise age
        obstacles against real seconds while the robot moves in simulated ones,
        and ``obstacle_ttl`` would mean something different in the sim than on
        the robot. Left as None it measures wall time, which is right for
        anything running live.

        ``scan_pose`` is the world-frame pose the scan was taken from, when
        that differs from ``pose`` (a scan from a frame captured before the
        robot moved on); left out, the scan is placed at ``pose``.
        """
        if pose is not None:
            with self._lock:
                self._robot_pose = pose
        with self._lock:
            current = self._robot_pose
            goal = self._goal_world
            changed = self._goal_dirty or force_replan
            self._goal_dirty = False
            soft, self._soft_request = self._soft_request, None
        if soft is not None:
            self._apply_soft(soft)
        self._source_stamp = time.monotonic()
        self._cycle(scan, current, goal, changed, elapsed, scan_pose=scan_pose)
        return self.latest()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _heading_changed(old: float, new: float, epsilon: float = 0.05) -> bool:
    """Did a goal's optional final heading change? NaN means "do not care"."""
    old_set = not math.isnan(old)
    new_set = not math.isnan(new)
    if old_set != new_set:
        return True
    if not new_set:
        return False
    return abs(wrap_angle(new - old)) > epsilon


def _trim_to_robot(
    poses: List[Pose], pose: Pose, min_step: float = 0.05
) -> List[Pose]:
    """Re-anchor a path at the robot, dropping what it has driven past.

    The projection is onto the path's *segments*, not its vertices. Snapping to
    the nearest vertex instead looks equivalent and is not: when the robot sits
    partway along a segment, the nearest vertex is often the one behind it, so
    that vertex survives the trim and the path acquires a spurious first leg
    running backwards. A lookahead follower then walks forwards along a path
    that goes back before it goes on, lands its target roughly where it already
    is, and commands zero speed. The robot stops dead on a perfectly good
    route -- and if anything is holding the plan steady, it stops forever.
    """
    if len(poses) < 2:
        return []

    best_distance = float("inf")
    best_index = 1
    best_t = 0.0
    for i in range(1, len(poses)):
        ax, ay = poses[i - 1].x, poses[i - 1].y
        bx, by = poses[i].x, poses[i].y
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            continue
        t = max(0.0, min(1.0, ((pose.x - ax) * dx + (pose.y - ay) * dy) / denom))
        d = math.hypot(pose.x - (ax + t * dx), pose.y - (ay + t * dy))
        if d < best_distance:
            best_distance = d
            best_index = i
            best_t = t

    rest = list(poses[best_index:])
    # If the projection already sits on top of the next vertex, that vertex
    # would only add a zero-length leg.
    if rest and (
        best_t > 0.999
        or math.hypot(rest[0].x - pose.x, rest[0].y - pose.y) < min_step
    ):
        rest = rest[1:]

    if not rest:
        return []
    return [Pose(pose.x, pose.y, pose.theta)] + rest


def _cross_track_error(poses: List[Pose], pose: Pose) -> float:
    return min(math.hypot(p.x - pose.x, p.y - pose.y) for p in poses)


__all__ = ["PathPlanner"]
