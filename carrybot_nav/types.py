from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Cost encoding
# --------------------------------------------------------------------------- #

FREE: int = 0
MAX_TRAVERSAL: int = 253
UNKNOWN: int = 254
LETHAL: int = 255


def is_traversable(value: int) -> bool:
    """True for any cost the search is allowed to consider entering."""
    return value <= MAX_TRAVERSAL


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def wrap_angle(theta: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    theta = math.fmod(theta + math.pi, 2.0 * math.pi)
    if theta <= 0.0:
        theta += 2.0 * math.pi
    return theta - math.pi


@dataclass(frozen=True)
class Pose:
    """A planar pose in metres and radians.

    ``theta`` is the heading the robot should be facing *on arrival* at
    ``(x, y)``. For a differential drive that turns in place, the motion
    between two consecutive path poses is: rotate to ``poses[i+1].theta``,
    then drive straight to ``poses[i+1]`` position.
    """

    x: float
    y: float
    theta: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.theta

    def distance_to(self, other: "Pose") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def relative_to(self, origin: "Pose") -> "Pose":
        """Express this pose in the frame of ``origin``.

        Used to convert a world-frame path back into the robot-centric frame
        the rest of the system speaks.
        """
        dx = self.x - origin.x
        dy = self.y - origin.y
        c = math.cos(-origin.theta)
        s = math.sin(-origin.theta)
        return Pose(
            x=dx * c - dy * s,
            y=dx * s + dy * c,
            theta=wrap_angle(self.theta - origin.theta),
        )

    def compose(self, delta: "Pose") -> "Pose":
        """Apply a body-frame delta to this pose (odometry integration)."""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return Pose(
            x=self.x + delta.x * c - delta.y * s,
            y=self.y + delta.x * s + delta.y * c,
            theta=wrap_angle(self.theta + delta.theta),
        )


IDENTITY_POSE = Pose(0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Planner results
# --------------------------------------------------------------------------- #


class PlanStatus(Enum):
    """Why the current path looks the way it does."""

    #: A complete collision-free path to the requested goal.
    PATH_OK = "PATH_OK"
    #: The search ran out of budget. The path heads the right way but stops short.
    PARTIAL = "PARTIAL"
    #: Goal is unreachable or unobserved; path leads to the most promising
    #: frontier between known and unknown space instead.
    FRONTIER = "FRONTIER"
    #: The robot is already within the goal tolerance.
    AT_GOAL = "AT_GOAL"
    #: No usable path and no frontier worth driving to.
    NO_PATH = "NO_PATH"
    #: No goal set, or no map received yet.
    IDLE = "IDLE"


#: Statuses that mean "there is something to drive along".
DRIVABLE = frozenset({PlanStatus.PATH_OK, PlanStatus.PARTIAL, PlanStatus.FRONTIER})


@dataclass(frozen=True)
class PathSnapshot:
    """An immutable plan handed to the motion controller.

    Poses are given in the **robot frame at the instant the plan was made**,
    matching the robot-centric convention of the input. Because the robot keeps
    moving while the plan is in flight, ``reference_pose`` records the world
    pose that frame corresponds to; call :meth:`in_frame_of` with the robot's
    current world pose to re-express the path without waiting for a replan.
    """

    seq: int
    status: PlanStatus
    poses: Tuple[Pose, ...] = ()
    #: World pose the robot occupied when this plan was produced.
    reference_pose: Pose = IDENTITY_POSE
    #: Goal used for this plan, in the same robot frame as ``poses``.
    goal: Optional[Pose] = None
    #: Straight-line path length in metres.
    length_m: float = 0.0
    #: Wall-clock time the search took, in milliseconds.
    plan_time_ms: float = 0.0
    #: Nodes expanded by the search; useful for tuning ``max_expansions``.
    expansions: int = 0
    #: Monotonic timestamp (``time.monotonic()``) of the plan.
    stamp: float = 0.0
    #: True when this is the previous plan re-validated and re-referenced
    #: rather than a fresh search. Useful for spotting a planner that has
    #: gone stale because the map stopped changing near its route.
    reused: bool = False
    #: Monotonic time of what this plan answers: the capture of the camera
    #: frame its scan was built from, or, for a plan made only because the
    #: goal moved, the moment the goal was set. ``stamp - source_stamp`` is
    #: the frame-to-path latency. 0.0 when unknown.
    source_stamp: float = 0.0

    @property
    def is_drivable(self) -> bool:
        return self.status in DRIVABLE and len(self.poses) > 0

    def in_frame_of(self, current_world_pose: Pose) -> List[Pose]:
        """Re-express the path in the robot frame given a newer world pose."""
        world = [
            Pose(
                x=self.reference_pose.x
                + p.x * math.cos(self.reference_pose.theta)
                - p.y * math.sin(self.reference_pose.theta),
                y=self.reference_pose.y
                + p.x * math.sin(self.reference_pose.theta)
                + p.y * math.cos(self.reference_pose.theta),
                theta=wrap_angle(p.theta + self.reference_pose.theta),
            )
            for p in self.poses
        ]
        return [p.relative_to(current_world_pose) for p in world]


EMPTY_SNAPSHOT = PathSnapshot(seq=0, status=PlanStatus.IDLE)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class PlannerConfig:
    """Tuning for the whole stack. Every default is a CarryBot value."""

    # -- Grid geometry ----------------------------------------------------- #
    #: Edge length of one cell, in metres.
    resolution: float = 0.05
    #: Side length of the initial world grid, in metres. Grows automatically.
    initial_world_size_m: float = 20.0
    #: How much padding to add, in metres, each time the world grid grows.
    world_growth_m: float = 10.0

    # -- Robot geometry ---------------------------------------------------- #
    #: Chassis width (y, across the wheel axis), metres.
    robot_width: float = 0.20
    #: Chassis length (x, direction of travel), metres.
    robot_length: float = 0.20
    #: Extra clearance added to the footprint before inflation, metres.
    safety_margin: float = 0.05
    #: Beyond the hard inflation radius, cost decays over this distance so the
    #: planner prefers the middle of a corridor. Set to 0 to disable.
    cost_decay_distance: float = 0.25
    #: Peak cost applied just outside the lethal skirt (0..MAX_TRAVERSAL).
    decay_peak_cost: int = 120

    # -- Dynamic obstacles ------------------------------------------------- #
    #: How long an obstacle survives in the map without being re-observed,
    #: in seconds. A follow robot sees a narrow cone, so every person who
    #: walks across it leaves a LETHAL ghost the moment the robot turns away;
    #: without a lifetime those ghosts accumulate until the map is a maze of
    #: places people used to be. Reverting to UNKNOWN rather than FREE is the
    #: honest statement -- "something was here and I can no longer see" --
    #: and UNKNOWN already carries ``unknown_cost``, so the planner routes
    #: around it when there is room and through it when there is not.
    #: Set to 0 to disable and keep every obstacle forever.
    obstacle_ttl: float = 6.0
    #: Sightings needed before an obstacle is at full confidence and earns the
    #: whole ``obstacle_ttl``. Walls are re-observed on every frame they are in
    #: view and saturate immediately; a person crossing the cone once does not,
    #: so they fade in a fraction of the time.
    obstacle_confirmations: int = 3

    # -- Search ------------------------------------------------------------ #
    #: Give every heading its own search state. Exact turn accounting, but an
    #: eightfold larger state space and correspondingly more expansions. Off is
    #: the right default on a fast cadence: the compact form keeps the best
    #: heading reached per cell and charges turns from there, which costs a few
    #: percent of path quality for several times the speed.
    directional_states: bool = False
    #: Cost charged per 90 degrees of in-place rotation, expressed in units of
    #: "cost of driving one cell". Raise it to get straighter, lazier paths.
    turn_cost: float = 1.2
    #: Treat an UNKNOWN cell as if it carried this traversal cost (0..253
    #: scale). Raise it to make the robot hug known-free ground; set
    #: ``allow_unknown`` False to forbid unobserved space outright.
    unknown_cost: float = 80.0
    #: Whether the search may route through UNKNOWN cells at all.
    allow_unknown: bool = True
    #: While terrain mapping is switched off, obstacles already on the map are
    #: no longer kept up to date, so they stop blocking and cost this instead
    #: (0..253, the same scale as ``unknown_cost``). Routes still go round them
    #: when going round is cheap, and through them when nothing else gets
    #: there -- a stale wall is a guess, not a fact.
    stale_obstacle_cost: float = 200.0
    #: How strongly per-cell traversal cost influences the path, relative to
    #: distance. 1.0 means a cost-253 cell is ~253x more expensive than free.
    cost_weight: float = 0.008
    #: Greedy weight on the heuristic. 1.0 is optimal A* but flood-fills, since
    #: the true cost of a step through unknown or graded terrain exceeds the
    #: pure-distance heuristic. Keep this at or above
    #: ``1 + cost_weight * unknown_cost`` and the search stays close to greedy
    #: in uniform terrain while still routing properly around obstacles.
    heuristic_weight: float = 2.0
    #: Hard cap on node expansions per plan. Bounds worst-case latency; on
    #: overrun the planner returns its best partial path rather than blocking.
    max_expansions: int = 40_000
    #: Padding, in metres, added around the robot-goal bounding box to give the
    #: search room to detour. ``None`` searches the whole known world.
    search_radius_m: Optional[float] = 4.0
    #: Hard cap on the search window area in cells. Padding shrinks to respect
    #: it, so latency stays bounded however far away the goal is.
    max_window_cells: int = 90_000

    # -- Goal handling ----------------------------------------------------- #
    #: Distance within which the goal counts as reached, metres.
    goal_tolerance: float = 0.10
    #: How far a goal may move before it counts as a *different* goal, in
    #: metres. A goal that drifts with a tracked person is the same goal; one
    #: that jumps across the room is a new one.
    #:
    #: This matters more than it looks. A follower re-issues the goal every
    #: vision frame, and any movement at all used to mark the goal dirty --
    #: which disables path reuse and the stickiness gate, so every cycle became
    #: a fresh search and the path shimmered. Measured on the same route:
    #: setting the goal once gave 141 reuses and 3.6 mm of path deviation;
    #: re-issuing the *identical* goal each frame gave 0 reuses and 21.4 mm.
    #: Set to 0 to restore the old always-dirty behaviour.
    goal_epsilon: float = 0.15
    #: If the goal cell is blocked, accept any free cell within this radius.
    goal_snap_radius: float = 0.40
    #: How far to look for clear ground when the robot itself is standing in
    #: inflated space -- after odometry drift, or when a new observation grows
    #: an obstacle over the robot's own cell. Keep this at or below the
    #: inflation radius: snapping the start further than that can put it on the
    #: far side of a thin wall, and the plan then begins by driving through it.
    escape_radius: float = 0.20
    #: Cost of driving *through* a lethal cell, used only when the robot has no
    #: clear ground within ``escape_radius``. Large enough that any legitimate
    #: route wins, small enough that backing out beats sitting there forever.
    lethal_escape_cost: float = 400.0
    #: Weight on "closeness to the real goal" when choosing a frontier. Higher
    #: means the robot pushes toward the goal; lower means it explores nearby
    #: frontiers first.
    frontier_goal_bias: float = 3.0
    #: Once the planner commits to a frontier it keeps driving to that one
    #: until the robot gets there (at which point it stops being a frontier),
    #: it becomes unreachable, or this many cycles pass. Re-choosing every
    #: cycle is what makes an explorer oscillate between two doorways: each is
    #: briefly the better option and the robot never reaches either.
    frontier_commit_cycles: int = 40
    #: Ignore frontiers smaller than this many contiguous cells (sensor noise).
    min_frontier_cells: int = 3
    #: How many frontier cells the fallback sweep collects before choosing.
    #: More gives a better choice; fewer stops the sweep sooner.
    frontier_sample: int = 40

    # -- Path post-processing ---------------------------------------------- #
    #: Collapse collinear waypoints. Tolerance in metres of cross-track error.
    simplify_tolerance: float = 0.03
    #: Run line-of-sight shortcutting over the grid path (cheap "any-angle").
    smooth_path: bool = True
    #: Collapse waypoints closer together than this, in metres. Stops the path
    #: handing the follower a target it is already standing on.
    min_waypoint_spacing: float = 0.25

    # -- Threading --------------------------------------------------------- #
    #: Upper bound on replans per second. The worker sleeps to respect it.
    max_plan_rate_hz: float = 20.0
    #: Skip the search if the previous path is still valid and the goal is
    #: unchanged. This is what makes a from-scratch A* survive a fast update
    #: cadence: most cycles reveal nothing that invalidates the current route.
    #: Only complete paths are ever reused -- a partial or frontier-bound plan
    #: is provisional by definition and always gets re-searched.
    reuse_valid_path: bool = True
    #: Force a fresh search after this many consecutive reuses, so a plan can
    #: never go stale indefinitely while the map improves around it.
    max_reuse_cycles: int = 8
    #: Discount applied to steps that stay on the previous route, as a fraction
    #: of step cost. Without it, two ways round an obstacle that cost almost
    #: the same will swap places every time the map shifts by a cell, and the
    #: robot dithers on the spot instead of committing. 0 disables.
    path_hysteresis: float = 0.3
    #: How far from the previous route still counts as being on it, in metres.
    hysteresis_radius: float = 0.15
    #: If a freshly searched path stays within this distance of the one the
    #: robot is already following, over ``stickiness_horizon`` metres of the
    #: near field, it is treated as the same route and the incumbent is kept.
    #: Re-running A* every cycle produces a slightly different discretisation
    #: of the same route; adopting each one moves the follower's target for no
    #: reason, and the path visibly shimmers. Set to 0 to always take the new
    #: plan.
    path_stickiness: float = 0.10
    #: How far ahead the stickiness comparison looks, in metres. Only the near
    #: field matters -- it is what the follower steers by, and the far end of a
    #: plan through unexplored space is going to change anyway.
    stickiness_horizon: float = 2.0
    #: A freshly searched route has to beat the one the robot is already
    #: following by this fraction before the planner switches to it. This is a
    #: blunter version of ``path_hysteresis`` and off by default: the two
    #: together over-commit, and the robot clings to a route long after a
    #: better one has opened up. Raise it only if hysteresis alone leaves the
    #: robot dithering between two routes.
    switch_margin: float = 0.0

    # -- Derived ----------------------------------------------------------- #

    @property
    def circumscribed_radius(self) -> float:
        """Half-diagonal of the chassis.

        The robot turns in place, so the swept shape of any rotation is the
        circumscribed circle. Planning against that circle is the conservative
        (and cheap) choice for a near-square chassis.
        """
        return 0.5 * math.hypot(self.robot_width, self.robot_length)

    @property
    def inflation_radius(self) -> float:
        return self.circumscribed_radius + self.safety_margin

    def cells(self, metres: float) -> int:
        """Convert a distance to a whole number of cells, rounding up."""
        return int(math.ceil(metres / self.resolution - 1e-9))

    def validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if not (0.0 <= self.decay_peak_cost <= MAX_TRAVERSAL):
            raise ValueError(f"decay_peak_cost must be 0..{MAX_TRAVERSAL}")
        if not (1.0 <= self.stale_obstacle_cost <= MAX_TRAVERSAL):
            raise ValueError(f"stale_obstacle_cost must be 1..{MAX_TRAVERSAL}")
        if self.heuristic_weight < 1.0:
            raise ValueError("heuristic_weight < 1.0 makes A* slower, not faster")
        if self.max_expansions < 1000:
            raise ValueError("max_expansions is too small to be useful")
        if self.robot_width <= 0 or self.robot_length <= 0:
            raise ValueError("robot dimensions must be positive")
        if self.obstacle_ttl < 0.0:
            raise ValueError("obstacle_ttl must be >= 0 (0 disables decay)")
        if self.obstacle_confirmations < 1:
            raise ValueError("obstacle_confirmations must be at least 1")


__all__ = [
    "FREE",
    "MAX_TRAVERSAL",
    "UNKNOWN",
    "LETHAL",
    "DRIVABLE",
    "IDENTITY_POSE",
    "EMPTY_SNAPSHOT",
    "is_traversable",
    "wrap_angle",
    "Pose",
    "PlanStatus",
    "PathSnapshot",
    "PlannerConfig",
]
