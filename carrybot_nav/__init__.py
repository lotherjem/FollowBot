"""CarryBot navigation: concurrent grid pathfinding for a differential drive.

    from carrybot_nav import PathPlanner, PlannerConfig, Pose

    planner = PathPlanner(PlannerConfig()).start()
    planner.set_goal_cells(target_ix, target_iy)
    planner.submit_scan(local_terrain_array)
    plan = planner.latest()
"""

from .costmap import WorldCostmap
from .latency import PlanLatency
from .path import path_is_clear, path_length, total_rotation
from .planner import PathPlanner
from .search import AStarSearch, GridSearch, SearchResult
from .types import (
    FREE,
    LETHAL,
    MAX_TRAVERSAL,
    UNKNOWN,
    PathSnapshot,
    PlannerConfig,
    PlanStatus,
    Pose,
    wrap_angle,
)

__version__ = "0.1.0"

__all__ = [
    "AStarSearch",
    "FREE",
    "GridSearch",
    "LETHAL",
    "MAX_TRAVERSAL",
    "PathPlanner",
    "PlanLatency",
    "PathSnapshot",
    "PlanStatus",
    "PlannerConfig",
    "Pose",
    "SearchResult",
    "UNKNOWN",
    "WorldCostmap",
    "path_is_clear",
    "path_length",
    "total_rotation",
    "wrap_angle",
]
