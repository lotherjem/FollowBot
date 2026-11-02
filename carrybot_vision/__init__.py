"""CarryBot vision: stereo depth and a person to follow, into the planner.

The camera produces two things the navigation stack wants and the original
tracker only produced one of:

    from carrybot_nav import PathPlanner, PlannerConfig
    from carrybot_vision import CameraSource, VisionConfig, VisionPipeline

    planner = PathPlanner(PlannerConfig()).start()
    vision = VisionPipeline(
        CameraSource(index=0),
        VisionConfig(),
        planner=planner,
        pose_provider=lambda: (pose.x, pose.y, pose.theta),
    ).start()

    while True:
        snap = planner.latest()          # a route, around the furniture
        print(vision.status.summary())   # and who it is following

Calibrate before trusting any of it. ``VisionConfig().describe()`` prints the
range error the current settings imply; if that number is larger than the
standoff hysteresis, the robot will hunt and no amount of controller tuning
will fix it.
"""

from .camera import (
    CameraDevice,
    CameraNotFound,
    VideoMode,
    enumerate_cameras,
    open_camera,
    select_camera,
)
from .costmap import DepthProjector, ProjectionStats, box_mask
from .follow import FollowCommand, FollowConfig, FollowController, FollowState
from .head import HeadCommand, HeadConfig, HeadController, HeadMode
from .pipeline import (
    CameraSource,
    FrameSource,
    ListSource,
    VisionPipeline,
    VisionStatus,
)
from .stereo import (
    Rectification,
    StereoDepth,
    load_calibration,
    sample_depth,
    split_side_by_side,
)
from .tracker import Detection, MedianFilter, PersonTracker, parse_results, torso_point
from .types import CameraModel, TargetObservation, VisionConfig, bearing_to_robot_frame

__version__ = "0.1.0"

__all__ = [
    "CameraDevice",
    "CameraModel",
    "CameraNotFound",
    "CameraSource",
    "DepthProjector",
    "Detection",
    "FollowCommand",
    "FollowConfig",
    "FollowController",
    "FollowState",
    "HeadCommand",
    "HeadConfig",
    "HeadController",
    "HeadMode",
    "FrameSource",
    "ListSource",
    "MedianFilter",
    "PersonTracker",
    "ProjectionStats",
    "Rectification",
    "StereoDepth",
    "TargetObservation",
    "VisionConfig",
    "VideoMode",
    "VisionPipeline",
    "VisionStatus",
    "bearing_to_robot_frame",
    "box_mask",
    "enumerate_cameras",
    "open_camera",
    "select_camera",
    "load_calibration",
    "parse_results",
    "sample_depth",
    "split_side_by_side",
    "torso_point",
]
