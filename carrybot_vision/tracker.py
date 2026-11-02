from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple

import numpy as np

from .stereo import sample_depth
from .types import CameraModel, TargetObservation, VisionConfig

# COCO-17 pose indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


@dataclass
class Detection:
    """One person in one frame, in the coordinates of the image given to YOLO."""

    track_id: Optional[int]
    confidence: float
    bbox: Tuple[int, int, int, int]
    keypoints: np.ndarray          # (17, 2)
    keypoint_conf: np.ndarray      # (17,)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def torso_point(
    detection: Detection, min_confidence: float
) -> Tuple[float, float, str]:
    """Best available torso anchor, with the reason it was chosen.

    Hips first, then shoulders, then the centre of the box. Each candidate has
    to clear ``min_confidence`` -- an unseen keypoint is reported at the origin,
    not omitted, so trusting it puts the target in the image corner.
    """
    kp, conf = detection.keypoints, detection.keypoint_conf

    for name, (a, b) in (
        ("hips", (LEFT_HIP, RIGHT_HIP)),
        ("shoulders", (LEFT_SHOULDER, RIGHT_SHOULDER)),
    ):
        good = [i for i in (a, b) if conf[i] >= min_confidence]
        if len(good) == 2:
            return float(kp[good].mean(axis=0)[0]), float(kp[good].mean(axis=0)[1]), name
        if len(good) == 1:
            return float(kp[good[0], 0]), float(kp[good[0], 1]), f"{name} (one side)"

    x1, y1, x2, y2 = detection.bbox
    return (x1 + x2) / 2.0, y1 + 0.6 * (y2 - y1), "box centre"


class MedianFilter:
    """Short rolling median. Resettable, and immune to a single bad sample."""

    def __init__(self, size: int = 5):
        self.size = size
        self._values: Deque[float] = deque(maxlen=size)

    def update(self, value: float) -> float:
        if math.isfinite(value):
            self._values.append(value)
        if not self._values:
            return value
        return float(np.median(self._values))

    def reset(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


class PersonTracker:
    """YOLO pose + persistent id + robust range, producing observations."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        weights: str = "yolov8n-pose.pt",
        device: Optional[str] = None,
    ):
        self.cfg = config or VisionConfig()
        self.weights = weights
        self.device = device
        self._model = None

        self.locked_id: Optional[int] = None
        # Range is noisy and only steers the speed, so it keeps a short
        # median. Bearing is not filtered at all. A median of five frames at
        # 2-3 frames a second reported where the person was a second ago,
        # and everything that turns the robot -- facing, the out-of-sight
        # search, the heading compensation that assumes the bearing belongs
        # to the newest frame -- overshot by most of the turn: in a
        # simulated 45 degree turn to face someone, 42 degrees past them and
        # back. A box bearing does not jitter enough to need it.
        self._distance = MedianFilter(3)
        self._bearing = MedianFilter(1)
        #: Range x box width / image width, from sightings where stereo and
        #: the box agreed: what this person's width says about their range.
        #: Kept across losing the lock -- it is a property of people, near
        #: enough, not of one track.
        self._size: Deque[float] = deque(maxlen=20)
        #: (stamp, metres) of the last range accepted, for the jump check.
        self._last_range: Optional[Tuple[float, float]] = None
        self._last_seen = 0.0
        self.frames = 0
        self.sightings = 0
        self.last_reason = "not started"

    # ------------------------------------------------------------------ #

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            if self.cfg.cpu_threads > 0:
                try:
                    import torch

                    torch.set_num_threads(self.cfg.cpu_threads)
                except Exception:
                    pass
            self._model = YOLO(self.weights)
        return self._model

    def reset_lock(self) -> None:
        """Forget the current target, and the filters that described it."""
        self.locked_id = None
        self._distance.reset()
        self._bearing.reset()

    # ------------------------------------------------------------------ #

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run the model and normalise its output into `Detection`s."""
        model = self._load()
        results = model.track(
            image, persist=True, classes=[0], verbose=False, device=self.device,
            imgsz=self.cfg.detect_size,
        )
        return parse_results(results)

    def update(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        camera: CameraModel,
        stamp: Optional[float] = None,
        detections: Optional[List[Detection]] = None,
    ) -> TargetObservation:
        """One frame in, one observation out.

        ``depth`` and ``camera`` must be the map and model from
        :meth:`StereoDepth.compute` -- same resolution as each other, which is
        not necessarily the resolution of ``image``. The scaling between them
        is handled here rather than assumed.
        """
        stamp = time.monotonic() if stamp is None else stamp
        self.frames += 1

        if detections is None:
            detections = self.detect(image)

        detection = self._choose(detections)
        if detection is None:
            self.last_reason = "no person detected" if not detections else "target lost"
            if stamp - self._last_seen > 1.0:
                self.reset_lock()
            return TargetObservation(stamp=stamp, valid=False)

        self.locked_id = detection.track_id if detection.track_id is not None else self.locked_id
        u, v, source = torso_point(detection, self.cfg.min_keypoint_confidence)

        if depth is None:
            self.last_reason = "no depth this frame"
            # ``camera`` describes the depth scale, not necessarily the image's.
            return TargetObservation(
                track_id=self.locked_id,
                bearing=camera.bearing_of(u * camera.width / image.shape[1]),
                confidence=detection.confidence, stamp=stamp,
                bbox=detection.bbox, valid=False,
            )

        # The depth map may be at a different scale from the detection image.
        scale_u = depth.shape[1] / image.shape[1]
        scale_v = depth.shape[0] / image.shape[0]
        distance, samples = sample_depth(
            depth, u * scale_u, v * scale_v,
            half_patch=self.cfg.depth_patch,
            min_samples=self.cfg.min_depth_samples,
        )

        bearing = camera.bearing_of(u * scale_u)
        distance, how = self._judge_range(distance, detection.bbox, image.shape[1], stamp)

        if not math.isfinite(distance):
            self.last_reason = f"no usable depth at the torso ({samples} samples)"
            return TargetObservation(
                track_id=self.locked_id, bearing=bearing,
                confidence=detection.confidence, stamp=stamp,
                bbox=detection.bbox, depth_samples=samples, valid=False,
            )

        self._last_seen = stamp
        self.sightings += 1
        self.last_reason = f"tracking id {self.locked_id} via {source}" + (
            f", range from {how}" if how != "stereo" else "")

        return TargetObservation(
            track_id=self.locked_id,
            # The median smooths a range that grows, but someone coming
            # closer is believed at once: a frame late is 0.4 s nearer.
            distance=min(self._distance.update(distance), distance),
            bearing=self._bearing.update(bearing),
            confidence=detection.confidence,
            stamp=stamp,
            bbox=detection.bbox,
            depth_samples=samples,
            valid=True,
        )

    # ------------------------------------------------------------------ #

    #: Stereo and box-size ranges within this ratio of each other agree.
    RANGE_AGREE = (0.6, 1.7)
    #: Box-size scale is learned only from stereo ranges in this band, where
    #: stereo is good and the person fits in the frame.
    SCALE_BAND = (0.6, 4.0)
    SCALE_MIN_SAMPLES = 4
    #: Without a box-size scale yet, a range this much further or nearer
    #: than the last one, within a second, is held rather than believed.
    MAX_SPEED = 3.0

    def _box_range(self, bbox, image_width: int) -> float:
        if len(self._size) < self.SCALE_MIN_SAMPLES or not bbox:
            return float("nan")
        width = bbox[2] - bbox[0]
        if width < 4:
            return float("nan")
        return float(np.median(self._size)) * image_width / width

    def _judge_range(self, stereo: float, bbox, image_width: int,
                     stamp: float) -> Tuple[float, str]:
        """Stereo range, checked against the size of the person's box.

        Up close the person's disparity runs past the end of the search, and
        stereo either finds nothing or matches something else entirely -- 15
        or 35 metres, for someone half a metre away. Both used to read as the
        person being gone, or far off. Their box says otherwise: how wide a
        person is in the image goes with how far away they are, and the scale
        is learned from frames where stereo and the box agreed. When stereo
        has nothing, the box is used; when the two disagree, the nearer is
        taken. The ways each goes wrong are both "too far" -- stereo past the
        end of its search, a box narrowed by someone turning sideways or cut
        off by the edge of the image -- and too near only ever means stopping
        early.
        """
        box = self._box_range(bbox, image_width)
        clipped = bool(bbox) and (bbox[0] <= 2 or bbox[2] >= image_width - 3)
        how = "stereo"
        if math.isfinite(stereo) and math.isfinite(box):
            if not self.RANGE_AGREE[0] <= stereo / box <= self.RANGE_AGREE[1]:
                if box < stereo:
                    stereo, how = box, "box size, stereo disagreed"
        elif not math.isfinite(stereo) and math.isfinite(box):
            stereo, how = box, "box size, no stereo"
        elif math.isfinite(stereo) and self._last_range is not None:
            then, last = self._last_range
            dt = stamp - then
            if 0 < dt < 1.0 and abs(stereo - last) > 0.5 + self.MAX_SPEED * dt:
                # Held, not re-stamped: a real step is believed after a second.
                return last, "the last range, stereo jumped"
        if (how == "stereo" and math.isfinite(stereo) and bbox and not clipped
                and self.SCALE_BAND[0] <= stereo <= self.SCALE_BAND[1]):
            self._size.append(stereo * (bbox[2] - bbox[0]) / image_width)
        if math.isfinite(stereo):
            self._last_range = (stamp, stereo)
        return stereo, how

    def _choose(self, detections: List[Detection]) -> Optional[Detection]:
        """Stay with the locked id; otherwise take the most prominent person."""
        usable = [d for d in detections if d.confidence >= self.cfg.min_confidence]
        if not usable:
            return None

        if self.locked_id is not None:
            for d in usable:
                if d.track_id == self.locked_id:
                    return d
            return None  # the lock is still held; wait for them to reappear

        # Acquiring: biggest box, which is the nearest person and the one most
        # likely to be the operator standing in front of the robot.
        return max(usable, key=lambda d: d.area)


def parse_results(results: Any) -> List[Detection]:
    """Normalise an ultralytics `Results` list into plain dataclasses.

    Separated out so the selection and geometry above can be tested without
    ultralytics, torch, or a camera.
    """
    out: List[Detection] = []
    if not results:
        return out
    result = results[0]

    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or len(boxes) == 0:
        return out

    xyxy = np.asarray(boxes.xyxy)
    conf = np.asarray(boxes.conf) if boxes.conf is not None else np.ones(len(xyxy))
    ids = np.asarray(boxes.id) if getattr(boxes, "id", None) is not None else None

    if keypoints is not None and getattr(keypoints, "xy", None) is not None:
        kxy = np.asarray(keypoints.xy)
        kconf = (
            np.asarray(keypoints.conf)
            if getattr(keypoints, "conf", None) is not None
            else np.ones(kxy.shape[:2])
        )
    else:
        kxy = np.zeros((len(xyxy), 17, 2))
        kconf = np.zeros((len(xyxy), 17))

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (int(v) for v in xyxy[i][:4])
        out.append(
            Detection(
                track_id=int(ids[i]) if ids is not None else None,
                confidence=float(conf[i]),
                bbox=(x1, y1, x2, y2),
                keypoints=kxy[i],
                keypoint_conf=kconf[i],
            )
        )
    return out


__all__ = [
    "Detection",
    "MedianFilter",
    "PersonTracker",
    "parse_results",
    "torso_point",
]
