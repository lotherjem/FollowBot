from __future__ import annotations

import json
import platform
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = "carrybot-recording/1"


def save_recording(path: str, frames: Sequence[np.ndarray],
                   meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Write frames and metadata. Returns the metadata actually written."""
    import cv2

    encoded: List[np.ndarray] = []
    for frame in frames:
        ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise ValueError("could not encode a frame")
        encoded.append(buf.reshape(-1))
    lengths = np.array([len(b) for b in encoded], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    blob = np.concatenate(encoded) if encoded else np.zeros(0, np.uint8)

    written = {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": f"{platform.system()} {platform.release()}",
        "opencv": getattr(cv2, "__version__", "?"),
        "frames": len(encoded),
        "shape": list(frames[0].shape) if len(frames) else [],
    }
    written.update(meta or {})
    np.savez(path, blob=blob, offsets=offsets,
             meta=np.array(json.dumps(written, default=str)))
    return written


def load_recording(path: str) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """Read a recording in any of the known layouts. Returns (frames, meta)."""
    import cv2

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"blob", "offsets"} <= keys:
            blob, offsets = data["blob"], data["offsets"]
            meta = json.loads(str(data["meta"])) if "meta" in keys else {}
            frames = [cv2.imdecode(blob[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR)
                      for i in range(len(offsets) - 1)]
            return frames, meta
        if "frames" in keys:
            return list(data["frames"]), {"format": "raw, no metadata"}
        per_frame = sorted(k for k in keys if k.startswith("f") and k[1:].isdigit())
        if per_frame:
            frames = [cv2.imdecode(data[k], cv2.IMREAD_COLOR) for k in per_frame]
            return frames, {"format": "per-frame encoded, no metadata"}
    raise ValueError(f"{path} is not a recording this code knows how to read")


def describe(meta: Dict[str, object]) -> str:
    """One line saying where a recording came from."""
    if "backend" not in meta:
        return f"{meta.get('format', 'unknown format')} -- capture settings unknown"
    # Media Foundation reports junk for the format -- sometimes a single
    # control character -- so only a printable code is worth showing.
    fourcc = str(meta.get("fourcc") or "")
    fourcc = fourcc if fourcc.isprintable() and fourcc.strip() else "format unreported"
    return (f"{meta.get('shape', '?')} via {meta.get('backend')}/{fourcc} "
            f"at {float(meta.get('fps', 0)):.1f} fps on {meta.get('platform')}, "
            f"{meta.get('created')}")


__all__ = ["save_recording", "load_recording", "describe", "FORMAT"]
