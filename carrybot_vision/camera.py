from __future__ import annotations

import contextlib
import glob
import json
import os
import time
import platform
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Keep GStreamer out of OpenCV's automatic backend choice. Nothing here asks
# for it, but anything that opens a capture without naming a backend gets it
# first on Raspberry Pi OS, and it segfaults on the stereo camera. OpenCV reads
# this when it first builds its backend list, which is after this import.
if platform.system() == "Linux":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")

# --------------------------------------------------------------------------- #
# V4L2, by hand
# --------------------------------------------------------------------------- #
#
# The ioctl request numbers encode the size of their argument struct, so if the
# constants below match the kernel's published values then the struct layouts
# packed further down must be right too. There is a test asserting exactly that,
# which is the cheapest possible guard against a silently misaligned struct.

_IOC_NRBITS, _IOC_TYPEBITS = 8, 8
_IOC_SIZEBITS, _IOC_DIRBITS = 14, 2
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2


def _ioc(direction: int, kind: str, number: int, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (ord(kind) << _IOC_TYPESHIFT)
        | (number << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _ior(kind: str, number: int, size: int) -> int:
    return _ioc(_IOC_READ, kind, number, size)


def _iowr(kind: str, number: int, size: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, kind, number, size)


#: struct v4l2_capability: driver[16] card[32] bus_info[32] + 3 u32 + 3 u32
_CAPABILITY = "16s32s32sIII12x"
#: struct v4l2_fmtdesc: index type flags description[32] pixelformat reserved[4]
_FMTDESC = "III32sI16x"
#: struct v4l2_frmsizeenum: index pixel_format type + union(24) + reserved[2]
_FRMSIZEENUM = "III24s8x"

VIDIOC_QUERYCAP = _ior("V", 0, struct.calcsize(_CAPABILITY))
VIDIOC_ENUM_FMT = _iowr("V", 2, struct.calcsize(_FMTDESC))
VIDIOC_ENUM_FRAMESIZES = _iowr("V", 74, struct.calcsize(_FRMSIZEENUM))

V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_META_CAPTURE = 0x00800000
V4L2_CAP_DEVICE_CAPS = 0x80000000
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FRMSIZE_TYPE_DISCRETE = 1


def fourcc_code(cv2, label: str) -> int:
    """``MJPG`` -> the integer OpenCV wants, across OpenCV 4.x spellings.

    ``cv2.VideoWriter_fourcc`` is the old free function and
    ``cv2.VideoWriter.fourcc`` the newer static method; builds differ over which
    they expose, and getting an AttributeError here would fail the camera open
    for a reason that has nothing to do with cameras.
    """
    if len(label) != 4:
        raise ValueError(
            f"{label!r} is not a FOURCC: it must be exactly four characters. "
            f"Codec names like 'MJPEG' are not codes -- the code is 'MJPG'."
        )
    if hasattr(cv2, "VideoWriter_fourcc"):
        return int(cv2.VideoWriter_fourcc(*label))
    return int(cv2.VideoWriter.fourcc(*label))


def fourcc_to_str(value: int) -> str:
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


def fourcc_of(cap) -> str:
    """The pixel format a capture is *actually* delivering.

    Asking for MJPG and checking only that the resolution came back right is not
    enough, and the gap is expensive. An uncompressed 2560x720 frame is 3.7 MB;
    over USB 2.0 that caps out around 6 fps, where the same camera in MJPG does
    30. The resolution is identical either way, so the only way to tell is to
    read the format back.
    """
    import cv2

    try:
        return fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    except Exception:
        return ""


def measure_fps(cap, frames: int = 30, warmup: int = 5) -> float:
    """Actual delivered frame rate, not the number the camera claims."""

    for _ in range(warmup):
        cap.read()
    got = 0
    began = time.monotonic()
    for _ in range(frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            got += 1
    elapsed = time.monotonic() - began
    return got / elapsed if elapsed > 0 else 0.0


#: Formats known to be compressed, so known to fit a stereo mode down a USB 2.0
#: cable. An allowlist rather than a list of uncompressed formats, because the
#: safe default for something unrecognised is "assume it is not compressed".
#: Getting that backwards makes the probe reassure you about a stream it has
#: never heard of -- and backends do return junk here: OpenCV 5 on Windows hands
#: back a single unprintable character for MSMF.
KNOWN_COMPRESSED = {"MJPG", "JPEG", "H264", "H265", "HEVC", "AVC1", "X264"}

#: How long to wait before retrying an open that failed. DirectShow does not
#: release a camera the instant it is closed, so an open straight after a
#: close can fail for no reason of its own.
SETTLE_SECONDS = 0.5

#: What a UVC camera can move uncompressed over USB 2.0, bytes per second:
#: three 1024-byte isochronous packets per 125 us microframe.
USB2_BYTES_PER_SECOND = 24.576e6


def wire_compressed(fourcc: str, width: int, height: int, fps: float) -> bool:
    """Is the stream compressed on the wire, whatever the backend claims?

    A known compressed FOURCC settles it. So does the frame rate: raw YUY2 is
    two bytes a pixel, and if the measured rate would need well over what
    USB 2.0 can carry, the camera cannot be sending it raw. That matters
    because Media Foundation hands back its *decoded output* format -- or
    nothing at all -- rather than what crossed the cable, and the allowlist
    alone once failed a 30 fps stream as "uncompressed" on that basis. The
    1.5x margin keeps a slightly optimistic frame-rate measurement from
    tipping a genuinely raw stream over the line.
    """
    if fourcc and fourcc.upper() in KNOWN_COMPRESSED:
        return True
    if width <= 0 or height <= 0 or fps <= 0:
        return False
    return fps * width * height * 2 > 1.5 * USB2_BYTES_PER_SECOND
#   Four characters each, because that is what a FOURCC is. "MJPEG" is the
#   name of the codec and is not a valid code -- putting it in a candidate
#   list is how this module once passed five characters to a function that
#   takes four.

#: Common uncompressed formats, for a clearer message when one turns up.
UNCOMPRESSED = {"YUY2", "YUYV", "UYVY", "NV12", "RGB3", "BGR3", "I420", "YV12", ""}


@contextlib.contextmanager
def quiet_opencv(cv2):
    """Silence OpenCV's backend chatter while deliberately probing things.

    A probe that walks every index and every backend produces pages of scary
    warnings about the combinations that do not work, which is the whole point
    of walking them. Printing them buries the actual findings.
    """
    previous = None
    try:
        previous = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            try:
                cv2.utils.logging.setLogLevel(previous)
            except Exception:
                pass


def backends_for_platform(cv2) -> List[Tuple[str, int]]:
    """Capture backends to try, best first for this platform.

    On Windows the order matters: DirectShow frequently ignores a FOURCC request
    and hands back an uncompressed stream regardless, while Media Foundation
    honours it. Trying MSMF first is the difference between 5 fps and 30.
    """
    if platform.system() == "Windows":
        return [("MSMF", getattr(cv2, "CAP_MSMF", 0)),
                ("DSHOW", getattr(cv2, "CAP_DSHOW", 0))]
    if platform.system() == "Darwin":
        return [("AVFOUNDATION", getattr(cv2, "CAP_AVFOUNDATION", 0))]
    # Linux: V4L2 and nothing else. "default" (CAP_ANY) used to follow it as a
    # fallback, but on Raspberry Pi OS's OpenCV, CAP_ANY means GStreamer, and
    # GStreamer handed a /dev/v4l path builds a half-made pipeline and
    # segfaults on release -- taking the whole process with it, after every
    # V4L2 strategy had already been surveyed. A fallback that kills the
    # program is worse than none. Only an OpenCV built without V4L2 gets it.
    if getattr(cv2, "CAP_V4L2", 0):
        return [("V4L2", cv2.CAP_V4L2)]
    return [("default", 0)]


@dataclass(frozen=True)
class OpenInfo:
    """How a capture was actually obtained, as opposed to how it was asked for."""

    backend: str = ""
    requested_fourcc: str = ""
    fourcc: str = ""
    mode: Optional[VideoMode] = None
    reported_fps: float = 0.0

    @property
    def fourcc_honoured(self) -> bool:
        return bool(self.fourcc) and self.fourcc == self.requested_fourcc

    #: Fraction of verification frames that arrived. A backend that opens and
    #: then returns nothing scores 0 here, which is the whole reason it exists.
    liveness: float = 1.0
    #: Frames per second actually measured, when it was measured.
    measured_fps: float = 0.0

    @property
    def compressed(self) -> bool:
        fps = self.measured_fps or self.reported_fps
        w, h = (self.mode.width, self.mode.height) if self.mode else (0, 0)
        return wire_compressed(self.fourcc, w, h, fps)

    @property
    def compression_inferred(self) -> bool:
        """Compressed by the frame-rate argument, not by a reported format."""
        return self.compressed and self.fourcc.upper() not in KNOWN_COMPRESSED

    @property
    def format_known(self) -> bool:
        fmt = self.fourcc.upper()
        return fmt in KNOWN_COMPRESSED or fmt in UNCOMPRESSED

    @property
    def alive(self) -> bool:
        return self.liveness >= 0.75

    def describe(self) -> str:
        fmt = self.fourcc if self.format_known and self.fourcc else "unknown format"
        if self.requested_fourcc and not self.fourcc_honoured:
            fmt += f" (asked for {self.requested_fourcc})"
        rate = f", {self.measured_fps:.1f} fps" if self.measured_fps else ""
        return f"{self.mode} {fmt} via {self.backend}{rate}"


# --------------------------------------------------------------------------- #
# What we found
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoMode:
    width: int
    height: int
    pixel_format: str = ""

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    @property
    def looks_side_by_side(self) -> bool:
        """Twice as wide as it is tall, or wider.

        A single 16:9 sensor is 1.78:1. Two of them side by side is 3.56:1.
        Nothing that is not a stereo pair (or a panorama camera) advertises a
        mode at 2:1 or beyond, so the aspect ratio alone identifies the format.
        """
        return self.aspect >= 2.0 and self.width >= 1280

    def __str__(self) -> str:
        fmt = f" {self.pixel_format}" if self.pixel_format else ""
        return f"{self.width}x{self.height}{fmt}"


@dataclass
class CameraDevice:
    """One capture device, and everything we could learn about it."""

    #: What to hand OpenCV. A path on Linux, a stringified index elsewhere.
    path: str
    #: Index within the platform's enumeration, when there is one.
    index: Optional[int] = None
    name: str = ""
    driver: str = ""
    bus_info: str = ""
    vendor_id: str = ""
    product_id: str = ""
    serial: str = ""
    #: A path that survives a reboot and a different USB port order.
    stable_path: Optional[str] = None
    modes: Tuple[VideoMode, ...] = ()
    is_capture: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def usb_id(self) -> str:
        if self.vendor_id and self.product_id:
            return f"{self.vendor_id}:{self.product_id}"
        return ""

    @property
    def stereo_modes(self) -> Tuple[VideoMode, ...]:
        return tuple(m for m in self.modes if m.looks_side_by_side)

    @property
    def best_stereo_mode(self) -> Optional[VideoMode]:
        modes = self.stereo_modes
        return max(modes, key=lambda m: m.width * m.height) if modes else None

    @property
    def looks_stereo(self) -> bool:
        return bool(self.stereo_modes)

    @property
    def open_target(self):
        """What to pass to ``cv2.VideoCapture``: stable path where possible."""
        if self.stable_path:
            return self.stable_path
        if self.path and not self.path.isdigit():
            return self.path
        return self.index if self.index is not None else 0

    def describe(self) -> str:
        bits = [f"{self.path}"]
        if self.stable_path:
            bits.append(f"-> {self.stable_path}")
        head = "  ".join(bits)
        lines = [f"{head}"]
        if self.name:
            lines.append(f"    name      {self.name}")
        if self.driver or self.bus_info:
            lines.append(f"    driver    {self.driver or '?'}  bus {self.bus_info or '?'}")
        if self.usb_id:
            serial = f"  serial {self.serial}" if self.serial else ""
            lines.append(f"    usb       {self.usb_id}{serial}")
        if self.modes:
            shown = ", ".join(str(m) for m in self._headline_modes())
            lines.append(f"    modes     {shown}")
        lines.append(
            f"    stereo    {'yes -- ' + str(self.best_stereo_mode) if self.looks_stereo else 'no wide mode advertised'}"
        )
        for note in self.notes:
            lines.append(f"    note      {note}")
        return "\n".join(lines)

    def _headline_modes(self, limit: int = 6) -> Sequence[VideoMode]:
        ranked = sorted(self.modes, key=lambda m: (m.looks_side_by_side, m.width * m.height),
                        reverse=True)
        return ranked[:limit]


class CameraNotFound(RuntimeError):
    """No camera could be identified. Carries the enumeration for the message."""

    def __init__(self, message: str, devices: Sequence[CameraDevice] = ()):
        self.devices = list(devices)
        if self.devices:
            listing = "\n".join(d.describe() for d in self.devices)
            message = f"{message}\n\nWhat is actually present:\n{listing}"
        else:
            message = f"{message}\n\nNo capture devices were found at all."
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Linux / Raspberry Pi OS enumeration
# --------------------------------------------------------------------------- #


def _sysfs_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


def _usb_ancestry(node: Path) -> Dict[str, str]:
    """Walk up from a video node's device link to the USB device that owns it.

    The interesting files (``idVendor`` and friends) live on the USB *device*,
    several directories above the V4L2 node, so this climbs until it finds them.
    """
    info: Dict[str, str] = {}
    try:
        current = (node / "device").resolve()
    except OSError:
        return info
    for _ in range(8):
        if (current / "idVendor").exists():
            info["vendor_id"] = _sysfs_text(current / "idVendor")
            info["product_id"] = _sysfs_text(current / "idProduct")
            info["serial"] = _sysfs_text(current / "serial")
            info["product"] = _sysfs_text(current / "product")
            info["manufacturer"] = _sysfs_text(current / "manufacturer")
            return info
        if current.parent == current:
            break
        current = current.parent
    return info


def _ioctl():
    """The ``fcntl`` module, imported only where it exists.

    POSIX-only, and this module has to import cleanly on Windows and macOS --
    where there is no V4L2 to talk to anyway, so nothing below is ever reached.
    Importing it at the top of the file made the whole package unimportable on
    Windows, which no amount of testing on Linux was ever going to reveal.
    """
    import fcntl

    return fcntl


def _query_capability(fd: int) -> Tuple[str, str, str, int]:
    fcntl = _ioctl()
    buf = bytearray(struct.calcsize(_CAPABILITY))
    fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf, True)
    driver, card, bus, _version, caps, device_caps = struct.unpack(_CAPABILITY, bytes(buf))
    effective = device_caps if caps & V4L2_CAP_DEVICE_CAPS else caps

    def clean(raw: bytes) -> str:
        return raw.split(b"\x00", 1)[0].decode(errors="replace").strip()

    return clean(driver), clean(card), clean(bus), effective


def _enum_modes(fd: int) -> Tuple[VideoMode, ...]:
    fcntl = _ioctl()
    modes: List[VideoMode] = []
    for fmt_index in range(32):
        buf = bytearray(struct.calcsize(_FMTDESC))
        struct.pack_into("II", buf, 0, fmt_index, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            fcntl.ioctl(fd, VIDIOC_ENUM_FMT, buf, True)
        except OSError:
            break
        _idx, _type, _flags, _desc, pixelformat = struct.unpack(_FMTDESC, bytes(buf))
        label = fourcc_to_str(pixelformat)
        for size_index in range(64):
            sbuf = bytearray(struct.calcsize(_FRMSIZEENUM))
            struct.pack_into("II", sbuf, 0, size_index, pixelformat)
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, sbuf, True)
            except OSError:
                break
            _i, _pf, size_type, union = struct.unpack(_FRMSIZEENUM, bytes(sbuf))
            if size_type != V4L2_FRMSIZE_TYPE_DISCRETE:
                # Stepwise or continuous: record the maximum, which is what the
                # first two fields of the stepwise union hold.
                max_w, max_h = struct.unpack_from("II", union, 8)
                modes.append(VideoMode(max_w, max_h, label))
                break
            width, height = struct.unpack_from("II", union, 0)
            modes.append(VideoMode(width, height, label))
    # De-duplicate, keeping order.
    seen, unique = set(), []
    for mode in modes:
        key = (mode.width, mode.height, mode.pixel_format)
        if key not in seen:
            seen.add(key)
            unique.append(mode)
    return tuple(unique)


def _stable_paths() -> Dict[str, str]:
    """Map /dev/videoN -> /dev/v4l/by-id/... for every symlink that exists."""
    mapping: Dict[str, str] = {}
    for link in glob.glob("/dev/v4l/by-id/*"):
        try:
            mapping.setdefault(os.path.realpath(link), link)
        except OSError:
            continue
    return mapping


def enumerate_linux() -> List[CameraDevice]:
    """Every V4L2 node, with capabilities and advertised modes. Linux only."""
    devices: List[CameraDevice] = []
    stable = _stable_paths()

    for node in sorted(Path("/sys/class/video4linux").glob("video*"),
                       key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0)):
        dev_path = f"/dev/{node.name}"
        name = _sysfs_text(node / "name")
        usb = _usb_ancestry(node)

        driver = bus = ""
        caps = 0
        modes: Tuple[VideoMode, ...] = ()
        notes: List[str] = []
        try:
            fd = os.open(dev_path, os.O_RDWR | os.O_NONBLOCK)
        except OSError as exc:
            notes.append(f"could not open ({exc.strerror}) -- check the 'video' group")
            devices.append(CameraDevice(
                path=dev_path, index=_index_of(node.name), name=name,
                vendor_id=usb.get("vendor_id", ""), product_id=usb.get("product_id", ""),
                serial=usb.get("serial", ""),
                stable_path=stable.get(dev_path), is_capture=True, notes=notes,
            ))
            continue
        try:
            driver, card, bus, caps = _query_capability(fd)
            name = card or name
            if caps & V4L2_CAP_VIDEO_CAPTURE:
                modes = _enum_modes(fd)
        except OSError as exc:
            notes.append(f"V4L2 query failed ({exc}); falling back to sysfs only")
        finally:
            os.close(fd)

        is_capture = bool(caps & V4L2_CAP_VIDEO_CAPTURE)
        if caps and not is_capture:
            kind = "metadata" if caps & V4L2_CAP_META_CAPTURE else "not a capture device"
            notes.append(f"{kind} -- this is the node naive index-scanning opens by mistake")

        devices.append(CameraDevice(
            path=dev_path,
            index=_index_of(node.name),
            name=name,
            driver=driver,
            bus_info=bus,
            vendor_id=usb.get("vendor_id", ""),
            product_id=usb.get("product_id", ""),
            serial=usb.get("serial", ""),
            stable_path=stable.get(dev_path),
            modes=modes,
            is_capture=is_capture,
            notes=notes,
        ))
    return devices


def _index_of(node_name: str) -> Optional[int]:
    digits = "".join(c for c in node_name if c.isdigit())
    return int(digits) if digits else None


# --------------------------------------------------------------------------- #
# Everywhere else: probe indices through OpenCV
# --------------------------------------------------------------------------- #


def enumerate_opencv(max_index: int = 8,
                     probe_modes: Sequence[Tuple[int, int]] = ()) -> List[CameraDevice]:
    """Open indices one by one and record what each reports.

    This is all that is available off Linux. It is slower and less informative
    -- there is no vendor id and no advertised mode list -- so the camera is
    identified by asking each device to produce the mode we want and seeing
    which one actually does.
    """
    import cv2

    if not probe_modes:
        probe_modes = ((2560, 720), (2560, 960), (3840, 1080), (1280, 480), (1920, 1080))

    backends = []
    if platform.system() == "Windows":
        backends = [("DSHOW", getattr(cv2, "CAP_DSHOW", 0)),
                    ("MSMF", getattr(cv2, "CAP_MSMF", 0))]
    devices: List[CameraDevice] = []

    with quiet_opencv(cv2):
      for index in range(max_index):
          opened = None
          used = "default"
          for label, api in backends or [("default", 0)]:
              cap = cv2.VideoCapture(index, api) if api else cv2.VideoCapture(index)
              if cap.isOpened():
                  opened, used = cap, label
                  break
              cap.release()
          if opened is None:
              continue

          native = VideoMode(int(opened.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(opened.get(cv2.CAP_PROP_FRAME_HEIGHT)))
          found = [native]
          for width, height in probe_modes:
              opened.set(cv2.CAP_PROP_FRAME_WIDTH, width)
              opened.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
              got = VideoMode(int(opened.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(opened.get(cv2.CAP_PROP_FRAME_HEIGHT)))
              if got.width and got not in found:
                  found.append(got)
          opened.release()

          devices.append(CameraDevice(
              path=str(index), index=index, name=f"index {index} ({used})",
              driver=used, modes=tuple(found),
              notes=["mode list is what the device accepted when asked, not an "
                     "advertised capability list"],
          ))

    return devices


def enumerate_cameras(max_index: int = 8) -> List[CameraDevice]:
    """Every capture device this machine has, best effort, no dependencies."""
    if platform.system() == "Linux" and Path("/sys/class/video4linux").is_dir():
        return enumerate_linux()
    return enumerate_opencv(max_index=max_index)


# --------------------------------------------------------------------------- #
# Choosing one
# --------------------------------------------------------------------------- #

#: Where ``--probe --pin`` records the camera it identified.
PIN_FILE = Path(
    os.environ.get("CARRYBOT_CAMERA_PIN",
                   str(Path.home() / ".config" / "carrybot" / "camera.json"))
)
#: Override everything: a device path, an index, or "vid:pid".
ENV_OVERRIDE = "CARRYBOT_CAMERA"


def load_pin(path: Optional[Path] = None) -> Dict[str, str]:
    target = Path(path) if path else PIN_FILE
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_pin(device: CameraDevice, path: Optional[Path] = None,
             strategy: Optional[Dict[str, object]] = None) -> Path:
    target = Path(path) if path else PIN_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "usb_id": device.usb_id,
        "serial": device.serial,
        "name": device.name,
        "stable_path": device.stable_path or "",
    }
    if strategy:
        # Remembering *how* to open it, not just which one it is. Deriving the
        # working backend and format takes several seconds of trying; the robot
        # should not repeat that on every boot.
        payload["strategy"] = {
            "backend": strategy.get("backend", ""),
            "fourcc": strategy.get("fourcc", ""),
            "fourcc_first": bool(strategy.get("fourcc_first", True)),
        }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target


def _matches_pin(device: CameraDevice, pin: Dict[str, str]) -> bool:
    if not pin:
        return False
    if pin.get("usb_id") and device.usb_id == pin["usb_id"]:
        # Several identical cameras: the serial breaks the tie when there is one.
        if pin.get("serial") and device.serial and pin["serial"] != device.serial:
            return False
        return True
    if pin.get("stable_path") and device.stable_path == pin["stable_path"]:
        return True
    return False


def select_camera(
    devices: Optional[Sequence[CameraDevice]] = None,
    override: Optional[str] = None,
    pin: Optional[Dict[str, str]] = None,
    require_stereo: bool = True,
) -> Tuple[CameraDevice, str]:
    """Pick the stereo camera. Returns ``(device, why)``.

    Raises :class:`CameraNotFound` rather than falling back to index 0, because
    a robot that quietly navigates on the wrong camera is worse than one that
    refuses to start.
    """
    devices = list(devices) if devices is not None else enumerate_cameras()
    override = override if override is not None else os.environ.get(ENV_OVERRIDE)
    pin = load_pin() if pin is None else pin

    capture = [d for d in devices if d.is_capture]

    # 1. Explicit override, whatever it costs.
    if override:
        for d in capture:
            if override in (d.path, d.stable_path, str(d.index), d.usb_id):
                return d, f"{ENV_OVERRIDE}={override}"
        if override.isdigit() or override.startswith("/dev/"):
            return CameraDevice(path=override,
                                index=int(override) if override.isdigit() else None,
                                name=f"forced: {override}",
                                notes=["forced by override; nothing verified"]), \
                   f"{ENV_OVERRIDE}={override} (not enumerated -- taken on trust)"
        raise CameraNotFound(f"{ENV_OVERRIDE}={override!r} matched no device", devices)

    # 2. The camera the probe pinned.
    for d in capture:
        if _matches_pin(d, pin):
            return d, f"pinned {pin.get('usb_id') or pin.get('stable_path')}"

    # 3. Capability: the one advertising a side-by-side mode.
    stereo = [d for d in capture if d.looks_stereo]
    if len(stereo) == 1:
        return stereo[0], f"only device advertising a stereo mode ({stereo[0].best_stereo_mode})"
    if len(stereo) > 1:
        best = max(stereo, key=lambda d: d.best_stereo_mode.width * d.best_stereo_mode.height)
        return best, (f"{len(stereo)} stereo-capable devices; took the highest "
                      f"resolution ({best.best_stereo_mode}) -- pin one to be sure")

    if require_stereo:
        raise CameraNotFound(
            "No camera advertises a side-by-side stereo mode (at least 1280 wide "
            "and twice as wide as it is tall).\n"
            "Either this is not the stereo camera, or it needs a different mode "
            f"requested. Set {ENV_OVERRIDE} to force one.",
            devices,
        )

    # 4. Exactly one capture device and we are not being fussy.
    if len(capture) == 1:
        return capture[0], "the only capture device present"
    raise CameraNotFound("Could not tell which camera to use.", devices)


def _try_open(cv2, target, api: int, mode: VideoMode, fourcc: str,
              fourcc_first: bool, buffer_size: int, verify: int = 8,
              time_it: bool = False):
    """One attempt. Returns ``(cap, OpenInfo)`` or ``(None, reason)``.

    ``fourcc_first`` exists because backends disagree about the order: some only
    honour a format change while the resolution is at its default, others reset
    the format when the resolution changes.

    ``verify`` is the important argument. An earlier version of this accepted a
    backend on the strength of **one** successful frame, and Media Foundation on
    Windows will give you exactly that and then fail on every subsequent grab --
    so a camera that had been working at 5 fps was replaced by one delivering
    nothing at all. Opening is not the same as capturing, and the only way to
    tell them apart is to keep reading for a while.
    """

    cap = cv2.VideoCapture(target, api) if api else cv2.VideoCapture(target)
    if not cap.isOpened():
        cap.release()
        return None, "would not open"

    def set_format():
        if not fourcc:
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_code(cv2, fourcc))
        except ValueError:
            # A malformed candidate is a bug in the candidate list, not a
            # reason to abandon a camera that would otherwise open.
            pass

    def set_size():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)

    if fourcc_first:
        set_format()
        set_size()
    else:
        set_size()
        set_format()

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    except Exception:
        pass

    got = VideoMode(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if (got.width, got.height) != (mode.width, mode.height):
        cap.release()
        return None, f"delivered {got} instead of {mode}"

    # Read a run of frames. The frame is the authority on shape, and the run is
    # the authority on whether this backend actually works.
    good = 0
    shape = None
    began = time.monotonic()
    for _ in range(max(2, verify)):
        ok, frame = cap.read()
        if ok and frame is not None:
            good += 1
            shape = frame.shape[:2]
    elapsed = time.monotonic() - began
    attempts = max(2, verify)
    liveness = good / attempts

    if good == 0:
        cap.release()
        return None, "opened at the right mode but every frame read failed"
    if liveness < 0.75:
        cap.release()
        return None, f"only {good}/{attempts} frames arrived -- unreliable"
    if shape and (shape[1], shape[0]) != (mode.width, mode.height):
        cap.release()
        return None, f"property said {got} but frames are {shape[1]}x{shape[0]}"

    try:
        reported = float(cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        reported = 0.0

    measured = (good / elapsed) if (time_it and elapsed > 0) else 0.0
    return cap, OpenInfo(backend="", requested_fourcc=fourcc,
                         fourcc=fourcc_of(cap), mode=got, reported_fps=reported,
                         liveness=liveness, measured_fps=measured)


def open_strategies(cv2, fourcc: str,
                    backend: Optional[str] = None) -> List[Tuple[str, int, str, bool]]:
    """Every (backend, format, ordering) worth trying, in a sensible order.

    ``backend`` restricts the search to one named backend. That is a diagnostic
    tool, not a preference: it exists so two backends can be compared on the
    same scene, which is the only way to tell whether they deliver the same
    geometry.
    """
    formats = [fourcc] if fourcc else [""]
    if fourcc and fourcc.upper() == "MJPG":
        formats.append("JPEG")           # some drivers register it under this code
    formats.append("")                   # and finally: whatever the camera likes
    out = []
    for name, api in backends_for_platform(cv2):
        if backend and name.upper() != backend.upper():
            continue
        for fmt in formats:
            for first in (True, False):
                out.append((name, api, fmt, first))
    return out


@dataclass(frozen=True)
class Strategy:
    """One way of opening a camera, and how well it worked when tried alone."""

    backend: str = ""
    api: int = 0
    fourcc: str = ""
    fourcc_first: bool = True
    liveness: float = 0.0
    fps: float = 0.0
    negotiated: str = ""
    why_not: str = ""
    width: int = 0
    height: int = 0

    @property
    def works(self) -> bool:
        return self.liveness >= 0.75

    @property
    def compressed(self) -> bool:
        return wire_compressed(self.negotiated, self.width, self.height, self.fps)

    @property
    def label(self) -> str:
        order = "format first" if self.fourcc_first else "size first"
        return f"{self.backend}/{self.fourcc or 'any'} ({order})"

    def as_pin(self) -> Dict[str, object]:
        return {"backend": self.backend, "fourcc": self.fourcc,
                "fourcc_first": self.fourcc_first}


def survey_strategies(cv2, device: CameraDevice, mode: VideoMode,
                      fourcc: str = "MJPG", verify: int = 10,
                      buffer_size: int = 1,
                      backend: Optional[str] = None,
                      settle: Optional[float] = None) -> List[Strategy]:
    """Try every way of opening the camera, **one at a time**, and score each."""
    settle = SETTLE_SECONDS if settle is None else settle
    results: List[Strategy] = []
    with quiet_opencv(cv2):
        for name, api, fmt, first in open_strategies(cv2, fourcc, backend):
            cap, outcome = _try_open(cv2, device.open_target, api, mode, fmt,
                                     first, buffer_size, verify, time_it=True)
            if cap is None and settle > 0:
                # One more go after a pause. DirectShow in particular does not
                # let go of a camera the instant it is released, so a strategy
                # tried straight after another can fail for no reason of its
                # own: the same camera's MJPG path worked at 24.6 fps in one
                # pass and "would not open" in the next, and losing it that way
                # left the robot on raw YUY2 at 5 fps.
                time.sleep(settle)
                cap, outcome = _try_open(cv2, device.open_target, api, mode, fmt,
                                         first, buffer_size, verify, time_it=True)
            if cap is None:
                results.append(Strategy(backend=name, api=api, fourcc=fmt,
                                        fourcc_first=first, why_not=str(outcome)))
                continue
            try:
                results.append(Strategy(
                    backend=name, api=api, fourcc=fmt, fourcc_first=first,
                    liveness=outcome.liveness, fps=outcome.measured_fps,
                    negotiated=outcome.fourcc,
                    width=mode.width, height=mode.height,
                ))
            finally:
                cap.release()
    return results


def rank_strategies(results: Sequence[Strategy],
                    prefer_compressed: bool = True) -> List[Strategy]:
    """Working first, then compressed, then frame rate.

    Liveness is a gate rather than a term: a compressed stream that cannot
    sustain capture is worth less than an uncompressed one that can, however
    good its bandwidth would have been.
    """
    working = [r for r in results if r.works]
    if prefer_compressed:
        return sorted(working, key=lambda r: (r.compressed, r.fps), reverse=True)
    return sorted(working, key=lambda r: r.fps, reverse=True)


def open_camera(
    device: Optional[CameraDevice] = None,
    mode: Optional[VideoMode] = None,
    fourcc: str = "MJPG",
    buffer_size: int = 1,
    prefer_compressed: bool = True,
    strategy: Optional[Dict[str, object]] = None,
    verify: int = 10,
    backend: Optional[str] = None,
):
    """Open a device at a stereo mode. Returns ``(cap, mode, info)``.

    ``backend`` forces one named backend and fails rather than falling back to
    another: a diagnostic that quietly substituted a different backend would
    answer a question nobody asked.

    Survey, decide, then open the winner on its own. Candidates are measured
    rather than ranked by reputation -- and measured in isolation, because
    probing a camera that is already open is a good way to break the capture you
    are about to return.

    ``strategy`` short-circuits the survey with a previously recorded winner,
    which is what ``probe --pin`` saves so the robot does not re-derive it on
    every start.
    """
    import cv2

    if device is None:
        device, _why = select_camera()
    mode = mode or device.best_stereo_mode
    if mode is None:
        raise CameraNotFound(f"{device.path} advertises no side-by-side mode", [device])

    def commit(chosen: Strategy):
        """Open one strategy for real, and prove it again on its own."""
        with quiet_opencv(cv2):
            cap, outcome = _try_open(cv2, device.open_target, chosen.api, mode,
                                     chosen.fourcc, chosen.fourcc_first,
                                     buffer_size, verify)
            if cap is None and SETTLE_SECONDS > 0:
                # The survey has only just released this camera.
                time.sleep(SETTLE_SECONDS)
                cap, outcome = _try_open(cv2, device.open_target, chosen.api, mode,
                                         chosen.fourcc, chosen.fourcc_first,
                                         buffer_size, verify)
        if cap is None:
            return None, f"{chosen.label} worked while surveying but not on reopen: {outcome}"
        return cap, OpenInfo(
            backend=chosen.backend, requested_fourcc=fourcc,
            fourcc=outcome.fourcc, mode=outcome.mode,
            reported_fps=outcome.reported_fps, liveness=outcome.liveness,
            measured_fps=chosen.fps,
        )

    attempts: List[str] = []

    if strategy and backend and str(strategy.get("backend", "")).upper() != backend.upper():
        strategy = None                  # the pin names a different backend
    if strategy:
        allowed = {name.upper() for name, _api in backends_for_platform(cv2)}
        if str(strategy.get("backend", "")).upper() not in allowed:
            # A pin saved by an older version, naming a backend no longer
            # tried here -- "default" on Linux was GStreamer, which crashes.
            attempts.append(f"ignored the pinned {strategy.get('backend')!r} backend: "
                            "not used on this platform any more; run probe --pin again")
            strategy = None
    if strategy:
        pinned = Strategy(
            backend=str(strategy.get("backend", "")),
            api=getattr(cv2, f"CAP_{strategy.get('backend', '')}", 0),
            fourcc=str(strategy.get("fourcc", fourcc)),
            fourcc_first=bool(strategy.get("fourcc_first", True)),
            liveness=1.0,
        )
        cap, info = commit(pinned)
        if cap is not None:
            return cap, info.mode, info
        attempts.append(f"pinned {pinned.label}: {info}")

    survey = survey_strategies(cv2, device, mode, fourcc, verify, buffer_size, backend)
    ranked = rank_strategies(survey, prefer_compressed)

    for chosen in ranked:
        cap, info = commit(chosen)
        if cap is not None:
            return cap, info.mode, info
        attempts.append(str(info))

    for failed in survey:
        if not failed.works:
            attempts.append(f"{failed.label}: {failed.why_not or 'no frames'}")

    raise CameraNotFound(
        f"could not open {target_name(device)} at {mode}. Every strategy tried:\n  "
        + "\n  ".join(attempts),
        [device],
    )


def target_name(device: CameraDevice) -> str:
    return str(device.open_target)


__all__ = [
    "CameraDevice",
    "CameraNotFound",
    "ENV_OVERRIDE",
    "PIN_FILE",
    "VideoMode",
    "enumerate_cameras",
    "enumerate_linux",
    "enumerate_opencv",
    "KNOWN_COMPRESSED",
    "wire_compressed",
    "OpenInfo",
    "UNCOMPRESSED",
    "Strategy",
    "open_strategies",
    "quiet_opencv",
    "rank_strategies",
    "survey_strategies",
    "backends_for_platform",
    "fourcc_code",
    "fourcc_of",
    "fourcc_to_str",
    "measure_fps",
    "load_pin",
    "open_camera",
    "save_pin",
    "select_camera",
]
