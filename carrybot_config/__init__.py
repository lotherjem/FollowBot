from __future__ import annotations

import dataclasses
import difflib
import json
import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from carrybot_drive.config import DriveConfig, EncoderChannel, Gains, MotorChannel
from carrybot_drive.follower import FollowerConfig
from carrybot_drive.servo import ServoConfig
from carrybot_nav import PlannerConfig
from carrybot_vision.follow import FollowConfig
from carrybot_vision.head import HeadConfig
from carrybot_vision.types import CameraModel, VisionConfig

from ._docs import field_docs

ENV_VAR = "CARRYBOT_CONFIG"
FILE_NAME = "carrybot.toml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """A config file that cannot be used as written."""


# --------------------------------------------------------------------------- #
# The one section that has no home elsewhere
# --------------------------------------------------------------------------- #


@dataclass
class LiveConfig:
    """Defaults for the live interface (``python -m tools.simui --live``)."""

    #: Speed caps the interface starts with, for every mode including the
    #: autonomous ones. The sliders can change them per session; the drive's
    #: own max_linear_speed / max_angular_speed are the hard ceiling above
    #: them. Low on purpose: raise them once the robot has earned it.
    max_linear: float = 0.25
    max_angular: float = 1.2
    #: Keep following if the interface goes quiet. Off: the robot disarms a
    #: second after the page stops polling. On: follow-me carries on with the
    #: phone in your pocket, and only the robot's own logic stops it.
    follow_offline: bool = False
    #: Seconds without contact from the interface before it disarms.
    client_timeout: float = 1.0
    #: Seconds a manual drive command stays good if no stop arrives. Letting
    #: go sends a stop at once; this only catches a stop that is lost. The
    #: page repeats commands every 50 ms while held, so 0.15 s rides out two
    #: late ones. Raise it a little if driving stutters over Wi-Fi.
    teleop_timeout: float = 0.15
    #: Seconds without a processed camera frame before follow-me holds.
    vision_timeout: float = 1.0
    #: A pan servo is fitted. It is driven only while the motors are on: a
    #: modelled servo would report angles the camera never turned to. --servo
    #: also sets it.
    servo_fitted: bool = False
    #: Camera override: a device path, an index, or vid:pid. Empty finds the
    #: stereo camera automatically (see CAMERA.md). --device overrides this.
    camera: str = ""
    #: YOLO pose weights for the person tracker.
    weights: str = "yolov8n-pose.pt"
    #: Password for the interface. Empty means none, which is only allowed
    #: while it listens on this machine alone (--host 127.0.0.1): anyone who
    #: can reach the page can drive the robot, so the server refuses to open it
    #: to the network without one unless started with --no-password. It is
    #: stored here as plain text and sent once at login over plain HTTP, which
    #: is fine on your own LAN and not fine anywhere else -- keep this file
    #: private and never expose the port to the internet.
    password: str = ""
    #: Folder for driving logs: one CSV per armed session, a row per control
    #: tick -- the person's range, the plan, the command, each wheel's speed
    #: and duty. Relative paths are from where the server is started. Empty
    #: turns logging off.
    log_dir: str = "logs"


# --------------------------------------------------------------------------- #
# What goes in the file, and in what order
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Section:
    name: str
    cls: type
    title: str
    intro: str
    #: Fields stored in radians, written in degrees as ``<name>_deg``.
    angles: Tuple[str, ...] = ()
    #: Fields that default to None, with the type a value must have.
    optional: Tuple[Tuple[str, type], ...] = ()
    #: Fields to leave out of this section (held elsewhere).
    skip: Tuple[str, ...] = ()
    #: Fields to list first, because they are the ones to change.
    first: Tuple[str, ...] = ()


#: Notes for fields that have no ``#:`` comment of their own in the code.
EXTRA_DOCS: Dict[Tuple[str, str], str] = {
    ("drive", "chip"): "gpiochip number. 0 on a Pi 4; the Pi 5 exposes its header on 4 on older kernels.",
    ("drive", "max_duty"): "Highest duty ever written, percent.",
    ("drive", "control_rate_hz"): "Velocity loop rate. Loop overruns in the live readouts mean the Pi cannot keep up.",
    ("drive", "max_angular_speed"): "Hard ceiling on turn rate, rad/s. Every command is clamped to these two.",
    ("drive", "default_linear_speed"): "Speeds and accelerations for the blocking move/turn helpers (the bench console).",
    ("drive", "angle_tolerance_deg"): "A turn finishes within this of target.",
    ("servo", "max_pulse_us"): "(see min_pulse_us)",
    ("servo", "max_angle_deg"): "(see min_angle_deg)",
    ("follower", "max_linear"): "Path-following speed limits, m/s and rad/s. The live interface's caps sit on top.",
    ("head", "search_expand"): "Seconds for the search sweep to widen from search_start to the full max_angle.",
    ("vision", "block_size"): "SGBM matching block, pixels, odd. Bigger is smoother and blurs edges.",
    ("vision", "speckle_window"): "SGBM speckle filter: blobs smaller than this many pixels, varying by more than speckle_range, are discarded.",
    ("camera", "width"): "One eye's resolution: half the side-by-side frame's width.",
    ("camera", "focal_px"): "Focal length in pixels at that resolution. UNMEASURED -- solve it with probe --focal at two or more distances.",
    ("camera", "cx"): "Principal point. Left out, it is the image centre.",
}

#: Where the code comment reads wrong in a config file, this replaces it.
DOC_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("drive", "standby"): "GPIO wired to the TB6612's STBY pin. Leave it commented out if STBY is "
                          "strapped to VCC -- but then the driver can never be forced into a "
                          "known-safe state from software (DRIVE.md). Wire it and set this if you can.",
    ("camera", "cx"): "Principal point, pixels. Leave these commented out for the image centre; "
                      "only a real calibration will tell you otherwise.",
    ("camera", "disparity_offset"): None,   # the code comment is fine once cleaned
}

NESTED = {
    "left_motor": MotorChannel, "right_motor": MotorChannel,
    "left_encoder": EncoderChannel, "right_encoder": EncoderChannel,
    "left_gains": Gains, "right_gains": Gains,
}

SECTIONS: Tuple[Section, ...] = (
    Section(
        "drive", DriveConfig, "Drive base",
        "Wheels, encoders, motors and their control loops. The geometry at the "
        "top is what turns encoder counts into metres; every distance the robot "
        "reports is wrong by whatever factor it is wrong by. Measure it with "
        "`python -m carrybot_drive.cli` (c = distance and track, f = motor "
        "gains, p = check each wheel counts up going forward).",
        angles=("angle_tolerance_rad",),
        optional=(("standby", int),),
        first=("encoder_cpr", "quadrature_multiplier", "gear_ratio",
               "wheel_diameter_m", "distance_scale", "track_width_m"),
    ),
    Section(
        "camera", CameraModel, "Stereo camera",
        "What the camera is and where it is bolted. The calibration values come "
        "from `python -m carrybot_vision.probe`: --offsets gives row_offset, and "
        "--focal at two or more measured distances gives focal_px and "
        "disparity_offset. Until then every range is provisional. Lock the lenses "
        "before calibrating -- one moved 16 px between two sessions.",
        angles=("mount_pitch",),
        optional=(("cx", float), ("cy", float)),
        first=("focal_px", "baseline", "disparity_offset", "row_offset",
               "mount_height", "mount_pitch", "mount_forward", "pan_offset"),
    ),
    Section(
        "servo", ServoConfig, "Pan servo",
        "The servo under the camera. Check its direction before anything else: "
        "in the live interface, Head -> By hand -> +20 deg must turn the camera "
        "to the robot's LEFT. If it goes right, swap min_pulse_us and "
        "max_pulse_us.",
        angles=("min_angle", "max_angle", "limit", "deadband", "offset"),
    ),
    Section(
        "live", LiveConfig, "Live interface",
        "Defaults for `python -m tools.simui --live`. Nothing here switches the "
        "real motors on: that is the MOTORS switch on the page (which asks, and "
        "is blocked while anything is wrong), or --drive hardware.",
    ),
    Section(
        "follow", FollowConfig, "Follow-me",
        "Turning a tracked person into a goal: how far behind them to sit, and "
        "what to do when they are lost.",
    ),
    Section(
        "head", HeadConfig, "Head behaviour",
        "How the pan head tracks, searches and surveys. Set enabled = true once "
        "the servo is fitted (and [live] servo_fitted, or --servo, for the live interface).",
        angles=("max_angle", "track_deadband", "body_deadband", "body_deadband_still",
                "search_start", "memory_bin", "survey_forward_margin"),
        first=("enabled", "max_angle"),
    ),
    Section(
        "follower", FollowerConfig, "Path follower",
        "Turns the planner's path into a velocity command.",
        angles=("pivot_threshold", "heading_tolerance"),
    ),
    Section(
        "planner", PlannerConfig, "Planner",
        "The costmap and A*. robot_width and robot_length are the robot's "
        "footprint and matter most here: too small and it clips doorframes, too "
        "big and it will not fit through them.",
        first=("robot_width", "robot_length", "safety_margin", "resolution"),
    ),
    Section(
        "vision", VisionConfig, "Vision processing",
        "Stereo matching, the ground plane, and how depth becomes obstacles. "
        "grid_resolution must equal [planner] resolution.",
        angles=("bearing_bin",),
        skip=("camera",),
    ),
)
SECTION_BY_NAME = {s.name: s for s in SECTIONS}


def _file_key(section: Section, name: str) -> str:
    """The key a field is written under: angles get a ``_deg`` name."""
    if name in section.angles:
        return (name[:-4] if name.endswith("_rad") else name) + "_deg"
    return name


def _field_for(section: Section, key: str) -> Tuple[Optional[str], bool]:
    """Map a file key back to ``(field name, is_degrees)``."""
    fields = {f.name for f in dataclasses.fields(section.cls)} - set(section.skip)
    for name in section.angles:
        if key == _file_key(section, name):
            return name, True
    if key in fields and key not in section.angles:
        return key, False
    return None, False


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #


@dataclass
class RobotConfig:
    """Everything, loaded. ``vision.camera`` is the same object as ``camera``."""

    drive: DriveConfig = field(default_factory=DriveConfig)
    camera: CameraModel = field(default_factory=CameraModel)
    servo: ServoConfig = field(default_factory=ServoConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    follow: FollowConfig = field(default_factory=FollowConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    #: The file these came from, or None for the built-in defaults.
    source: Optional[Path] = None
    #: Things that are allowed but probably not what you meant.
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.vision.camera = self.camera

    def describe_source(self) -> str:
        if self.source is None:
            return "no carrybot.toml found -- using built-in defaults"
        return f"config: {self.source}"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _toml():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    try:
        import tomli
        return tomli
    except ImportError as exc:
        raise ConfigError("reading carrybot.toml needs Python 3.11+, or "
                          "`pip install tomli` on older versions") from exc


def find_config_file(explicit: Optional[os.PathLike] = None) -> Optional[Path]:
    """The config file to use, following the search order above."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"{path}: no such config file")
        return path
    env = os.environ.get(ENV_VAR)
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise ConfigError(f"{ENV_VAR}={env}: no such config file")
        return path
    for candidate in (Path.cwd() / FILE_NAME, PROJECT_ROOT / FILE_NAME,
                      Path.home() / ".config" / "carrybot" / FILE_NAME):
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Optional[os.PathLike] = None) -> RobotConfig:
    """Load the robot's configuration. See the module docstring."""
    found = find_config_file(path)
    if found is None:
        return RobotConfig()
    try:
        with open(found, "rb") as handle:
            data = _toml().load(handle)
    except ConfigError:
        raise
    except Exception as exc:              # tomllib.TOMLDecodeError and friends
        raise ConfigError(f"{found}: not valid TOML: {exc}") from exc
    config = config_from_dict(data, where=str(found))
    config.source = found
    return config


def config_from_dict(data: Dict[str, Any], where: str = "<config>") -> RobotConfig:
    """Build a RobotConfig from parsed TOML, strictly."""
    unknown = [k for k in data if k not in SECTION_BY_NAME]
    for key in unknown:
        raise ConfigError(f"{where}: unknown section [{key}]{_suggest(key, SECTION_BY_NAME)}")

    built: Dict[str, Any] = {}
    for section in SECTIONS:
        table = data.get(section.name, {})
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: [{section.name}] must be a table")
        built[section.name] = _build_section(section, table, where)

    config = RobotConfig(**built)
    _validate(config, where)
    return config


def _build_section(section: Section, table: Dict[str, Any], where: str):
    defaults = section.cls()
    optional = dict(section.optional)
    kwargs: Dict[str, Any] = {}
    for key, value in table.items():
        label = f"{where}: [{section.name}] {key}"
        name, degrees = _field_for(section, key)
        if name is None:
            hint = ""
            if key in section.angles:
                hint = f" -- angles are in degrees here: write {_file_key(section, key)}"
            else:
                valid = [_file_key(section, f.name) for f in dataclasses.fields(section.cls)
                         if f.name not in section.skip]
                hint = _suggest(key, valid)
            raise ConfigError(f"{label}: unknown setting{hint}")
        if name in NESTED:
            if not isinstance(value, dict):
                raise ConfigError(f"{label}: must be a table, [{section.name}.{key}]")
            kwargs[name] = _build_nested(NESTED[name], getattr(defaults, name), value,
                                         f"{where}: [{section.name}.{key}]")
            continue
        current = getattr(defaults, name)
        kind = optional.get(name) if current is None else type(current)
        coerced = _coerce(value, kind, label)
        kwargs[name] = math.radians(coerced) if degrees else coerced
    try:
        return section.cls(**kwargs) if kwargs else defaults
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: [{section.name}] {exc}") from exc


def _build_nested(cls, default, table: Dict[str, Any], label: str):
    values = dataclasses.asdict(default)
    for key, value in table.items():
        if key not in values:
            raise ConfigError(f"{label} {key}: unknown setting{_suggest(key, values)}")
        values[key] = _coerce(value, type(values[key]), f"{label} {key}")
    return cls(**values)


def _coerce(value: Any, kind: Optional[type], label: str):
    if kind is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{label}: must be true or false, not {value!r}")
    if kind is int:
        if isinstance(value, bool):
            raise ConfigError(f"{label}: must be a whole number, not {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ConfigError(f"{label}: must be a whole number, not {value!r}")
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{label}: must be a number, not {value!r}")
        return float(value)
    if kind is str:
        if not isinstance(value, str):
            raise ConfigError(f"{label}: must be a quoted string, not {value!r}")
        return value
    return value


def _suggest(key: str, choices) -> str:
    close = difflib.get_close_matches(key, list(choices), n=1, cutoff=0.6)
    return f" -- did you mean {close[0]!r}?" if close else ""


def _validate(config: RobotConfig, where: str) -> None:
    """Each class's own checks, then the ones that span sections."""
    for name, obj in (("drive", config.drive), ("servo", config.servo),
                      ("vision", config.vision), ("planner", config.planner)):
        check = getattr(obj, "validate", None)
        if check is None:
            continue
        try:
            check()
        except ValueError as exc:
            raise ConfigError(f"{where}: [{name}] {exc}") from exc

    drive_pins = _drive_pins(config.drive)
    if config.servo.pin in drive_pins:
        raise ConfigError(f"{where}: [servo] pin {config.servo.pin} is already used "
                          f"by the drive ({drive_pins[config.servo.pin]})")
    if abs(config.planner.resolution - config.vision.grid_resolution) > 1e-9:
        raise ConfigError(
            f"{where}: [planner] resolution ({config.planner.resolution}) and "
            f"[vision] grid_resolution ({config.vision.grid_resolution}) must match: "
            "the vision layer writes cells the planner reads")

    warn = config.warnings
    d = config.drive
    if config.live.max_linear > d.max_linear_speed:
        warn.append(f"[live] max_linear {config.live.max_linear} is above [drive] "
                    f"max_linear_speed {d.max_linear_speed}; the drive will clamp it")
    if config.follower.max_linear > d.max_linear_speed:
        warn.append(f"[follower] max_linear {config.follower.max_linear} is above "
                    f"[drive] max_linear_speed {d.max_linear_speed}; the drive will clamp it")
    if config.head.enabled and not config.live.servo_fitted:
        warn.append("[head] enabled but [live] servo_fitted is false: the live "
                    "interface will keep the head fixed unless started with --servo")


def _drive_pins(d: DriveConfig) -> Dict[int, str]:
    pins = {}
    for side in ("left", "right"):
        m = getattr(d, f"{side}_motor")
        e = getattr(d, f"{side}_encoder")
        pins.update({m.pwm: f"{side} motor pwm", m.in1: f"{side} motor in1",
                     m.in2: f"{side} motor in2", e.a: f"{side} encoder a",
                     e.b: f"{side} encoder b"})
    if d.standby is not None:
        pins[d.standby] = "standby"
    return pins


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


HEADER = """\
# CarryBot configuration -- everything about THIS robot.
#
# Every value below is the built-in default. Change the ones that differ for
# your robot; anything you delete falls back to the default. Unknown keys are
# errors (with a suggestion), so a typo cannot be silently ignored.
# Angles are in DEGREES (keys ending _deg). Distances in metres, times in
# seconds, turn rates in rad/s.
#
# Found at: --config PATH, then $CARRYBOT_CONFIG, then ./carrybot.toml, then
# this project's folder, then ~/.config/carrybot/carrybot.toml.
# Check what is in effect:   python -m carrybot_config
#
# ---------------------------------------------------------------------------
# START HERE -- the values that are placeholders until you measure them
# ---------------------------------------------------------------------------
#  [drive]  gear_ratio, wheel_diameter_m     from the motor datasheet / calipers
#           distance_scale, track_width_m    python -m carrybot_drive.cli -> c
#           [drive.*_gains] kv, ks           python -m carrybot_drive.cli -> f
#           motor/encoder pins and invert    python -m carrybot_drive.cli -> p
#  [camera] row_offset                       python -m carrybot_vision.probe --offsets
#           focal_px, disparity_offset       ... --focal 1.0 --focal 2.2 --focal 4.0
#           mount_height/_pitch_deg/_forward tape measure, on the robot
#  [servo]  pin, pulse widths, offset_deg    check direction: LIVE.md
#  [planner] robot_width, robot_length       tape measure: the footprint
#  [head]   enabled = true                   once the servo is fitted
#  [live]   servo_fitted = true              likewise
# ---------------------------------------------------------------------------
"""


def default_toml(config: Optional[RobotConfig] = None) -> str:
    """A complete, commented config file. Defaults unless ``config`` is given."""
    config = config or RobotConfig()
    out: List[str] = [HEADER]
    for section in SECTIONS:
        obj = getattr(config, section.name)
        docs, headings = field_docs(section.cls)
        out.append("")
        out.append("#" + "=" * 77)
        out.append(f"# {section.title}")
        out.extend(_comment(section.intro))
        out.append("#" + "=" * 77)
        out.append(f"[{section.name}]")

        names = [f.name for f in dataclasses.fields(section.cls)
                 if f.name not in section.skip]
        ordered = [n for n in section.first if n in names] + \
                  [n for n in names if n not in section.first]
        nested = [n for n in ordered if n in NESTED]
        optional = dict(section.optional)

        for name in ordered:
            if name in nested:
                continue
            key = _file_key(section, name)
            if headings.get(name) and name not in section.first[1:]:
                out.append("")
                out.append(f"# -- {headings[name]} --")
            note = docs.get(name) or []
            extra = EXTRA_DOCS.get((section.name, key)) or EXTRA_DOCS.get((section.name, name))
            text = DOC_OVERRIDES.get((section.name, name)) or (
                " ".join(note) if note else (extra or ""))
            if name in section.angles:
                # The code documents radians; the file is in degrees.
                text = re.sub(r"\bin radians\b", "in degrees", text)
                text = re.sub(r"\bradians\b", "degrees", text)
                if "degree" not in text and not text.startswith("(see"):
                    text = (text + " Degrees.").strip()
            out.append("")
            out.extend(_comment(text))
            value = getattr(obj, name)
            declared = next(f for f in dataclasses.fields(section.cls) if f.name == name)
            if name in optional and (declared.default is None and value is None
                                     or name in ("cx", "cy") and _is_centre(obj, name)):
                example = {"standby": "22", "cx": "640.0", "cy": "360.0"}.get(name, "0")
                out.append(f"# {key} = {example}")
                continue
            if name in section.angles:
                value = round(math.degrees(value), 4)
            out.append(f"{key} = {_format(value)}")

        for name in nested:
            sub = getattr(obj, name)
            out.append("")
            out.append(f"[{section.name}.{name}]")
            note = docs.get(name)
            if note:
                out.extend(_comment(" ".join(note)))
            elif name.endswith("_gains"):
                out.extend(_comment(
                    "Velocity loop gains, duty % per rad/s. Measure kv and ks with "
                    "the f command; kv = 0 means feedforward is uncalibrated."))
            for key, value in dataclasses.asdict(sub).items():
                out.append(f"{key} = {_format(value)}")
    return "\n".join(out) + "\n"


def _is_centre(camera: CameraModel, name: str) -> bool:
    centre = camera.width / 2.0 if name == "cx" else camera.height / 2.0
    return abs(getattr(camera, name) - centre) < 1e-9


def _comment(text: str) -> List[str]:
    text = text.replace("``", "`").replace("**", "").replace(":attr:", "").replace(":meth:", "")
    return ["# " + line for line in textwrap.wrap(text, 76)] if text else []


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        value = round(value, 9)
        if value.is_integer() and abs(value) < 1e15:
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"cannot write {value!r} to TOML")


def differences(config: RobotConfig) -> List[str]:
    """Every setting that differs from the built-in default, as ``[s] key = v``."""
    base = RobotConfig()
    out = []
    for section in SECTIONS:
        now, was = getattr(config, section.name), getattr(base, section.name)
        for f in dataclasses.fields(section.cls):
            if f.name in section.skip:
                continue
            a, b = getattr(now, f.name), getattr(was, f.name)
            if dataclasses.is_dataclass(a):
                for key, value in dataclasses.asdict(a).items():
                    if value != getattr(b, key):
                        out.append(f"[{section.name}.{f.name}] {key} = {_format(value)}")
                continue
            # Angles go through degrees rounded to 4 places in the file, so an
            # untouched one comes back within 1e-6 rad of the default.
            tol = 1e-6 if f.name in section.angles else 1e-12
            if a != b and not (isinstance(a, float) and isinstance(b, float)
                               and math.isclose(a, b, rel_tol=1e-9, abs_tol=tol)):
                key = _file_key(section, f.name)
                if f.name == "password":
                    out.append(f"[{section.name}] {key} = (set, not shown)")
                    continue
                shown = round(math.degrees(a), 4) if f.name in section.angles else a
                out.append(f"[{section.name}] {key} = {_format(shown) if shown is not None else '(unset)'}")
    return out


__all__ = [
    "ConfigError", "LiveConfig", "RobotConfig", "SECTIONS",
    "config_from_dict", "default_toml", "differences", "find_config_file",
    "load_config",
]
