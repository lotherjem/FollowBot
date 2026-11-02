"""Show, check or generate the robot's config file.

    python -m carrybot_config                    # which file, what differs, key numbers
    python -m carrybot_config --config my.toml   # check a particular file
    python -m carrybot_config --write            # create ./carrybot.toml with every default
    python -m carrybot_config --write --force    # ...even over an existing one
    python -m carrybot_config --dump             # the effective config, as a full file
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from . import FILE_NAME, ConfigError, default_toml, differences, load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m carrybot_config",
                                description=__doc__.split("\n")[0])
    p.add_argument("--config", metavar="PATH", help="the file to read")
    p.add_argument("--write", nargs="?", const=FILE_NAME, metavar="PATH",
                   help=f"write a complete default file (default ./{FILE_NAME})")
    p.add_argument("--force", action="store_true", help="overwrite with --write")
    p.add_argument("--dump", action="store_true",
                   help="print the effective configuration as a complete file")
    args = p.parse_args(argv)

    if args.write:
        target = Path(args.write)
        if target.exists() and not args.force:
            print(f"{target} already exists; --force to replace it "
                  f"(your changes in it would be lost)")
            return 1
        target.write_text(default_toml(), encoding="utf-8")
        print(f"wrote {target}: every setting at its default, with notes")
        return 0

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    if args.dump:
        sys.stdout.write(default_toml(config))
        return 0

    print(config.describe_source())
    changed = differences(config)
    print()
    if changed:
        print(f"{len(changed)} setting(s) differ from the built-in defaults:")
        for line in changed:
            print(f"  {line}")
    else:
        print("every setting is at its built-in default")
    for warning in config.warnings:
        print(f"\nwarning: {warning}")

    d, c = config.drive, config.camera
    print("\nWhat that works out to:")
    print(f"  {d.counts_per_wheel_rev:.0f} counts per wheel turn, "
          f"{1.0 / d.metres_per_count:.0f} counts per metre "
          f"({d.metres_per_count * 1000:.4f} mm per count)")
    print(f"  effective wheel diameter {d.effective_wheel_diameter_m * 1000:.2f} mm, "
          f"track {d.track_width_m * 1000:.1f} mm")
    print(f"  camera HFOV {math.degrees(c.hfov):.1f} deg, baseline {c.baseline * 1000:.0f} mm, "
          f"nearest measurable range {config.vision.near_limit:.2f} m")
    if not (c.disparity_offset or c.row_offset):
        print("  camera NOT calibrated: set row_offset and disparity_offset (CAMERA.md)")
    if d.left_gains.kv == 0.0 or d.right_gains.kv == 0.0:
        print("  motor feedforward NOT calibrated: kv = 0 (python -m carrybot_drive.cli, f)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
