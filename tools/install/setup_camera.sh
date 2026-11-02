#!/usr/bin/env bash
# Set up the stereo camera on Raspberry Pi OS.
#
#   sudo tools/install/setup_camera.sh            # detect, then install the rule
#   sudo tools/install/setup_camera.sh 32e4 9422  # or name the ids yourself
#
# What it does, and why each step is needed:
#
#   * puts the invoking user in the `video` group -- without it every
#     /dev/video* open fails with EACCES, and OpenCV reports that as an
#     unhelpful "can't open camera by index"
#   * writes a udev rule so the camera is always /dev/carrybot-stereo,
#     pinned to the *capture* node rather than the metadata one
#   * reloads udev and reports what the symlink resolved to
#
# It does not install Python packages; requirements.txt covers those, and on
# Pi OS you want the distribution's python3-opencv rather than the wheel.

set -euo pipefail

RULE=/etc/udev/rules.d/99-carrybot-camera.rules
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_USER="${SUDO_USER:-$USER}"

if [[ $EUID -ne 0 ]]; then
  echo "needs root to write $RULE and reload udev -- re-run with sudo" >&2
  exit 1
fi

VENDOR="${1:-}"
PRODUCT="${2:-}"

if [[ -z "$VENDOR" || -z "$PRODUCT" ]]; then
  echo "== detecting the stereo camera =="
  PIN="$(eval echo ~"$TARGET_USER")/.config/carrybot/camera.json"
  if [[ -r "$PIN" ]]; then
    VENDOR="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('usb_id','').split(':')[0])" "$PIN" || true)"
    PRODUCT="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('usb_id','').split(':')[-1])" "$PIN" || true)"
    echo "   from $PIN: ${VENDOR:-?}:${PRODUCT:-?}"
  fi
fi

if [[ -z "$VENDOR" || -z "$PRODUCT" ]]; then
  echo "   no pinned camera found. Run this first, as $TARGET_USER:" >&2
  echo "       python3 -m carrybot_vision.probe --pin" >&2
  echo "   or pass the ids directly:  sudo $0 <idVendor> <idProduct>" >&2
  exit 1
fi

echo "== adding $TARGET_USER to the video group =="
if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx video; then
  echo "   already a member"
else
  usermod -aG video "$TARGET_USER"
  echo "   done -- $TARGET_USER must log out and back in for it to take effect"
fi

echo "== installing $RULE =="
sed -e "s/VVVV/$VENDOR/" -e "s/PPPP/$PRODUCT/" \
    "$HERE/99-carrybot-camera.rules" > "$RULE"
chmod 0644 "$RULE"
udevadm control --reload-rules
udevadm trigger --subsystem-match=video4linux
sleep 1

echo "== result =="
if [[ -e /dev/carrybot-stereo ]]; then
  echo "   /dev/carrybot-stereo -> $(readlink -f /dev/carrybot-stereo)"
  echo
  echo "   Point the stack at it with:"
  echo "       export CARRYBOT_CAMERA=/dev/carrybot-stereo"
  echo "   or leave it unset -- the pinned USB id is found automatically."
else
  echo "   /dev/carrybot-stereo did not appear." >&2
  echo "   Check the camera is plugged in and that these ids are right:" >&2
  echo "       $VENDOR:$PRODUCT" >&2
  echo "   Compare against:  ls -l /dev/v4l/by-id/ ; lsusb" >&2
  exit 1
fi
