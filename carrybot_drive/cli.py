from __future__ import annotations

import argparse
import math
import select
import sys
import threading
import time
from typing import Callable, List, NamedTuple, Optional, Tuple

from .controller import DriveController


class Console:
    def __init__(self, drive: DriveController, simulated: bool = False):
        self.drive = drive
        self.cfg = drive.cfg
        self.simulated = simulated
        #: Fraction of max speed, matching the original's "move_speed" percent.
        self.speed_fraction = 0.6

    # ------------------------------------------------------------------ #
    # Interruptible moves
    # ------------------------------------------------------------------ #

    def _run_move(self, label: str, action: Callable[[], bool]) -> None:
        """Run a blocking move while watching stdin so it can be cut short.

        The move itself runs on a worker thread. That is not for speed -- it is
        so the console keeps reading input while the robot is moving, which is
        the difference between "press Enter to stop" and "wait and hope".
        """
        print(f"{label}  (Enter to stop)")
        result: List[Optional[bool]] = [None]

        def worker() -> None:
            try:
                result[0] = action()
            except Exception as exc:  # never let a move kill the console
                print(f"\nmove failed: {exc}")
                result[0] = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while thread.is_alive():
                if _stdin_ready(0.1):
                    sys.stdin.readline()
                    self.drive.abort()
                    print("  aborted")
                    break
        except KeyboardInterrupt:
            self.drive.abort()
            print("  interrupted")
        thread.join(timeout=3.0)

        state = self.drive.state()
        if result[0]:
            print(f"  done. {state.summary()}")
        else:
            print(f"  stopped. {state.summary()}")
        if state.fault:
            print(f"\n  THE DRIVE STOPPED ITSELF: {state.fault}")
            print("  Nothing will move until you type r to clear it.")

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def execute(self, cmd: str, value: Optional[int]) -> bool:
        """Returns False to quit."""
        cfg = self.cfg
        state = self.drive.state()
        if state.emergency_stopped and cmd in "wsadqemcf" and cmd:
            # Every move would return at once with the motors coasting, which
            # looks exactly like a robot that will not drive. Say why instead.
            print(f"The drive is latched stopped: {state.fault or 'emergency stop'}")
            print("Fix the cause, then type r to clear it.")
            return True
        counts = value if value is not None else 100
        metres = cfg.counts_to_metres(counts)
        linear = self.speed_fraction * cfg.max_linear_speed
        angular = self.speed_fraction * cfg.max_angular_speed

        if cmd == "w":
            self._run_move(
                f"FORWARD {counts} counts = {metres * 1000:.0f} mm @ {linear:.2f} m/s",
                lambda: self.drive.move_distance(metres, linear),
            )
        elif cmd == "s":
            self._run_move(
                f"BACK {counts} counts = {metres * 1000:.0f} mm @ {linear:.2f} m/s",
                lambda: self.drive.move_distance(-metres, linear),
            )
        elif cmd in ("a", "d"):
            side = "left" if cmd == "a" else "right"
            turn = math.degrees(metres / cfg.track_width_m)
            # The original mapped 'a' to the left motor and 'd' to the right,
            # and noted the physical labels felt backwards. Rather than guess,
            # print which wheel drives and which way the robot ends up facing.
            direction = "clockwise" if side == "left" else "anticlockwise"
            self._run_move(
                f"ARC on the {side} wheel, {counts} counts = {metres * 1000:.0f} mm "
                f"-> {turn:.0f}deg {direction}",
                lambda: self.drive.arc_about_wheel(side, metres, linear),
            )
        elif cmd in ("q", "e"):
            radians = metres / cfg.track_width_m
            if cmd == "e":
                radians = -radians
            self._run_move(
                f"SPIN {'left' if cmd == 'q' else 'right'} {counts} counts = "
                f"{math.degrees(radians):.0f}deg @ {angular:.2f} rad/s",
                lambda: self.drive.turn_in_place(radians, angular),
            )
        elif cmd == "z":
            self.speed_fraction = max(0.05, min(1.0, (value or 60) / 100.0))
            print(
                f"speed {self.speed_fraction * 100:.0f}% "
                f"= {self.speed_fraction * cfg.max_linear_speed:.2f} m/s"
            )
        elif cmd == "b":
            cfg.brake_on_stop = bool(value)
            print(f"stopping mode: {'brake' if cfg.brake_on_stop else 'coast'}")
        elif cmd == "p":
            print(self.drive.state().summary())
        elif cmd == "o":
            self.drive.reset_odometry()
            print("odometry and counts zeroed")
        elif cmd == "k":
            print(cfg.describe())
            print(f"left gains  : {cfg.left_gains}")
            print(f"right gains : {cfg.right_gains}")
        elif cmd == "t":
            self._telemetry()
        elif cmd == "m":
            self._raw_duty(value)
        elif cmd == "c":
            self._calibrate_geometry(counts)
        elif cmd == "f":
            self._calibrate_feedforward()
        elif cmd == "r":
            reason = self.drive.state().fault
            self.drive.clear_emergency_stop()
            print("cleared" + (f" (it was: {reason})" if reason else ""))
        elif cmd == "h":
            print(__doc__)
        elif cmd == "x":
            return False
        else:
            print(f"unknown command {cmd!r} -- 'h' for help")
        return True

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def _telemetry(self) -> None:
        print("streaming -- Enter to stop")
        try:
            while not _stdin_ready(0.2):
                print(f"\r{self.drive.state().summary()}", end="", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            if _stdin_ready(0):
                sys.stdin.readline()
            print()

    def _raw_duty(self, value: Optional[int]) -> None:
        """Each motor channel on its own, open loop. Proves the wiring.

        One channel at a time, because running both together hides the most
        dangerous mistake there is: encoders on the wrong wheels. Driving
        straight, both wheels turn the same way, so each speed loop reads the
        other wheel and is none the wiser. The first pivot, each loop pushes
        harder the faster the robot spins -- full power, and it keeps spinning
        with nothing commanded. Run alone, each channel has to move *its own*
        encoder, and the wheel you see turn says which part is swapped.
        """
        duty = value if value is not None else 30
        print(f"\nWiring test: each motor channel alone, open loop at {duty}% for "
              f"1.5 s. WHEELS OFF THE GROUND. 'left' is the robot's left, seen "
              f"from behind.")
        seen = {}
        for channel in ("left", "right"):
            try:
                input(f"\nEnter to run the {channel.upper()} motor channel "
                      "(Ctrl-C to cancel)... ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            before = self.drive.state()
            end = time.monotonic() + 1.5
            try:
                while time.monotonic() < end:
                    self.drive.set_open_loop_duty(
                        duty if channel == "left" else 0.0,
                        duty if channel == "right" else 0.0)
                    time.sleep(0.02)
            except KeyboardInterrupt:
                print("  interrupted")
                return
            finally:
                self.drive.clear_open_loop()
            time.sleep(0.3)
            after = self.drive.state()
            moved = (after.left_counts - before.left_counts,
                     after.right_counts - before.right_counts)
            print(f"  encoder counts moved: left {moved[0]:+d}, right {moved[1]:+d}")
            wheel = _ask_choice("  Which wheel turned? [l]eft / [r]ight / [n]either: ",
                                {"l": "left", "r": "right", "n": None})
            forward = (_ask_choice("  Forwards? [y/n]: ", {"y": True, "n": False})
                       if wheel else None)
            seen[channel] = WiringObservation(wheel, forward, moved[0], moved[1])
        print()
        for line in diagnose_wiring(seen, self.cfg):
            print(line)

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def _calibrate_geometry(self, counts: int) -> None:
        """Measure metres-per-count, then track width.

        These two numbers are the entire bridge between encoder counts and the
        metres the planner thinks in. Every distance the robot reports is wrong
        by whatever factor these are wrong by, and no amount of control tuning
        compensates for it.
        """
        cfg = self.cfg
        counts = max(200, counts)
        nominal = cfg.counts_to_metres(counts)

        print(f"\nStep 1 of 2: distance. Driving {counts} counts "
              f"(nominally {nominal * 1000:.0f} mm) in a straight line.")
        print("Put the robot against a mark first. Enter to start, or 'skip'.")
        if input("> ").strip().lower() == "skip":
            return

        before = self.drive.state()
        self._run_move(
            "calibration drive",
            lambda: self.drive.move_distance(
                nominal, self.speed_fraction * cfg.max_linear_speed
            ),
        )
        after = self.drive.state()
        actual_counts = 0.5 * (
            (after.left_counts - before.left_counts)
            + (after.right_counts - before.right_counts)
        )

        measured = _ask_float("Measured distance in mm (blank to skip): ")
        if measured and actual_counts:
            mpc = (measured / 1000.0) / actual_counts
            # The ratio of true to reported distance, applied on top of the
            # scale already in use -- so running this twice converges rather
            # than undoing the first correction.
            scale = cfg.distance_scale * mpc / cfg.metres_per_count
            diameter = mpc * cfg.counts_per_wheel_rev / math.pi
            reported = actual_counts * cfg.metres_per_count
            print(f"\n  counts actually turned : {actual_counts:.0f}")
            print(f"  odometry said          : {reported * 1000:.1f} mm, "
                  f"tape says {measured:.1f} mm")
            print(f"  metres per count       : {mpc * 1000:.5f} mm "
                  f"(was {cfg.metres_per_count * 1000:.5f})")
            print(f"  effective wheel diam.  : {diameter * 1000:.2f} mm "
                  f"(nominal {cfg.wheel_diameter_m * 1000:.1f})")
            print("\n  Put this in carrybot.toml, under [drive]:")
            print(f"\n    distance_scale = {scale:.5f}")
            if not 0.75 <= scale <= 1.25:
                print("\n  That is a long way from 1. Check gear_ratio and "
                      "encoder_cpr before trusting it -- a wrong ratio looks "
                      "exactly like this, and the scale would only hide it.")

        print("\nStep 2 of 2: track width. Spinning 360 degrees on the spot.")
        print("Mark the robot's facing first. Enter to start, or 'skip'.")
        if input("> ").strip().lower() == "skip":
            return

        self._run_move(
            "calibration spin",
            lambda: self.drive.turn_in_place(
                2 * math.pi, self.speed_fraction * cfg.max_angular_speed
            ),
        )
        error = _ask_float(
            "Degrees it actually turned (360 = perfect, blank to skip): "
        )
        if error and error > 0:
            corrected = corrected_track_width(cfg.track_width_m, error)
            print(f"\n  track width : {corrected * 1000:.2f} mm "
                  f"(was {cfg.track_width_m * 1000:.1f})")
            print("\n  Put this in carrybot.toml, under [drive]:")
            print(f"\n    track_width_m = {corrected:.5f}")
            print("  Turning short of 360 means the wheels act further apart "
                  "than configured -- on the floor they scrub, so this is often "
                  "wider than the tape says. Repeat once after applying it.")

    def _calibrate_feedforward(self) -> None:
        """Sweep duty, measure wheel speed, fit ``duty = kv * speed + ks``.

        Feedforward is what makes the velocity loop respond in a tenth of a
        second instead of two. Uncalibrated the loop still works -- the
        integrator finds the same duty eventually -- it is just slow, and slow
        velocity control shows up as a robot that wanders off a straight line
        before correcting.
        """
        print("\nWheels off the ground, please. Sweeping duty on both sides.")
        print("Enter to start, anything else to cancel.")
        if input("> ").strip():
            return

        duties = [10, 20, 30, 40, 50, 60, 70, 85, 100]
        samples: List[Tuple[float, float, float]] = []

        try:
            for duty in duties:
                end = time.monotonic() + 1.2
                while time.monotonic() < end:
                    self.drive.set_open_loop_duty(duty, duty)
                    time.sleep(0.02)
                state = self.drive.state()
                samples.append((duty, state.left_speed, state.right_speed))
                print(f"  {duty:3d}% -> left {state.left_speed:6.2f}  "
                      f"right {state.right_speed:6.2f} rad/s")
        except KeyboardInterrupt:
            print("  cancelled")
            return
        finally:
            self.drive.clear_open_loop()

        for name, index in (("left", 1), ("right", 2)):
            points = [(s[index], s[0]) for s in samples if abs(s[index]) > 0.2]
            fit = _least_squares(points)
            if fit is None:
                print(f"\n  {name}: not enough motion to fit. Is it wired up?")
                continue
            kv, ks = fit
            print(f"\n  In carrybot.toml:\n    [drive.{name}_gains]\n"
                  f"    kv = {kv:.3f}\n    ks = {max(0.0, ks):.1f}")
            top = max(abs(s[index]) for s in samples)
            print(f"    top speed {top:.1f} rad/s "
                  f"= {top * self.cfg.wheel_radius_m:.2f} m/s at the wheel")
        print("\n  ks is the duty needed to break stiction; if it lands far "
              "from min_duty, set min_duty to match it.")

    # ------------------------------------------------------------------ #

    def loop(self) -> None:
        print(__doc__)
        print(self.cfg.describe())
        if self.simulated:
            print("\n*** SIMULATION -- no GPIO is being touched ***")
        if self.cfg.left_gains.kv == 0.0:
            print("\nNote: feedforward is uncalibrated (kv = 0). Control will "
                  "work but respond slowly. Run 'f' to fix that.")
        if self.cfg.standby is None:
            print("Note: STBY is strapped to VCC, so there is no hardware "
                  "kill. See DRIVE.md.")

        while True:
            try:
                line = input("\n> ")
            except EOFError:
                break
            try:
                if not self._dispatch(line):
                    break
            except KeyboardInterrupt:
                self.drive.abort()
                print("  interrupted")

    def _dispatch(self, line: str) -> bool:
        i = 0
        while i < len(line):
            char = line[i].lower()
            i += 1
            if char.isspace():
                continue

            digits = ""
            while i < len(line) and line[i].isdigit():
                digits += line[i]
                i += 1

            if not self.execute(char, int(digits) if digits else None):
                return False
        return True


# --------------------------------------------------------------------------- #


def _stdin_ready(timeout: float) -> bool:
    """True if a keypress is waiting. Always False when stdin is not a tty.

    Piped input is *always* "ready", so without the tty check every move would
    abort the instant it started when the console is driven from a script.
    """
    if not sys.stdin.isatty():
        if timeout > 0:
            time.sleep(timeout)
        return False
    try:
        return bool(select.select([sys.stdin], [], [], timeout)[0])
    except (OSError, ValueError):
        time.sleep(timeout)
        return False


def corrected_track_width(configured: float, turned_degrees: float) -> float:
    """The track width that makes odometry agree with a measured spin.

    The spin stops when odometry says 360 degrees, and odometry's turn is the
    wheels' travel difference divided by the track width. The robot really
    turned ``turned_degrees`` for that same travel, so the width that explains
    it is the configured one scaled by 360 / turned: a robot that stopped
    short acts wider than configured.

    This was once written the other way up (scaled by turned / 360), which
    made a short spin *narrower* still. Every turn rate the robot was asked
    for then came out a fraction of what was asked, and odometry reported
    turns several times larger than the robot made.
    """
    if turned_degrees <= 0:
        raise ValueError("turned_degrees must be positive")
    return configured * 360.0 / turned_degrees


def _ask_choice(prompt: str, options: dict):
    while True:
        try:
            text = input(prompt).strip().lower()[:1]
        except EOFError:
            return None
        if text in options:
            return options[text]


class WiringObservation(NamedTuple):
    """What one motor channel did when run alone."""

    wheel: Optional[str]          # the physical wheel that turned, or None
    forward: Optional[bool]       # did it turn forwards
    left_counts: int              # how far each encoder channel moved
    right_counts: int


#: Fewer counts than this on an encoder is "did not move".
_MOVED = 20


def diagnose_wiring(seen: dict, cfg) -> List[str]:
    """Turn two single-channel runs into config changes, physical wheel first.

    Every fix is phrased so it can be applied in one edit of carrybot.toml:
    swap the pin numbers between two sections, then flip ``invert`` in the
    section that now belongs to the wheel concerned.
    """
    lines: List[str] = []
    fixes: List[str] = []
    other = {"left": "right", "right": "left"}

    dead = [ch for ch, o in seen.items() if o.wheel is None]
    if len(dead) == 2:
        left, right = cfg.left_motor, cfg.right_motor
        return [
            "Neither channel turned a wheel on its own. If the motors drive "
            "normally otherwise, the PWM wires are crossed: PWMA is on the pin "
            "[drive.right_motor] calls pwm and PWMB on the left one's. Each "
            "channel then takes its power from the other's PWM pin -- held at "
            "0% here, so nothing turns. Driving, the wheel the loop pushes "
            "harder slows down and the other speeds up: one wheel driving, the "
            "robot swerving. Fix: under [drive.left_motor] set "
            f"pwm = {right.pwm}, and under [drive.right_motor] set pwm = {left.pwm}. "
            "Then run m again.",
            "If they do not drive otherwise either: VM, STBY and the motor wires "
            "(tools/motor_pin_test.py walks through them).",
        ]
    for ch in dead:
        lines.append(f"The {ch} motor channel turned no wheel: check its wiring "
                     f"(IN1/IN2/PWM for [drive.{ch}_motor], the motor's two wires, VM).")
    if len(seen) < 2 or dead:
        return lines or ["Not enough to go on -- run m again."]

    if seen["left"].wheel == seen["right"].wheel:
        return [f"Both channels turned the {seen['left'].wheel} wheel. That is not a "
                "swap; check that the two motors are on separate driver outputs."]

    motors_swapped = seen["left"].wheel == "right"
    if motors_swapped:
        fixes.append("Swap the pin numbers (pwm, in1, in2) between [drive.left_motor] "
                     "and [drive.right_motor]. The left channel drives the right wheel.")

    encoders = {}                 # physical wheel -> (encoder channel, signed counts)
    for ch, o in seen.items():
        counts = {"left": o.left_counts, "right": o.right_counts}
        moved = max(counts, key=lambda k: abs(counts[k]))
        if abs(counts[moved]) < _MOVED:
            lines.append(f"The {o.wheel} wheel turned but neither encoder counted: its "
                         "encoder is unplugged or unpowered.")
            continue
        if abs(counts[other[moved]]) >= max(_MOVED, abs(counts[moved]) // 3):
            lines.append(f"Running one wheel moved BOTH encoders ({counts['left']:+d}, "
                         f"{counts['right']:+d}): an encoder signal is shared or shorted.")
        encoders[o.wheel] = (moved, counts[moved] * (1 if o.forward else -1))

    if len(encoders) == 2:
        if encoders["left"][0] == "right" and encoders["right"][0] == "left":
            fixes.append("Swap the pin numbers (a, b) between [drive.left_encoder] and "
                         "[drive.right_encoder]. Each wheel's encoder is on the other "
                         "wheel's pins -- the cause of a robot that spins at full power "
                         "on its first turn and will not stop.")
        elif encoders["left"][0] == encoders["right"][0]:
            lines.append("Both wheels moved the same encoder channel; the other is "
                         "not connected.")

    for wheel in ("left", "right"):
        o = next(o for o in seen.values() if o.wheel == wheel)
        if not o.forward:
            current = getattr(cfg, f"{wheel}_motor").invert
            fixes.append(f"Set invert = {str(not current).lower()} under "
                         f"[drive.{wheel}_motor]: that wheel ran backwards.")
        if wheel in encoders and encoders[wheel][1] < 0:
            current = getattr(cfg, f"{wheel}_encoder").invert
            fixes.append(f"Set invert = {str(not current).lower()} under "
                         f"[drive.{wheel}_encoder]: its counts fall when the wheel "
                         "goes forwards, which makes the speed loop run away.")

    if fixes:
        lines.append("Change carrybot.toml, in this order, then run m again:")
        lines += [f"  {i}. {fix}" for i, fix in enumerate(fixes, 1)]
    elif not lines:
        lines.append("Wiring matches the config: each channel drives its own wheel "
                     "forwards, and each wheel's encoder counts up.")
    return lines


def _ask_float(prompt: str) -> Optional[float]:
    try:
        text = input(prompt).strip()
    except EOFError:
        return None
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        print("  not a number, skipping")
        return None


def _least_squares(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Fit ``y = m*x + c`` over (speed, duty) pairs."""
    n = len(points)
    if n < 3:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denominator = n * sxx - sx * sx
    if abs(denominator) < 1e-9:
        return None
    m = (n * sxy - sx * sy) / denominator
    c = (sy - m * sx) / n
    return m, c


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CarryBot drive console")
    parser.add_argument("--sim", action="store_true", help="run against the simulator")
    parser.add_argument("--rate", type=float, default=None, help="control rate in Hz")
    parser.add_argument("--config", metavar="PATH", default=None,
                        help="robot config file (default: carrybot.toml, found "
                             "as described in `python -m carrybot_config`)")
    args = parser.parse_args(argv)

    from carrybot_config import ConfigError, load_config

    try:
        loaded = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    print(loaded.describe_source())
    config = loaded.drive
    if args.rate:
        config.control_rate_hz = args.rate

    simulator = None
    if args.sim:
        from .backend import SimBackend

        simulator = SimBackend(config)
        backend = simulator
    else:
        from .backend import LgpioBackend

        backend = LgpioBackend(config.chip)

    drive = DriveController(config, backend)
    try:
        drive.start()
        if simulator is not None:
            simulator.start_thread()
        Console(drive, simulated=args.sim).loop()
    except KeyboardInterrupt:
        pass
    finally:
        drive.close()
        if simulator is not None:
            simulator.stop_thread()
        print("\nMotors stopped, GPIO released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
