#!/usr/bin/env python3
""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN_SECONDS = 1.5


def ask(question: str) -> bool:
    while True:
        answer = input(f"  {question} [y/n] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


class Pins:
    """The handful of lgpio calls this needs, with everything released on exit."""

    def __init__(self, lgpio, chip: int, pins):
        self.lg = lgpio
        self.h = lgpio.gpiochip_open(chip)
        self.pins = list(pins)
        for pin in self.pins:
            lgpio.gpio_claim_output(self.h, pin, 0)

    def write(self, pin: int, level: int) -> None:
        self.lg.gpio_write(self.h, pin, level)

    def pwm(self, pin: int, frequency: int, duty: float) -> None:
        self.lg.tx_pwm(self.h, pin, frequency, duty)

    def coast(self) -> None:
        for pin in self.pins:
            try:
                self.lg.tx_pwm(self.h, pin, 0, 0)        # stop any software PWM
            except Exception:
                pass
            self.lg.gpio_write(self.h, pin, 0)

    def close(self) -> None:
        try:
            self.coast()
        finally:
            for pin in self.pins:
                try:
                    self.lg.gpio_free(self.h, pin)
                except Exception:
                    pass
            self.lg.gpiochip_close(self.h)


def run_steady(pins: Pins, motor, forward: bool) -> None:
    """IN1/IN2 for the direction and PWM held HIGH: no PWM involved at all."""
    pins.write(motor.in1, 1 if forward else 0)
    pins.write(motor.in2, 0 if forward else 1)
    pins.write(motor.pwm, 1)
    time.sleep(RUN_SECONDS)
    pins.coast()


def run_pwm(pins: Pins, motor, frequency: int, duty: float) -> None:
    """Forward, with lgpio software PWM on the PWM pin -- what the drive uses."""
    pins.write(motor.in1, 1)
    pins.write(motor.in2, 0)
    pins.pwm(motor.pwm, frequency, duty)
    time.sleep(RUN_SECONDS)
    pins.coast()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="config file (default: carrybot.toml)")
    parser.add_argument("--duty", type=float, default=60.0, help="PWM duty for step 2, percent")
    args = parser.parse_args(argv)

    from carrybot_config import ConfigError, load_config

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    try:
        import lgpio
    except ImportError:
        print("lgpio is not installed -- run this on the robot's Pi "
              "(sudo apt install python3-lgpio)")
        return 2

    d = config.drive
    motors = {"left": d.left_motor, "right": d.right_motor}
    used = [p for m in motors.values() for p in (m.in1, m.in2, m.pwm)]
    if d.standby is not None:
        used.append(d.standby)

    print(config.describe_source())
    for side, m in motors.items():
        print(f"  {side:5s} motor: IN1 = GPIO{m.in1}, IN2 = GPIO{m.in2}, PWM = GPIO{m.pwm}")
    print(f"  standby (STBY): {f'GPIO{d.standby}' if d.standby is not None else 'not on a GPIO -- must be tied to 3.3 V'}")
    print("\nWHEELS OFF THE GROUND. Each step runs one motor for 1.5 s.")
    input("Enter to start, Ctrl-C to quit... ")

    try:
        pins = Pins(lgpio, d.chip, used)
    except Exception as exc:
        print(f"\ncould not claim the pins: {exc}")
        print("  Is the drive console or the live interface still running? Only one "
              "program can hold a pin. Otherwise: is this user in the gpio group?")
        return 2

    results = {}
    try:
        if d.standby is not None:
            pins.write(d.standby, 1)

        print("\nFirst, with every pin low the driver is in coast.")
        free = ask("Can you turn each wheel by hand now (stiffly is fine -- it is a gearbox)?")

        for side, m in motors.items():
            print(f"\n== {side} motor, pins held steady (no PWM) ==")
            run_steady(pins, m, forward=True)
            steady_fwd = ask(f"Did the {side} wheel spin?")
            run_steady(pins, m, forward=False)
            steady_rev = ask(f"Did it spin the other way this time?")
            print(f"== {side} motor, software PWM at {args.duty:.0f}%, "
                  f"{d.pwm_frequency} Hz (what the drive uses) ==")
            try:
                run_pwm(pins, m, d.pwm_frequency, args.duty)
                pwm_ok = ask(f"Did the {side} wheel spin?")
            except Exception as exc:
                print(f"  tx_pwm failed: {exc}")
                pwm_ok = False
            results[side] = (steady_fwd, steady_rev, pwm_ok)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 1
    finally:
        pins.close()
        print("\nAll pins released, motors coasting.")

    print("\n" + "=" * 70)
    print(diagnose(results, free, d))
    return 0


def diagnose(results, free: bool, d) -> str:
    lines = []
    steady = {side: r[0] or r[1] for side, r in results.items()}
    pwm = {side: r[2] for side, r in results.items()}

    if not free:
        lines.append(
            "The wheels are stiff even with every pin low. If they will not turn at "
            "all, something mechanical is binding -- or STBY and the IN pins are not "
            "where the config says and the driver is still braking. Check that "
            "nothing else was holding the pins.")
    if not any(steady.values()):
        lines += [
            "Neither motor turns even with its pins held steady, so it is not the "
            "code and not PWM. Check, in this order:",
            "  1. VM: the motor battery to the TB6612's VM pin, switched on. VCC "
            "alone (3.3 V from the Pi) powers the logic and can still brake, but "
            "cannot drive.",
            "  2. GND: the battery's negative and the Pi's ground joined.",
            "  3. STBY: tied to 3.3 V" + (f", or GPIO{d.standby} reaching it" if d.standby is not None else "")
            + ". Low puts the driver in standby.",
            f"  4. PWMA to GPIO{d.left_motor.pwm} and PWMB to GPIO{d.right_motor.pwm}. If "
            "PWM never reaches the chip, every drive command reads as SHORT BRAKE -- "
            "exactly 'locked and won't drive'.",
            "  With a meter: during a step, AIN1/BIN1 and PWMA/PWMB should read 3.3 V "
            "at the chip's pins, and VM should read your battery voltage.",
            "  If the robot does drive in the live program -- but straight driving "
            "goes on one wheel -- PWMA and PWMB are crossed: here the other motor's "
            "PWM pin is held low, so neither turns. Swap the two pwm numbers "
            f"(GPIO{d.left_motor.pwm} and GPIO{d.right_motor.pwm}) between "
            "[drive.left_motor] and [drive.right_motor].",
        ]
    else:
        for side, ok in steady.items():
            m = d.left_motor if side == "left" else d.right_motor
            if not ok:
                lines.append(
                    f"The {side} motor does not turn but the other does: check that "
                    f"side's wires -- GPIO{m.in1}/{m.in2} to its IN1/IN2, GPIO{m.pwm} "
                    f"to its PWM pin, and the motor to its two output pins.")
        if any(steady[s] and not pwm[s] for s in steady):
            lines.append(
                "Steady pins drive it but software PWM does not. Check "
                "/boot/firmware/config.txt for a 'dtoverlay=pwm' line claiming "
                f"GPIO{d.left_motor.pwm}/{d.right_motor.pwm}, and update lgpio "
                "(sudo apt install --only-upgrade python3-lgpio). Try --duty 100.")
        if all(steady.values()) and all(pwm.values()):
            lines.append(
                "Both motors drive both ways, with and without PWM: the wiring and the "
                "driver are fine, so the fault is in how the drive code is being used. "
                "Run `python -m carrybot_drive.cli`, type m30, and send me the state "
                "line it prints -- the duty and motor mode in it say where it stops.")
    for side, (fwd, rev, _p) in results.items():
        if fwd != rev:
            lines.append(f"The {side} motor turns one way only: one of its IN pins is "
                         "not reaching the driver.")
    return "\n".join(lines) if lines else "No conclusion -- the answers contradict each other."


if __name__ == "__main__":
    raise SystemExit(main())
