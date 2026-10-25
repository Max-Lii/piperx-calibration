"""Connect to two Piper-X CAN interfaces and print feedback without motion."""
from __future__ import annotations

import argparse
import json
import time

from pyAgxArm import PiperFW

from .piper_x import PiperXConfig, PiperXDualDriver
from .safety_audit import assert_deployment_safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Piper-X dual-arm read-only CAN probe")
    parser.add_argument("--left-can", required=True)
    parser.add_argument("--right-can", required=True)
    parser.add_argument("--firmware", choices=(PiperFW.DEFAULT, PiperFW.V183, PiperFW.V188, PiperFW.V189), required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--period", type=float, default=0.25)
    args = parser.parse_args()
    if args.left_can == args.right_can:
        raise SystemExit("left and right CAN interfaces must be different")
    if args.samples < 1 or args.period <= 0:
        raise SystemExit("samples and period must be positive")
    assert_deployment_safe()
    driver = PiperXDualDriver(PiperXConfig(args.left_can, args.right_can, args.firmware))
    try:
        driver.connect()
        for index in range(args.samples):
            print(json.dumps({"sample": index + 1, **driver.readonly_diagnostics()}, ensure_ascii=False), flush=True)
            if index + 1 < args.samples:
                time.sleep(args.period)
    finally:
        # disconnect() closes host-side CAN threads only. It does not send an
        # SDK disable/reset/emergency-stop command.
        driver.close()


if __name__ == "__main__":
    main()
