#!/usr/bin/env python3
"""Demo 1: non-flight communication check for 3 Tello + Raspberry Pi gateways.

This demo does not start motors.  It verifies the communication chain:
    main PC -> Raspberry Pi gateway -> paired Tello -> Raspberry Pi -> main PC

It sends SDK command, battery?, and status to each gateway and logs replies.
"""
from __future__ import annotations

import argparse
import time

from no_tracker_common import (
    JsonlLogger,
    add_common_args,
    any_error,
    close_all,
    connect_all,
    load_config,
    log_path,
    make_clients,
    parallel_call,
    print_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="No-tracker non-flight communication demo")
    add_common_args(parser)
    parser.add_argument("--repeat", type=int, default=1, help="number of battery/status repeats")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between repeats")
    parser.add_argument("--skip-command", action="store_true", help="do not send SDK command")
    args = parser.parse_args()

    cfg = load_config(args.config)
    clients = make_clients(cfg)
    lp = log_path(cfg, args, "demo_01_comm_check")
    print(f"log: {lp}")

    with JsonlLogger(lp) as logger:
        logger.write("demo_start", demo="demo_01_comm_check", config=cfg)
        try:
            connect_all(clients, logger)

            if not args.skip_command:
                results = parallel_call(clients, lambda c: c.command(), logger, event="command_all")
                print_results("command_all", results)
                if any_error(results):
                    return 2

            for r in range(args.repeat):
                print(f"\n--- repeat {r + 1}/{args.repeat} ---")
                b = parallel_call(clients, lambda c: c.query("battery?"), logger, event="battery_all")
                print_results("battery_all", b)

                s = parallel_call(clients, lambda c: c.status(), logger, event="status_all")
                print_results("status_all", s)

                if r + 1 < args.repeat:
                    time.sleep(args.interval)

            logger.write("demo_end", ok=True)
            print("\nDemo 1 finished: communication path is reachable if all rows are ok.")
            return 0
        finally:
            close_all(clients)


if __name__ == "__main__":
    raise SystemExit(main())
