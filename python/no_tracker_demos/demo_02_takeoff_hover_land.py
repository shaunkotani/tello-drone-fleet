#!/usr/bin/env python3
"""Demo 2: three-drone takeoff-hover-land without an external tracker.

This demo performs flight.  It deliberately sends only rc 0 0 0 0 during the
hover phase, so it does not rely on position feedback.

Use a large open area, propeller guards, and be ready to stop the program.
"""
from __future__ import annotations

import argparse
import time

from no_tracker_common import (
    JsonlLogger,
    add_common_args,
    any_error,
    cfg_rate_hz,
    cfg_rc_limit,
    close_all,
    connect_all,
    hover_loop,
    load_config,
    log_path,
    make_clients,
    parallel_call,
    print_results,
    require_yes,
    safe_land_all,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="No-tracker takeoff-hover-land demo")
    add_common_args(parser)
    parser.add_argument("--duration", type=float, default=8.0, help="hover duration in seconds")
    parser.add_argument("--rate-hz", dest="rate_hz", type=float, default=None, help="rc send rate")
    parser.add_argument("--rc-limit", type=int, default=None, help="rc limit, only 0 is sent in this demo")
    parser.add_argument("--takeoff-settle", type=float, default=5.0, help="seconds to wait after takeoff")
    args = parser.parse_args()

    require_yes(args, "This demo will take off all connected Tello drones, hover, and land.")

    cfg = load_config(args.config)
    clients = make_clients(cfg)
    rc_limit = cfg_rc_limit(cfg, args, default=25)
    rate_hz = cfg_rate_hz(cfg, args, "rate_hz", default=10.0)
    lp = log_path(cfg, args, "demo_02_takeoff_hover_land")
    print(f"log: {lp}")

    with JsonlLogger(lp) as logger:
        logger.write("demo_start", demo="demo_02_takeoff_hover_land", config=cfg,
                     duration_s=args.duration, rate_hz=rate_hz, rc_limit=rc_limit)
        try:
            connect_all(clients, logger)

            results = parallel_call(clients, lambda c: c.command(), logger, event="command_all")
            print_results("command_all", results)
            if any_error(results):
                return 2

            b = parallel_call(clients, lambda c: c.query("battery?"), logger, event="battery_all")
            print_results("battery_all", b)
            if any_error(b):
                return 2

            t = parallel_call(clients, lambda c: c.takeoff(), logger, event="takeoff_all")
            print_results("takeoff_all", t)
            if any_error(t):
                safe_land_all(clients, logger)
                return 3

            print(f"waiting {args.takeoff_settle:.1f}s for takeoff stabilization")
            time.sleep(args.takeoff_settle)

            print(f"hovering for {args.duration:.1f}s at {rate_hz:.1f} Hz")
            hover_loop(
                clients=clients,
                duration_s=args.duration,
                rate_hz=rate_hz,
                rc_limit=rc_limit,
                logger=logger,
                actions_fn=lambda elapsed, k: [[0.0, 0.0, 0.0, 0.0] for _ in clients],
                event_prefix="hover",
            )

            safe_land_all(clients, logger)
            logger.write("demo_end", ok=True)
            print("\nDemo 2 finished: takeoff-hover-land command path was verified.")
            return 0
        except KeyboardInterrupt:
            logger.write("keyboard_interrupt")
            safe_land_all(clients, logger)
            return 130
        except Exception as exc:
            logger.write("demo_error", error=str(exc))
            safe_land_all(clients, logger)
            raise
        finally:
            close_all(clients)


if __name__ == "__main__":
    raise SystemExit(main())
