#!/usr/bin/env python3
"""Demo 3: synchronized role-based yaw motion without an external tracker.

The drones take off, hover, then receive different yaw roles while horizontal
channels remain zero.  This checks role assignment and synchronized periodic rc
command delivery without requiring position feedback.
"""
from __future__ import annotations

import argparse
import time
from typing import List

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


def role_actions(n: int, yaw_norm: float) -> List[List[float]]:
    """Return [lr, fb, ud, yaw] actions for the role demo."""
    base: List[List[float]] = []
    for i in range(n):
        if i % 3 == 0:
            yaw = yaw_norm
        elif i % 3 == 1:
            yaw = -yaw_norm
        else:
            yaw = 0.0
        base.append([0.0, 0.0, 0.0, yaw])
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="No-tracker synchronized yaw role demo")
    add_common_args(parser)
    parser.add_argument("--duration", type=float, default=10.0, help="yaw phase duration in seconds")
    parser.add_argument("--pre-hover", type=float, default=3.0, help="hover seconds before yaw")
    parser.add_argument("--post-hover", type=float, default=3.0, help="hover seconds after yaw")
    parser.add_argument("--rate-hz", dest="rate_hz", type=float, default=None, help="rc send rate")
    parser.add_argument("--rc-limit", type=int, default=None, help="maximum absolute rc value")
    parser.add_argument("--yaw-norm", type=float, default=0.20,
                        help="normalized yaw action. Actual rc yaw = rc_limit * yaw_norm")
    parser.add_argument("--takeoff-settle", type=float, default=5.0, help="seconds to wait after takeoff")
    args = parser.parse_args()

    require_yes(args, "This demo will take off all connected Tello drones and rotate them in place.")

    cfg = load_config(args.config)
    clients = make_clients(cfg)
    rc_limit = cfg_rc_limit(cfg, args, default=25)
    rate_hz = cfg_rate_hz(cfg, args, "rate_hz", default=10.0)
    lp = log_path(cfg, args, "demo_03_sync_yaw_roles")
    print(f"log: {lp}")

    with JsonlLogger(lp) as logger:
        logger.write("demo_start", demo="demo_03_sync_yaw_roles", config=cfg,
                     duration_s=args.duration, rate_hz=rate_hz, rc_limit=rc_limit,
                     yaw_norm=args.yaw_norm)
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

            if args.pre_hover > 0:
                print(f"pre-hover {args.pre_hover:.1f}s")
                hover_loop(
                    clients, args.pre_hover, rate_hz, rc_limit, logger,
                    actions_fn=lambda elapsed, k: [[0.0, 0.0, 0.0, 0.0] for _ in clients],
                    event_prefix="pre_hover",
                )

            actions = role_actions(len(clients), args.yaw_norm)
            print("yaw role actions:")
            for c, a in zip(clients, actions):
                print(f"  {c.name}: {a}")
            logger.write("yaw_roles", actions={c.name: a for c, a in zip(clients, actions)})

            hover_loop(
                clients=clients,
                duration_s=args.duration,
                rate_hz=rate_hz,
                rc_limit=rc_limit,
                logger=logger,
                actions_fn=lambda elapsed, k: actions,
                event_prefix="yaw_role",
            )

            if args.post_hover > 0:
                print(f"post-hover {args.post_hover:.1f}s")
                hover_loop(
                    clients, args.post_hover, rate_hz, rc_limit, logger,
                    actions_fn=lambda elapsed, k: [[0.0, 0.0, 0.0, 0.0] for _ in clients],
                    event_prefix="post_hover",
                )

            safe_land_all(clients, logger)
            logger.write("demo_end", ok=True)
            print("\nDemo 3 finished: role-specific synchronized yaw command delivery was verified.")
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
