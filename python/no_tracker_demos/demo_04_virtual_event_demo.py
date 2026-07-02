#!/usr/bin/env python3
"""Demo 4: virtual event-triggered coordination demo without an external tracker.

This demo runs a small virtual consensus/event-trigger calculation on the main
PC.  It does not use real drone positions.  The event pattern is reflected on
real Tello drones as safe yaw pulses only when --fly is specified.

Default mode is non-flight dry-run: it connects to the gateways, queries battery,
runs the virtual algorithm, and logs the per-agent events/actions without sending
flight commands.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from typing import List, Sequence, Tuple

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


def consensus_step(theta: List[float], alpha: float) -> List[float]:
    """One ring-neighbor consensus-like update."""
    n = len(theta)
    out = []
    for i in range(n):
        left = theta[(i - 1) % n]
        right = theta[(i + 1) % n]
        out.append(theta[i] + alpha * ((left - theta[i]) + (right - theta[i])))
    return out


def event_threshold(q: int, eps0: float, rho: float, eps_min: float) -> float:
    return eps0 * (rho ** q) + eps_min


def event_to_action(event: bool, delta: float, yaw_norm: float) -> List[float]:
    if not event:
        return [0.0, 0.0, 0.0, 0.0]
    # Positive delta -> clockwise pulse, negative delta -> counter-clockwise pulse.
    sign = 1.0 if delta >= 0 else -1.0
    return [0.0, 0.0, 0.0, sign * yaw_norm]


def run_virtual_algorithm(
    n: int,
    steps: int,
    alpha: float,
    eps0: float,
    rho: float,
    eps_min: float,
    yaw_norm: float,
    seed: int,
) -> List[Tuple[int, List[float], List[bool], List[List[float]]]]:
    rng = random.Random(seed)
    theta = [rng.uniform(-0.8, 0.8) for _ in range(n)]
    sent = list(theta)
    records: List[Tuple[int, List[float], List[bool], List[List[float]]]] = []
    for q in range(steps):
        # A small common target oscillation makes events visible even after partial consensus.
        drive = 0.03 * math.sin(0.25 * q)
        theta = [x + drive for x in consensus_step(theta, alpha=alpha)]
        eps = event_threshold(q, eps0=eps0, rho=rho, eps_min=eps_min)
        events: List[bool] = []
        actions: List[List[float]] = []
        for i, val in enumerate(theta):
            delta = val - sent[i]
            ev = abs(delta) >= eps
            events.append(ev)
            actions.append(event_to_action(ev, delta, yaw_norm=yaw_norm))
            if ev:
                sent[i] = val
        records.append((q, list(theta), events, actions))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="No-tracker virtual event-triggered coordination demo")
    add_common_args(parser)
    parser.add_argument("--fly", action="store_true",
                        help="take off and reflect virtual events as yaw pulses")
    parser.add_argument("--steps", type=int, default=60, help="virtual coordination steps")
    parser.add_argument("--rate-hz", dest="rate_hz", type=float, default=None, help="rc/log rate")
    parser.add_argument("--rc-limit", type=int, default=None, help="maximum absolute rc value")
    parser.add_argument("--yaw-norm", type=float, default=0.18, help="yaw pulse normalized value")
    parser.add_argument("--alpha", type=float, default=0.20, help="virtual consensus gain")
    parser.add_argument("--eps0", type=float, default=0.18, help="initial event threshold")
    parser.add_argument("--rho", type=float, default=0.98, help="event threshold decay")
    parser.add_argument("--eps-min", type=float, default=0.04, help="minimum event threshold")
    parser.add_argument("--seed", type=int, default=7, help="virtual state random seed")
    parser.add_argument("--takeoff-settle", type=float, default=5.0, help="seconds to wait after takeoff")
    args = parser.parse_args()

    if args.fly:
        require_yes(args, "This demo will take off all connected Tello drones and reflect virtual events as yaw pulses.")

    cfg = load_config(args.config)
    clients = make_clients(cfg)
    rc_limit = cfg_rc_limit(cfg, args, default=25)
    rate_hz = cfg_rate_hz(cfg, args, "rate_hz", default=5.0)
    lp = log_path(cfg, args, "demo_04_virtual_event_demo")
    print(f"log: {lp}")

    with JsonlLogger(lp) as logger:
        logger.write("demo_start", demo="demo_04_virtual_event_demo", config=cfg,
                     fly=args.fly, steps=args.steps, rate_hz=rate_hz, rc_limit=rc_limit)
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

            records = run_virtual_algorithm(
                n=len(clients),
                steps=args.steps,
                alpha=args.alpha,
                eps0=args.eps0,
                rho=args.rho,
                eps_min=args.eps_min,
                yaw_norm=args.yaw_norm,
                seed=args.seed,
            )

            if not args.fly:
                print("\nDry-run mode: no takeoff and no rc commands are sent.")
                for q, theta, events, actions in records:
                    logger.write("virtual_step", q=q, theta=theta, events=events, actions=actions)
                    if q < 10 or q == args.steps - 1:
                        print(f"q={q:03d} events={events} theta={[round(x, 3) for x in theta]}")
                    time.sleep(1.0 / rate_hz)
                logger.write("demo_end", ok=True, fly=False)
                print("\nDemo 4 dry-run finished: virtual coordination/event logs were generated.")
                return 0

            t = parallel_call(clients, lambda c: c.takeoff(), logger, event="takeoff_all")
            print_results("takeoff_all", t)
            if any_error(t):
                safe_land_all(clients, logger)
                return 3

            print(f"waiting {args.takeoff_settle:.1f}s for takeoff stabilization")
            time.sleep(args.takeoff_settle)

            dt = 1.0 / rate_hz
            start = time.time()
            for q, theta, events, actions in records:
                step_start = time.time()
                for c, a in zip(clients, actions):
                    reply = c.rc(a, rc_limit=rc_limit)
                    logger.write("virtual_flight_step", q=q, elapsed_s=time.time() - start,
                                 drone=c.name, theta=theta, events=events,
                                 action=a, rc_limit=rc_limit, reply=reply.raw)
                print(f"q={q:03d} events={events} actions={actions}")
                sleep_s = dt - (time.time() - step_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            # Stop yaw pulses before landing.
            hover_loop(
                clients=clients,
                duration_s=2.0,
                rate_hz=rate_hz,
                rc_limit=rc_limit,
                logger=logger,
                actions_fn=lambda elapsed, k: [[0.0, 0.0, 0.0, 0.0] for _ in clients],
                event_prefix="final_hover",
            )
            safe_land_all(clients, logger)
            logger.write("demo_end", ok=True, fly=True)
            print("\nDemo 4 flight mode finished: virtual event-trigger results were reflected as yaw pulses.")
            return 0
        except KeyboardInterrupt:
            logger.write("keyboard_interrupt")
            if args.fly:
                safe_land_all(clients, logger)
            return 130
        except Exception as exc:
            logger.write("demo_error", error=str(exc))
            if args.fly:
                safe_land_all(clients, logger)
            raise
        finally:
            close_all(clients)


if __name__ == "__main__":
    raise SystemExit(main())
