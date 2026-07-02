#!/usr/bin/env python3
"""Common utilities for no-external-tracker Tello + Raspberry Pi demos.

These demos talk to the Raspberry Pi gateway created in the previous patch:
    raspberry_pi/tello_gateway.py

Protocol:
    newline-delimited JSON over TCP from the main PC to each Raspberry Pi.

The gateway then sends Tello SDK UDP commands to its paired Tello.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]

# The gateway client/spec/reply implementation now lives in tello_fleet.  Keep
# the historical names here (as aliases) so the demos import unchanged.
from tello_fleet import (  # noqa: E402
    DroneClient as PiGatewayClient,
    GatewaySpec,
    GatewayReply,
    FleetError,
    specs_from_config,
)


class DemoError(RuntimeError):
    """Raised when a demo cannot continue safely (non-client errors)."""


class JsonlLogger:
    """Tiny JSONL logger."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> None:
        rec = {"t_wall": time.time(), "event": event}
        rec.update(fields)
        self._fp.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(path: str | os.PathLike[str]) -> JsonDict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def gateway_specs_from_config(cfg: JsonDict) -> List[GatewaySpec]:
    gateways = cfg.get("gateways")
    specs: List[GatewaySpec] = []
    if gateways:
        for i, g in enumerate(gateways):
            specs.append(GatewaySpec(
                name=str(g.get("name", f"tello-{i+1}")),
                host=str(g["host"]),
                port=int(g.get("port", cfg.get("pi_port", 10000))),
            ))
        return specs

    # Backward-compatible format used by the previous real-control config.
    hosts = cfg.get("pi_hosts")
    if not hosts:
        raise DemoError("config requires either 'gateways' or 'pi_hosts'")
    port = int(cfg.get("pi_port", 10000))
    for i, host in enumerate(hosts):
        specs.append(GatewaySpec(name=f"tello-{i+1}", host=str(host), port=port))
    return specs


def make_clients(cfg: JsonDict) -> List[PiGatewayClient]:
    timeout_s = float(cfg.get("pi_timeout_s", 1.0))
    return [PiGatewayClient(spec, timeout_s=timeout_s) for spec in gateway_specs_from_config(cfg)]


def close_all(clients: Iterable[PiGatewayClient]) -> None:
    for c in clients:
        c.close()


def connect_all(clients: Sequence[PiGatewayClient], logger: Optional[JsonlLogger] = None) -> None:
    for c in clients:
        start = time.time()
        c.connect()
        elapsed = time.time() - start
        if logger:
            logger.write("pc_connect", drone=c.name, host=c.spec.host, port=c.spec.port, elapsed_s=elapsed)


def parallel_call(
    clients: Sequence[PiGatewayClient],
    fn: Callable[[PiGatewayClient], GatewayReply],
    logger: Optional[JsonlLogger] = None,
    event: str = "parallel_call",
) -> List[Tuple[str, Optional[GatewayReply], Optional[str]]]:
    """Run the same client operation on all gateways in parallel."""
    results: List[Tuple[str, Optional[GatewayReply], Optional[str]]] = []
    with ThreadPoolExecutor(max_workers=len(clients)) as ex:
        fut_to_client = {ex.submit(fn, c): c for c in clients}
        for fut in as_completed(fut_to_client):
            c = fut_to_client[fut]
            try:
                reply = fut.result()
                results.append((c.name, reply, None))
                if logger:
                    logger.write(event, drone=c.name, ok=True, elapsed_s=reply.elapsed_s, reply=reply.raw)
            except Exception as exc:
                results.append((c.name, None, str(exc)))
                if logger:
                    logger.write(event, drone=c.name, ok=False, error=str(exc))
    results.sort(key=lambda x: x[0])
    return results


def print_results(title: str, results: Sequence[Tuple[str, Optional[GatewayReply], Optional[str]]]) -> None:
    print(f"\n[{title}]")
    for name, reply, err in results:
        if err:
            print(f"  {name}: ERROR {err}")
        else:
            raw = reply.raw if reply is not None else {}
            shown = raw.get("resp", raw.get("cmd", raw.get("status", raw)))
            print(f"  {name}: ok elapsed={reply.elapsed_s:.3f}s reply={shown}")


def any_error(results: Sequence[Tuple[str, Optional[GatewayReply], Optional[str]]]) -> bool:
    return any(err is not None for _, _, err in results)


def require_yes(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "yes", False):
        return
    print("\nSAFETY CONFIRMATION")
    print(message)
    ans = input("Type YES to continue: ").strip()
    if ans != "YES":
        raise DemoError("aborted by user")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/no_tracker_demo_config.example.json",
                        help="JSON config file for Pi gateway addresses")
    parser.add_argument("--log-dir", default=None,
                        help="directory for JSONL logs. Default: config log_dir or ./logs")
    parser.add_argument("--yes", action="store_true",
                        help="skip interactive safety confirmation for flight demos")


def log_path(cfg: JsonDict, args: argparse.Namespace, demo_name: str) -> Path:
    base = args.log_dir or cfg.get("log_dir", "logs")
    return Path(base) / f"{timestamp()}_{demo_name}.jsonl"


def cfg_rc_limit(cfg: JsonDict, args: argparse.Namespace, default: int = 25) -> int:
    value = getattr(args, "rc_limit", None)
    if value is not None:
        return int(value)
    return int(cfg.get("rc_limit", default))


def cfg_rate_hz(cfg: JsonDict, args: argparse.Namespace, key: str, default: float) -> float:
    value = getattr(args, key, None)
    if value is not None:
        return float(value)
    return float(cfg.get(key, default))


def battery_summary(results: Sequence[Tuple[str, Optional[GatewayReply], Optional[str]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, reply, err in results:
        out[name] = {"error": err} if err else {"resp": reply.raw.get("resp", "")}
    return out


def safe_land_all(clients: Sequence[PiGatewayClient], logger: Optional[JsonlLogger] = None) -> None:
    print("\n[safety] sending land to all drones")
    results = parallel_call(clients, lambda c: c.land(), logger, event="land_all")
    print_results("land_all", results)


def hover_loop(
    clients: Sequence[PiGatewayClient],
    duration_s: float,
    rate_hz: float,
    rc_limit: int,
    logger: JsonlLogger,
    actions_fn: Callable[[float, int], Sequence[Sequence[float]]],
    event_prefix: str,
) -> None:
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    dt = 1.0 / rate_hz
    start = time.time()
    k = 0
    next_t = start
    while True:
        now = time.time()
        elapsed = now - start
        if elapsed >= duration_s:
            break
        actions = actions_fn(elapsed, k)
        if len(actions) != len(clients):
            raise DemoError("actions_fn returned wrong number of actions")
        for c, a in zip(clients, actions):
            try:
                reply = c.rc(a, rc_limit=rc_limit)
                logger.write(event_prefix + "_rc", step=k, elapsed_s=elapsed,
                             drone=c.name, action=list(a), rc_limit=rc_limit, reply=reply.raw)
            except Exception as exc:
                logger.write(event_prefix + "_rc", step=k, elapsed_s=elapsed,
                             drone=c.name, action=list(a), rc_limit=rc_limit, error=str(exc))
                raise
        k += 1
        next_t += dt
        sleep_s = next_t - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
