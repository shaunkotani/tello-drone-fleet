#!/usr/bin/env python3
"""Live monitor for Raspberry Pi Tello gateways.

Polls each gateway's ``status`` message over TCP and shows, per drone, whether:
  * the gateway itself is reachable (the Pi / service is up), and
  * the Tello link is healthy (up / down), with battery and last-seen age.

Transitions (link lost / restored, gateway reachable / unreachable) are printed
as timestamped ALERT lines so a glance at the terminal tells you the fleet state.

Usage:
    python pi_monitor.py --config configs/pc_real_config.example.json
    python pi_monitor.py --config configs/pc_real_config.example.json --interval 1.0

Config may use either form (same as the demos / real-control config):
    {"gateways": [{"name": "tello-1", "host": "192.168.50.101", "port": 10000}, ...]}
    {"pi_hosts": ["192.168.50.101", ...], "pi_port": 10000}
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any, Dict, List, Optional


# --- gateway link states (mirror raspberry_pi/tello_gateway.py) -------------
LINK_UP = "up"
LINK_DOWN = "down"
LINK_INIT = "init"
UNREACHABLE = "unreachable"  # monitor-side pseudo-state: cannot reach the gateway


@dataclass
class GatewaySpec:
    name: str
    host: str
    port: int = 10000


def load_specs(config_path: str) -> List[GatewaySpec]:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    gateways = cfg.get("gateways")
    if gateways:
        return [
            GatewaySpec(
                name=str(g.get("name", f"tello-{i + 1}")),
                host=str(g["host"]),
                port=int(g.get("port", cfg.get("pi_port", 10000))),
            )
            for i, g in enumerate(gateways)
        ]
    hosts = cfg.get("pi_hosts")
    if not hosts:
        raise SystemExit("config needs either 'gateways' or 'pi_hosts'")
    port = int(cfg.get("pi_port", 10000))
    return [GatewaySpec(name=f"tello-{i + 1}", host=str(h), port=port)
            for i, h in enumerate(hosts)]


def query_status(spec: GatewaySpec, timeout_s: float) -> Optional[Dict[str, Any]]:
    """Open a short-lived TCP connection, request status, return the status dict.

    Returns None if the gateway cannot be reached or gave a malformed reply.
    """
    try:
        with socket.create_connection((spec.host, spec.port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(b'{"type":"status"}\n')
            f = sock.makefile("r", encoding="utf-8", newline="\n")
            line = f.readline()
        if not line:
            return None
        reply = json.loads(line)
        if not reply.get("ok"):
            return None
        return reply.get("status", {})
    except (OSError, json.JSONDecodeError):
        return None


# --- rendering --------------------------------------------------------------
class Color:
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"


def _enable_windows_ansi() -> None:
    """Best-effort enable of ANSI escape processing on Windows terminals."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004 on the stdout handle (-11).
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def link_of(status: Optional[Dict[str, Any]]) -> str:
    if status is None:
        return UNREACHABLE
    return str(status.get("tello_link", LINK_INIT))


def _colorize(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{Color.RESET}" if use_color else text


def _fmt_uptime(uptime: Optional[float]) -> str:
    """Human-readable, fixed-ish width uptime so it stays legible when the
    gateway runs for days (raw seconds would grow without bound)."""
    if uptime is None:
        return "-"
    s = int(uptime)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def render_row(spec: GatewaySpec, status: Optional[Dict[str, Any]], use_color: bool) -> str:
    link = link_of(status)
    color = {
        LINK_UP: Color.GREEN,
        LINK_DOWN: Color.RED,
        UNREACHABLE: Color.MAGENTA,
        LINK_INIT: Color.YELLOW,
    }.get(link, Color.YELLOW)
    label = _colorize(f"{link.upper():<11}", color, use_color)

    if status is None:
        detail = _colorize("gateway not reachable", Color.DIM, use_color)
    else:
        batt = status.get("battery") or "?"
        age = status.get("last_seen_age_s")
        uptime = status.get("uptime_s")
        sdk = "sdk" if status.get("sdk_mode") else "no-sdk"
        armed = " ARMED" if status.get("armed") else ""
        age_str = f"{age}s" if age is not None else "-"
        detail = f"batt={batt:>3}%  age={age_str:<6} {sdk} up={_fmt_uptime(uptime):<6}{armed}"
    return f"  {spec.name:<10} {spec.host:<15} {label} {detail}"


def print_table(specs: List[GatewaySpec], statuses: Dict[str, Optional[Dict[str, Any]]],
                use_color: bool) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] fleet status")
    for spec in specs:
        print(render_row(spec, statuses.get(spec.name), use_color))
    sys.stdout.flush()


def print_alert(name: str, prev: str, cur: str, use_color: bool) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    if cur == LINK_UP and prev in (LINK_DOWN, UNREACHABLE, LINK_INIT):
        msg, color = f"{name}: TELLO LINK RESTORED (was {prev})", Color.GREEN
    elif cur == LINK_DOWN:
        msg, color = f"{name}: TELLO LINK LOST", Color.RED
    elif cur == UNREACHABLE:
        msg, color = f"{name}: GATEWAY UNREACHABLE", Color.MAGENTA
    else:
        msg, color = f"{name}: {prev} -> {cur}", Color.YELLOW
    line = _colorize(f"*** [{stamp}] {msg} ***", color, use_color)
    print(line)
    sys.stdout.flush()


def main() -> int:
    p = argparse.ArgumentParser(description="Live monitor for Raspberry Pi Tello gateways")
    p.add_argument("--config", required=True, help="JSON config with gateways / pi_hosts")
    p.add_argument("--interval", type=float, default=1.0, help="poll interval in seconds")
    p.add_argument("--timeout", type=float, default=1.0, help="per-gateway TCP timeout in seconds")
    p.add_argument("--heartbeat", type=float, default=15.0,
                   help="reprint the full table at least this often even without changes")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = p.parse_args()

    use_color = (not args.no_color) and sys.stdout.isatty()
    if use_color:
        _enable_windows_ansi()

    specs = load_specs(args.config)
    prev_link: Dict[str, str] = {spec.name: LINK_INIT for spec in specs}

    print(f"Monitoring {len(specs)} gateway(s), interval={args.interval}s. Ctrl+C to stop.")
    print_table(specs, {s.name: None for s in specs}, use_color)
    last_table = time.time()

    try:
        while True:
            statuses: Dict[str, Optional[Dict[str, Any]]] = {}
            changed = False
            for spec in specs:
                status = query_status(spec, args.timeout)
                statuses[spec.name] = status
                cur = link_of(status)
                if cur != prev_link[spec.name]:
                    print_alert(spec.name, prev_link[spec.name], cur, use_color)
                    prev_link[spec.name] = cur
                    changed = True

            now = time.time()
            if changed or (now - last_table) >= args.heartbeat:
                print_table(specs, statuses, use_color)
                last_table = now

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
