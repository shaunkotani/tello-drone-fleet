"""`fleet` command-line interface: quick fleet health from the desktop PC.

    python -m tello_fleet status --config configs/pc_real_config.example.json
    python -m tello_fleet doctor --config configs/pc_real_config.example.json

``status`` prints a one-shot table (UP / DOWN / UNREACHABLE + battery).
``doctor`` runs pass/fail pre-flight checks and exits non-zero if any drone fails.
For a continuously refreshing view use ``python pi_monitor.py`` instead.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import sys

from .fleet import Fleet


class _Color:
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"


def _enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def _paint(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{_Color.RESET}" if use_color else text


def _link_color(link: str) -> str:
    return {
        "up": _Color.GREEN,
        "down": _Color.RED,
        "unreachable": _Color.MAGENTA,
    }.get(link, _Color.YELLOW)


def _cmd_status(fleet: Fleet, use_color: bool) -> int:
    statuses = fleet.status_all()
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] fleet status")
    for c in fleet.clients:
        st = statuses.get(c.name)
        if st is None:
            link = "unreachable"
            detail = "gateway not reachable"
        else:
            link = str(st.get("tello_link", "init"))
            batt = st.get("battery") or "?"
            age = st.get("last_seen_age_s")
            sdk = "sdk" if st.get("sdk_mode") else "no-sdk"
            age_str = f"{age}s" if age is not None else "-"
            detail = f"batt={batt:>3}%  age={age_str:<6} {sdk} up={st.get('uptime_s')}s"
        label = _paint(f"{link.upper():<11}", _link_color(link), use_color)
        print(f"  {c.name:<10} {c.host:<15} {label} {detail}")
    return 0


def _cmd_doctor(fleet: Fleet, use_color: bool) -> int:
    report = fleet.doctor()
    all_ok = True
    print("fleet doctor")
    for entry in report:
        ok = entry.get("ok", False)
        all_ok &= ok
        tag = _paint("PASS", _Color.GREEN, use_color) if ok else _paint("FAIL", _Color.RED, use_color)
        issues = "; ".join(entry.get("issues", [])) or "ready"
        extra = ""
        if entry.get("reachable"):
            extra = (f"link={entry.get('tello_link')} sdk={entry.get('sdk_mode')} "
                     f"batt={entry.get('battery') or '?'}%")
        print(f"  [{tag}] {entry['name']:<10} {entry['host']:<15} {extra:<28} {issues}")
    print("all ready." if all_ok else "one or more drones NOT ready.")
    return 0 if all_ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fleet", description="Raspberry Pi Tello gateway fleet CLI")
    p.add_argument("command", choices=["status", "doctor"], help="what to run")
    p.add_argument("--config", required=True, help="JSON config with gateways / pi_hosts")
    p.add_argument("--timeout", type=float, default=1.0, help="per-gateway TCP timeout in seconds")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = p.parse_args(argv)

    use_color = (not args.no_color) and sys.stdout.isatty()
    if use_color:
        _enable_windows_ansi()

    fleet = Fleet.from_config(args.config, timeout_s=args.timeout)
    if args.command == "status":
        return _cmd_status(fleet, use_color)
    return _cmd_doctor(fleet, use_color)


if __name__ == "__main__":
    raise SystemExit(main())
