#!/usr/bin/env python3
"""Run a conservative real-flight baseline through Raspberry Pi gateways.

This is a first real-machine test program.  It does not train.  It reads an
external tracker stream, computes PD slot tracking actions, passes them through
RealSafetyShield, and sends them to the Pi gateways.

Example:
    python run_real_baseline.py --config configs/pc_real_config.example.json --takeoff
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from tello_rl.config import EnvConfig
from tello_rl.real import PiTelloGroup, UdpJsonTracker, RealTelloEnv


_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n <= 1e-9:
        return v
    return v * (max_norm / n)


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def world_vel_to_action(v_world: np.ndarray, yaw: float, v_max_fb: float, v_max_lr: float) -> np.ndarray:
    """World horizontal velocity -> normalized action [lr, fb, 0, 0]."""
    c, s = math.cos(yaw), math.sin(yaw)
    fb = c * v_world[0] + s * v_world[1]
    lr = -s * v_world[0] + c * v_world[1]
    return np.array([lr / max(v_max_lr, 1e-6), fb / max(v_max_fb, 1e-6), 0.0, 0.0], dtype=float)


def pd_slot_controller(env: RealTelloEnv, kp_xy: float, kp_h: float, kp_yaw: float,
                       v_xy_limit: float, ud_limit: float, yaw_limit: float) -> np.ndarray:
    """Conservative PD-like slot tracker for the real environment."""
    cfg = env.cfg
    actions = np.zeros((env.N, 4), dtype=float)
    for i, st in enumerate(env.states):
        e = env.slots[i] - st.r
        v_world = clip_norm(kp_xy * e, v_xy_limit)
        a = world_vel_to_action(v_world, st.psi, cfg.v_max_fb, cfg.v_max_lr)

        # Altitude hold.
        a[2] = float(np.clip(kp_h * (cfg.h_ref - st.h), -ud_limit, ud_limit))

        # Yaw: look at target.
        d = env.ro - st.r
        yaw_ref = math.atan2(d[1], d[0])
        a[3] = float(np.clip(kp_yaw * wrap_pi(yaw_ref - st.psi), -yaw_limit, yaw_limit))
        actions[i] = np.clip(a, -1.0, 1.0)
    return actions


def write_log_header(writer: csv.writer, n: int) -> None:
    header = ["t", "k", "done", "reason"]
    for i in range(n):
        header += [
            f"r{i}_x", f"r{i}_y", f"h{i}", f"yaw{i}",
            f"slot{i}_x", f"slot{i}_y",
            f"a{i}_lr", f"a{i}_fb", f"a{i}_ud", f"a{i}_yaw",
            f"obj{i}", f"c{i}_1", f"c{i}_2", f"c{i}_3", f"c{i}_4", f"c{i}_5",
            f"safety{i}",
        ]
    header += ["ro_x", "ro_y", "vo_x", "vo_y"]
    writer.writerow(header)


def write_log_row(writer: csv.writer, env: RealTelloEnv, actions: np.ndarray,
                  obj: np.ndarray, con: np.ndarray, done: bool, info: Dict[str, Any]) -> None:
    row = [time.time(), env.k, int(done), info.get("reason", "")]
    reasons = info.get("safety_reasons", ["" for _ in range(env.N)])
    for i, st in enumerate(env.states):
        row += [
            float(st.r[0]), float(st.r[1]), float(st.h), float(st.psi),
            float(env.slots[i, 0]), float(env.slots[i, 1]),
            float(actions[i, 0]), float(actions[i, 1]), float(actions[i, 2]), float(actions[i, 3]),
            float(obj[i]),
            float(con[i, 0]), float(con[i, 1]), float(con[i, 2]), float(con[i, 3]), float(con[i, 4]),
            reasons[i] if i < len(reasons) else "",
        ]
    row += [float(env.ro[0]), float(env.ro[1]), float(env.vo[0]), float(env.vo[1])]
    writer.writerow(row)


def build_env_from_config(cfg_json: Dict[str, Any], cli_args) -> RealTelloEnv:
    env_kwargs = cfg_json.get("env", {})
    cfg = EnvConfig(**env_kwargs)
    if cli_args.N is not None:
        cfg.N = cli_args.N
    if cli_args.case is not None:
        cfg.case = cli_args.case
    if cli_args.H is not None:
        cfg.H = cli_args.H
    if cli_args.h_ref is not None:
        cfg.h_ref = cli_args.h_ref
    if cli_args.R is not None:
        cfg.R = cli_args.R

    pi_hosts = cfg_json["pi_hosts"]
    if len(pi_hosts) != cfg.N:
        raise ValueError(f"config has {len(pi_hosts)} pi_hosts but cfg.N={cfg.N}")
    pi_port = int(cfg_json.get("pi_port", 10000))
    pi_timeout = float(cfg_json.get("pi_timeout_s", 1.0))
    group = PiTelloGroup.from_hosts(pi_hosts, port=pi_port, timeout_s=pi_timeout)

    tracker_cfg = cfg_json.get("tracker", {})
    tracker = UdpJsonTracker(
        n_drones=cfg.N,
        host=tracker_cfg.get("host", "0.0.0.0"),
        port=int(tracker_cfg.get("port", 15000)),
        max_age_s=float(tracker_cfg.get("max_age_s", 0.30)),
    )
    rc_limit = int(cli_args.rc_limit if cli_args.rc_limit is not None else cfg_json.get("rc_limit", 30))
    return RealTelloEnv(cfg, group, tracker, rc_limit=rc_limit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config file for PC-side real experiment")
    parser.add_argument("--takeoff", action="store_true", help="send takeoff to all Tello drones before control loop")
    parser.add_argument("--no-land", action="store_true", help="do not automatically land at the end")
    parser.add_argument("--duration", type=float, default=20.0, help="maximum control duration [s]")
    parser.add_argument("--log", default="runs/real_baseline.csv")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--case", type=int, choices=[1, 2], default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--h-ref", type=float, default=None)
    parser.add_argument("--R", type=float, default=None)
    parser.add_argument("--rc-limit", type=int, default=None)
    parser.add_argument("--kp-xy", type=float, default=0.6)
    parser.add_argument("--kp-h", type=float, default=0.8)
    parser.add_argument("--kp-yaw", type=float, default=0.7)
    parser.add_argument("--v-xy-limit", type=float, default=0.20)
    parser.add_argument("--ud-limit", type=float, default=0.25)
    parser.add_argument("--yaw-limit", type=float, default=0.25)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg_json = load_json(args.config)
    env = build_env_from_config(cfg_json, args)

    out_path = Path(args.log)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        env.connect()
        obs = env.reset()
        print(f"connected; obs shape={obs.shape}; rc_limit={env.rc_limit}")
        if args.takeoff:
            print("takeoff all")
            env.takeoff()
            time.sleep(5.0)
            env.reset()

        t0 = time.time()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            write_log_header(writer, env.N)
            done = False
            info: Dict[str, Any] = {"reason": ""}
            while not done and not _stop_requested and time.time() - t0 < args.duration:
                raw = pd_slot_controller(env, args.kp_xy, args.kp_h, args.kp_yaw,
                                         args.v_xy_limit, args.ud_limit, args.yaw_limit)
                _, obj, con, exe, done, info = env.step(raw)
                write_log_row(writer, env, exe, obj, con, done, info)
                f.flush()
                if env.k % 10 == 0:
                    slot_err = np.mean([np.linalg.norm(env.states[i].r - env.slots[i]) for i in range(env.N)])
                    print(f"k={env.k:04d} slot_err={slot_err:.3f} reason={info.get('reason','')} safety={info.get('safety_reasons')}")

        print(f"finished: reason={info.get('reason','')} log={out_path}")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        try:
            env.hover()
            time.sleep(0.3)
        except Exception:
            pass
        if not args.no_land:
            try:
                print("land all")
                env.land()
            except Exception as exc:
                print(f"land failed: {exc}", file=sys.stderr)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
