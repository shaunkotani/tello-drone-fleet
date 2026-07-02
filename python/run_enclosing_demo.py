#!/usr/bin/env python3
"""3 台の Tello による対象物の協調取り囲みデモ (原稿 10.7 段階4)。

ArUco 外部計測 (run_aruco_tracker.py -> UdpJsonTracker) で得た全機と対象物の位置を
用いて閉ループでスロット追従し、対象物の周りに半径 R の正三角形フォーメーションを
作る。低レベル姿勢は使わず、Raspberry Pi ゲートウェイ経由で rc a b c d だけを送る。

制御は式(30)/(53) の PD スロット追従 (静止対象物 = Case 1):
    u_i = Kp (r_slot_i - r_i)          (世界水平速度)
に高度保持と対象物注視 yaw を加え、RealSafetyShield を通して rc へ変換する。

構成 (原稿 10.3 の分担):
    [天井カメラ]->run_aruco_tracker.py --UDP--> UdpJsonTracker --> 本デモ(制御)
    本デモ --rc--> PiTelloGroup --> 各 Raspberry Pi --> 各 Tello

前提:
    1. run_aruco_tracker.py が起動し、UdpJsonTracker が受信できている
    2. 各 Raspberry Pi ゲートウェイが起動し、各 Tello に接続済み
    3. 対象物マーカ (config ids.target) が床/台車に貼られ計測できている
       (固定点を囲むだけなら --center X Y で対象物マーカ不要)

例:
    # 原点(0,0)を囲む (対象物マーカ不要・最も安全な初回)
    python run_enclosing_demo.py --config configs/pc_real_config.example.json \
        --center 0 0 --takeoff --duration 30
    # ArUco 対象物マーカを囲む
    python run_enclosing_demo.py --config configs/pc_real_config.example.json \
        --takeoff --duration 30
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tello_rl.config import EnvConfig
from tello_rl import formation as fm
from tello_rl.real import (
    PiTelloGroup, UdpJsonTracker, TrackerTimeout,
    RealSafetyShield, RealSafetyConfig,
)


_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n <= 1e-9:
        return v
    return v * (max_norm / n)


def world_vel_to_action(v_world: np.ndarray, yaw: float,
                        v_max_fb: float, v_max_lr: float) -> np.ndarray:
    """世界水平速度 -> 正規化行動 [lr, fb, 0, 0] (式(31) の逆回転)。"""
    c, s = math.cos(yaw), math.sin(yaw)
    fb = c * v_world[0] + s * v_world[1]
    lr = -s * v_world[0] + c * v_world[1]
    return np.array([lr / max(v_max_lr, 1e-6), fb / max(v_max_fb, 1e-6), 0.0, 0.0], dtype=float)


def enclosing_actions(states, center: np.ndarray, slots: np.ndarray, cfg: EnvConfig,
                      kp_xy: float, kp_h: float, kp_yaw: float,
                      v_xy_limit: float, ud_limit: float, yaw_limit: float) -> np.ndarray:
    """PD スロット追従 + 高度保持 + 対象物注視 yaw。raw action (N,4) を返す。"""
    N = len(states)
    raw = np.zeros((N, 4))
    for i, st in enumerate(states):
        # 水平: スロットへ寄せる (式(30), 静止対象物なので vo=0)
        v_world = clip_norm(kp_xy * (slots[i] - st.r), v_xy_limit)
        a = world_vel_to_action(v_world, st.psi, cfg.v_max_fb, cfg.v_max_lr)
        # 高度保持
        a[2] = float(np.clip(kp_h * (cfg.h_ref - st.h), -ud_limit, ud_limit))
        # 対象物を正面に (式(90) 下)
        yaw_ref = math.atan2(center[1] - st.r[1], center[0] - st.r[0])
        a[3] = float(np.clip(kp_yaw * wrap_pi(yaw_ref - st.psi), -yaw_limit, yaw_limit))
        raw[i] = np.clip(a, -1.0, 1.0)
    return raw


def enclosing_metrics(states, center: np.ndarray, slots: np.ndarray,
                      edge_ref: np.ndarray, cfg: EnvConfig) -> Dict[str, float]:
    """取り囲み誤差 (式(133)-(135)) と安全指標を計算する。"""
    r = np.stack([s.r for s in states])
    N = len(states)
    e_slot = float(np.mean([np.linalg.norm(r[i] - slots[i]) for i in range(N)]))
    e_R = float(np.mean([abs(np.linalg.norm(r[i] - center) - cfg.R) for i in range(N)]))
    e_edge = float(np.mean([
        abs(np.linalg.norm(r[i] - r[(i + 1) % N]) - edge_ref[i]) for i in range(N)
    ]))
    dmin = min(np.linalg.norm(r[i] - r[j]) for i in range(N) for j in range(i + 1, N))
    h = [s.h for s in states]
    return {"E_slot": e_slot, "E_R": e_R, "E_edge": e_edge,
            "d_min": float(dmin), "h_min": float(min(h)), "h_max": float(max(h))}


def build_env_config(cfg_json: Dict[str, Any], args) -> EnvConfig:
    cfg = EnvConfig(**cfg_json.get("env", {}))
    cfg.case = 1                      # デモは静止対象物の正多角形取り囲み
    cfg.target_mode = "static"
    if args.N is not None:
        cfg.N = args.N
    if args.R is not None:
        cfg.R = args.R
    if args.h_ref is not None:
        cfg.h_ref = args.h_ref
    return cfg


def battery_check(group: PiTelloGroup, min_pct: float) -> bool:
    """全機 battery? を問い合わせ、閾値未満があれば False。"""
    ok = True
    for c in group.clients:
        try:
            rep = c.query("battery?")
            resp = str(rep.raw.get("resp", rep.raw.get("cmd", ""))).strip()
            pct = float("".join(ch for ch in resp if ch.isdigit()) or "0")
            mark = "OK" if pct >= min_pct else "LOW"
            print(f"  {c.name}: battery={pct:.0f}% [{mark}]")
            ok &= pct >= min_pct
        except Exception as exc:
            print(f"  {c.name}: battery query failed: {exc}")
            ok = False
    return ok


def confirm(args, message: str) -> None:
    if args.yes:
        return
    print("\n=== 安全確認 ===")
    print(message)
    if input("続行するには YES と入力: ").strip() != "YES":
        raise SystemExit("ユーザーにより中断")


def parallel(group: PiTelloGroup, fn: Callable[[Any], Any], long_timeout_s: float,
             restore_timeout_s: float) -> List[Tuple[str, Optional[str]]]:
    """takeoff/land を全機ほぼ同時に実行する。

    takeoff/land はドローンが動作完了まで応答を返さないため、実行中だけ
    タイムアウトを ``long_timeout_s`` へ延ばし、終了後に制御用の
    ``restore_timeout_s`` へ戻す。順次だと待ち時間で自動着陸しうるため並列。
    """
    group.set_timeout(long_timeout_s)
    errs: List[Tuple[str, Optional[str]]] = []
    try:
        with ThreadPoolExecutor(max_workers=len(group.clients)) as ex:
            futs = {ex.submit(fn, c): c for c in group.clients}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    fut.result()
                    errs.append((c.name, None))
                except Exception as exc:  # noqa: BLE001 - report per-drone
                    errs.append((c.name, str(exc)))
    finally:
        group.set_timeout(restore_timeout_s)
    errs.sort(key=lambda x: x[0])
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description="3-Tello cooperative enclosing demo (ArUco closed loop)")
    p.add_argument("--config", required=True, help="PC-side JSON config (pi_hosts/tracker/env)")
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="囲む固定点[m]。省略時は ArUco 対象物マーカを囲む")
    p.add_argument("--takeoff", action="store_true", help="制御前に全機 takeoff")
    p.add_argument("--no-land", action="store_true", help="終了時に自動着陸しない")
    p.add_argument("--duration", type=float, default=30.0, help="取り囲み制御の最大時間[s]")
    p.add_argument("--settle", type=float, default=6.0, help="takeoff 後の整定待ち[s]")
    p.add_argument("--motion-timeout", type=float, default=12.0,
                   help="takeoff/land 応答待ちのタイムアウト[s] (離着陸は数秒かかる)")
    p.add_argument("--airborne-h", type=float, default=0.30,
                   help="離陸成功とみなす高度[m] (ArUco計測で確認)")
    p.add_argument("--airborne-delta", type=float, default=0.25,
                   help="離陸前高度からの上昇量[m] (これ以上で離陸成功とみなす)")
    p.add_argument("--rate", type=float, default=10.0, help="制御レート[Hz] (10-20 推奨)")
    p.add_argument("--rc-limit", type=int, default=None, help="rc 絶対値上限 (config 既定)")
    p.add_argument("--min-battery", type=float, default=20.0, help="離陸に必要な最低電池[%]")
    p.add_argument("--kp-xy", type=float, default=0.6)
    p.add_argument("--kp-h", type=float, default=0.8)
    p.add_argument("--kp-yaw", type=float, default=0.7)
    p.add_argument("--v-xy-limit", type=float, default=0.20, help="水平速度上限[m/s]")
    p.add_argument("--ud-limit", type=float, default=0.20)
    p.add_argument("--yaw-limit", type=float, default=0.25)
    p.add_argument("--d-drone-stop", type=float, default=None,
                   help="機体間の停止距離[m] (既定0.45)。下げると近づけるが衝突リスク増")
    p.add_argument("--d-drone-warn", type=float, default=None,
                   help="機体間の減衰開始距離[m] (既定0.60)。d-drone-stop 以上にすること")
    p.add_argument("--log", default="runs/enclosing_demo.csv")
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--R", type=float, default=None)
    p.add_argument("--h-ref", type=float, default=None)
    p.add_argument("--yes", action="store_true", help="対話式の安全確認をスキップ")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    cfg_json = load_json(args.config)
    cfg = build_env_config(cfg_json, args)

    # --- ゲートウェイ / トラッカ / 安全シールド ---
    pi_hosts = cfg_json["pi_hosts"]
    if len(pi_hosts) != cfg.N:
        print(f"[error] pi_hosts={len(pi_hosts)} but N={cfg.N}", file=sys.stderr)
        return 2
    group = PiTelloGroup.from_hosts(
        pi_hosts, port=int(cfg_json.get("pi_port", 10000)),
        timeout_s=float(cfg_json.get("pi_timeout_s", 1.0)))
    tcfg = cfg_json.get("tracker", {})
    tracker = UdpJsonTracker(n_drones=cfg.N, host=tcfg.get("host", "0.0.0.0"),
                             port=int(tcfg.get("port", 15000)),
                             max_age_s=float(tcfg.get("max_age_s", 0.30)))
    rc_limit = int(args.rc_limit if args.rc_limit is not None
                   else cfg_json.get("rc_limit", 30))
    scfg = RealSafetyConfig(rc_limit=rc_limit)
    if args.d_drone_stop is not None:
        scfg.d_drone_stop = args.d_drone_stop
    if args.d_drone_warn is not None:
        scfg.d_drone_warn = args.d_drone_warn
    if scfg.d_drone_warn < scfg.d_drone_stop:
        print(f"[warn] d_drone_warn({scfg.d_drone_warn}) < d_drone_stop({scfg.d_drone_stop}); "
              f"警告帯が無効です", file=sys.stderr)
    shield = RealSafetyShield(cfg, scfg)
    control_timeout = float(cfg_json.get("pi_timeout_s", 1.0))  # 制御ループ用の短いタイムアウト

    dt = 1.0 / max(1e-3, args.rate)
    out_path = Path(args.log)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"enclosing demo | N={cfg.N} R={cfg.R}m h_ref={cfg.h_ref}m rc_limit={rc_limit} "
          f"rate={args.rate:.0f}Hz center={'ArUco target' if args.center is None else args.center}")

    landed = False
    log_fp = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_fp)
    writer.writerow(["t", "k", "phase", "E_slot", "E_R", "E_edge", "d_min", "h_min", "h_max",
                     "center_x", "center_y", "reasons"])
    phi = 0.0
    try:
        # --- 接続 ---
        group.connect_all()
        tracker.start()
        print("接続完了。SDK mode 初期化...")
        group.broadcast({"type": "command"})

        # --- トラッカ確認 ---
        try:
            states, target, _ = tracker.get_states(timeout_s=2.0)
            print(f"トラッカ受信 OK: {len(states)} 機, target r={np.round(target.r, 3)}")
        except TrackerTimeout as exc:
            print(f"[error] トラッカ受信なし: {exc}\n  run_aruco_tracker.py が起動し UDP 送信中か確認",
                  file=sys.stderr)
            return 2

        # --- バッテリ確認 ---
        print("バッテリ確認:")
        if not battery_check(group, args.min_battery) and not args.yes:
            confirm(args, f"電池不足の機体があります (最低 {args.min_battery:.0f}% 推奨)。")

        # --- 離陸 ---
        if args.takeoff:
            confirm(args, f"{cfg.N} 台を離陸させ、目標点の周囲 R={cfg.R}m に取り囲みます。\n"
                          f"周囲の安全を確認してください。")
            # 離陸前の高度を記録 (上昇量の判定に使う)
            h0 = {i: 0.0 for i in range(cfg.N)}
            try:
                s0, _, _ = tracker.get_states(timeout_s=1.0)
                h0 = {i: s0[i].h for i in range(cfg.N)}
            except TrackerTimeout:
                pass

            print("takeoff (並列)...")
            errs = parallel(group, lambda c: c.takeoff(),
                            long_timeout_s=args.motion_timeout, restore_timeout_s=control_timeout)
            for n, e in errs:
                if e:
                    # ok パケットは落ちやすい。応答未確認でも高度で判定する。
                    print(f"[warn] {n}: takeoff 応答未確認 ({e}); 高度で離陸を確認します")

            # 離陸確認 + 整定を 1 ループで。ゲートウェイの watchdog(1.5s無rcで着陸)を
            # 避けるため、高度保持 rc を送り続けながら全機の上昇を確認する。
            print(f"離陸確認+整定 (高度>={args.airborne_h}m または +{args.airborne_delta}m を確認)...")
            airborne: set = set()
            t_s = time.time()
            deadline = t_s + max(args.motion_timeout, args.settle + 2.0)
            verified = False
            while time.time() < deadline and not _stop:
                try:
                    states, _, _ = tracker.get_states(timeout_s=0.5)
                except TrackerTimeout:
                    group.hover_all()  # keep-alive
                    time.sleep(dt)
                    continue
                raw = np.zeros((cfg.N, 4))
                for i, st in enumerate(states):
                    raw[i, 2] = float(np.clip(args.kp_h * (cfg.h_ref - st.h),
                                              -args.ud_limit, args.ud_limit))
                    if st.h >= args.airborne_h or (st.h - h0.get(i, 0.0)) >= args.airborne_delta:
                        airborne.add(i)
                res = shield.filter(raw, states, tracker_age_s=None)
                group.send_actions(res.actions, rc_limit=rc_limit)
                if len(airborne) == cfg.N and (time.time() - t_s) >= args.settle:
                    verified = True
                    break
                time.sleep(dt)

            not_air = [i for i in range(cfg.N) if i not in airborne]
            if not_air:
                print(f"[error] 離陸を高度で確認できない機体 {not_air} "
                      f"(高度が上がっていない) -> land", file=sys.stderr)
                raise RuntimeError("takeoff not verified by altitude")
            print(f"全機の離陸を高度で確認 (OK){' 整定完了' if verified else ''}")

        # --- 取り囲み制御ループ ---
        print("取り囲み制御開始 (Ctrl+C で安全停止)")
        vo = np.zeros(2)  # 静止対象物
        t0 = time.time()
        k = 0
        last_print = 0.0
        consecutive_hard = 0
        while not _stop and (time.time() - t0) < args.duration:
            try:
                states, target, battery = tracker.get_states(timeout_s=2.0 * cfg.dt)
                age = None
            except TrackerTimeout:
                # 計測喪失: ホバリング後に着陸へ
                print("[safety] トラッカ喪失 -> hover -> land")
                group.hover_all()
                break

            center = np.asarray(args.center, dtype=float) if args.center is not None else target.r
            phi = fm.update_heading(phi, vo, cfg)              # 静止なので phi 据え置き
            slots, _ = fm.compute_slots(center, vo, phi, cfg)  # 式(44)
            edge_ref = fm.edge_ref_lengths(slots)

            raw = enclosing_actions(states, center, slots, cfg, args.kp_xy, args.kp_h,
                                    args.kp_yaw, args.v_xy_limit, args.ud_limit, args.yaw_limit)
            res = shield.filter(raw, states, target_r=center, tracker_age_s=age)
            group.send_actions(res.actions, rc_limit=rc_limit)

            m = enclosing_metrics(states, center, slots, edge_ref, cfg)

            # 幾何的な非常停止 (シールドの上のさらなる保険)
            if m["d_min"] < shield.scfg.d_drone_stop * 0.7 or m["h_max"] > cfg.z_max:
                consecutive_hard += 1
            else:
                consecutive_hard = 0
            if consecutive_hard >= 5:
                print(f"[safety] 危険継続 (d_min={m['d_min']:.2f}) -> land")
                break

            writer.writerow([f"{time.time():.3f}", k, "enclose",
                             f"{m['E_slot']:.3f}", f"{m['E_R']:.3f}", f"{m['E_edge']:.3f}",
                             f"{m['d_min']:.3f}", f"{m['h_min']:.3f}", f"{m['h_max']:.3f}",
                             f"{center[0]:.3f}", f"{center[1]:.3f}", "|".join(res.reasons)])
            log_fp.flush()

            now = time.time()
            if now - last_print >= 0.5:
                print(f"t={now - t0:5.1f}s k={k:04d} | E_slot={m['E_slot']:.3f} "
                      f"E_R={m['E_R']:.3f} E_edge={m['E_edge']:.3f} | d_min={m['d_min']:.2f} "
                      f"h=[{m['h_min']:.2f},{m['h_max']:.2f}] | {res.reasons}")
                last_print = now
            k += 1
            time.sleep(dt)

        print("制御ループ終了")
    except KeyboardInterrupt:
        print("割り込み")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
    finally:
        # ホバリング -> 着陸
        try:
            group.hover_all()
            time.sleep(0.3)
        except Exception:
            pass
        if not args.no_land:
            try:
                print("land all (並列・完了待ち)")
                errs = parallel(group, lambda c: c.land(),
                                long_timeout_s=args.motion_timeout, restore_timeout_s=control_timeout)
                failed = [(n, e) for n, e in errs if e]
                if failed:
                    print(f"[error] land failed: {failed}; sending emergency", file=sys.stderr)
                    group.emergency_all()
                else:
                    landed = True
            except Exception as exc:
                print(f"[error] land failed: {exc}; sending emergency", file=sys.stderr)
                group.emergency_all()
        try:
            group.close_all()
        finally:
            tracker.stop()
        log_fp.close()
        print(f"log: {out_path} landed={landed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
