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
    PiTelloGroup, UdpJsonTracker, TrackerTimeout, TrackerBounds,
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


def fov_box_at_height(bounds: TrackerBounds, h: float) -> Tuple[float, float, float, float]:
    """床の視野四隅とカメラ位置から、高度 h での視野内接矩形を返す。

    カメラ視野は錐台なので、高度 h の断面は床の四隅をカメラ直下点へ
    (Hc - h)/Hc 倍に相似縮小したもの。その内接軸平行矩形 (各軸の中央 2 値)
    を保守的な飛行可能域として返す。return (x_min, x_max, y_min, y_max)。
    """
    cam = np.asarray(bounds.camera, dtype=float)
    hc = float(cam[2])
    if hc <= h + 0.05:
        raise ValueError(f"カメラ高 {hc:.2f}m が飛行高度 {h:.2f}m 以下です (外部パラメータを確認)")
    k = (hc - h) / hc
    pts = cam[:2] + (np.asarray(bounds.corners, dtype=float) - cam[:2]) * k
    xs = np.sort(pts[:, 0])
    ys = np.sort(pts[:, 1])
    return float(xs[1]), float(xs[-2]), float(ys[1]), float(ys[-2])


def plan_formation_fit(cfg: EnvConfig, scfg: RealSafetyConfig,
                       center: np.ndarray) -> Tuple[float, float]:
    """安全ボックスに収まる最大半径 R_max と最適スロット位相を返す。

    スロットは center + R*u(a_k), a_k = phase + 2πk/N。全スロットが壁 warn
    マージンの内側に入る最大 R を、位相を振って最大化する (周期 2π/N)。
    center がマージン内側にない場合は (0, 0) を返す。
    """
    m = scfg.d_wall_warn
    lo = np.array([cfg.x_min + m, cfg.y_min + m])
    hi = np.array([cfg.x_max - m, cfg.y_max - m])
    c = np.asarray(center, dtype=float)
    if np.any(c <= lo) or np.any(c >= hi):
        return 0.0, 0.0
    best_r, best_phase = 0.0, 0.0
    for phase in np.linspace(0.0, 2.0 * math.pi / cfg.N, 91):
        r_phase = float("inf")
        for k in range(cfg.N):
            a = phase + 2.0 * math.pi * k / cfg.N
            u = (math.cos(a), math.sin(a))
            for d in range(2):
                if u[d] > 1e-9:
                    r_phase = min(r_phase, (hi[d] - c[d]) / u[d])
                elif u[d] < -1e-9:
                    r_phase = min(r_phase, (lo[d] - c[d]) / u[d])
        if r_phase > best_r:
            best_r, best_phase = r_phase, float(phase)
    return best_r, best_phase


def formation_r_min(cfg: EnvConfig, scfg: RealSafetyConfig,
                    spacing_margin: float = 0.10, target_margin: float = 0.10
                    ) -> Tuple[float, float, float]:
    """シールドと恒久的に干渉しない最小の取り囲み半径とその内訳。

    機体間隔 2R sin(π/N) が d_drone_warn を余裕をもって上回り、かつスロットが
    d_target_stop の外側にあること。return (r_min, r_spacing, r_target)。
    """
    r_spacing = (scfg.d_drone_warn + spacing_margin) / (2.0 * math.sin(math.pi / cfg.N))
    r_target = scfg.d_target_stop + target_margin
    return max(r_spacing, r_target), r_spacing, r_target


def assign_slots(r: np.ndarray, slots: np.ndarray,
                 center: Optional[np.ndarray] = None) -> List[int]:
    """中心周りの角度順序を保存する巡回割当を返す。

    perm[i] = 機体 i が向かうスロットの添字。機体を中心周りの角度で並べ、
    スロットも角度で並べて、N 通りの巡回シフトのうち総距離最小を選ぶ。
    順序が保存されるため各機は自分の角度セクター内を動くだけでよく、
    「2 機の間 (スロット間隔 < 2*d_drone_stop) をすり抜ける」不可能な経路や
    交差経路が発生しない。単純な最近傍割当だと、遠い側のスロットを引いた
    機体が他機の間を通れず詰む (シールドが正しく通さない) ことがある。
    """
    N = len(slots)
    c = np.mean(slots, axis=0) if center is None else np.asarray(center, dtype=float)
    order_d = sorted(range(N), key=lambda i: math.atan2(r[i][1] - c[1], r[i][0] - c[0]))
    order_s = sorted(range(N), key=lambda k: math.atan2(slots[k][1] - c[1], slots[k][0] - c[0]))
    best: List[int] = list(range(N))
    best_cost = float("inf")
    for shift in range(N):
        perm = [0] * N
        for pos in range(N):
            perm[order_d[pos]] = order_s[(pos + shift) % N]
        cost = sum(float(np.linalg.norm(r[i] - slots[perm[i]])) for i in range(N))
        if cost < best_cost:
            best, best_cost = perm, cost
    return best


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


def parallel_takeoff(group: PiTelloGroup, long_timeout_s: float, restore_timeout_s: float,
                     hover_period_s: float = 0.4) -> List[Tuple[str, Optional[str]]]:
    """takeoff を全機ほぼ同時に実行し、完了機には残りを待つ間 hover を送り続ける。

    ゲートウェイは takeoff 完了時点で rc watchdog が武装される。全機の応答が
    揃うのを待つだけだと、遅い機体のタイムアウト (最大 long_timeout_s) を待つ間に
    先に離陸した機体への rc が途絶え、watchdog (land-timeout 1.5s) に着陸させ
    られてしまう。そこで応答が返った機体から順に hover keep-alive を送る。
    """
    group.set_timeout(long_timeout_s)
    errs: List[Tuple[str, Optional[str]]] = []
    done_clients: List[Any] = []
    try:
        with ThreadPoolExecutor(max_workers=len(group.clients)) as ex:
            pending = {ex.submit(c.takeoff): c for c in group.clients}
            while pending:
                for fut in [f for f in pending if f.done()]:
                    c = pending.pop(fut)
                    try:
                        fut.result()
                        errs.append((c.name, None))
                    except Exception as exc:  # noqa: BLE001 - report per-drone
                        errs.append((c.name, str(exc)))
                    done_clients.append(c)
                if not pending:
                    break
                for c in done_clients:
                    try:
                        c.hover()
                    except Exception:
                        pass
                time.sleep(hover_period_s)
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
    p.add_argument("--motion-timeout", type=float, default=15.0,
                   help="takeoff/land 応答待ちのタイムアウト[s] "
                        "(ゲートウェイ内部の takeoff 待ちは 12s なので余裕を持たせる)")
    p.add_argument("--takeoff-retries", type=int, default=1,
                   help="高度が上がらない機体への takeoff 再送回数 (0 で無効)")
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
    p.add_argument("--d-target-stop", type=float, default=None,
                   help="対象物への停止距離[m] (既定0.30)。床マーカを囲むだけなら "
                        "0.15 程度まで下げられる (R_min を支配しがち)")
    p.add_argument("--wall-stop", type=float, default=None,
                   help="視野/壁端の押し戻しマージン[m] (FOV自動フィット時の既定 0.15)")
    p.add_argument("--wall-warn", type=float, default=None,
                   help="視野/壁端の減衰マージン[m] (FOV自動フィット時の既定 0.30)")
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
    if args.d_target_stop is not None:
        scfg.d_target_stop = args.d_target_stop
    if scfg.d_drone_warn < scfg.d_drone_stop:
        print(f"[warn] d_drone_warn({scfg.d_drone_warn}) < d_drone_stop({scfg.d_drone_stop}); "
              f"警告帯が無効です", file=sys.stderr)
    if scfg.d_drone_stop < 0.25:
        print(f"[warn] d_drone_stop={scfg.d_drone_stop}m は Tello の機体サイズ (~0.2m) と"
              f"ダウンウォッシュを考えると危険です。0.30m 以上を強く推奨", file=sys.stderr)
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

        # --- カメラ視野の自動フィット (四隅マーカ + カメラ姿勢による錐台補正) ---
        bounds = tracker.get_bounds(timeout_s=1.0)
        # シールドは h_min_real + h_warn_margin 未満への降下を止めるため、
        # 実飛行高度はこの下限を下回らない。FOV も実飛行高度で計算する。
        h_floor = scfg.h_min_real + scfg.h_warn_margin
        h_fit = max(cfg.h_ref, h_floor)
        if cfg.h_ref < h_floor:
            print(f"[warn] h_ref={cfg.h_ref}m はシールドの高度下限 {h_floor:.2f}m 未満です。"
                  f"実飛行高度は ~{h_floor:.2f}m になり、FOV 計算にも {h_fit:.2f}m を使います",
                  file=sys.stderr)
        if bounds is not None:
            cam = bounds.camera
            box = fov_box_at_height(bounds, h_fit)  # ValueError は外側で捕捉
            if box[0] >= box[1] or box[2] >= box[3]:
                print(f"[error] 四隅マーカから有効な視野矩形を作れません: {np.round(box, 2)}",
                      file=sys.stderr)
                return 2
            cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max = box
            # 視野端は物理壁ではない (越えてもマーカ喪失 -> 自動着陸) のでマージンは緩め
            scfg.d_wall_stop = 0.15 if args.wall_stop is None else args.wall_stop
            scfg.d_wall_warn = 0.30 if args.wall_warn is None else args.wall_warn
            print(f"FOV 自動フィット: カメラ ({cam[0]:+.2f},{cam[1]:+.2f}) 高さ {cam[2]:.2f}m, "
                  f"h={h_fit:.2f}m での視野 x=[{cfg.x_min:.2f},{cfg.x_max:.2f}] "
                  f"y=[{cfg.y_min:.2f},{cfg.y_max:.2f}]")
        else:
            if args.wall_stop is not None:
                scfg.d_wall_stop = args.wall_stop
            if args.wall_warn is not None:
                scfg.d_wall_warn = args.wall_warn
            print(f"[warn] トラッカに四隅 bounds なし (config の ids.corners 未設定?)。"
                  f"config の範囲を使用: x=[{cfg.x_min},{cfg.x_max}] y=[{cfg.y_min},{cfg.y_max}]",
                  file=sys.stderr)

        # --- 取り囲み計画: 実行可否判定と R / スロット位相の自動調整 ---
        plan_center = (np.asarray(args.center, dtype=float) if args.center is not None
                       else target.r.copy())
        r_max, phase = plan_formation_fit(cfg, scfg, plan_center)
        r_min, r_min_sp, r_min_tg = formation_r_min(cfg, scfg)
        if r_max < r_min:
            cx = 0.5 * (cfg.x_min + cfg.x_max)
            cy = 0.5 * (cfg.y_min + cfg.y_max)
            dominant = (f"機体間隔 (--d-drone-warn={scfg.d_drone_warn}) が支配的"
                        if r_min_sp >= r_min_tg else
                        f"対象物距離 (--d-target-stop={scfg.d_target_stop}) が支配的")
            print(f"[error] この空間では N={cfg.N} の取り囲みが成立しません: "
                  f"R_max={r_max:.2f}m < R_min={r_min:.2f}m\n"
                  f"  R_min の内訳: 機体間隔 {r_min_sp:.2f}m / 対象物距離 {r_min_tg:.2f}m -> {dominant}\n"
                  f"  対処: 支配的な方のマージンを下げる / --wall-warn を下げる /\n"
                  f"        h_ref を下げる (視野が広がる) / "
                  f"対象物を視野中心 ({cx:+.2f},{cy:+.2f}) 付近へ移動する", file=sys.stderr)
            return 2
        r_req = cfg.R
        cfg.R = float(min(max(cfg.R, r_min), r_max))
        if abs(cfg.R - r_req) > 1e-6:
            print(f"[warn] R={r_req}m を実行可能範囲 [{r_min:.2f}, {r_max:.2f}] に合わせて "
                  f"{cfg.R:.2f}m に調整しました")
        phi = phase - getattr(cfg, "alpha0", 0.0)
        spacing = 2.0 * cfg.R * math.sin(math.pi / cfg.N)
        print(f"取り囲み計画: R={cfg.R:.2f}m (可行域 [{r_min:.2f}, {r_max:.2f}]), "
              f"位相={math.degrees(phase):.0f}deg, 機体間={spacing:.2f}m, "
              f"中心=({plan_center[0]:+.2f},{plan_center[1]:+.2f})")

        # --- バッテリ確認 ---
        print("バッテリ確認:")
        if not battery_check(group, args.min_battery) and not args.yes:
            confirm(args, f"電池不足の機体があります (最低 {args.min_battery:.0f}% 推奨)。")

        # --- 離陸 ---
        if args.takeoff:
            confirm(args, f"{cfg.N} 台を離陸させ、目標点の周囲 R={cfg.R:.2f}m に取り囲みます。\n"
                          f"周囲の安全を確認してください。")
            # 離陸前の高度を記録 (上昇量の判定に使う)
            h0 = {i: 0.0 for i in range(cfg.N)}
            try:
                s0, _, _ = tracker.get_states(timeout_s=1.0)
                h0 = {i: s0[i].h for i in range(cfg.N)}
            except TrackerTimeout:
                pass

            print("takeoff (並列)...")
            errs = parallel_takeoff(group, long_timeout_s=args.motion_timeout,
                                    restore_timeout_s=control_timeout)
            for n, e in errs:
                if e:
                    # ok パケットは落ちやすい。応答未確認でも高度で判定する。
                    print(f"[warn] {n}: takeoff 応答未確認 ({e}); 高度で離陸を確認します")

            # 離陸確認 + 整定を 1 ループで。ゲートウェイの watchdog(1.5s無rcで着陸)を
            # 避けるため、高度保持 rc を送り続けながら全機の上昇を確認する。
            # 高度が上がらない機体には takeoff を再送する (コマンドの UDP ロス対策)。
            print(f"離陸確認+整定 (高度>={args.airborne_h}m または +{args.airborne_delta}m を確認)...")
            airborne: set = set()
            retry_futs: Dict[int, Any] = {}      # index -> in-flight takeoff future
            retries_left = {i: max(0, args.takeoff_retries) for i in range(cfg.N)}
            retry_grace_s = 3.0  # 応答ロストでも実際は上昇中、という機体を待つ猶予
            retry_ex = ThreadPoolExecutor(max_workers=cfg.N)

            def _retry_takeoff(c):
                # 再送中はこのクライアントのソケットを再送スレッドが専有する
                # (メインループは rc 送信をスキップする)
                c.set_timeout(args.motion_timeout)
                try:
                    return c.takeoff()
                finally:
                    c.set_timeout(control_timeout)

            def _collect_retries() -> None:
                for i, fut in list(retry_futs.items()):
                    if fut.done():
                        retry_futs.pop(i)
                        name = group.clients[i].name
                        try:
                            fut.result()
                            print(f"[info] {name}: takeoff 再送に応答 ok")
                        except Exception as exc:  # noqa: BLE001
                            print(f"[warn] {name}: takeoff 再送も応答未確認 ({exc}); "
                                  f"高度で確認を継続")

            def _hover_except_retrying() -> None:
                for i, c in enumerate(group.clients):
                    if i not in retry_futs:
                        try:
                            c.hover()
                        except Exception:
                            pass

            t_s = time.time()
            deadline = t_s + max(args.motion_timeout, args.settle + 2.0)
            verified = False
            try:
                while time.time() < deadline and not _stop:
                    _collect_retries()
                    try:
                        states, _, _ = tracker.get_states(timeout_s=0.5)
                    except TrackerTimeout:
                        _hover_except_retrying()  # keep-alive
                        time.sleep(dt)
                        continue
                    raw = np.zeros((cfg.N, 4))
                    for i, st in enumerate(states):
                        raw[i, 2] = float(np.clip(args.kp_h * (cfg.h_ref - st.h),
                                                  -args.ud_limit, args.ud_limit))
                        if st.h >= args.airborne_h or (st.h - h0.get(i, 0.0)) >= args.airborne_delta:
                            airborne.add(i)
                    # 高度が上がらない機体へ takeoff を再送
                    if (time.time() - t_s) >= retry_grace_s:
                        for i in range(cfg.N):
                            if i in airborne or i in retry_futs or retries_left[i] <= 0:
                                continue
                            retries_left[i] -= 1
                            print(f"[warn] {group.clients[i].name}: 高度が上がらないため "
                                  f"takeoff を再送します")
                            retry_futs[i] = retry_ex.submit(_retry_takeoff, group.clients[i])
                            deadline = max(deadline,
                                           time.time() + args.motion_timeout + args.settle)
                    res = shield.filter(raw, states, tracker_age_s=None)
                    for i, c in enumerate(group.clients):
                        if i not in retry_futs:  # 再送中の機体はソケット使用中なのでスキップ
                            c.rc(res.actions[i], rc_limit=rc_limit)
                    if len(airborne) == cfg.N and not retry_futs \
                            and (time.time() - t_s) >= args.settle:
                        verified = True
                        break
                    time.sleep(dt)
            finally:
                # 再送スレッドが残ったまま先へ進むと land と同一ソケットを共有して
                # しまうため、他機に hover を送りつつ完了を待つ
                while retry_futs:
                    _collect_retries()
                    _hover_except_retrying()
                    time.sleep(dt)
                retry_ex.shutdown(wait=False)

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
        slot_perm: Optional[List[int]] = None  # 初回に最近傍割当を決めて固定
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
            if slot_perm is None:
                slot_perm = assign_slots(np.stack([s.r for s in states]), slots, center)
                if slot_perm != list(range(cfg.N)):
                    print(f"slot assignment: drone i -> slot {slot_perm} (角度順序保存割当)")
            slots = slots[slot_perm]
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
