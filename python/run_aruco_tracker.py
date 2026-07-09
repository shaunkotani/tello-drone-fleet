#!/usr/bin/env python3
"""ArUco 外部計測トラッカ (原稿 10.2-10.3 節)。

天井カメラから 3 台の Tello + 対象物の ArUco マーカを検出し、世界座標系 (ENU) の
位置・速度・yaw を推定して、``tello_rl.real.UdpJsonTracker`` が受信する UDP JSON を
毎フレーム送信する。

事前準備:
  1. python calibrate_camera.py --make-board         # 校正ボード生成 (印刷)
  2. python calibrate_camera.py --capture --out configs/camera_calib.npz
  3. 各 Tello と対象物に ArUco マーカを貼る (ID を config の ids と一致させる)
  4. 床に世界原点マーカ (world_marker_id) を置く

実行:
  python run_aruco_tracker.py --config configs/aruco_tracker_config.example.json

  # 設定なしで素早く試す (キャリブレーションファイルだけ指定):
  python run_aruco_tracker.py --calib configs/camera_calib.npz --camera 0 \
      --drone-ids 0,1,2 --target-id 5 --world-marker-id 10 --marker-len 0.10

操作: ウィンドウ上で 'q' または ESC で終了。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import cv2

from tello_rl.real.aruco_tracker import (
    ArucoTracker,
    ArucoTrackerConfig,
    CameraIntrinsics,
    UdpPublisher,
    draw_overlay,
)


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def build_config(args: argparse.Namespace) -> ArucoTrackerConfig:
    if args.config:
        cfg = ArucoTrackerConfig.from_json(args.config)
    else:
        cfg = ArucoTrackerConfig()
    # CLI 上書き
    if args.calib is not None:
        cfg.calibration = args.calib
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.dict is not None:
        cfg.dictionary = args.dict
    if args.marker_len is not None:
        cfg.marker_length_m = args.marker_len
    if args.drone_ids is not None:
        cfg.drone_ids = _parse_int_list(args.drone_ids)
    if args.target_id is not None:
        cfg.target_id = args.target_id
    if args.world_marker_id is not None:
        cfg.world_marker_id = args.world_marker_id
    if args.host is not None:
        cfg.out_host = args.host
    if args.port is not None:
        cfg.out_port = args.port
    if args.no_viz:
        cfg.viz = False
    return cfg


def open_camera(cfg: ArucoTrackerConfig) -> cv2.VideoCapture:
    # Windows では CAP_DSHOW が安定しやすい。
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {cfg.camera_index}")
    return cap


def main() -> int:
    p = argparse.ArgumentParser(description="ArUco external measurement tracker")
    p.add_argument(
        "--config", help="JSON config (configs/aruco_tracker_config.example.json)"
    )
    p.add_argument("--calib", help="camera calibration .npz/.json (overrides config)")
    p.add_argument("--camera", type=int, default=None, help="camera index")
    p.add_argument("--dict", default=None, help="ArUco dictionary, e.g. DICT_4X4_50")
    p.add_argument(
        "--marker-len", type=float, default=None, help="drone/target marker length [m]"
    )
    p.add_argument(
        "--drone-ids", default=None, help="comma list of drone ArUco IDs, e.g. 0,1,2"
    )
    p.add_argument("--target-id", type=int, default=None, help="target ArUco ID")
    p.add_argument(
        "--world-marker-id", type=int, default=None, help="floor origin marker ID"
    )
    p.add_argument("--host", default=None, help="UDP destination host")
    p.add_argument("--port", type=int, default=None, help="UDP destination port")
    p.add_argument("--no-viz", action="store_true", help="disable preview window")
    p.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="preview window width in px (capture resolution is unchanged)",
    )
    p.add_argument(
        "--print-every", type=int, default=30, help="console status every N frames"
    )
    args = p.parse_args()

    cfg = build_config(args)
    try:
        intr = CameraIntrinsics.load(cfg.calibration)
    except FileNotFoundError:
        print(
            f"[error] calibration not found: {cfg.calibration}\n"
            f"        run: python calibrate_camera.py --capture --out {cfg.calibration}",
            file=sys.stderr,
        )
        return 2

    publisher = UdpPublisher(cfg.out_host, cfg.out_port)
    cap = open_camera(cfg)

    # 校正解像度と実フレームサイズの照合。K は解像度に比例するため、ずれたまま
    # 使うと位置・高度が黙って比例して狂う (RMS には現れない)。
    ok0 = False
    for _ in range(50):
        ok0, frame0 = cap.read()
        if ok0:
            break
        time.sleep(0.05)
    if ok0:
        fh, fw = frame0.shape[:2]
        if intr.image_size is None:
            print(
                f"[warn] calibration file has no image_size (old format); cannot verify "
                f"it matches capture {fw}x{fh}. Re-save with: "
                f"python calibrate_camera.py --analyze --out {cfg.calibration}",
                file=sys.stderr,
            )
        elif tuple(intr.image_size) != (fw, fh):
            print(
                f"[warn] calibration resolution {intr.image_size[0]}x{intr.image_size[1]} "
                f"!= capture {fw}x{fh}; scaling K to match",
                file=sys.stderr,
            )
            intr = intr.matched_to(fw, fh)

    tracker = ArucoTracker(cfg, intr)

    if cfg.viz:
        # Resizable window sized to fit the screen, independent of capture
        # resolution (a 1920x1080 capture would otherwise overflow an FHD monitor).
        cv2.namedWindow("aruco_tracker", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        disp_w = min(args.display_width, cfg.width)
        disp_h = int(round(cfg.height * disp_w / cfg.width))
        cv2.resizeWindow("aruco_tracker", disp_w, disp_h)
        # ウィンドウ枠のアスペクト比を固定する (OpenCV は枠比率をロックできないため、
        # ループ内でサイズ変更を検知して補正する)。
        win_aspect = cfg.width / cfg.height
        last_win = (disp_w, disp_h)

    period = 1.0 / max(1e-3, cfg.rate_hz)
    print(
        f"tracker -> udp {cfg.out_host}:{cfg.out_port} @ {cfg.rate_hz:.0f} Hz | "
        f"dict={cfg.dictionary} drones={cfg.drone_ids} target={cfg.target_id} "
        f"world_marker={cfg.world_marker_id}"
    )
    if cfg.world_marker_id is None and not tracker.world.valid:
        print(
            "[warn] no world_marker_id and no stored extrinsics: world frame undefined; "
            "set extrinsics in config or use --world-marker-id",
            file=sys.stderr,
        )

    frame_count = 0
    sent_count = 0
    try:
        while True:
            t_loop = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[warn] camera read failed", file=sys.stderr)
                time.sleep(0.05)
                continue

            stamp = time.time()
            payload, poses = tracker.process_frame(frame, stamp)
            if payload is not None:
                publisher.send(payload)
                sent_count += 1

            frame_count += 1
            if args.print_every and frame_count % args.print_every == 0:
                # Raw detected IDs (any of the configured dictionary), to tell
                # "camera sees no markers" from "sees markers but wrong IDs".
                raw_ids = sorted(
                    tracker.detector.detect(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ).keys()
                )
                if payload is not None:
                    status = " ".join(
                        f"d{d['id']}{'' if d['seen'] else '!'}"
                        for d in payload["drones"]
                    )
                elif not tracker.world.valid:
                    status = f"WORLD-FRAME-NOT-SET (need world_marker id={cfg.world_marker_id} in view)"
                else:
                    status = "no-payload (not all drone markers seen)"
                print(
                    f"frame={frame_count} sent={sent_count} detected_ids={raw_ids} "
                    f"markers={len(poses)} {status}"
                )

            if cfg.viz:
                draw_overlay(frame, tracker, poses, payload)
                cv2.imshow("aruco_tracker", frame)
                # マウスでウィンドウを変形させても縦横比を維持する: 変化した辺から
                # もう一方の辺を計算し直して resize する。
                try:
                    _, _, w, h = cv2.getWindowImageRect("aruco_tracker")
                    if w > 0 and h > 0:
                        pw, ph = last_win
                        if w != pw:  # 横幅が変わった -> 高さを合わせる
                            h2 = max(1, int(round(w / win_aspect)))
                            if h2 != h:
                                cv2.resizeWindow("aruco_tracker", w, h2)
                            last_win = (w, h2)
                        elif h != ph:  # 高さが変わった -> 横幅を合わせる
                            w2 = max(1, int(round(h * win_aspect)))
                            if w2 != w:
                                cv2.resizeWindow("aruco_tracker", w2, h)
                            last_win = (w2, h)
                except cv2.error:
                    pass
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            # レート制御
            sleep = period - (time.time() - t_loop)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cap.release()
        publisher.close()
        if cfg.viz:
            cv2.destroyAllWindows()
    print(f"done: frames={frame_count} sent={sent_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
