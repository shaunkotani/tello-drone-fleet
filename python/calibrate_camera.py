#!/usr/bin/env python3
"""カメラ内部パラメータ校正 + ArUco マーカ/ボード生成。

ArUco の姿勢推定 (高度・水平位置) には正しいカメラ行列 K と歪み係数が必要なので、
外部計測トラッカ (run_aruco_tracker.py) の前にこのスクリプトで校正する。

ChArUco ボードを使った校正:
  1. ボード画像を生成して A4 などに等倍印刷 (拡大縮小なし):
       python calibrate_camera.py --make-board --out-image charuco_board.png
  2. 印刷したボードを色々な角度・位置で写しながら校正:
       python calibrate_camera.py --capture --out configs/camera_calib.npz
     スペースキーで 1 枚取り込み、15-25 枚集めたら 'c' で校正実行、'q' で中断。

ドローン/対象物用のマーカ画像生成:
       python calibrate_camera.py --make-markers --ids 0,1,2,5,10 --marker-px 600

注意: 印刷後に実際の一辺長 [m] を定規で測り、config の marker_length_m /
world_marker_length_m に反映すること。スケールがずれると高度・位置が比例してずれる。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import cv2.aruco as aruco
import numpy as np

from tello_rl.real.aruco_tracker import CameraIntrinsics


# ChArUco ボード既定パラメータ (印刷しやすい A4 想定)
DEF_SQUARES_X = 5
DEF_SQUARES_Y = 7
DEF_SQUARE_LEN = 0.035   # [m] チェッカー 1 マス
DEF_MARKER_LEN = 0.026   # [m] マス内 ArUco
DEF_DICT = "DICT_4X4_50"


def make_board(dictionary: str, squares_x: int, squares_y: int,
               square_len: float, marker_len: float):
    d = aruco.getPredefinedDictionary(getattr(aruco, dictionary))
    return aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, d), d


def cmd_make_board(args: argparse.Namespace) -> int:
    board, _ = make_board(args.dict, args.squares_x, args.squares_y,
                          args.square_len, args.marker_len)
    px_per_m = args.dpi / 0.0254
    w = int(round(args.squares_x * args.square_len * px_per_m))
    h = int(round(args.squares_y * args.square_len * px_per_m))
    img = board.generateImage((w, h), marginSize=int(round(px_per_m * 0.005)))
    out = args.out_image
    cv2.imwrite(out, img)
    print(f"saved board image: {out} ({w}x{h}px @ {args.dpi}dpi)")
    print(f"  squares={args.squares_x}x{args.squares_y} square_len={args.square_len}m "
          f"marker_len={args.marker_len}m dict={args.dict}")
    print("  -> 等倍 (100%) で印刷し、印刷後に実寸を測って校正パラメータと一致させること")
    return 0


def cmd_make_markers(args: argparse.Namespace) -> int:
    d = aruco.getPredefinedDictionary(getattr(aruco, args.dict))
    ids = [int(x) for x in args.ids.split(",") if x.strip() != ""]
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        img = aruco.generateImageMarker(d, mid, args.marker_px)
        # 白枠を付けて検出を安定化
        border = args.marker_px // 8
        canvas = np.full((args.marker_px + 2 * border, args.marker_px + 2 * border),
                         255, dtype=np.uint8)
        canvas[border:border + args.marker_px, border:border + args.marker_px] = img
        path = outdir / f"marker_{args.dict}_{mid}.png"
        cv2.imwrite(str(path), canvas)
        print(f"saved {path}")
    print(f"{len(ids)} markers ({args.dict}). 印刷後に実寸 [m] を測り config に反映すること。")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    board, dictionary = make_board(args.dict, args.squares_x, args.squares_y,
                                   args.square_len, args.marker_len)
    charuco_detector = aruco.CharucoDetector(board)

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[error] cannot open camera {args.camera}", file=sys.stderr)
        return 2

    all_obj: List[np.ndarray] = []
    all_img: List[np.ndarray] = []
    image_size = None
    print("SPACE=取り込み  c=校正実行  q=中断")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
            disp = frame.copy()
            n = 0 if ch_ids is None else len(ch_ids)
            if n > 0:
                aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids)
            cv2.putText(disp, f"captured={len(all_img)} corners={n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("calibrate", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("aborted")
                return 1
            if key == ord(" ") and n >= 6:
                obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
                if obj_pts is not None and len(obj_pts) >= 6:
                    all_obj.append(obj_pts)
                    all_img.append(img_pts)
                    print(f"  captured frame {len(all_img)} ({len(obj_pts)} points)")
            if key == ord("c"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(all_obj) < args.min_frames:
        print(f"[error] only {len(all_obj)} frames; need >= {args.min_frames}", file=sys.stderr)
        return 2

    print(f"calibrating on {len(all_obj)} frames...")
    rms, K, dist, _, _ = cv2.calibrateCamera(all_obj, all_img, image_size, None, None)
    print(f"  RMS reprojection error = {rms:.4f} px")
    print(f"  K =\n{K}")
    print(f"  dist = {dist.ravel()}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    CameraIntrinsics(K=np.asarray(K), dist=np.asarray(dist)).save(str(out))
    print(f"saved calibration: {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="camera calibration / ArUco generation")
    p.add_argument("--make-board", action="store_true", help="generate ChArUco board image")
    p.add_argument("--make-markers", action="store_true", help="generate ArUco marker images")
    p.add_argument("--capture", action="store_true", help="capture frames and calibrate")

    p.add_argument("--dict", default=DEF_DICT)
    p.add_argument("--squares-x", type=int, default=DEF_SQUARES_X)
    p.add_argument("--squares-y", type=int, default=DEF_SQUARES_Y)
    p.add_argument("--square-len", type=float, default=DEF_SQUARE_LEN, help="[m]")
    p.add_argument("--marker-len", type=float, default=DEF_MARKER_LEN, help="[m]")

    # make-board
    p.add_argument("--out-image", default="charuco_board.png")
    p.add_argument("--dpi", type=float, default=300.0)
    # make-markers
    p.add_argument("--ids", default="0,1,2,5,10")
    p.add_argument("--marker-px", type=int, default=600)
    p.add_argument("--out-dir", default="markers")
    # capture
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--out", default="configs/camera_calib.npz")

    args = p.parse_args()
    if args.make_board:
        return cmd_make_board(args)
    if args.make_markers:
        return cmd_make_markers(args)
    if args.capture:
        return cmd_capture(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
