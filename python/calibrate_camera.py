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

精度が頭打ち (RMS が下がらない) のときの追い込み手順:
  1. 撮影は 1 回だけ (点は --points-out に自動保存される):
       python calibrate_camera.py --capture
     (既定は 1080p。広角レンズで床端の分解能が足りないときだけ
      --width 3840 --height 2160 で 4K 校正に上げる)
  2. 残差の「形」でボトルネックを切り分ける (再撮影不要):
       python calibrate_camera.py --analyze --points configs/calib_points.npz
     - 端ほど残差大 (radial ratio 高) => 歪みモデル不足 => 手順 3
     - 半径に依らずばらつく => 検出ノイズ/ボード反り/ブレ => Phase 1 (物理側)
  3. モデルを段階的に比較して限界効用が頭打ちのものを選ぶ:
       python calibrate_camera.py --analyze --compare
       python calibrate_camera.py --analyze --rational --out configs/camera_calib.npz
     (--rational / --thin-prism / --tilted は --capture にも付けられる)
  4. 外れフレームが報告されたら除外して再校正 (番号は診断出力の frame 番号):
       python calibrate_camera.py --analyze --drop 10,12
       python calibrate_camera.py --analyze --auto-drop --out configs/camera_calib.npz
  ※ --analyze は --out を明示したときだけ保存する (診断だけなら何も上書きしない)。

外部パラメータ (world<-camera) の計測・保存 (基準マーカが常時見えない運用向け):
       python calibrate_camera.py --extrinsics --world-marker-id 10 --world-marker-len 0.15
     床の基準マーカを複数フレーム撮って平均し、R_wc/t_wc を JSON に書き出す。
     --tracker-config を渡すとその設定 JSON の extrinsics 節を直接更新する。

注意1: 色収差 (chromatic aberration) 補正は calibrateCamera の RMS を下げない
  (グレースケール検出への当てはまり誤差なので単色幾何モデルの外)。まず上記
  Phase 1/2 を尽くすこと。
注意2: 印刷後に実際の一辺長 [m] を定規で測り、config の marker_length_m /
  world_marker_length_m に反映すること。スケールがずれると高度・位置が比例して
  ずれる (この誤差は RMS には一切現れない)。最終精度は pose 実測で確認する。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import cv2.aruco as aruco
import numpy as np

from tello_rl.real.aruco_tracker import (
    CameraIntrinsics,
    MarkerDetector,
    WorldFrame,
    estimate_pose_single,
)


# ChArUco ボード既定パラメータ (印刷しやすい A4 想定)
DEF_SQUARES_X = 5
DEF_SQUARES_Y = 7
DEF_SQUARE_LEN = 0.035   # [m] チェッカー 1 マス
DEF_MARKER_LEN = 0.026   # [m] マス内 ArUco
DEF_DICT = "DICT_4X4_50"

# --capture の校正保存先の既定。--analyze は --out 明示時のみ保存する (診断だけの
# つもりの実行で校正ファイルを上書きしないため、argparse の default にはしない)。
DEF_CALIB_OUT = "configs/camera_calib.npz"


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


def make_charuco_detector(board):
    """サブピクセル精緻化を有効にした ChArUco 検出器を作る。

    マーカ検出段のコーナー精緻化 (CORNER_REFINE_SUBPIX) を入れると、内部の
    チェッカー交点補間が安定し、コーナー局在ノイズが減って RMS が下がる。
    古い OpenCV 署名では charucoParams/detectorParams を受けないので握りつぶす。
    """
    det_params = aruco.DetectorParameters()
    det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    det_params.cornerRefinementWinSize = 5
    try:
        return aruco.CharucoDetector(
            board, charucoParams=aruco.CharucoParameters(), detectorParams=det_params)
    except TypeError:
        return aruco.CharucoDetector(board)


def build_calib_flags(args: argparse.Namespace) -> int:
    """--rational / --thin-prism / --tilted から calibrateCamera のフラグを組む。"""
    flags = 0
    if getattr(args, "rational", False):
        flags |= cv2.CALIB_RATIONAL_MODEL
    if getattr(args, "thin_prism", False):
        flags |= cv2.CALIB_THIN_PRISM_MODEL
    if getattr(args, "tilted", False):
        flags |= cv2.CALIB_TILTED_MODEL
    return flags


def _model_name(flags: int) -> str:
    parts = ["base(k1,k2,p1,p2,k3)"]
    if flags & cv2.CALIB_RATIONAL_MODEL:
        parts.append("rational(k4,k5,k6)")
    if flags & cv2.CALIB_THIN_PRISM_MODEL:
        parts.append("thin-prism(s1..s4)")
    if flags & cv2.CALIB_TILTED_MODEL:
        parts.append("tilted(taux,tauy)")
    return " + ".join(parts)


def residual_report(all_obj, all_img, K, dist, rvecs, tvecs, image_size,
                    n_bins: int = 6,
                    frame_ids: Optional[List[int]] = None) -> List[int]:
    """再投影残差を「フレーム毎」「半径依存」で表示し、A/B を切り分ける。

    - 半径 (画像中心=主点からの距離) が増えるほど残差が増える → 歪みモデル不足 (Phase 2)
    - 半径に依らずばらつく → 検出ノイズ/ボード反り/ブレ (Phase 1)
    - 特定フレームだけ突出 → 外れフレーム除去

    frame_ids は表示に使う元ファイル上のフレーム番号 (--drop 後も番号が安定する)。
    return: 外れフレーム番号のリスト (rms > mean+2std)。
    """
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = int(image_size[0]), int(image_size[1])
    max_r = math.hypot(max(cx, w - cx), max(cy, h - cy))

    per_frame = []          # (frame_id, rms_i, n_pts)
    radii: List[float] = []
    errs: List[float] = []
    px_x: List[float] = []  # 全フレームの画像点 (占有マップ用)
    px_y: List[float] = []
    for i, (obj, img) in enumerate(zip(all_obj, all_img)):
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        pts = np.asarray(img, dtype=np.float64).reshape(-1, 2)
        e = np.linalg.norm(proj - pts, axis=1)
        fid = frame_ids[i] if frame_ids is not None else i
        per_frame.append((fid, float(np.sqrt(np.mean(e ** 2))), len(e)))
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / max_r
        radii.extend(r.tolist())
        errs.extend(e.tolist())
        px_x.extend(pts[:, 0].tolist())
        px_y.extend(pts[:, 1].tolist())

    radii_a = np.asarray(radii)
    errs_a = np.asarray(errs)

    print("\n--- residual diagnostics ---")

    # (1) 2D 占有マップ: 画像をセルに割り、各セルに点が来たかを地図表示する。
    # 半径正規化は 16:9 の異方性で誤判定するため、実画素座標で 2 次元評価する。
    cols = 12
    rows = max(1, int(round(cols * h / w)))
    gx = np.clip((np.asarray(px_x) / w * cols).astype(int), 0, cols - 1)
    gy = np.clip((np.asarray(px_y) / h * rows).astype(int), 0, rows - 1)
    occ = np.zeros((rows, cols), dtype=int)
    for xx, yy in zip(gx, gy):
        occ[yy, xx] += 1
    filled = int((occ > 0).sum())
    total = rows * cols
    cov_pct = 100.0 * filled / total
    print(f"coverage map ({cols}x{rows} cells, '#'=撮れた '.'=空, {cov_pct:.0f}% filled):")
    for ry in range(rows):
        print("  " + "".join("#" if occ[ry, cx_] > 0 else "." for cx_ in range(cols)))

    # 端・四隅のセルが埋まっているかを個別に評価 (歪みが強く pose に効く領域)
    edge_cells = []
    for ry in range(rows):
        for cx_ in range(cols):
            if ry in (0, rows - 1) or cx_ in (0, cols - 1):
                edge_cells.append(occ[ry, cx_] > 0)
    edge_cov = 100.0 * sum(edge_cells) / len(edge_cells)
    corners = [occ[0, 0], occ[0, cols - 1], occ[rows - 1, 0], occ[rows - 1, cols - 1]]
    n_corner = sum(1 for c in corners if c > 0)
    coverage_ok = cov_pct >= 80.0 and edge_cov >= 70.0 and n_corner >= 3
    print(f"  edge cells filled = {edge_cov:.0f}%   corner cells = {n_corner}/4")
    if not coverage_ok:
        # 空いている辺/隅を具体的に案内
        empty_edges = []
        if occ[0, :].sum() < occ.sum() / rows * 0.3:
            empty_edges.append("上")
        if occ[rows - 1, :].sum() < occ.sum() / rows * 0.3:
            empty_edges.append("下")
        if occ[:, 0].sum() < occ.sum() / cols * 0.3:
            empty_edges.append("左")
        if occ[:, cols - 1].sum() < occ.sum() / cols * 0.3:
            empty_edges.append("右")
        hint = ("・".join(empty_edges) + "の端/隅") if empty_edges else "空きセル ('.')"
        print(f"  ! coverage 不足: {hint} に盤面を動かして撮り足す "
              f"(ChArUco は盤が画面外にはみ出て部分的でも検出できる)")

    # (2) 半径依存 (歪みモデル不足の切り分け)。coverage OK のときだけ結論を出す。
    print("\nradial dependence (0=center .. 1=corner):")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means = []
    for b in range(n_bins):
        m = (radii_a >= edges[b]) & (radii_a < edges[b + 1] if b < n_bins - 1
                                     else radii_a <= edges[b + 1])
        if m.any():
            mean_e = float(errs_a[m].mean())
            bin_means.append(mean_e)
            bar = "#" * int(round(mean_e / max(errs_a.mean(), 1e-9) * 20))
            print(f"  r[{edges[b]:.2f}-{edges[b+1]:.2f}]  mean={mean_e:5.2f}px  "
                  f"n={int(m.sum()):5d}  {bar}")
    if not coverage_ok:
        print("  -> coverage が不足しているため半径依存の判定は保留。"
              "まず端まで撮ってから再評価")
    elif len(bin_means) >= 2:
        ratio = bin_means[-1] / max(bin_means[0], 1e-9)
        if ratio > 1.8:
            print(f"  -> 端が中心の {ratio:.1f}x。歪みモデル不足の疑い "
                  f"=> Phase 2 (--rational から試す)")
        else:
            print(f"  -> 半径依存は弱い ({ratio:.1f}x)。検出ノイズ/ボード反り主体 "
                  f"=> Phase 1 (ボード固定・サブピクセル・ブレ対策)")

    # (3) フレーム毎 RMS と外れフレーム
    rms_vals = np.asarray([f[1] for f in per_frame])
    thr = float(rms_vals.mean() + 2.0 * rms_vals.std())
    print(f"\nper-frame RMS: mean={rms_vals.mean():.3f} std={rms_vals.std():.3f} "
          f"max={rms_vals.max():.3f} px  (outlier thr=mean+2std={thr:.3f})")
    worst = sorted(per_frame, key=lambda t: t[1], reverse=True)[:5]
    print("  worst frames:")
    for idx, rms_i, n_pts in worst:
        flag = "  <== outlier" if rms_i > thr else ""
        print(f"    frame {idx:3d}: RMS={rms_i:5.3f}px  pts={n_pts}{flag}")
    outliers = [idx for idx, rms_i, _ in per_frame if rms_i > thr]
    if outliers:
        drop_arg = ",".join(str(i) for i in outliers)
        print(f"  -> 外れフレーム {outliers} を除いて再校正すると改善する可能性 "
              f"(--analyze --drop {drop_arg} または --auto-drop)")

    # (4) データ量ガイド: 4K は 1 枚あたりの点が少ないので枚数で稼ぐ
    avg_pts = int(round(float(np.mean([f[2] for f in per_frame]))))
    if len(per_frame) < 20:
        print(f"\n! frames={len(per_frame)} は少ない (4K は 20-30 枚推奨)。"
              f"1 枚 {avg_pts} 点しか無いので枚数で稼ぐこと")

    return outliers


def calibrate_and_report(all_obj, all_img, image_size, flags: int,
                         frame_ids: Optional[List[int]] = None):
    """指定フラグで校正し、残差診断まで出す。return (rms, K, dist, outliers)。"""
    print(f"\ncalibrating on {len(all_obj)} frames  model = {_model_name(flags)}")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, image_size, None, None, flags=flags)
    print(f"  RMS reprojection error = {rms:.4f} px")
    print(f"  K =\n{K}")
    print(f"  dist = {dist.ravel()}")
    outliers = residual_report(all_obj, all_img, K, dist, rvecs, tvecs, image_size,
                               frame_ids=frame_ids)
    return rms, K, dist, outliers


def _save_points(path: str, all_obj, all_img, image_size) -> None:
    np.savez(path,
             obj=np.array(all_obj, dtype=object),
             img=np.array(all_img, dtype=object),
             image_size=np.asarray(image_size, dtype=np.int64))
    print(f"saved captured points: {path} ({len(all_obj)} frames)  "
          f"-> 再撮影せず --analyze で別モデルを試せます")


def _save_points_if_any(args: argparse.Namespace, all_obj, all_img, image_size) -> None:
    """取得済みコーナー点があれば保存する (中断・枚数不足でもデータを失わない)。"""
    if args.points_out and all_obj and image_size is not None:
        pts_path = Path(args.points_out)
        pts_path.parent.mkdir(parents=True, exist_ok=True)
        _save_points(str(pts_path), all_obj, all_img, image_size)


def _load_points(path: str):
    data = np.load(path, allow_pickle=True)
    # 同形状フレームは object 配列としてスタックされ dtype=object で戻るため、
    # calibrateCamera が要求する float32 に各要素を明示キャストする。
    all_obj = [np.asarray(x, dtype=np.float32).reshape(-1, 1, 3) for x in data["obj"]]
    all_img = [np.asarray(x, dtype=np.float32).reshape(-1, 1, 2) for x in data["img"]]
    image_size = tuple(int(x) for x in data["image_size"])
    return all_obj, all_img, image_size


def cmd_analyze(args: argparse.Namespace) -> int:
    """保存済みコーナー点を再校正し、モデルを比較する (再撮影不要)。

    保存は --out を明示したときのみ (診断だけの実行で校正ファイルを上書きしない)。
    """
    all_obj, all_img, image_size = _load_points(args.points)
    print(f"loaded {len(all_obj)} frames from {args.points}  image_size={image_size}")

    # --drop: 診断出力の frame 番号 (= 保存ファイル内の順序) で除外する
    keep = list(range(len(all_obj)))
    if args.drop:
        drop = {int(x) for x in args.drop.split(",") if x.strip() != ""}
        unknown = drop - set(keep)
        if unknown:
            print(f"[warn] --drop のフレーム {sorted(unknown)} は存在しない "
                  f"(0..{len(keep) - 1})", file=sys.stderr)
        keep = [i for i in keep if i not in drop]
        print(f"dropped frames {sorted(drop - unknown)} -> {len(keep)} frames")
    if len(keep) < 4:
        print(f"[error] {len(keep)} frames では校正できない (>= 4 必要)", file=sys.stderr)
        return 2

    def subset(ids: List[int]):
        return [all_obj[i] for i in ids], [all_img[i] for i in ids]

    if args.compare:
        # モデルを段階的に増やして限界効用を見る
        obj_k, img_k = subset(keep)
        presets = [
            ("base", 0),
            ("+rational", cv2.CALIB_RATIONAL_MODEL),
            ("+rational+thin-prism",
             cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL),
            ("+rational+thin-prism+tilted",
             cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL),
        ]
        summary = []
        for name, flags in presets:
            rms, _, _ = cv2.calibrateCamera(
                obj_k, img_k, image_size, None, None, flags=flags)[:3]
            summary.append((name, rms))
        print("\n=== model comparison (RMS px, lower=better) ===")
        base = summary[0][1]
        for name, rms in summary:
            print(f"  {name:32s} RMS={rms:.4f}  (Δ vs base = {rms - base:+.4f})")
        print("  注意: 項を増やせば RMS は必ず下がる。改善幅が頭打ちのモデルを選ぶこと")
        print("        (過学習回避)。最終判断は pose 実測で。")
        return 0

    flags = build_calib_flags(args)
    obj_k, img_k = subset(keep)
    rms, K, dist, outliers = calibrate_and_report(obj_k, img_k, image_size, flags,
                                                  frame_ids=keep)
    if args.auto_drop and outliers:
        keep2 = [i for i in keep if i not in outliers]
        if len(keep2) >= 4:
            print(f"\n--auto-drop: 外れフレーム {outliers} を除外して再校正")
            obj_k, img_k = subset(keep2)
            rms, K, dist, _ = calibrate_and_report(obj_k, img_k, image_size, flags,
                                                   frame_ids=keep2)
        else:
            print(f"[warn] --auto-drop で {len(keep2)} 枚になるため除外しない",
                  file=sys.stderr)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        CameraIntrinsics(K=np.asarray(K), dist=np.asarray(dist),
                         image_size=(int(image_size[0]), int(image_size[1]))).save(str(out))
        print(f"saved calibration: {out}")
    else:
        print(f"\n(診断のみ。この結果を保存するには --out {DEF_CALIB_OUT} を付ける)")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    board, dictionary = make_board(args.dict, args.squares_x, args.squares_y,
                                   args.square_len, args.marker_len)
    charuco_detector = make_charuco_detector(board)

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
    win = "calibrate"
    # WINDOW_NORMAL でウィンドウを自由にリサイズ可能にする
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, args.win_width, args.win_height)
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
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("aborted")
                _save_points_if_any(args, all_obj, all_img, image_size)
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

    # 撮影点を先に保存する。枚数不足で終わっても再撮影の続きから --analyze で使える
    _save_points_if_any(args, all_obj, all_img, image_size)

    if len(all_obj) < args.min_frames:
        print(f"[error] only {len(all_obj)} frames; need >= {args.min_frames}", file=sys.stderr)
        return 2

    flags = build_calib_flags(args)
    rms, K, dist, _ = calibrate_and_report(all_obj, all_img, image_size, flags)
    out = Path(args.out or DEF_CALIB_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    CameraIntrinsics(K=np.asarray(K), dist=np.asarray(dist),
                     image_size=(int(image_size[0]), int(image_size[1]))).save(str(out))
    print(f"saved calibration: {out}")
    return 0


def cmd_extrinsics(args: argparse.Namespace) -> int:
    """床の基準マーカを撮影して外部パラメータ (world<-camera) を計測・保存する。

    基準マーカが常時見えない運用 (ドローンや対象物で隠れる等) では、トラッカは
    設定 JSON の extrinsics 節 (R_wc, t_wc) を使う。このコマンドは校正済み K/dist で
    マーカ姿勢を複数フレーム推定し、平均した R_wc/t_wc を書き出す。カメラもマーカも
    静止している前提 (回転は行列和の SVD 直交化で平均する)。
    """
    if args.world_marker_id is None:
        print("[error] --extrinsics には --world-marker-id が必要", file=sys.stderr)
        return 2
    marker_len = args.world_marker_len
    if marker_len is None or marker_len <= 0:
        print("[error] --world-marker-len [m] を指定すること (印刷後の実測値)",
              file=sys.stderr)
        return 2
    try:
        intr = CameraIntrinsics.load(args.calib)
    except FileNotFoundError:
        print(f"[error] calibration not found: {args.calib} "
              f"(先に --capture で校正すること)", file=sys.stderr)
        return 2
    detector = MarkerDetector(args.dict)

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[error] cannot open camera {args.camera}", file=sys.stderr)
        return 2

    win = "extrinsics"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, args.win_width, args.win_height)
    print(f"基準マーカ id={args.world_marker_id} を {args.samples} フレーム収集する "
          f"(q=中断)")

    samples_R: List[np.ndarray] = []
    samples_t: List[np.ndarray] = []
    matched = False
    try:
        while len(samples_t) < args.samples:
            ok, frame = cap.read()
            if not ok:
                continue
            if not matched:
                fh, fw = frame.shape[:2]
                if intr.image_size is None:
                    print(f"[warn] 校正ファイルに image_size が無い (旧形式)。"
                          f"フレーム {fw}x{fh} と一致しているか検証できない",
                          file=sys.stderr)
                elif tuple(intr.image_size) != (fw, fh):
                    print(f"[warn] 校正解像度 {intr.image_size} != フレーム {fw}x{fh}: "
                          f"K をスケーリングして使用", file=sys.stderr)
                    intr = intr.matched_to(fw, fh)
                matched = True
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = detector.detect(gray)
            if args.world_marker_id in det:
                rvec, tvec = estimate_pose_single(
                    det[args.world_marker_id], marker_len, intr)
                wf = WorldFrame()
                wf.set_from_reference_marker(rvec, tvec)
                samples_R.append(wf.R_wc)
                samples_t.append(wf.t_wc)
                cv2.drawFrameAxes(frame, intr.K, intr.dist, rvec.reshape(3, 1),
                                  tvec.reshape(3, 1), marker_len * 0.5, 2)
            cv2.putText(frame, f"samples={len(samples_t)}/{args.samples}  q=abort",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("aborted", file=sys.stderr)
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    t_arr = np.asarray(samples_t)
    t_wc = t_arr.mean(axis=0)
    t_std = t_arr.std(axis=0)
    # 回転平均: 近接した回転群なら行列和の SVD 直交化が最尤に近い
    U, _, Vt = np.linalg.svd(np.sum(np.asarray(samples_R), axis=0))
    R_wc = U @ Vt
    if np.linalg.det(R_wc) < 0:
        R_wc = U @ np.diag([1.0, 1.0, -1.0]) @ Vt

    print(f"\nカメラ世界位置 t_wc = [{t_wc[0]:+.3f}, {t_wc[1]:+.3f}, {t_wc[2]:+.3f}] m "
          f"(カメラ高 = {t_wc[2]:.3f} m)")
    print(f"ばらつき std = [{t_std[0]:.4f}, {t_std[1]:.4f}, {t_std[2]:.4f}] m")
    if float(t_std.max()) > 0.02:
        print("[warn] std が 2cm 超: マーカの反射/ブレ/検出不安定の疑い。"
              "照明・マーカ固定を見直して再計測を推奨", file=sys.stderr)

    ext = {"R_wc": R_wc.tolist(), "t_wc": t_wc.tolist()}
    if args.tracker_config:
        cfg_path = Path(args.tracker_config)
        raw = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        raw["extrinsics"] = ext
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"updated extrinsics in {cfg_path}")
    else:
        out = Path(args.extrinsics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"extrinsics": ext}, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out}")
        print("  -> トラッカ設定 JSON の extrinsics 節にコピーするか、"
              "--tracker-config で直接更新できる")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="camera calibration / ArUco generation")
    p.add_argument("--make-board", action="store_true", help="generate ChArUco board image")
    p.add_argument("--make-markers", action="store_true", help="generate ArUco marker images")
    p.add_argument("--capture", action="store_true", help="capture frames and calibrate")
    p.add_argument("--analyze", action="store_true",
                   help="recalibrate saved points (--points) without recapturing")
    p.add_argument("--extrinsics", action="store_true",
                   help="measure world<-camera extrinsics from the floor reference marker")

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
    p.add_argument("--width", type=int, default=1920,
                   help="撮影幅 [px] (既定 1080p)。4K 校正は --width 3840 --height 2160")
    p.add_argument("--height", type=int, default=1080, help="撮影高 [px]")
    p.add_argument("--win-width", type=int, default=960, help="表示ウィンドウ初期幅 [px]")
    p.add_argument("--win-height", type=int, default=540, help="表示ウィンドウ初期高さ [px]")
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--out", default=None,
                   help=f"校正結果の保存先 (.npz)。--capture の既定は {DEF_CALIB_OUT}、"
                        f"--analyze は明示指定時のみ保存 (診断だけなら上書きしない)")
    p.add_argument("--points-out", default="configs/calib_points.npz",
                   help="撮影したコーナー点の保存先 (--analyze で再利用)。空文字で無効")

    # 歪みモデル拡張フラグ (capture / analyze 共通)。半径依存の残差に効く。
    p.add_argument("--rational", action="store_true",
                   help="CALIB_RATIONAL_MODEL (k4,k5,k6): 広角の強い放射歪み")
    p.add_argument("--thin-prism", action="store_true",
                   help="CALIB_THIN_PRISM_MODEL (s1..s4): レンズ偏心・非対称")
    p.add_argument("--tilted", action="store_true",
                   help="CALIB_TILTED_MODEL (taux,tauy): センサ傾き (Scheimpflug)")

    # analyze
    p.add_argument("--points", default="configs/calib_points.npz",
                   help="--analyze が読む保存済みコーナー点")
    p.add_argument("--compare", action="store_true",
                   help="--analyze でモデルを段階的に比較 (限界効用を見る)")
    p.add_argument("--drop", default="",
                   help="--analyze で除外するフレーム番号 (カンマ区切り、診断出力の番号)")
    p.add_argument("--auto-drop", action="store_true",
                   help="--analyze で RMS > mean+2std のフレームを自動除外して再校正")

    # extrinsics
    p.add_argument("--calib", default=DEF_CALIB_OUT,
                   help="--extrinsics が使う校正ファイル (.npz/.json)")
    p.add_argument("--world-marker-id", type=int, default=None,
                   help="床の世界原点マーカの ArUco ID")
    p.add_argument("--world-marker-len", type=float, default=None,
                   help="世界原点マーカの一辺 [m] (印刷後の実測値)")
    p.add_argument("--samples", type=int, default=60,
                   help="--extrinsics で平均するフレーム数")
    p.add_argument("--tracker-config", default=None,
                   help="extrinsics 節を直接更新するトラッカ設定 JSON")
    p.add_argument("--extrinsics-out", default="configs/extrinsics.json",
                   help="--tracker-config を使わないときの計測結果の保存先")

    args = p.parse_args()
    if args.make_board:
        return cmd_make_board(args)
    if args.make_markers:
        return cmd_make_markers(args)
    if args.capture:
        return cmd_capture(args)
    if args.analyze:
        return cmd_analyze(args)
    if args.extrinsics:
        return cmd_extrinsics(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
