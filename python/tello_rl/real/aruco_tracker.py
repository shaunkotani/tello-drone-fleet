"""ArUco による外部計測 (State Estimator フロントエンド)。

原稿 10.2-10.3 節の「天井カメラ + ArUco/AprilTag」外部計測系の送信側実装。
天井カメラで各 Tello (3 台) と対象物に貼った ArUco マーカを検出し、世界座標系
(ENU: 水平 r=[x,y], 高度 h=z, yaw=psi) での位置・姿勢・速度を推定して、
``real_tracker.UdpJsonTracker`` が受信する UDP JSON を 1 フレーム 1 パケットで送る。

座標系
------
* 世界座標は ENU 型 (原稿 2 節)。水平面 r=[x,y] [m]、高度 h=z [m]、yaw=psi [rad]。
* 世界原点は床に置いた基準マーカ (``world_marker_id``) のマーカ座標系で定義する。
  マーカは床に水平に置き z 軸を天井 (上) に向けるので、その座標系をそのまま ENU
  世界系とみなせる。基準マーカが毎フレーム見えない構成では、設定ファイルに保存した
  外部パラメータ (R_wc, t_wc) を使う。

出力スキーマ (``configs/tracker_udp_schema.example.json`` と一致)
-------------------------------------------------------------------
``{"t":<epoch>, "drones":[{"id","r","v","h","hdot","yaw","psidot","battery",
"seen","age"}...], "target":{"r","v","seen","age"}}``

単位: 位置 [m]、速度 [m/s]、角度 [rad]、角速度 [rad/s]。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    import cv2.aruco as aruco
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "aruco_tracker requires opencv-contrib-python (cv2.aruco). "
        "Install with: pip install opencv-contrib-python"
    ) from exc


# --------------------------------------------------------------------------- #
# カメラ内部パラメータ
# --------------------------------------------------------------------------- #
@dataclass
class CameraIntrinsics:
    """カメラ行列 K (3x3) と歪み係数 dist。``calibrate_camera.py`` で生成。"""

    K: np.ndarray
    dist: np.ndarray

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "CameraIntrinsics":
        path = str(path)
        if path.endswith(".npz"):
            data = np.load(path)
            K = np.asarray(data["camera_matrix"], dtype=np.float64)
            dist = np.asarray(data["dist_coeffs"], dtype=np.float64)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
            dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(1, -1)
        else:
            raise ValueError(f"unsupported calibration format: {path} (use .npz or .json)")
        return CameraIntrinsics(K=K.reshape(3, 3), dist=dist.reshape(1, -1))

    def save(self, path: str | os.PathLike[str]) -> None:
        path = str(path)
        if path.endswith(".npz"):
            np.savez(path, camera_matrix=self.K, dist_coeffs=self.dist)
        elif path.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"camera_matrix": self.K.tolist(), "dist_coeffs": self.dist.ravel().tolist()},
                    f, indent=2,
                )
        else:
            raise ValueError(f"unsupported calibration format: {path}")


# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
@dataclass
class ArucoTrackerConfig:
    # camera
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    calibration: str = "configs/camera_calib.npz"

    # aruco
    dictionary: str = "DICT_4X4_50"
    marker_length_m: float = 0.10          # ドローン/対象物マーカの一辺 [m]
    world_marker_id: Optional[int] = None  # 床に置く世界原点マーカの ID (任意)
    world_marker_length_m: Optional[float] = None  # 原点マーカの一辺 (異なる場合)

    # ID 割り当て: drones[k] の ArUco ID をドローン index k に対応させる
    drone_ids: List[int] = field(default_factory=lambda: [0, 1, 2])
    target_id: Optional[int] = 5

    # 保存済み外部パラメータ (world<-camera)。world_marker を使わないとき必須。
    R_wc: Optional[List[List[float]]] = None
    t_wc: Optional[List[float]] = None

    # 平滑化 (原稿 10.3: 移動平均/Kalman の代わりに EMA + 有限差分)
    pos_ema: float = 0.5        # 位置 EMA の新測定重み (1=平滑化なし)
    vel_ema: float = 0.4        # 速度 EMA の新測定重み
    hold_timeout_s: float = 0.5  # マーカ未検出を「最後の値で保持」する上限 [s]

    # 出力 (UdpJsonTracker へ)
    out_host: str = "127.0.0.1"
    out_port: int = 15000
    rate_hz: float = 30.0

    # 可視化
    viz: bool = True

    @staticmethod
    def from_json(path: str | os.PathLike[str]) -> "ArucoTrackerConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cam = raw.get("camera", {})
        ar = raw.get("aruco", {})
        ids = raw.get("ids", {})
        ext = raw.get("extrinsics", {})
        sm = raw.get("smoothing", {})
        out = raw.get("output", {})
        return ArucoTrackerConfig(
            camera_index=int(cam.get("index", 0)),
            width=int(cam.get("width", 1280)),
            height=int(cam.get("height", 720)),
            fps=float(cam.get("fps", 30.0)),
            calibration=str(cam.get("calibration", "configs/camera_calib.npz")),
            dictionary=str(ar.get("dictionary", "DICT_4X4_50")),
            marker_length_m=float(ar.get("marker_length_m", 0.10)),
            world_marker_id=_opt_int(ar.get("world_marker_id")),
            world_marker_length_m=_opt_float(ar.get("world_marker_length_m")),
            drone_ids=[int(x) for x in ids.get("drones", [0, 1, 2])],
            target_id=_opt_int(ids.get("target", 5)),
            R_wc=ext.get("R_wc"),
            t_wc=ext.get("t_wc"),
            pos_ema=float(sm.get("pos_ema", 0.5)),
            vel_ema=float(sm.get("vel_ema", 0.4)),
            hold_timeout_s=float(sm.get("hold_timeout_s", 0.5)),
            out_host=str(out.get("host", "127.0.0.1")),
            out_port=int(out.get("port", 15000)),
            rate_hz=float(out.get("rate_hz", 30.0)),
            viz=bool(raw.get("viz", True)),
        )


def _opt_int(v: Any) -> Optional[int]:
    return None if v is None else int(v)


def _opt_float(v: Any) -> Optional[float]:
    return None if v is None else float(v)


# --------------------------------------------------------------------------- #
# 世界座標系 (world <- camera) 変換
# --------------------------------------------------------------------------- #
class WorldFrame:
    """world <- camera の剛体変換 p_world = R_wc @ p_cam + t_wc を保持する。"""

    def __init__(self, R_wc: Optional[np.ndarray] = None, t_wc: Optional[np.ndarray] = None):
        self.R_wc = np.eye(3) if R_wc is None else np.asarray(R_wc, dtype=float).reshape(3, 3)
        self.t_wc = np.zeros(3) if t_wc is None else np.asarray(t_wc, dtype=float).reshape(3)
        self.valid = R_wc is not None

    def set_from_reference_marker(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """基準マーカ姿勢 (camera<-marker) から world<-camera を更新する。

        基準マーカ座標系をそのまま世界系 (ENU) とみなす。solvePnP は
        p_cam = R_cm @ p_marker + t_cm を返すので、その逆が world<-camera。

        平面マーカの法線方向には符号曖昧性があり、solvePnP が世界 z 軸を下向きに
        選ぶことがある。天井カメラは必ず床より上にあるという物理的拘束から、
        カメラの世界高度 t_wc[2] が負なら世界 x 軸まわりに 180 度反転して z 上向き
        (ENU) を強制する。これによりドローン高度 h が常に正、床上の対象物が h≈0
        となる。床マーカの面内向き (East/North) はマーカの置き方で物理的に合わせる。
        """
        R_cm, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
        t_cm = np.asarray(tvec, dtype=float).reshape(3)
        R_wc = R_cm.T
        t_wc = -R_cm.T @ t_cm
        if t_wc[2] < 0.0:
            flip = np.diag([1.0, -1.0, -1.0])  # 世界 x 軸まわり 180 度 (z 上向き強制)
            R_wc = flip @ R_wc
            t_wc = flip @ t_wc
        self.R_wc = R_wc
        self.t_wc = t_wc
        self.valid = True

    def transform(self, rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """マーカ姿勢 (camera<-marker) を世界系へ。return (pos_world(3,), R_wm(3x3))。"""
        R_cm, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
        t_cm = np.asarray(tvec, dtype=float).reshape(3)
        pos_world = self.R_wc @ t_cm + self.t_wc
        R_wm = self.R_wc @ R_cm
        return pos_world, R_wm


def pose_to_state(pos_world: np.ndarray, R_wm: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """世界系のマーカ姿勢 -> (r[x,y], h=z, yaw)。

    床に水平なマーカは z 軸が上を向くので、yaw は世界 x 軸まわりの回転として
    R_wm の第 1 列 (マーカ x 軸の世界表現) の atan2 で得る。
    """
    r = pos_world[:2].astype(float)
    h = float(pos_world[2])
    yaw = float(math.atan2(R_wm[1, 0], R_wm[0, 0]))
    return r, h, yaw


# --------------------------------------------------------------------------- #
# マーカ検出 + 単マーカ姿勢推定 (OpenCV 4.7+ 新 API)
# --------------------------------------------------------------------------- #
class MarkerDetector:
    def __init__(self, dictionary_name: str):
        if not hasattr(aruco, dictionary_name):
            raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
        self.dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
        params = aruco.DetectorParameters()
        # サブピクセル精緻化で姿勢推定を安定化
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(self.dictionary, params)

    def detect(self, gray: np.ndarray) -> Dict[int, np.ndarray]:
        """グレースケール画像 -> {id: corners(4,2)}。corners は TL,TR,BR,BL 順。"""
        corners, ids, _ = self.detector.detectMarkers(gray)
        out: Dict[int, np.ndarray] = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            out[int(i)] = c.reshape(4, 2).astype(np.float64)
        return out


def estimate_pose_single(corners: np.ndarray, marker_len: float,
                         intr: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """平面正方マーカの姿勢を solvePnP(IPPE_SQUARE) で推定。

    estimatePoseSingleMarkers は OpenCV 4.7 で廃止されたため自前で実装する。
    マーカ座標系: 中心原点、辺長 marker_len、z 軸はマーカ面の外向き。
    """
    s = float(marker_len) / 2.0
    obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed for marker")
    return rvec.reshape(3), tvec.reshape(3)


# --------------------------------------------------------------------------- #
# 平滑化 (位置 EMA + 有限差分速度 + yaw アンラップ)
# --------------------------------------------------------------------------- #
class _Smoother:
    """1 マーカ分の状態推定。EMA で平滑化し、有限差分で速度を出す (原稿 10.3)。"""

    def __init__(self, pos_ema: float, vel_ema: float):
        self.a_pos = float(pos_ema)
        self.a_vel = float(vel_ema)
        self.r: Optional[np.ndarray] = None
        self.h = 0.0
        self.yaw = 0.0
        self.v = np.zeros(2)
        self.hdot = 0.0
        self.psidot = 0.0
        self.last_stamp: Optional[float] = None
        self.last_seen_stamp: float = 0.0

    @property
    def initialized(self) -> bool:
        return self.r is not None

    def update_seen(self, r: np.ndarray, h: float, yaw: float, stamp: float) -> None:
        if self.r is None:
            self.r = r.astype(float)
            self.h = float(h)
            self.yaw = float(yaw)
        else:
            ap = self.a_pos
            r_new = ap * r + (1.0 - ap) * self.r
            h_new = ap * h + (1.0 - ap) * self.h
            dyaw = math.atan2(math.sin(yaw - self.yaw), math.cos(yaw - self.yaw))
            yaw_new = self.yaw + ap * dyaw
            if self.last_stamp is not None:
                dt = max(1e-3, stamp - self.last_stamp)
                v_raw = (r_new - self.r) / dt
                hdot_raw = (h_new - self.h) / dt
                psidot_raw = (ap * dyaw) / dt
                av = self.a_vel
                self.v = av * v_raw + (1.0 - av) * self.v
                self.hdot = av * hdot_raw + (1.0 - av) * self.hdot
                self.psidot = av * psidot_raw + (1.0 - av) * self.psidot
            self.r = r_new
            self.h = h_new
            self.yaw = yaw_new
        self.last_stamp = stamp
        self.last_seen_stamp = stamp

    def hold(self, stamp: float) -> None:
        """未検出フレーム: 位置は据え置き、速度は徐々に 0 へ。"""
        self.v *= 0.5
        self.hdot *= 0.5
        self.psidot *= 0.5
        self.last_stamp = stamp

    def age(self, stamp: float) -> float:
        return float(stamp - self.last_seen_stamp) if self.initialized else float("inf")


# --------------------------------------------------------------------------- #
# トラッカ本体
# --------------------------------------------------------------------------- #
class ArucoTracker:
    """1 フレームを処理して UDP 送信用の状態 dict を作る (カメラ I/O は持たない)。"""

    def __init__(self, cfg: ArucoTrackerConfig, intr: CameraIntrinsics):
        self.cfg = cfg
        self.intr = intr
        self.detector = MarkerDetector(cfg.dictionary)
        self.world = WorldFrame(
            R_wc=None if cfg.R_wc is None else np.asarray(cfg.R_wc, dtype=float),
            t_wc=None if cfg.t_wc is None else np.asarray(cfg.t_wc, dtype=float),
        )
        self.world_len = cfg.world_marker_length_m or cfg.marker_length_m
        self.drone_sm = {i: _Smoother(cfg.pos_ema, cfg.vel_ema) for i in range(len(cfg.drone_ids))}
        self.target_sm = _Smoother(cfg.pos_ema, cfg.vel_ema)
        # ArUco ID -> ドローン index
        self.id_to_drone = {int(mid): idx for idx, mid in enumerate(cfg.drone_ids)}

    def process_frame(self, frame: np.ndarray, stamp: Optional[float] = None
                      ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """frame(BGR) -> (payload or None, {id:(rvec,tvec)} 検出姿勢)。

        payload は世界座標が未確定 (基準マーカ未検出かつ外部パラメータ未設定) のとき None。
        """
        if stamp is None:
            stamp = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # 1) 世界座標系の確定。基準マーカが見えればそれで更新。
        if self.cfg.world_marker_id is not None and self.cfg.world_marker_id in detections:
            rvec, tvec = estimate_pose_single(
                detections[self.cfg.world_marker_id], self.world_len, self.intr)
            poses[self.cfg.world_marker_id] = (rvec, tvec)
            self.world.set_from_reference_marker(rvec, tvec)
        if not self.world.valid:
            return None, poses  # まだ世界座標を決められない

        # 2) 各ドローン / 対象物マーカを世界系へ。
        seen_drones = set()
        for mid, corners in detections.items():
            if mid == self.cfg.world_marker_id:
                continue
            is_drone = mid in self.id_to_drone
            is_target = (self.cfg.target_id is not None and mid == self.cfg.target_id)
            if not (is_drone or is_target):
                continue
            rvec, tvec = estimate_pose_single(corners, self.cfg.marker_length_m, self.intr)
            poses[mid] = (rvec, tvec)
            pos_w, R_wm = self.world.transform(rvec, tvec)
            r, h, yaw = pose_to_state(pos_w, R_wm)
            if is_drone:
                idx = self.id_to_drone[mid]
                self.drone_sm[idx].update_seen(r, h, yaw, stamp)
                seen_drones.add(idx)
            else:
                self.target_sm.update_seen(r, h, yaw, stamp)

        # 3) 未検出は保持。
        for idx, sm in self.drone_sm.items():
            if idx not in seen_drones:
                sm.hold(stamp)
        if self.cfg.target_id is None or self.cfg.target_id not in detections:
            self.target_sm.hold(stamp)

        payload = self._build_payload(stamp)
        return payload, poses

    def _build_payload(self, stamp: float) -> Optional[Dict[str, Any]]:
        drones: List[Dict[str, Any]] = []
        for idx in range(len(self.cfg.drone_ids)):
            sm = self.drone_sm[idx]
            if not sm.initialized:
                # まだ一度も観測されていないドローンがあると受信側が成立しない。
                return None
            drones.append({
                "id": idx,
                "r": [float(sm.r[0]), float(sm.r[1])],
                "v": [float(sm.v[0]), float(sm.v[1])],
                "h": float(sm.h),
                "hdot": float(sm.hdot),
                "yaw": float(sm.yaw),
                "psidot": float(sm.psidot),
                "battery": 100.0,  # 外部計測ではバッテリ不明。Tello Driver 側で上書き。
                "seen": bool(sm.age(stamp) <= self.cfg.hold_timeout_s),
                "age": round(sm.age(stamp), 4),
            })
        target_sm = self.target_sm
        if target_sm.initialized:
            target = {
                "r": [float(target_sm.r[0]), float(target_sm.r[1])],
                "v": [float(target_sm.v[0]), float(target_sm.v[1])],
                "seen": bool(target_sm.age(stamp) <= self.cfg.hold_timeout_s),
                "age": round(target_sm.age(stamp), 4),
            }
        else:
            target = {"r": [0.0, 0.0], "v": [0.0, 0.0], "seen": False, "age": float("inf")}
        return {"t": float(stamp), "drones": drones, "target": target}


# --------------------------------------------------------------------------- #
# UDP 送信
# --------------------------------------------------------------------------- #
class UdpPublisher:
    """状態 dict を JSON にして UDP で 1 パケット送信する。"""

    def __init__(self, host: str, port: int):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, self.addr)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 可視化
# --------------------------------------------------------------------------- #
def draw_overlay(frame: np.ndarray, tracker: ArucoTracker,
                 poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 payload: Optional[Dict[str, Any]]) -> np.ndarray:
    """検出マーカの軸と推定状態を描画する (デバッグ用)。"""
    intr = tracker.intr
    for mid, (rvec, tvec) in poses.items():
        length = tracker.world_len if mid == tracker.cfg.world_marker_id else tracker.cfg.marker_length_m
        cv2.drawFrameAxes(frame, intr.K, intr.dist, rvec.reshape(3, 1), tvec.reshape(3, 1),
                          length * 0.5, 2)
    lines = []
    if not tracker.world.valid:
        lines.append("world frame: NOT set (show reference marker)")
    if payload is not None:
        for d in payload["drones"]:
            tag = "" if d["seen"] else " [stale]"
            lines.append("drone{}: r=({:+.2f},{:+.2f}) h={:.2f} yaw={:+.0f}deg{}".format(
                d["id"], d["r"][0], d["r"][1], d["h"], math.degrees(d["yaw"]), tag))
        t = payload["target"]
        lines.append("target: r=({:+.2f},{:+.2f}){}".format(
            t["r"][0], t["r"][1], "" if t["seen"] else " [stale]"))
    else:
        lines.append("payload: waiting for all drone markers...")
    y = 24
    for ln in lines:
        cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26
    return frame
