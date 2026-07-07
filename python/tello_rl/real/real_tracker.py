"""External tracker interface for real Tello experiments.

The recommended first implementation is an external tracker process that sends one
UDP JSON packet per frame to this module.

Expected UDP JSON schema
------------------------
{
  "t": 1710000000.123,
  "drones": [
    {"id":0, "r":[0.1, 0.2], "h":1.0, "yaw":0.0,
     "v":[0.0,0.0], "hdot":0.0, "psidot":0.0, "battery":95},
    ...
  ],
  "target": {"r":[0.0,0.0], "v":[0.0,0.0]}
}

Units:
    position [m], velocity [m/s], angle [rad], angular velocity [rad/s].
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..command_model import DroneState


class TrackerTimeout(RuntimeError):
    """Raised when no fresh tracker frame is available."""


@dataclass
class RealDroneState:
    id: int
    r: np.ndarray
    v: np.ndarray
    h: float
    hdot: float
    yaw: float
    psidot: float
    battery: float = 100.0
    stamp: float = 0.0

    def to_command_model_state(self) -> DroneState:
        return DroneState(self.r.copy(), self.v.copy(), self.h, self.hdot, self.yaw, self.psidot)


@dataclass
class RealTargetState:
    r: np.ndarray
    v: np.ndarray
    stamp: float = 0.0


@dataclass
class TrackerFrame:
    stamp: float
    drones: List[RealDroneState]
    target: RealTargetState


@dataclass
class TrackerBounds:
    """カメラ視野の床上四隅とカメラ位置 (aruco_tracker の "bounds" フィールド)。"""
    corners: np.ndarray  # (M,2) 床レベルでの視野四隅 [m] (通常 M=4)
    camera: np.ndarray   # (3,) カメラ位置 [cx, cy, Hc] [m]
    stamp: float = 0.0


class UdpJsonTracker:
    """Receive real-time pose estimates from an external UDP JSON tracker.

    The class stores only the latest complete frame.  Velocities are estimated by
    finite differences if the incoming JSON omits them.
    """

    def __init__(self, n_drones: int, host: str = "0.0.0.0", port: int = 15000,
                 max_age_s: float = 0.30, recv_buffer: int = 65535):
        self.n_drones = int(n_drones)
        self.host = host
        self.port = int(port)
        self.max_age_s = float(max_age_s)
        self.recv_buffer = int(recv_buffer)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[TrackerFrame] = None
        self._latest_bounds: Optional[TrackerBounds] = None
        self._prev_drones: Dict[int, RealDroneState] = {}
        self._prev_target: Optional[RealTargetState] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="UdpJsonTracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._sock = None
        self._thread = None

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running.is_set():
            try:
                data, _ = self._sock.recvfrom(self.recv_buffer)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                frame = self._parse_frame(msg)
                bounds = self._parse_bounds(msg, frame.stamp)
            except Exception:
                # Ignore malformed frames.  The control loop will timeout if no good
                # frame arrives.
                continue
            with self._lock:
                self._latest = frame
                if bounds is not None:
                    self._latest_bounds = bounds

    def _parse_frame(self, msg: Dict[str, Any]) -> TrackerFrame:
        stamp = float(msg.get("t", time.time()))
        drones_raw = msg.get("drones", [])
        if len(drones_raw) < self.n_drones:
            raise ValueError(f"tracker frame has {len(drones_raw)} drones, expected {self.n_drones}")

        drones: List[RealDroneState] = []
        for item in drones_raw:
            drone_id = int(item["id"])
            r = np.asarray(item["r"], dtype=float).reshape(2)
            h = float(item.get("h", 0.0))
            yaw = float(item.get("yaw", item.get("psi", 0.0)))
            prev = self._prev_drones.get(drone_id)
            dt = max(1e-6, stamp - prev.stamp) if prev is not None else 1e-6
            if "v" in item:
                v = np.asarray(item["v"], dtype=float).reshape(2)
            elif prev is not None:
                v = (r - prev.r) / dt
            else:
                v = np.zeros(2)
            if "hdot" in item:
                hdot = float(item["hdot"])
            elif prev is not None:
                hdot = (h - prev.h) / dt
            else:
                hdot = 0.0
            if "psidot" in item:
                psidot = float(item["psidot"])
            elif "yaw_rate" in item:
                psidot = float(item["yaw_rate"])
            elif prev is not None:
                dyaw = np.arctan2(np.sin(yaw - prev.yaw), np.cos(yaw - prev.yaw))
                psidot = float(dyaw / dt)
            else:
                psidot = 0.0
            battery = float(item.get("battery", 100.0))
            st = RealDroneState(drone_id, r, v, h, hdot, yaw, psidot, battery, stamp)
            drones.append(st)

        drones.sort(key=lambda s: s.id)
        drones = drones[:self.n_drones]
        for st in drones:
            self._prev_drones[st.id] = st

        target_raw = msg.get("target", {"r": [0.0, 0.0], "v": [0.0, 0.0]})
        tr = np.asarray(target_raw.get("r", [0.0, 0.0]), dtype=float).reshape(2)
        if "v" in target_raw:
            tv = np.asarray(target_raw["v"], dtype=float).reshape(2)
        elif self._prev_target is not None:
            dt = max(1e-6, stamp - self._prev_target.stamp)
            tv = (tr - self._prev_target.r) / dt
        else:
            tv = np.zeros(2)
        target = RealTargetState(tr, tv, stamp)
        self._prev_target = target
        return TrackerFrame(stamp=stamp, drones=drones, target=target)

    @staticmethod
    def _parse_bounds(msg: Dict[str, Any], stamp: float) -> Optional[TrackerBounds]:
        raw = msg.get("bounds")
        if not isinstance(raw, dict):
            return None
        try:
            corners = np.asarray(raw["corners"], dtype=float).reshape(-1, 2)
            camera = np.asarray(raw["camera"], dtype=float).reshape(3)
        except Exception:
            return None
        if corners.shape[0] < 3:
            return None
        return TrackerBounds(corners=corners, camera=camera, stamp=stamp)

    def get_bounds(self, timeout_s: float = 0.0) -> Optional[TrackerBounds]:
        """最新の視野 bounds を返す。未受信なら timeout_s まで待ち、なければ None。

        bounds は毎フレーム同梱とは限らないため、一度受信した値を保持し続ける
        (四隅マーカは静止物なので古さは問題にならない)。
        """
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            with self._lock:
                b = self._latest_bounds
            if b is not None or time.time() >= deadline:
                return b
            time.sleep(0.02)

    def get_latest(self, timeout_s: Optional[float] = None) -> TrackerFrame:
        """Return the latest fresh frame.

        Parameters
        ----------
        timeout_s:
            Maximum wall-clock time to wait for a fresh frame.  ``None`` uses
            ``max_age_s``.
        """
        deadline = time.time() + (self.max_age_s if timeout_s is None else float(timeout_s))
        while True:
            with self._lock:
                frame = self._latest
            now = time.time()
            if frame is not None and now - frame.stamp <= self.max_age_s:
                return frame
            if now >= deadline:
                age = None if frame is None else now - frame.stamp
                raise TrackerTimeout(f"no fresh tracker frame; latest age={age}")
            time.sleep(0.005)

    def get_states(self, timeout_s: Optional[float] = None) -> Tuple[List[DroneState], RealTargetState, np.ndarray]:
        frame = self.get_latest(timeout_s=timeout_s)
        states = [d.to_command_model_state() for d in frame.drones]
        battery = np.asarray([d.battery for d in frame.drones], dtype=float)
        return states, frame.target, battery

    def __enter__(self) -> "UdpJsonTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
