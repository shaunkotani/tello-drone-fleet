"""
3.2 実装用の SDK コマンドレベルモデル (式(16)-(26))

正規化行動 a=[a_lr,a_fb,a_ud,a_yaw] in [-1,1]^4 を、機体座標の速度指令へ変換し
(式(18)-(19))、世界座標へ回転 (式(20))、一次遅れ+積分で状態更新する (式(21)-(26))。

低レベルの推力・姿勢角は使わず、実機 rc a b c d に対応する抽象行動だけを用いる。
"""
from __future__ import annotations
import math
import numpy as np
from .config import EnvConfig


class DroneState:
    """単機の状態。"""
    __slots__ = ("r", "v", "h", "hdot", "psi", "psidot")

    def __init__(self, r, v, h, hdot, psi, psidot):
        self.r = np.asarray(r, dtype=float)        # 水平位置 [m] (2,)
        self.v = np.asarray(v, dtype=float)        # 水平速度 (世界) [m/s] (2,)
        self.h = float(h)                           # 高度 [m]
        self.hdot = float(hdot)                     # 上下速度 [m/s]
        self.psi = float(psi)                       # yaw [rad]
        self.psidot = float(psidot)                 # yaw 角速度 [rad/s]

    def copy(self):
        return DroneState(self.r.copy(), self.v.copy(), self.h,
                          self.hdot, self.psi, self.psidot)


def action_to_world_vel_cmd(a: np.ndarray, psi: float, cfg: EnvConfig) -> np.ndarray:
    """正規化行動の前後・左右成分 → 世界座標の水平速度指令 (式(18),(20))。
    a=[a_lr,a_fb,a_ud,a_yaw]。return v^cmd_world (2,)。"""
    vb_x = cfg.v_max_fb * a[1]                      # 前後 (式(18))
    vb_y = cfg.v_max_lr * a[0]                      # 左右 (式(18))
    c, s = math.cos(psi), math.sin(psi)
    # 式(20): world = R(psi) * body
    vx = c * vb_x - s * vb_y
    vy = s * vb_x + c * vb_y
    return np.array([vx, vy])


def step_command_model(st: DroneState, a: np.ndarray, cfg: EnvConfig,
                       rng: np.random.Generator | None = None) -> DroneState:
    """1 ステップ前進 (式(21)-(26))。a は executed action in [-1,1]^4。"""
    dt = cfg.dt
    # 速度指令
    v_cmd = action_to_world_vel_cmd(a, st.psi, cfg)            # 世界座標水平
    vz_cmd = cfg.v_max_ud * a[2]                                # 式(19)
    psidot_cmd = cfg.omega_max_psi * a[3]                       # 式(19)

    # ノイズ (8.6)
    wv = np.zeros(2); wh = 0.0; wpsi = 0.0
    if rng is not None and cfg.wind_acc_std > 0:
        wv = rng.normal(0.0, cfg.wind_acc_std, size=2)

    new = st.copy()
    # 水平: 式(21),(22)
    new.v = st.v + (dt / cfg.tau_v) * (v_cmd - st.v) + dt * wv
    new.r = st.r + dt * new.v
    # 高度: 式(23),(24)
    new.hdot = st.hdot + (dt / cfg.tau_z) * (vz_cmd - st.hdot) + dt * wh
    new.h = st.h + dt * new.hdot
    # yaw: 式(25),(26)
    new.psidot = st.psidot + (dt / cfg.tau_psi) * (psidot_cmd - st.psidot) + dt * wpsi
    new.psi = st.psi + dt * new.psidot
    return new
