"""
対象物軌道 (8.2-8.4)。静止・等速・円軌道・折れ線を切替。
CMDP 状態 s_k には対象物軌道内部状態 zeta^o_k (式(80)) を含めて Markov 性を保つ。
"""
from __future__ import annotations
import math
import numpy as np
from .config import EnvConfig


class Target:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.k = 0
        self.ro = np.zeros(2)
        self.vo = np.zeros(2)
        self.reset()

    def reset(self):
        cfg = self.cfg
        self.k = 0
        if cfg.target_mode == "circle":
            A, w = cfg.circle_A, cfg.circle_omega
            self.ro = np.array([A, 0.0])
            self.vo = np.array([0.0, A * w])
        elif cfg.target_mode == "const_vel":
            self.ro = np.array([-1.0, 0.0])
            self.vo = np.array([cfg.v_tar, 0.0])
        elif cfg.target_mode == "polyline":
            self.ro = np.zeros(2)
            self.vo = np.array([cfg.v_tar, 0.0])
        else:  # static
            self.ro = np.zeros(2)
            self.vo = np.zeros(2)
        return self.ro.copy(), self.vo.copy()

    def step(self):
        """次時刻へ。return (ro,(2,), vo,(2,), zeta_o(dict))."""
        cfg = self.cfg
        self.k += 1
        k, dt = self.k, cfg.dt
        if cfg.target_mode == "circle":               # 式(139)
            A, w = cfg.circle_A, cfg.circle_omega
            self.ro = np.array([A * math.cos(w * k * dt), A * math.sin(w * k * dt)])
            self.vo = np.array([-A * w * math.sin(w * k * dt),
                                A * w * math.cos(w * k * dt)])
            zeta = {"mode": "circle", "phase": w * k * dt}
        elif cfg.target_mode == "polyline":           # 式(140)
            seg = (k // cfg.poly_K_seg) % 4
            v = cfg.v_tar
            self.vo = {0: np.array([v, 0.0]), 1: np.array([0.0, v]),
                       2: np.array([-v, 0.0]), 3: np.array([0.0, -v])}[seg]
            self.ro = self.ro + dt * self.vo
            zeta = {"mode": "polyline", "seg": int(seg),
                    "seg_step": int(k % cfg.poly_K_seg)}
        elif cfg.target_mode == "const_vel":          # 式(136)
            self.ro = self.ro + dt * self.vo
            # 領域境界で折り返し
            if self.ro[0] > cfg.x_max - 0.5 or self.ro[0] < cfg.x_min + 0.5:
                self.vo[0] *= -1.0
            zeta = {"mode": "const_vel"}
        else:                                          # static
            zeta = {"mode": "static"}
        return self.ro.copy(), self.vo.copy(), zeta
