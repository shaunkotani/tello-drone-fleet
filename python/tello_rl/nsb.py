"""
4.3 逆運動学に基づく参照速度 (式(53)-(76))

* reference_velocity : 目的コストの参照速度 v_ref (式(53)) を有限差分形で計算。
  d/dt r_slot = vo + R*alpha_dot[..] を r_slot の有限差分で評価し、Kp スロット誤差を加える。
* NSBController     : damped pseudoinverse + 零空間射影の優先順位付き NSB (式(56)-(64))。
  比較ベースライン / 初期教師 として用いる。Case1=(σe,σc,σR[,σo]), Case2=(σe^dir,σc^dir,σo)。
"""
from __future__ import annotations
import math
import numpy as np
from .config import EnvConfig
from . import formation as fm


def reference_velocity(slots_k: np.ndarray, slots_kp1: np.ndarray,
                       r: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """v_ref (式(53)) を有限差分で計算。return (N,2)。"""
    v_ff = (slots_kp1 - slots_k) / cfg.dt              # vo + R*alpha_dot[-sin,cos] 相当
    return v_ff + cfg.Kp_slot * (slots_k - r)          # 式(53)


def clip_reference(v_ref: np.ndarray, cfg: EnvConfig, eps_v: float = 1e-3) -> np.ndarray:
    """参照速度の飽和 (式(91))。return (N,2)。"""
    out = v_ref.copy()
    for i in range(out.shape[0]):
        n = np.linalg.norm(out[i])
        scale = min(1.0, cfg.v_max_norm / max(n, eps_v))
        out[i] = scale * out[i]
    return out


def damped_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    """damped pseudoinverse J^T (J J^T + lam^2 I)^-1 (式(57))。"""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(m))


class NSBController:
    """優先順位付き NSB 速度 v_NSB (式(58)-(64)) を計算するベースライン制御器。"""

    def __init__(self, cfg: EnvConfig, lam: float = 0.1,
                 lam_c=1.0, lam_R=1.0, lam_e=1.0, lam_o=1.0):
        self.cfg = cfg
        self.lam = lam
        self.gains = dict(c=lam_c, R=lam_R, e=lam_e, o=lam_o)

    # --- タスク・ヤコビ・ドリフトの構築 ---
    def _tasks_case1(self, r, ro, vo, slots, slot_vel, with_slot):
        cfg = self.cfg
        N = cfg.N
        rbar = r.mean(axis=0)
        ellR = 2.0 * cfg.R * math.sin(math.pi / N)

        tasks = []
        # 高優先: 辺長 σe (式(67),(76))
        sigma_e = np.array([0.5 * (np.linalg.norm(r[i] - r[(i + 1) % N]) ** 2 - ellR ** 2)
                            for i in range(N)])
        Je = np.zeros((N, 2 * N))
        for i in range(N):
            ip = (i + 1) % N
            Je[i, 2 * i:2 * i + 2] = (r[i] - r[ip])
            Je[i, 2 * ip:2 * ip + 2] = -(r[i] - r[ip])
        tasks.append(("e", sigma_e, np.zeros(N), Je, np.zeros(N)))

        # 重心 σc (式(65),(74),(69))
        sigma_c = rbar - ro
        Jc = np.tile(np.eye(2), (1, N)) / N
        b_c = -vo
        tasks.append(("c", sigma_c, np.zeros(2), Jc, b_c))

        # 半径 σR (式(66),(75),(69))
        sigma_R = np.array([0.5 * (np.linalg.norm(r[i] - ro) ** 2 - cfg.R ** 2)
                            for i in range(N)])
        JR = np.zeros((N, 2 * N))
        b_R = np.zeros(N)
        for i in range(N):
            JR[i, 2 * i:2 * i + 2] = (r[i] - ro)
            b_R[i] = -float((r[i] - ro) @ vo)
        tasks.append(("R", sigma_R, np.zeros(N), JR, b_R))

        if with_slot:                                   # 低優先 σo (式(68))
            sigma_o = (r - slots).reshape(-1)
            Jo = np.eye(2 * N)
            b_o = -slot_vel.reshape(-1)
            tasks.append(("o", sigma_o, np.zeros(2 * N), Jo, b_o))
        return tasks

    def _tasks_case2(self, r, ro, vo, slots, slot_vel, r_bar_slot, r_bar_slot_vel):
        cfg = self.cfg
        N = cfg.N
        rbar = r.mean(axis=0)
        # 辺長 σe^dir (式(71),(72))
        ell = fm.edge_ref_lengths(slots)
        sigma_e = np.array([0.5 * (np.linalg.norm(r[i] - r[(i + 1) % N]) ** 2 - ell[i] ** 2)
                            for i in range(N)])
        Je = np.zeros((N, 2 * N))
        for i in range(N):
            ip = (i + 1) % N
            Je[i, 2 * i:2 * i + 2] = (r[i] - r[ip])
            Je[i, 2 * ip:2 * ip + 2] = -(r[i] - r[ip])
        # 辺長ドリフト b_e^dir = -ell*ell_dot は有限差分で外部供給が必要なため近似0
        tasks = [("e", sigma_e, np.zeros(N), Je, np.zeros(N))]
        # 重心 σc^dir (式(70))
        sigma_c = rbar - r_bar_slot
        Jc = np.tile(np.eye(2), (1, N)) / N
        tasks.append(("c", sigma_c, np.zeros(2), Jc, -r_bar_slot_vel))
        # スロット σo (式(70))
        sigma_o = (r - slots).reshape(-1)
        Jo = np.eye(2 * N)
        tasks.append(("o", sigma_o, np.zeros(2 * N), Jo, -slot_vel.reshape(-1)))
        return tasks

    def compute(self, r, ro, vo, slots, slot_vel,
                r_bar_slot=None, r_bar_slot_vel=None, with_slot=True) -> np.ndarray:
        """v_NSB を計算。return (N,2)。"""
        cfg = self.cfg
        if cfg.case == 1:
            tasks = self._tasks_case1(r, ro, vo, slots, slot_vel, with_slot)
        else:
            if r_bar_slot is None:
                r_bar_slot = slots.mean(axis=0)
            if r_bar_slot_vel is None:
                r_bar_slot_vel = slot_vel.mean(axis=0)
            tasks = self._tasks_case2(r, ro, vo, slots, slot_vel,
                                      r_bar_slot, r_bar_slot_vel)
        # 再帰 NSB (式(59)-(64))
        n = 2 * cfg.N
        Nproj = np.eye(n)
        v = np.zeros(n)
        for name, sigma, sigma_d, J, b in tasks:
            lam_T = self.gains[name]
            u = sigma_d + lam_T * (sigma_d - sigma) - b   # 式(58) (σ_d_dot=0)
            Jt = J @ Nproj                                 # 式(60)
            Jt_pinv = damped_pinv(Jt, self.lam)
            dv = Nproj @ Jt_pinv @ (u - J @ v)             # 式(61)
            v = v + dv                                     # 式(62)
            Nproj = Nproj @ (np.eye(n) - Jt_pinv @ Jt)     # 式(63)
        return v.reshape(cfg.N, 2)                          # 式(64)
