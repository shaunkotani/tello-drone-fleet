"""
4 協調取り囲み制御 (4.1, 4.2)
進行方向角 phi のローパス (式(40)-(42))、正多角形スロット (式(43)-(44))、
進行方向依存配置 (式(45)-(52)) を実装する。

スロット phi_k, r_slot は CMDP 状態 s_k の環境内部状態 (5.1 式(80)) として保持する。
"""
from __future__ import annotations
import math
import numpy as np
from .config import EnvConfig


def wrap_pi(a: float) -> float:
    """角度を (-pi, pi] に折り返す。"""
    return math.atan2(math.sin(a), math.cos(a))


def update_heading(phi_prev: float, vo: np.ndarray, cfg: EnvConfig) -> float:
    """進行方向角 phi の更新 (式(40)-(42))。
    速度が小さいときは更新しない。差分を (-pi,pi] に折り返してローパス。"""
    speed = float(np.linalg.norm(vo))
    if speed < cfg.v_eps:
        return phi_prev
    phi_raw = math.atan2(vo[1], vo[0])                  # 式(40)
    d_phi = wrap_pi(phi_raw - phi_prev)                 # 式(41)
    return phi_prev + cfg.lambda_phi * d_phi            # 式(42)


def slots_case1(ro: np.ndarray, phi: float, cfg: EnvConfig) -> np.ndarray:
    """正多角形スロット (式(43)-(44))。return shape (N,2)."""
    N = cfg.N
    out = np.zeros((N, 2))
    for i in range(N):
        a = phi + cfg.alpha0 + 2.0 * math.pi * i / N    # 式(43)
        out[i] = ro + cfg.R * np.array([math.cos(a), math.sin(a)])  # 式(44)
    return out


def slots_case2(ro: np.ndarray, vo: np.ndarray, phi: float, cfg: EnvConfig):
    """進行方向依存配置 (式(45)-(52))。
    return (slots(N,2), r_bar_slot(2,), c(2,))."""
    N = cfg.N
    nu = float(np.linalg.norm(vo))
    nu_bar = nu / cfg.v_scale                            # 式(45)
    alpha_raw = (2.0 * math.pi / N) * (cfg.cv ** (-nu_bar))  # 式(45): (2pi/N) c^{-nu_bar}
    alpha_k = max(cfg.alpha_min, alpha_raw)              # 式(46)
    out = np.zeros((N, 2))
    for i in range(N):
        # 進行方向側を密にするため中心化 (式(49))
        a = phi - (N - 1) / 2.0 * alpha_k + i * alpha_k
        out[i] = ro + cfg.R * np.array([math.cos(a), math.sin(a)])  # 式(50)
    r_bar_slot = out.mean(axis=0)                        # 式(52)
    c = r_bar_slot - ro                                  # 式(52)
    return out, r_bar_slot, c


def compute_slots(ro: np.ndarray, vo: np.ndarray, phi: float, cfg: EnvConfig):
    """Case に応じてスロットと重心参照を返す。
    return (slots(N,2), r_cg_ref(2,))  where r_cg_ref は式(87)。"""
    if cfg.case == 1:
        slots = slots_case1(ro, phi, cfg)
        r_cg_ref = ro.copy()                             # 式(87) Case1
    else:
        slots, r_bar_slot, _ = slots_case2(ro, vo, phi, cfg)
        r_cg_ref = r_bar_slot                            # 式(87) Case2
    return slots, r_cg_ref


def edge_ref_lengths(slots: np.ndarray) -> np.ndarray:
    """フォーメーショングラフ(サイクル)上の目標辺長 l_edge_ref_{i,i+} (式(88))。
    return shape (N,) : 各 i について i と i+ の目標辺長。"""
    N = slots.shape[0]
    out = np.zeros(N)
    for i in range(N):
        ip = (i + 1) % N
        out[i] = np.linalg.norm(slots[i] - slots[ip])
    return out
