"""
5.2 目的コスト (式(92)) と 5.3 安全制約コスト (式(95)-(99))

CTDE: これらは訓練時に環境(=全状態を知る側)が計算し各エージェントへ返す。
* objective_cost : ステップ目的コスト l_i (式(92))。
* constraint_cost: 制約コストベクトル C_i in R^5 (式(95)-(99))。
"""
from __future__ import annotations
import math
import numpy as np
from .config import EnvConfig
from . import command_model as cm


def wall_distance_vec(r: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """4方向の壁余裕ベクトル d^wall,vec (式(81))。return (4,)."""
    return np.array([r[0] - cfg.x_min, cfg.x_max - r[0],
                     r[1] - cfg.y_min, cfg.y_max - r[1]])


def objective_cost(i, r_next, ro_next, slots_next, r_cg_ref_next, edge_ref_next,
                   v_xy_cmd, v_ref_clip, a, a_prev, h_next, psi_next, psi_ref_next,
                   cfg: EnvConfig) -> float:
    """エージェント i の 1 ステップ目的コスト l_i (式(92))。
    r_next:(N,2) 次状態水平位置, slots_next:(N,2), v_xy_cmd:(2,) 世界座標速度指令,
    v_ref_clip:(2,), a/a_prev:(4,) executed action。"""
    N = cfg.N
    ri = r_next[i]
    R = cfg.R
    rbar = r_next.mean(axis=0)

    # slot 誤差
    c_slot = cfg.w_slot * np.sum((ri - slots_next[i]) ** 2) / R ** 2
    # 半径誤差
    c_R = cfg.w_R * ((np.linalg.norm(ri - ro_next) - R) / R) ** 2
    # 重心誤差
    c_c = cfg.w_c * np.sum((rbar - r_cg_ref_next) ** 2) / R ** 2
    # 辺長誤差 (i-, i+)
    c_e = 0.0
    for jp in ((i - 1) % N, (i + 1) % N):
        # j のフォーメーション目標辺長: i と隣接 jp の辺長基準
        ell_ref = (edge_ref_next[i] if jp == (i + 1) % N
                   else edge_ref_next[(i - 1) % N])
        ell_ref = max(ell_ref, 1e-6)
        c_e += ((np.linalg.norm(ri - r_next[jp]) - ell_ref) / ell_ref) ** 2
    c_e *= cfg.w_e
    # 参照速度追従
    c_v = cfg.w_v * np.sum((v_xy_cmd - v_ref_clip) ** 2) / cfg.v_max_norm ** 2
    # 入力・入力変化
    c_u = cfg.w_u * np.sum(a ** 2)
    c_du = cfg.w_du * np.sum((a - a_prev) ** 2)
    # 高度
    c_h = cfg.w_h * ((h_next - cfg.h_ref) / cfg.h_scale) ** 2
    # yaw
    c_psi = cfg.w_psi * (1.0 - math.cos(psi_next - psi_ref_next))

    return float(c_slot + c_R + c_c + c_e + c_v + c_u + c_du + c_h + c_psi)


def yaw_reference(i, r_next, ro_next, phi, cfg: EnvConfig) -> float:
    """yaw 参照 psi_ref (式(90))。"""
    if cfg.yaw_ref_mode == "heading":
        return phi
    d = ro_next - r_next[i]                              # 対象物注視
    return math.atan2(d[1], d[0])


def constraint_cost(i, r_next, ro_next, h_next, a, cfg: EnvConfig) -> np.ndarray:
    """エージェント i の制約コストベクトル C_i (式(95)-(99))。return (5,)."""
    N = cfg.N
    ri = r_next[i]
    # C_{i,1}: 機体間衝突 (全ペア最小距離; CTDE centralized safety evaluator)
    dmin = min(np.linalg.norm(ri - r_next[j]) for j in range(N) if j != i)
    c1 = max(0.0, 1.0 - dmin / cfg.d_drone)
    # C_{i,2}: 対象物過接近
    c2 = max(0.0, 1.0 - np.linalg.norm(ri - ro_next) / cfg.d_target)
    # C_{i,3}: 壁逸脱 (最短壁距離 式(97))
    dwall_min = float(np.min(wall_distance_vec(ri, cfg)))
    c3 = max(0.0, 1.0 - dwall_min / cfg.d_safe_wall)
    # C_{i,4}: 高度逸脱
    c4 = max(0.0, (abs(h_next - cfg.h_ref) - cfg.e_h) / cfg.e_h)
    # C_{i,5}: 入力飽和
    c5 = max(0.0, (np.max(np.abs(a)) - cfg.a_safe) / (1.0 - cfg.a_safe))
    return np.array([c1, c2, c3, c4, c5])
