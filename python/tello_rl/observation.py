"""
5.1 局所観測 o_i (式(82))

固定長ベクトル。最大観測近傍数 n_max_obs を決め、距離の近い順に近傍を並べ、
存在しないスロットは 0 + mask=0 を入れる。方策実行時の唯一の入力 (decentralized execution)。
"""
from __future__ import annotations
import math
import numpy as np
from .config import EnvConfig
from . import command_model as cm
from . import costs as ct


def obs_dim(cfg: EnvConfig) -> int:
    n_obs = cfg.resolved_n_max_obs()
    # (ri-ro)/Rmax 2, (ri-slot)/Rmax 2, vi 2, vo 2, sin/cos psi 2,
    # psidot 1, (h-href) 1, hdot 1, 近傍 5*n_obs, 壁 4, batt 1, a_prev 4, role 2
    return 2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 5 * n_obs + 4 + 1 + 4 + 2


def build_observation(i, states, ro, vo, slots, a_prev_i, battery_i, cfg: EnvConfig):
    """エージェント i の観測 o_i (式(82)) を返す。
    states: List[DroneState], a_prev_i:(4,) 前回 executed action。"""
    N = cfg.N
    n_obs = cfg.resolved_n_max_obs()
    st = states[i]
    feats = []
    feats += list((st.r - ro) / cfg.R_max)                       # ri-ro
    feats += list((st.r - slots[i]) / cfg.R_max)                 # ri-slot
    feats += list(st.v / cfg.v_max_norm)                         # vi
    feats += list(vo / cfg.vo_max_norm)                          # vo
    feats += [math.sin(st.psi), math.cos(st.psi)]                # yaw
    feats += [st.psidot / cfg.omega_max_psi]                     # psidot
    feats += [(st.h - cfg.h_ref) / cfg.h_scale]                  # 高度誤差
    feats += [st.hdot / cfg.vz_max_norm]                         # 上下速度

    # 近傍 (距離順、mask 付き)
    others = [j for j in range(N) if j != i]
    others.sort(key=lambda j: np.linalg.norm(states[j].r - st.r))
    for ell in range(n_obs):
        if ell < len(others):
            j = others[ell]; m = 1.0
            feats += list(m * (states[j].r - st.r) / cfg.R_max)
            feats += list(m * (states[j].v - st.v) / cfg.v_max_norm)
            feats += [m]
        else:
            feats += [0.0, 0.0, 0.0, 0.0, 0.0]

    # 壁 4方向
    feats += list(ct.wall_distance_vec(st.r, cfg) / cfg.d_wall_max)
    # バッテリ
    feats += [battery_i / 100.0]
    # 前回入力
    feats += list(a_prev_i)
    # 役割番号
    feats += [math.sin(2 * math.pi * i / N), math.cos(2 * math.pi * i / N)]

    return np.asarray(feats, dtype=np.float32)
