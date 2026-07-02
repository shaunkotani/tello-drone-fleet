"""
7.1 重み行列とイベントトリガ (式(119)-(122))

通信グラフ G_comm (サイクル/完全) の近傍と、差分型混合の係数 omega_mix_ij = w_ij (式(121)) を構築。
Algorithm 1 step 7 の差分型混合 z_i = x_i + sum_j omega_mix_ij (x_hat_j - x_hat_i) に用いる。
"""
from __future__ import annotations
import numpy as np
from .config import TrainConfig


def build_graph(N: int, kind: str = "cycle"):
    """近傍リスト N_comm_i を返す。"""
    nbr = [[] for _ in range(N)]
    if kind == "complete":
        for i in range(N):
            nbr[i] = [j for j in range(N) if j != i]
    else:  # cycle
        for i in range(N):
            nbr[i] = sorted({(i - 1) % N, (i + 1) % N} - {i})
    return nbr


def mixing_coeffs(N: int, nbr, rule: str = "cycle"):
    """差分型混合係数 omega_mix_ij (オフ対角のみ) を返す dict[(i,j)]->w。"""
    omega = {}
    if rule == "cycle":                                # 式(119): w_{i,i±}=4/9
        for i in range(N):
            for j in nbr[i]:
                omega[(i, j)] = 4.0 / 9.0
    else:                                              # Metropolis 式(120)
        deg = [len(nbr[i]) for i in range(N)]
        for i in range(N):
            for j in nbr[i]:
                omega[(i, j)] = 1.0 / (1.0 + max(deg[i], deg[j]))
    return omega


def trigger_threshold(eps0: float, eps_min: float, rho: float, q: int) -> float:
    """イベントトリガしきい値 (式(122)): eps0 * rho^q + eps_min。"""
    return eps0 * (rho ** q) + eps_min
