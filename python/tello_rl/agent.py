"""
6 各エージェントが持つ変数 (表4)

Agent i は以下を保持する:
  - 方策パラメータ theta_i (TanhGaussianPolicy)
  - 価値関数パラメータ xi_i (ValueNetwork; 通信しないローカル変数)
  - 乗数スタック mu_i in R^{N*m} (全制約乗数の局所コピー; 自分の更新は i ブロックへ)
  - 送信済み値 theta_hat_i, mu_hat_i (イベント発生時のみ更新)
近傍からの保存値は学習器側の last_broadcast レジストリ経由で参照する。
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .networks import TanhGaussianPolicy, ValueNetwork


class Agent:
    def __init__(self, i: int, obs_dim: int, act_dim: int, N: int, m_i: int,
                 hidden: int, value_lr: float, device: str = "cpu"):
        self.i = i
        self.N = N
        self.m_i = m_i
        self.device = device
        self.policy = TanhGaussianPolicy(obs_dim, act_dim, hidden).to(device)
        self.value = ValueNetwork(obs_dim, hidden).to(device)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        # 乗数スタック mu_i in R^{N*m_i}, 初期 0 (式: mu_{i,0} in M)
        self.mu = np.zeros(N * m_i)
        # 送信済み値 (初期は現在値)
        self.theta_hat = self.get_theta_vec()
        self.mu_hat = self.mu.copy()

    # --- 方策パラメータのベクトル入出力 (consensus/projection 用) ---
    def get_theta_vec(self) -> np.ndarray:
        return parameters_to_vector(self.policy.parameters()).detach().cpu().numpy()

    def set_theta_vec(self, vec: np.ndarray):
        t = torch.as_tensor(vec, dtype=torch.float32, device=self.device)
        vector_to_parameters(t, self.policy.parameters())

    def own_block(self) -> slice:
        """自分の制約乗数ブロック mu^(i)_i の slice。"""
        return slice(self.i * self.m_i, (self.i + 1) * self.m_i)
