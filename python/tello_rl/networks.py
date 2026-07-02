"""
方策・価値ネットワーク (5.1, 表5)。
* TanhGaussianPolicy: 平均 m と log_std を出力し tanh で [-1,1]^4 に押し込む方策 (式(83),(84))。
* ValueNetwork     : advantage 推定用の価値関数 V_xi (式(126),(130))。
MLP: input-128-128-action / input-128-128-1。
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class TanhGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, o):
        h = self.body(o)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def dist(self, o):
        mean, log_std = self.forward(o)
        return mean, log_std.exp()

    def sample(self, o):
        """raw action a~ を tanh-squashed Gaussian からサンプル (式(84))。
        return a_tilde, log_prob, pre_tanh_entropy。"""
        mean, std = self.dist(o)
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        a = torch.tanh(u)
        # tanh 補正付き log-prob
        logp = normal.log_prob(u).sum(-1)
        logp -= torch.log(1.0 - a.pow(2) + 1e-6).sum(-1)
        # エントロピー(ガウス分): H(pi) の近似 (式(128))
        ent = normal.entropy().sum(-1)
        return a, logp, ent

    def log_prob(self, o, a):
        """与えられた raw action a の log-prob を再評価 (勾配計算用)。"""
        mean, std = self.dist(o)
        normal = torch.distributions.Normal(mean, std)
        a_clip = torch.clamp(a, -1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(a_clip)
        logp = normal.log_prob(u).sum(-1)
        logp -= torch.log(1.0 - a_clip.pow(2) + 1e-6).sum(-1)
        ent = normal.entropy().sum(-1)
        return logp, ent


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, o):
        return self.net(o).squeeze(-1)
