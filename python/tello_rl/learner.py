"""
7 事象駆動型コンセンサス primal-dual 学習 (Algorithm 1) + 7.2 ロールアウト勾配推定 (式(123)-(130))

* primal 方向 g_hat_i (式(127),(128)): コスト advantage に基づく局所 actor 勾配 (descent)。
* dual   方向 h_hat_i (式(124)): 正規化割引制約和 - d_i。自ブロックへ。
* 差分型コンセンサス混合 (式 step7) + projected descent-ascent (step9) + イベントトリガ (step11)。

注意 (7.2): A は「コスト」advantage なので primal は descent (theta <- z - eta*g)。
報酬最大化型を流用する場合の符号反転をここでは避け、コスト最小化として直接実装する。
"""
from __future__ import annotations
import numpy as np
import torch
from .config import EnvConfig, TrainConfig
from .agent import Agent
from . import consensus as cs
from . import observation as obsmod


def kappa_gamma_H(gamma: float, H: int) -> float:
    """正規化係数 kappa_{gamma,H} (式(101))。"""
    if gamma >= 1.0:
        return 1.0 / H
    return (1.0 - gamma) / (1.0 - gamma ** H)


class Learner:
    def __init__(self, env_factory, ecfg: EnvConfig, tcfg: TrainConfig):
        self.ecfg = ecfg
        self.tcfg = tcfg
        self.env = env_factory()
        torch.manual_seed(tcfg.seed)
        np.random.seed(tcfg.seed)
        self.N = ecfg.N
        self.m = ecfg.m_i
        od = obsmod.obs_dim(ecfg)
        self.agents = [Agent(i, od, 4, self.N, self.m, tcfg.hidden,
                             tcfg.value_lr, tcfg.device) for i in range(self.N)]
        self.nbr = cs.build_graph(self.N, tcfg.comm_graph)
        self.omega = cs.mixing_coeffs(self.N, self.nbr, tcfg.weight_rule)
        self.kappa = kappa_gamma_H(ecfg.gamma, ecfg.H)
        self.d_i = np.asarray(ecfg.d_i, dtype=float)
        self.device = tcfg.device

    # ------------------------------------------------------------------
    def collect_rollouts(self):
        """Algorithm 3 inner: N_roll エピソードをオンポリシーで収集。"""
        ecfg, tcfg = self.ecfg, self.tcfg
        eps = []
        for _ in range(tcfg.N_roll):
            obs = self.env.reset()
            O, A, L, C = [], [], [], []
            POS, SLOT, EDGE = [], [], []
            done = False
            while not done:
                ot = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                raw = np.zeros((self.N, 4))
                with torch.no_grad():
                    for i in range(self.N):
                        a_i, _, _ = self.agents[i].policy.sample(ot[i:i + 1])
                        raw[i] = a_i.cpu().numpy()[0]
                nobs, obj, con, exe, done, info = self.env.step(raw)
                O.append(obs); A.append(raw); L.append(obj); C.append(con)
                POS.append(info.get("r")); SLOT.append(info.get("slots"))
                EDGE.append(info.get("edge_ref"))
                obs = nobs
            eps.append(dict(
                obs=np.asarray(O), act=np.asarray(A), lcost=np.asarray(L),
                ccost=np.asarray(C), pos=POS, slot=SLOT, edge=EDGE, T=len(O)))
        return eps

    # ------------------------------------------------------------------
    def _agent_grad_and_dual(self, i, eps, eta_ent):
        """エージェント i の primal 方向 g_hat_i (ベクトル) と dual 方向 h_hat_i を計算。"""
        ag = self.agents[i]
        ecfg = self.ecfg
        gamma = ecfg.gamma
        mu_block = ag.mu[ag.own_block()]                 # mu^(i)_i (式(125))

        # --- return R_hat と特徴を episode ごとに構築 ---
        obs_list, act_list, ret_list, gk_list = [], [], [], []
        Jc_eps = []                                      # dual 用 (式(123))
        for e in eps:
            T = e["T"]
            li = e["lcost"][:, i]                        # (T,)
            ci = e["ccost"][:, i, :]                     # (T,m)
            # stage cost: (1/N) l + kappa * mu^(i) . C   (式(125) の被加算項)
            stage = (1.0 / self.N) * li + self.kappa * (ci @ mu_block)
            R = np.zeros(T)
            acc = 0.0
            for k in range(T - 1, -1, -1):
                acc = stage[k] + gamma * acc
                R[k] = acc
            obs_list.append(e["obs"][:, i, :])
            act_list.append(e["act"][:, i, :])
            ret_list.append(R)
            gk_list.append(gamma ** np.arange(T))
            Jc_eps.append(self.kappa * np.sum(
                (gamma ** np.arange(T))[:, None] * ci, axis=0))   # 式(123)

        obs_all = torch.as_tensor(np.concatenate(obs_list), dtype=torch.float32,
                                  device=self.device)
        act_all = torch.as_tensor(np.concatenate(act_list), dtype=torch.float32,
                                  device=self.device)
        ret_all = torch.as_tensor(np.concatenate(ret_list), dtype=torch.float32,
                                  device=self.device)
        gk_all = torch.as_tensor(np.concatenate(gk_list), dtype=torch.float32,
                                 device=self.device)

        # --- advantage (式(126)) と正規化 ---
        with torch.no_grad():
            V = ag.value(obs_all)
        adv = ret_all - V                                # コスト advantage
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-6)

        # --- primal 勾配 (式(127)+(128)) ---
        logp, ent = ag.policy.log_prob(obs_all, act_all)
        n = obs_all.shape[0]
        pg_loss = (gk_all * adv_n.detach() * logp).sum() / self.tcfg.N_roll
        ent_loss = -eta_ent * (gk_all * ent).sum() / self.tcfg.N_roll
        loss = pg_loss + ent_loss
        ag.policy.zero_grad()
        loss.backward()
        g_vec = torch.cat([p.grad.reshape(-1) for p in ag.policy.parameters()])
        g_hat = g_vec.detach().cpu().numpy()

        # --- 価値関数更新 (式(130)) ---
        for _ in range(self.tcfg.value_steps):
            Vp = ag.value(obs_all)
            vloss = 0.5 * ((Vp - ret_all.detach()) ** 2).mean()
            ag.value_opt.zero_grad(); vloss.backward(); ag.value_opt.step()

        # --- dual 方向 (式(124)) ---
        h_hat = np.mean(np.stack(Jc_eps), axis=0) - self.d_i   # (m,)
        return g_hat, h_hat

    # ------------------------------------------------------------------
    def _project_ball(self, v):
        nrm = np.linalg.norm(v)
        if nrm <= self.tcfg.R_X:
            return v
        return v * (self.tcfg.R_X / nrm)

    def update(self, q, eps):
        """Algorithm 1 step 6-15 を全エージェントについて実行。"""
        tcfg = self.tcfg
        eta_theta = tcfg.eta_theta0 / np.sqrt(q + 1)
        eta_mu = tcfg.eta_mu
        eta_ent = tcfg.beta_ent0 * (tcfg.beta_ent_decay ** q)

        # step 6: 各エージェントの primal/dual 方向
        grads, duals = [], []
        for i in range(self.N):
            g, h = self._agent_grad_and_dual(i, eps, eta_ent)
            grads.append(g); duals.append(h)

        # step 7-9: コンセンサス混合 + projected descent-ascent
        theta_new, mu_new = [], []
        for i in range(self.N):
            ag = self.agents[i]
            theta_i = ag.get_theta_vec()
            # z_i (theta): 差分型混合 (式 step7)。theta_hat は最後に受信した値。
            z_theta = theta_i.copy()
            z_mu = ag.mu.copy()
            for j in self.nbr[i]:
                w = self.omega[(i, j)]
                z_theta += w * (self.agents[j].theta_hat - ag.theta_hat)
                z_mu += w * (self.agents[j].mu_hat - ag.mu_hat)
            # primal descent (式 step9)
            tn = self._project_ball(z_theta - eta_theta * grads[i])
            # dual ascent: E_i h_hat (自ブロックのみ) を加え clip (式(115),(116))
            E_h = np.zeros_like(ag.mu)
            E_h[ag.own_block()] = duals[i]
            mn = np.clip(z_mu + eta_mu * E_h, 0.0, tcfg.mu_max)
            theta_new.append(tn); mu_new.append(mn)

        # step 11-15: イベントトリガ (theta_hat/mu_hat 更新 = broadcast)
        eps_th = cs.trigger_threshold(tcfg.eps_theta0, tcfg.eps_theta_min,
                                      tcfg.rho_trigger, q)
        eps_mu = cs.trigger_threshold(tcfg.eps_mu0, tcfg.eps_mu_min,
                                      tcfg.rho_trigger, q)
        triggers = np.zeros(self.N)
        for i in range(self.N):
            ag = self.agents[i]
            d_th = np.linalg.norm(theta_new[i] - ag.theta_hat)
            d_mu = np.linalg.norm(mu_new[i] - ag.mu_hat)
            if d_th >= eps_th or d_mu >= eps_mu:         # 式(141)
                ag.theta_hat = theta_new[i].copy()
                ag.mu_hat = mu_new[i].copy()
                triggers[i] = 1.0

        # パラメータ適用
        for i in range(self.N):
            self.agents[i].set_theta_vec(theta_new[i])
            self.agents[i].mu = mu_new[i]

        return triggers

    # ------------------------------------------------------------------
    def enclosing_metrics(self, eps):
        """Eslot, ER, Eedge (式(133)-(135)) をロールアウト平均で算出。"""
        ecfg = self.ecfg
        Es, ER, Ee, cnt = 0.0, 0.0, 0.0, 0
        for e in eps:
            for k in range(e["T"]):
                if e["pos"][k] is None:
                    continue
                r = e["pos"][k]; sl = e["slot"][k]; ed = e["edge"][k]
                ro = None
                for i in range(self.N):
                    Es += np.linalg.norm(r[i] - sl[i])
                # ER は対象物中心。slots から ro を逆算できないため pos と slot のみで近似:
                # ここでは半径誤差を |‖r_i - centroid_slot‖ - R| で代用しない。
                ip = (np.arange(self.N) + 1) % self.N
                for i in range(self.N):
                    Ee += abs(np.linalg.norm(r[i] - r[ip[i]]) - ed[i])
                cnt += 1
        denom = max(cnt * self.N, 1)
        return dict(Eslot=Es / denom, Eedge=Ee / denom)

    def train_iteration(self, q):
        eps = self.collect_rollouts()
        triggers = self.update(q, eps)
        # メトリクス
        mean_obj = np.mean([e["lcost"].mean() for e in eps])
        # 正規化割引制約 (成分ごと最大, 式(123))
        Jc_max = np.zeros(self.m)
        for i in range(self.N):
            Jc = []
            for e in eps:
                T = e["T"]
                gk = self.ecfg.gamma ** np.arange(T)
                Jc.append(self.kappa * np.sum(gk[:, None] * e["ccost"][:, i, :], axis=0))
            Jc_max = np.maximum(Jc_max, np.mean(np.stack(Jc), axis=0))
        enc = self.enclosing_metrics(eps)
        return dict(q=q, mean_obj=float(mean_obj),
                    constraint_max=Jc_max.tolist(),
                    broadcast_ratio=float(triggers.mean()),
                    **enc)
