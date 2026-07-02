"""
純 Python 版の協調取り囲み環境。9.3 Algorithm 2 (Unity 1 ステップ処理) を Python で
ミラー実装したもの。Unity を用いずに Python 側 RL を検証でき、同時に Unity 側が
実装すべき計算の参照仕様にもなる。unity_env.UnityEnv と同一インターフェース。

CTDE: 目的コスト l_i / 制約コスト C_i / 局所観測 o_i を環境が全状態から計算して返す。
方策の入力は o_i のみ (decentralized execution)。raw action vs executed action を区別。
"""
from __future__ import annotations
import numpy as np
from .config import EnvConfig
from . import command_model as cm
from . import formation as fm
from . import costs as ct
from . import observation as obsmod
from .target import Target


class MockEnv:
    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.target = Target(cfg)
        self.N = cfg.N
        self.obs_dim = obsmod.obs_dim(cfg)
        self.act_dim = 4
        self._reset_internal()

    # ------------------------------------------------------------------
    def _reset_internal(self):
        cfg = self.cfg
        self.k = 0
        ro, vo = self.target.reset()
        self.ro, self.vo = ro, vo
        self.phi = 0.0
        self.slots, self.r_cg_ref = fm.compute_slots(ro, vo, self.phi, cfg)
        self.edge_ref = fm.edge_ref_lengths(self.slots)
        # ドローン初期化: スロット付近に微小ノイズ、高度 h_ref、ホバリング
        self.states = []
        for i in range(self.N):
            r0 = self.slots[i] + self.rng.normal(0, 0.05, size=2)
            d = ro - r0
            psi0 = np.arctan2(d[1], d[0])
            self.states.append(cm.DroneState(r0, np.zeros(2), cfg.h_ref, 0.0, psi0, 0.0))
        self.a_prev = np.zeros((self.N, 4))            # a_{i,-1}=0 (hover)
        self.battery = np.full(self.N, 100.0)

    def reset(self):
        self._reset_internal()
        return self._all_obs()

    # ------------------------------------------------------------------
    def _all_obs(self):
        return np.stack([
            obsmod.build_observation(i, self.states, self.ro, self.vo, self.slots,
                                     self.a_prev[i], self.battery[i], self.cfg)
            for i in range(self.N)])

    def _safety_shield(self, sk_states, raw):
        """安全シールド S_i (5.1)。数値実験では既定で恒等写像 (executed=clip(raw))。
        実機(10節)CBF-QP を入れる場合はここを差し替える。"""
        clipped = np.clip(raw, -1.0, 1.0)
        if not self.cfg.use_safety_shield:
            return clipped
        return clipped  # フック: ここに command-level 安全フィルタを実装可能

    # ------------------------------------------------------------------
    def step(self, raw_actions: np.ndarray):
        """raw_actions:(N,4)。Algorithm 2 の処理。
        return obs(N,od), obj_cost(N,), con_cost(N,5), executed(N,4), done, info"""
        cfg = self.cfg
        raw = np.asarray(raw_actions, dtype=float).reshape(self.N, 4)
        # 1. clip + 安全シールド → executed action
        a = self._safety_shield(self.states, raw)

        # 2. 目的コスト用に世界座標速度指令 v_xy_cmd (executed, 現時刻 psi)
        v_xy_cmd = np.stack([cm.action_to_world_vel_cmd(a[i], self.states[i].psi, cfg)
                             for i in range(self.N)])
        r_prev = np.stack([self.states[i].r for i in range(self.N)])

        # 3. ダイナミクス前進 (式(21)-(26))
        self.states = [cm.step_command_model(self.states[i], a[i], cfg, self.rng)
                       for i in range(self.N)]
        if cfg.meas_pos_noise > 0:                     # 計測ノイズ (8.6)
            for i in range(self.N):
                self.states[i].r += self.rng.normal(0, cfg.meas_pos_noise, size=2)

        # 4. 対象物更新
        slots_prev = self.slots
        self.ro, self.vo, _ = self.target.step()

        # 5. 進行方向角・スロット更新 (k+1)
        self.phi = fm.update_heading(self.phi, self.vo, cfg)
        self.slots, self.r_cg_ref = fm.compute_slots(self.ro, self.vo, self.phi, cfg)
        self.edge_ref = fm.edge_ref_lengths(self.slots)

        # 6. 参照速度 v_ref (式(53)) を有限差分で計算しクリップ (式(91))
        from . import nsb
        v_ref = nsb.reference_velocity(slots_prev, self.slots, r_prev, cfg)
        v_ref_clip = nsb.clip_reference(v_ref, cfg)

        r_next = np.stack([self.states[i].r for i in range(self.N)])

        # 7. 目的コスト・制約コスト
        obj = np.zeros(self.N)
        con = np.zeros((self.N, cfg.m_i))
        for i in range(self.N):
            psi_ref = ct.yaw_reference(i, r_next, self.ro, self.phi, cfg)
            obj[i] = ct.objective_cost(
                i, r_next, self.ro, self.slots, self.r_cg_ref, self.edge_ref,
                v_xy_cmd[i], v_ref_clip[i], a[i], self.a_prev[i],
                self.states[i].h, self.states[i].psi, psi_ref, cfg)
            con[i] = ct.constraint_cost(i, r_next, self.ro, self.states[i].h, a[i], cfg)

        # 8. 観測・前回入力更新
        self.a_prev = a.copy()
        self.k += 1
        obs = self._all_obs()

        # 9. 終了判定 (region/collision/time)
        done, reason = self._check_done(r_next)
        info = {"reason": reason, "k": self.k, "ro": self.ro.copy(),
                "r": r_next.copy(), "slots": self.slots.copy(),
                "edge_ref": self.edge_ref.copy(), "phi": self.phi}
        return obs, obj, con, a, done, info

    def _check_done(self, r):
        cfg = self.cfg
        for i in range(self.N):
            if not (cfg.x_min <= r[i, 0] <= cfg.x_max and cfg.y_min <= r[i, 1] <= cfg.y_max):
                return True, "region"
            if not (cfg.z_min <= self.states[i].h <= cfg.z_max):
                return True, "altitude"
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if np.linalg.norm(r[i] - r[j]) < 0.15:
                    return True, "collision"
        if self.k >= cfg.H:
            return True, "time"
        return False, ""
