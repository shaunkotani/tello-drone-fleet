"""Command-level safety shield for real Tello experiments.

This is intentionally conservative.  It does not replace a formal CBF-QP safety
filter, but it gives a practical first layer for low-speed indoor tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..command_model import DroneState
from ..config import EnvConfig


@dataclass
class RealSafetyConfig:
    action_limit: float = 0.50          # clip normalized action to [-limit, limit]
    rc_limit: int = 30                  # default rc limit used by the Pi gateway
    d_drone_stop: float = 0.45          # engage repel if another drone is closer than this
    d_drone_release: float = 0.55       # repel latch releases above this (hysteresis)
    d_drone_warn: float = 0.60          # start damping motion toward neighbors
    d_target_stop: float = 0.30
    repel_speed: float = 0.15           # escape / push-back speed [m/s]
    approach_gain: float = 1.0          # 接近速度上限の勾配 [1/s] (v_max = gain*(d-d_stop))
    brake_gain: float = 1.5             # 実測接近速度の超過に対する先行ブレーキ係数
    d_wall_stop: float = 0.25
    d_wall_warn: float = 0.45
    h_min_real: float = 0.45
    h_max_real: float = 1.50
    h_warn_margin: float = 0.15
    min_up_action: float = 0.15
    min_down_action: float = -0.15
    stale_tracker_zero_action: bool = True


@dataclass
class SafetyResult:
    actions: np.ndarray
    reasons: List[str]
    hard_stop: bool = False


class RealSafetyShield:
    """Conservative action filter using externally measured states."""

    def __init__(self, cfg: EnvConfig, scfg: RealSafetyConfig | None = None):
        self.cfg = cfg
        self.scfg = scfg or RealSafetyConfig()
        # 機体ごとの近接リペル ラッチ (d_drone_stop で係合し d_drone_release で解除)
        self._drone_latch: Dict[int, bool] = {}

    @staticmethod
    def _clip_action(a: np.ndarray, limit: float) -> np.ndarray:
        return np.clip(np.asarray(a, dtype=float), -float(limit), float(limit))

    @staticmethod
    def _action_to_world_xy(a: np.ndarray, yaw: float, v_max_fb: float, v_max_lr: float) -> np.ndarray:
        # a=[lr, fb, ud, yaw]
        vb_x = v_max_fb * float(a[1])
        vb_y = v_max_lr * float(a[0])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * vb_x - s * vb_y, s * vb_x + c * vb_y], dtype=float)

    @staticmethod
    def _world_xy_to_action(v: np.ndarray, yaw: float, v_max_fb: float, v_max_lr: float,
                            ud: float, yaw_action: float) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        fb = c * v[0] + s * v[1]
        lr = -s * v[0] + c * v[1]
        return np.array([
            lr / max(v_max_lr, 1e-6),
            fb / max(v_max_fb, 1e-6),
            ud,
            yaw_action,
        ], dtype=float)

    def filter(self, raw_actions: Sequence[Sequence[float]], states: Sequence[DroneState],
               target_r: np.ndarray | None = None, tracker_age_s: float | None = None) -> SafetyResult:
        cfg, scfg = self.cfg, self.scfg
        actions = np.asarray(raw_actions, dtype=float).reshape(len(states), 4).copy()
        actions = self._clip_action(actions, scfg.action_limit)
        reasons = ["ok" for _ in states]
        hard_stop = False

        if tracker_age_s is not None and tracker_age_s > 2.0 * cfg.dt and scfg.stale_tracker_zero_action:
            return SafetyResult(np.zeros_like(actions), ["tracker_stale" for _ in states], hard_stop=True)

        for i, st in enumerate(states):
            reason_parts: List[str] = []
            a = actions[i].copy()
            v_world = self._action_to_world_xy(a, st.psi, cfg.v_max_fb, cfg.v_max_lr)

            # --- 近接制約の同時解決 ------------------------------------------
            # 逐次適用 (機体間投影 -> target 押し出し) では、後段の押し出しが
            # 前段の投影を素通りして隣機へ向かう成分を再導入する欠陥があった
            # (対象物と隣機に挟まれた機体が隣機へ押し込まれる)。ここでは:
            #   (1) warn 圏の接近速度を距離比例で制限 (d_drone_stop でゼロ)。
            #       応答遅れ・運動量があっても stop 境界の手前で軟着陸的に減速する。
            #   (2) repel / target 押し出しを合成した後にも (1) と内向き除去を
            #       反復適用し、押し出しが他の制約を破る経路を塞ぐ。
            #   (3) repel と target 押し出しが打ち消し合うサンドイッチでは
            #       接線方向 (対象物円に沿って隣機から遠ざかる向き) へ逃がす。
            neigh: List[Tuple[float, int, np.ndarray]] = []
            for j, sj in enumerate(states):
                if i == j:
                    continue
                dvec = st.r - sj.r
                neigh.append((float(np.linalg.norm(dvec)), j, dvec))

            damped_js: set = set()

            def _cap_neighbor_approach(v: np.ndarray) -> np.ndarray:
                # コマンドの接近成分を gain*(d - d_stop) に制限する。さらに実測の
                # 接近速度 (トラッカ計測) が上限を超えている分は brake_gain 倍で
                # コマンド上限から差し引く = 先行ブレーキ。機体の応答遅れで
                # コマンドを絞るだけでは止まらない運動量をここで殺す。
                # 上限は距離に線形なので遠方ペアには自然に無効果 (warn 判定不要)。
                for dist, j, dvec in neigh:
                    if dist > 1e-6:
                        toward = -dvec / dist
                        closing = float(np.dot(st.v - states[j].v, toward))
                        allowed = scfg.approach_gain * max(0.0, dist - scfg.d_drone_stop)
                        cmd_allowed = allowed - scfg.brake_gain * max(0.0, closing - allowed)
                        comp = float(np.dot(v, toward))
                        if comp > cmd_allowed + 1e-9:
                            v = v - (comp - cmd_allowed) * toward
                            if dist < scfg.d_drone_warn:
                                damped_js.add(j)
                return v

            in_target = False
            outward = np.zeros(2)
            dtarget = float("inf")
            if target_r is not None:
                tvec = st.r - np.asarray(target_r, dtype=float)
                dtarget = float(np.linalg.norm(tvec))
                if dtarget < scfg.d_target_stop and dtarget > 1e-6:
                    in_target = True
                    outward = tvec / dtarget

            def _remove_target_inward(v: np.ndarray) -> np.ndarray:
                comp_in = float(np.dot(v, -outward))
                if comp_in > 0.0:
                    v = v + comp_in * outward
                return v

            latched = False
            away = np.zeros(2)
            j_min = -1
            if neigh:
                d_min_i, j_min, dvec_min = min(neigh, key=lambda t: t[0])
                release = max(scfg.d_drone_release, scfg.d_drone_stop)
                latched = self._drone_latch.get(i, False)
                if d_min_i < scfg.d_drone_stop:
                    latched = True
                elif d_min_i >= release:
                    latched = False
                self._drone_latch[i] = latched
                if latched and d_min_i > 1e-6:
                    away = dvec_min / d_min_i

            # (1) PD 速度へ接近上限と内向き除去を適用
            v_world = _cap_neighbor_approach(v_world)
            if in_target:
                v_world = _remove_target_inward(v_world)
            # (2) 押し出しの合成と反復投影
            if latched:
                v_world = scfg.repel_speed * away  # PD は破棄して離反を最優先
                reason_parts.append(f"drone_repel:{j_min}")
                hard_stop = True
            if in_target:
                push = scfg.repel_speed * (1.0 - dtarget / scfg.d_target_stop)
                v_world = v_world + push * outward
                reason_parts.append("target_repel")
            for _ in range(2):
                v_world = _cap_neighbor_approach(v_world)
                if in_target:
                    v_world = _remove_target_inward(v_world)
            # (3) サンドイッチ: 打ち消し合ったら接線方向へ滑らせる
            if latched and in_target and float(np.linalg.norm(v_world)) < 0.05:
                tangent = np.array([-outward[1], outward[0]])
                if float(np.dot(tangent, away)) < 0.0:
                    tangent = -tangent
                v_world = _cap_neighbor_approach(scfg.repel_speed * tangent)
                reason_parts.append("target_slide")

            # Wall constraints: stop 圏では安全ボックス内向きの速度を強制し
            # (凍結ではなく押し戻し)、warn 圏では外向き成分を除去する。
            if st.r[0] < cfg.x_min + scfg.d_wall_stop:
                v_world[0] = max(v_world[0], scfg.repel_speed)
                reason_parts.append("wall_push_xmin")
                hard_stop = True
            elif st.r[0] > cfg.x_max - scfg.d_wall_stop:
                v_world[0] = min(v_world[0], -scfg.repel_speed)
                reason_parts.append("wall_push_xmax")
                hard_stop = True
            else:
                if st.r[0] < cfg.x_min + scfg.d_wall_warn and v_world[0] < 0.0:
                    v_world[0] = 0.0
                    reason_parts.append("wall_xmin_damp")
                if st.r[0] > cfg.x_max - scfg.d_wall_warn and v_world[0] > 0.0:
                    v_world[0] = 0.0
                    reason_parts.append("wall_xmax_damp")

            if st.r[1] < cfg.y_min + scfg.d_wall_stop:
                v_world[1] = max(v_world[1], scfg.repel_speed)
                reason_parts.append("wall_push_ymin")
                hard_stop = True
            elif st.r[1] > cfg.y_max - scfg.d_wall_stop:
                v_world[1] = min(v_world[1], -scfg.repel_speed)
                reason_parts.append("wall_push_ymax")
                hard_stop = True
            else:
                if st.r[1] < cfg.y_min + scfg.d_wall_warn and v_world[1] < 0.0:
                    v_world[1] = 0.0
                    reason_parts.append("wall_ymin_damp")
                if st.r[1] > cfg.y_max - scfg.d_wall_warn and v_world[1] > 0.0:
                    v_world[1] = 0.0
                    reason_parts.append("wall_ymax_damp")

            # 壁の減衰が軸成分を消した結果、隣機への接近成分が相対的に残る角の
            # ケースに備えて接近上限をもう一度通し、安全上最優先のハードな
            # 壁押し戻しを最後に再適用する (衝突回避 > 視野維持 の優先順)。
            v_world = _cap_neighbor_approach(v_world)
            if st.r[0] < cfg.x_min + scfg.d_wall_stop:
                v_world[0] = max(v_world[0], scfg.repel_speed)
            elif st.r[0] > cfg.x_max - scfg.d_wall_stop:
                v_world[0] = min(v_world[0], -scfg.repel_speed)
            if st.r[1] < cfg.y_min + scfg.d_wall_stop:
                v_world[1] = max(v_world[1], scfg.repel_speed)
            elif st.r[1] > cfg.y_max - scfg.d_wall_stop:
                v_world[1] = min(v_world[1], -scfg.repel_speed)

            for j in sorted(damped_js):
                reason_parts.append(f"drone_damp:{j}")

            # Altitude constraints.
            if st.h < scfg.h_min_real:
                a[2] = max(a[2], scfg.min_up_action)
                reason_parts.append("altitude_low")
            elif st.h < scfg.h_min_real + scfg.h_warn_margin:
                a[2] = max(a[2], 0.0)
                reason_parts.append("altitude_low_damp")

            if st.h > scfg.h_max_real:
                a[2] = min(a[2], scfg.min_down_action)
                reason_parts.append("altitude_high")
            elif st.h > scfg.h_max_real - scfg.h_warn_margin:
                a[2] = min(a[2], 0.0)
                reason_parts.append("altitude_high_damp")

            # Reconstruct action after any world-velocity damping.
            a2 = self._world_xy_to_action(v_world, st.psi, cfg.v_max_fb, cfg.v_max_lr, a[2], a[3])
            actions[i] = self._clip_action(a2, scfg.action_limit)
            reasons[i] = "+".join(reason_parts) if reason_parts else "ok"

        return SafetyResult(actions, reasons, hard_stop=hard_stop)
