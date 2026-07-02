"""Command-level safety shield for real Tello experiments.

This is intentionally conservative.  It does not replace a formal CBF-QP safety
filter, but it gives a practical first layer for low-speed indoor tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np

from ..command_model import DroneState
from ..config import EnvConfig


@dataclass
class RealSafetyConfig:
    action_limit: float = 0.50          # clip normalized action to [-limit, limit]
    rc_limit: int = 30                  # default rc limit used by the Pi gateway
    d_drone_stop: float = 0.45          # hover if another drone is closer than this
    d_drone_warn: float = 0.60          # start damping motion toward neighbors
    d_target_stop: float = 0.30
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

            # Drone-drone proximity: stop first; then damp motion toward nearby drones.
            for j, sj in enumerate(states):
                if i == j:
                    continue
                dvec = st.r - sj.r
                dist = float(np.linalg.norm(dvec))
                if dist < scfg.d_drone_stop:
                    a[:2] = 0.0
                    v_world[:] = 0.0
                    reason_parts.append(f"drone_stop:{j}")
                    hard_stop = True
                    break
                if dist < scfg.d_drone_warn and dist > 1e-6:
                    # Remove velocity component that moves toward the neighbor.
                    toward_neighbor = -dvec / dist
                    comp = float(np.dot(v_world, toward_neighbor))
                    if comp > 0.0:
                        v_world = v_world - comp * toward_neighbor
                        reason_parts.append(f"drone_damp:{j}")

            # Target proximity.
            if target_r is not None:
                dtarget = float(np.linalg.norm(st.r - np.asarray(target_r, dtype=float)))
                if dtarget < scfg.d_target_stop:
                    a[:2] = 0.0
                    v_world[:] = 0.0
                    reason_parts.append("target_stop")

            # Wall constraints: remove components moving out of the safe box.
            if st.r[0] < cfg.x_min + scfg.d_wall_stop or st.r[0] > cfg.x_max - scfg.d_wall_stop:
                a[:2] = 0.0
                v_world[:] = 0.0
                reason_parts.append("wall_stop")
                hard_stop = True
            else:
                if st.r[0] < cfg.x_min + scfg.d_wall_warn and v_world[0] < 0.0:
                    v_world[0] = 0.0
                    reason_parts.append("wall_xmin_damp")
                if st.r[0] > cfg.x_max - scfg.d_wall_warn and v_world[0] > 0.0:
                    v_world[0] = 0.0
                    reason_parts.append("wall_xmax_damp")

            if st.r[1] < cfg.y_min + scfg.d_wall_stop or st.r[1] > cfg.y_max - scfg.d_wall_stop:
                a[:2] = 0.0
                v_world[:] = 0.0
                reason_parts.append("wall_stop")
                hard_stop = True
            else:
                if st.r[1] < cfg.y_min + scfg.d_wall_warn and v_world[1] < 0.0:
                    v_world[1] = 0.0
                    reason_parts.append("wall_ymin_damp")
                if st.r[1] > cfg.y_max - scfg.d_wall_warn and v_world[1] > 0.0:
                    v_world[1] = 0.0
                    reason_parts.append("wall_ymax_damp")

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
