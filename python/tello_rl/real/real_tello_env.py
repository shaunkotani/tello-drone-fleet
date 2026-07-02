"""Real Tello environment using Raspberry Pi gateways.

This class has the same high-level API as MockEnv/UnityEnv:
    reset() -> obs
    step(raw_actions) -> obs, objective_costs, constraint_costs, executed_actions, done, info

Unlike MockEnv, ``step`` does not integrate a simulated command model.  It sends
commands to real Tello drones through Raspberry Pi gateways, then reads the next
state from an external tracker.
"""
from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np

from ..config import EnvConfig
from .. import command_model as cm
from .. import formation as fm
from .. import observation as obsmod
from .. import costs as ct
from .. import nsb

from .pi_client import PiTelloGroup
from .real_tracker import UdpJsonTracker, TrackerTimeout, RealTargetState
from .real_safety_shield import RealSafetyShield, RealSafetyConfig


class RealTelloEnv:
    """Real execution environment for N Tello drones.

    Parameters
    ----------
    cfg:
        Existing EnvConfig used by MockEnv/UnityEnv.
    pi_group:
        Group of Raspberry Pi gateway clients.  Its length must equal ``cfg.N``.
    tracker:
        External tracker object that returns real drone states and target state.
    rc_limit:
        Maximum absolute Tello SDK ``rc`` value sent by the gateways.
    safety:
        Optional action safety shield.  If omitted, a conservative shield is used.
    send_period_s:
        Time to wait after sending actions before reading the next tracker frame.
        If None, uses ``cfg.dt``.
    """

    def __init__(self, cfg: EnvConfig, pi_group: PiTelloGroup, tracker: UdpJsonTracker,
                 rc_limit: int = 30, safety: RealSafetyShield | None = None,
                 send_period_s: float | None = None, tracker_timeout_s: float = 0.5):
        self.cfg = cfg
        self.N = cfg.N
        if len(pi_group.clients) != self.N:
            raise ValueError(f"pi_group has {len(pi_group.clients)} clients, but cfg.N={self.N}")
        self.pi_group = pi_group
        self.tracker = tracker
        self.rc_limit = int(rc_limit)
        self.safety = safety or RealSafetyShield(cfg, RealSafetyConfig(rc_limit=self.rc_limit))
        self.send_period_s = cfg.dt if send_period_s is None else float(send_period_s)
        self.tracker_timeout_s = float(tracker_timeout_s)
        self.obs_dim = obsmod.obs_dim(cfg)
        self.act_dim = 4
        self.k = 0
        self.phi = 0.0
        self.states: List[cm.DroneState] = []
        self.battery = np.full(self.N, 100.0)
        self.ro = np.zeros(2)
        self.vo = np.zeros(2)
        self.slots = np.zeros((self.N, 2))
        self.r_cg_ref = np.zeros(2)
        self.edge_ref = np.zeros(self.N)
        self.a_prev = np.zeros((self.N, 4))
        self._last_tracker_stamp = 0.0

    # ------------------------------------------------------------------
    def connect(self) -> None:
        self.pi_group.connect_all()
        self.tracker.start()

    def close(self) -> None:
        try:
            self.pi_group.close_all()
        finally:
            self.tracker.stop()

    def takeoff(self) -> None:
        self.pi_group.takeoff_all()

    def land(self) -> None:
        self.pi_group.land_all()

    def emergency(self) -> None:
        self.pi_group.emergency_all()

    def hover(self) -> None:
        self.pi_group.hover_all()

    # ------------------------------------------------------------------
    def reset(self):
        """Synchronize environment state from the external tracker.

        This does not physically reset the drones.  Use takeoff/land outside this
        method as needed.
        """
        self.k = 0
        self.a_prev = np.zeros((self.N, 4))
        self.phi = 0.0
        self.hover()
        self._sync_from_tracker()
        self.slots, self.r_cg_ref = fm.compute_slots(self.ro, self.vo, self.phi, self.cfg)
        self.edge_ref = fm.edge_ref_lengths(self.slots)
        return self._all_obs()

    def _sync_from_tracker(self) -> None:
        states, target, battery = self.tracker.get_states(timeout_s=self.tracker_timeout_s)
        if len(states) != self.N:
            raise TrackerTimeout(f"tracker returned {len(states)} states, expected {self.N}")
        self.states = states
        self.ro = target.r.copy()
        self.vo = target.v.copy()
        self.battery = battery.copy()
        self._last_tracker_stamp = target.stamp

    def _all_obs(self) -> np.ndarray:
        return np.stack([
            obsmod.build_observation(i, self.states, self.ro, self.vo, self.slots,
                                     self.a_prev[i], self.battery[i], self.cfg)
            for i in range(self.N)
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, raw_actions: np.ndarray):
        cfg = self.cfg
        raw = np.asarray(raw_actions, dtype=float).reshape(self.N, 4)
        if not self.states:
            self._sync_from_tracker()

        states_prev = [s.copy() for s in self.states]
        r_prev = np.stack([s.r for s in states_prev])
        slots_prev = self.slots.copy()
        tracker_age = max(0.0, time.time() - self._last_tracker_stamp) if self._last_tracker_stamp else None

        # 1. Safety shield on real measured state.
        safety_result = self.safety.filter(raw, states_prev, target_r=self.ro, tracker_age_s=tracker_age)
        executed = safety_result.actions

        # 2. Commanded world velocity for cost evaluation.
        v_xy_cmd = np.stack([
            cm.action_to_world_vel_cmd(executed[i], states_prev[i].psi, cfg)
            for i in range(self.N)
        ])

        # 3. Send executed actions to all Pi gateways.
        self.pi_group.send_actions(executed, rc_limit=self.rc_limit)

        # 4. Wait one control period and read real next state from tracker.
        if self.send_period_s > 0:
            time.sleep(self.send_period_s)
        try:
            self._sync_from_tracker()
            tracker_ok = True
            tracker_error = ""
        except TrackerTimeout as exc:
            # Telemetry lost for at least one drone.  Fleet-wide safety policy:
            # land ALL drones (data-integrity / safety of a partially-blind
            # formation takes priority over continuing the experiment).
            self.pi_group.land_all_safe()
            tracker_ok = False
            tracker_error = str(exc)

        # If tracker failed, reuse previous states only to produce a well-formed reply.
        if not tracker_ok:
            self.states = states_prev

        # 5. Update heading and slots using the real target state.
        self.phi = fm.update_heading(self.phi, self.vo, cfg)
        self.slots, self.r_cg_ref = fm.compute_slots(self.ro, self.vo, self.phi, cfg)
        self.edge_ref = fm.edge_ref_lengths(self.slots)

        # 6. Reference velocity and costs.
        v_ref = nsb.reference_velocity(slots_prev, self.slots, r_prev, cfg)
        v_ref_clip = nsb.clip_reference(v_ref, cfg)
        r_next = np.stack([s.r for s in self.states])

        obj = np.zeros(self.N)
        con = np.zeros((self.N, cfg.m_i))
        for i in range(self.N):
            psi_ref = ct.yaw_reference(i, r_next, self.ro, self.phi, cfg)
            obj[i] = ct.objective_cost(
                i, r_next, self.ro, self.slots, self.r_cg_ref, self.edge_ref,
                v_xy_cmd[i], v_ref_clip[i], executed[i], self.a_prev[i],
                self.states[i].h, self.states[i].psi, psi_ref, cfg,
            )
            con[i] = ct.constraint_cost(i, r_next, self.ro, self.states[i].h, executed[i], cfg)

        self.a_prev = executed.copy()
        self.k += 1
        obs = self._all_obs()
        done, reason = self._check_done(r_next)
        if not tracker_ok:
            done, reason = True, "tracker_timeout"

        info = {
            "reason": reason,
            "tracker_error": tracker_error,
            "k": self.k,
            "ro": self.ro.copy(),
            "vo": self.vo.copy(),
            "r": r_next.copy(),
            "slots": self.slots.copy(),
            "edge_ref": self.edge_ref.copy(),
            "phi": self.phi,
            "safety_reasons": safety_result.reasons,
            "hard_stop": safety_result.hard_stop,
            "rc_limit": self.rc_limit,
        }
        return obs, obj, con, executed, done, info

    def _check_done(self, r: np.ndarray) -> Tuple[bool, str]:
        cfg = self.cfg
        for i, st in enumerate(self.states):
            if not (cfg.x_min <= r[i, 0] <= cfg.x_max and cfg.y_min <= r[i, 1] <= cfg.y_max):
                return True, "region"
            if not (cfg.z_min <= st.h <= cfg.z_max):
                return True, "altitude"
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if np.linalg.norm(r[i] - r[j]) < 0.15:
                    return True, "collision"
        if self.k >= cfg.H:
            return True, "time"
        return False, ""
