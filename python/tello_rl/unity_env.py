"""
Unity 環境への TCP/JSON クライアント。MockEnv と同一インターフェース
(reset() -> obs(N,od), step(raw) -> obs,obj,con,exe,done,info) を提供し、
Learner からそのまま差し替えられる。

使い方:
    env = UnityEnv(host="127.0.0.1", port=5005, cfg=ecfg)
    Learner(lambda: env, ecfg, tcfg)
Unity 側が先にサーバとして listen していること。
"""
from __future__ import annotations
import socket
import numpy as np
from .config import EnvConfig, to_dict
from . import protocol as proto


class UnityEnv:
    def __init__(self, cfg: EnvConfig, host: str = "127.0.0.1", port: int = 5005,
                 send_config: bool = True):
        self.cfg = cfg
        self.N = cfg.N
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if send_config:
            proto.send_msg(self.sock, {"cmd": "config", "config": to_dict(cfg)})
            proto.recv_msg(self.sock)  # ack

    def reset(self):
        proto.send_msg(self.sock, {"cmd": "reset"})
        rep = proto.recv_msg(self.sock)
        return np.asarray(rep["obs"], dtype=np.float32)

    def step(self, raw_actions: np.ndarray):
        raw = np.asarray(raw_actions, dtype=float).reshape(self.N, 4)
        proto.send_msg(self.sock, {"cmd": "step", "raw_actions": raw.tolist()})
        rep = proto.recv_msg(self.sock)
        obs = np.asarray(rep["obs"], dtype=np.float32)
        obj = np.asarray(rep["objective_costs"], dtype=float)
        con = np.asarray(rep["constraint_costs"], dtype=float)
        exe = np.asarray(rep["executed_actions"], dtype=float)
        done = bool(rep["done"])
        info = rep.get("info", {})
        # info の配列を numpy 化 (評価指標用)
        for key in ("r", "slots", "ro", "edge_ref"):
            if key in info and info[key] is not None:
                info[key] = np.asarray(info[key], dtype=float)
        return obs, obj, con, exe, done, info

    def close(self):
        try:
            proto.send_msg(self.sock, {"cmd": "close"})
        except Exception:
            pass
        self.sock.close()
