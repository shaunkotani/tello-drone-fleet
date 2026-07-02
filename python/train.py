#!/usr/bin/env python3
"""
9.4 Algorithm 3 メインループ (学習エントリ)。

  python train.py --env mock --target circle --case 1 --N 3 --Q 1000
  python train.py --env unity --host 127.0.0.1 --port 5005   # Unity を先に起動

MockEnv (純 Python) と UnityEnv (TCP/JSON) を切替可能。両者は同一インターフェース。
学習曲線 (目的コスト, 制約, 取り囲み誤差, 通信率) を CSV に出力する。
"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np

from tello_rl.config import EnvConfig, TrainConfig
from tello_rl.mock_env import MockEnv
from tello_rl.learner import Learner


def build_env_factory(args, ecfg):
    if args.env == "unity":
        from tello_rl.unity_env import UnityEnv
        return lambda: UnityEnv(ecfg, host=args.host, port=args.port)
    return lambda: MockEnv(ecfg, seed=args.seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["mock", "unity"], default="mock")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--case", type=int, default=1, choices=[1, 2])
    p.add_argument("--target", default="static",
                   choices=["static", "const_vel", "circle", "polyline"])
    p.add_argument("--H", type=int, default=200)
    p.add_argument("--Q", type=int, default=1000)
    p.add_argument("--N_roll", type=int, default=8)
    p.add_argument("--graph", default="cycle", choices=["cycle", "complete"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/run.csv")
    args = p.parse_args()

    ecfg = EnvConfig(N=args.N, case=args.case, target_mode=args.target, H=args.H)
    tcfg = TrainConfig(Q=args.Q, N_roll=args.N_roll, comm_graph=args.graph,
                       weight_rule=("cycle" if args.graph == "cycle" else "metropolis"),
                       seed=args.seed)

    learner = Learner(build_env_factory(args, ecfg), ecfg, tcfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["q", "mean_obj", "broadcast_ratio", "Eslot", "Eedge",
              "c1", "c2", "c3", "c4", "c5"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        for q in range(tcfg.Q):
            m = learner.train_iteration(q)
            cmax = m["constraint_max"]
            w.writerow([q, m["mean_obj"], m["broadcast_ratio"],
                        m["Eslot"], m["Eedge"], *cmax])
            if q % tcfg.log_every == 0 or q == tcfg.Q - 1:
                print(f"q={q:4d} obj={m['mean_obj']:.3f} "
                      f"Eslot={m['Eslot']:.3f} Eedge={m['Eedge']:.3f} "
                      f"bcast={m['broadcast_ratio']:.2f} "
                      f"cmax={np.round(cmax, 3).tolist()}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
