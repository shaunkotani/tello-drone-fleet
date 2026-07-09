"""楕円 (アフィン正多角形) スロット配置 — 円形隊列とは独立した追加実装。

原稿の正多角形スロット (式(43)-(44), :mod:`formation`.slots_case1) に線形変換
diag(R_a, R_b) を掛けた族。R_a = R_b = R で円に退化するため理論上は上位互換だが、
既存の円形隊列コードを変更しないよう別モジュールとして提供する
(run_enclosing_demo.py の ``--shape ellipse`` から使用)。

軸は安全ボックス (= FOV 自動フィット結果) に合わせて世界座標軸に固定する。
静止対象物 (Case 1) 用。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .config import EnvConfig
from .real.real_safety_shield import RealSafetyConfig


@dataclass
class EllipseSpec:
    """楕円スロットのパラメータ。R_a: x 半径 [m], R_b: y 半径 [m], phase: 位相 [rad]。"""
    R_a: float
    R_b: float
    phase: float


def slots_ellipse(ro: np.ndarray, spec: EllipseSpec, N: int) -> np.ndarray:
    """楕円スロット slot_i = ro + [R_a cos a_i, R_b sin a_i], a_i = phase + 2πi/N。

    return shape (N, 2)。R_a = R_b なら formation.slots_case1 (phi + alpha0 = phase)
    と一致する。
    """
    out = np.zeros((N, 2))
    for i in range(N):
        a = spec.phase + 2.0 * math.pi * i / N
        out[i] = ro + np.array([spec.R_a * math.cos(a), spec.R_b * math.sin(a)])
    return out


def min_edge_distance(slots: np.ndarray, p: np.ndarray) -> float:
    """点 p からスロット多角形の各辺までの最小距離 (囲みの頑健性指標)。

    中心対称な楕円に内接する正多角形の像なので、p = 中心なら常に多角形内部にある。
    """
    N = len(slots)
    p = np.asarray(p, dtype=float)
    best = float("inf")
    for i in range(N):
        a, b = slots[i], slots[(i + 1) % N]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-12:
            return 0.0
        t = min(1.0, max(0.0, float(np.dot(p - a, ab)) / L2))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def plan_ellipse_fit(cfg: EnvConfig, scfg: RealSafetyConfig, center: np.ndarray,
                     aspect_max: float = 2.0, spacing_margin: float = 0.10,
                     target_margin: float = 0.10, n_phase: int = 31,
                     n_r: int = 24) -> Optional[EllipseSpec]:
    """安全ボックスに収まる楕円スロットを探索する。

    円形プランナ (plan_formation_fit) の楕円版。位相と (R_a, R_b) のグリッドを
    探索し、制約:
        - 全スロットが壁 warn マージンの内側
        - 全ペアの機体間隔 >= d_drone_warn + spacing_margin (楕円は間隔が不均一)
        - 各スロットの対象物距離 >= d_target_stop + target_margin
        - 軸比 max/min <= aspect_max (細長すぎる形の抑制)
        - 対象物から多角形の辺までの距離 >= 0.05m (囲みとして成立)
    を満たす中で、score = 辺距離 (囲み頑健性) + 0.3 * 最小機体間隔 を最大化する。
    実行不可能なら None。
    """
    m = scfg.d_wall_warn
    lo = np.array([cfg.x_min + m, cfg.y_min + m])
    hi = np.array([cfg.x_max - m, cfg.y_max - m])
    c = np.asarray(center, dtype=float)
    if np.any(c <= lo) or np.any(c >= hi):
        return None

    N = cfg.N
    d_pair_min = scfg.d_drone_warn + spacing_margin
    d_target_min = scfg.d_target_stop + target_margin
    aspect_max = max(1.0, float(aspect_max))

    best: Optional[EllipseSpec] = None
    best_score = -float("inf")
    for phase in np.linspace(0.0, 2.0 * math.pi / N, n_phase):
        dirs = [(math.cos(phase + 2.0 * math.pi * k / N),
                 math.sin(phase + 2.0 * math.pi * k / N)) for k in range(N)]
        # 箱制約は軸ごとに分離可能: R_a*cos が x 範囲, R_b*sin が y 範囲に入る上限
        ra_box = float("inf")
        rb_box = float("inf")
        for cx, sy in dirs:
            if cx > 1e-9:
                ra_box = min(ra_box, (hi[0] - c[0]) / cx)
            elif cx < -1e-9:
                ra_box = min(ra_box, (lo[0] - c[0]) / cx)
            if sy > 1e-9:
                rb_box = min(rb_box, (hi[1] - c[1]) / sy)
            elif sy < -1e-9:
                rb_box = min(rb_box, (lo[1] - c[1]) / sy)
        if ra_box <= 0.0 or rb_box <= 0.0:
            continue
        for ra in np.linspace(ra_box / n_r, ra_box, n_r):
            for rb in np.linspace(rb_box / n_r, rb_box, n_r):
                if max(ra, rb) / min(ra, rb) > aspect_max:
                    continue
                spec = EllipseSpec(float(ra), float(rb), float(phase))
                slots = slots_ellipse(c, spec, N)
                d_pair = min(float(np.linalg.norm(slots[i] - slots[j]))
                             for i in range(N) for j in range(i + 1, N))
                if d_pair < d_pair_min:
                    continue
                if min(float(np.linalg.norm(s - c)) for s in slots) < d_target_min:
                    continue
                edge = min_edge_distance(slots, c)
                if edge < 0.05:
                    continue
                score = edge + 0.3 * d_pair
                if score > best_score:
                    best_score = score
                    best = spec
    return best
