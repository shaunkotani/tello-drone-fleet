"""
原稿「事象駆動型分散安全強化学習に基づく複数ドローンによる協調取り囲み実験」
の数値実験設定 (8.1 表5, 3.2 式(27)-(29), 5.2 式(93)-(94), 5.3 式(103)-(104),
7.1 式(122)) を一箇所にまとめた設定。

* パラメータ値は変更可。問題設定・アルゴリズムは原稿に従う。
* Unity 側 (EnvConfig 相当) と Python 学習側 (TrainConfig) の両方で参照する。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
from typing import List


@dataclass
class EnvConfig:
    """環境(コマンドレベルモデル+取り囲み制御+CMDP)の設定。Unity 側と共有。"""
    # --- 規模・座標 (2節, 表5) ---
    N: int = 3                       # エージェント数 (実機初期検証は3)
    # 飛行領域 Omega = [xmin,xmax]x[ymin,ymax]x[zmin,zmax] [m]
    x_min: float = -2.5
    x_max: float = 2.5
    y_min: float = -2.5
    y_max: float = 2.5
    z_min: float = 0.0
    z_max: float = 2.0
    h_ref: float = 1.0               # 目標高度 [m]
    R: float = 0.9                   # 取り囲み半径 [m] (N=3:0.9, N=5:1.1)

    # --- 制御周期・エピソード (表5) ---
    dt: float = 0.1                  # 制御周期 Delta t [s]
    H: int = 200                     # エピソード長 [step]
    gamma: float = 0.99              # 割引率

    # --- コマンドレベルモデル (3.2 式(27)-(29)) ---
    tau_v: float = 0.25              # 水平速度応答時定数 [s]
    tau_z: float = 0.30              # 上下速度応答時定数 [s]
    tau_psi: float = 0.20            # yaw 角速度応答時定数 [s]
    v_max_fb: float = 0.4            # 前後最大速度 [m/s]
    v_max_lr: float = 0.4            # 左右最大速度 [m/s]
    v_max_ud: float = 0.25           # 上下最大速度 [m/s]
    omega_max_psi: float = math.radians(60.0)  # yaw 最大角速度 [rad/s]
    sigma_sdk_lr: int = +1           # 左右符号規約 (3.2/10.4): 機体左方向を正とする

    # ノイズ (8.6 domain randomization 用; 0 で無効)
    wind_acc_std: float = 0.0        # 風外乱の水平加速度ノイズ [m/s^2]
    meas_pos_noise: float = 0.0      # 位置計測ノイズ標準偏差 [m]
    meas_yaw_noise: float = 0.0      # yaw 計測ノイズ標準偏差 [rad]
    comm_delay_steps: int = 0        # 行動適用の遅延 [step]

    # --- 取り囲み制御 (4節) ---
    case: int = 1                    # 1: 正多角形, 2: 進行方向依存配置
    alpha0: float = 0.0              # スロット1の方向オフセット (式(43))
    v_eps: float = 0.05              # 進行方向角を更新する速度しきい値 (式(41))
    lambda_phi: float = 0.3          # 進行方向角ローパス係数 (式(42), 0<lam<=1)
    v_scale: float = 0.3             # Case2 無次元化代表速度 (式(45))
    cv: float = 2.0                  # Case2 速度依存指数係数 (>1, 式(45))
    alpha_min: float = 0.6           # Case2 最小角度間隔 (式(46)-(47))
    Kp_slot: float = 1.0             # 参照速度のスロット誤差ゲイン (式(53))
    yaw_ref_mode: str = "look_target"  # "look_target"(式90下) or "heading"(式90上)

    # --- 観測正規化 (2節, 5.1 式(82)) ---
    R_max: float = 3.0               # 相対位置正規化長 [m]
    v_max_norm: float = 0.4          # 水平速度正規化 [m/s] (= v_max_fb)
    vo_max_norm: float = 0.3         # 対象物速度正規化 [m/s]
    vz_max_norm: float = 0.25        # 上下速度正規化 [m/s]
    d_wall_max: float = 2.5          # 壁距離特徴量正規化長 [m]
    h_scale: float = 1.0             # 高度誤差正規化 [m]
    n_max_obs: int = -1              # 観測近傍数 (<0 なら N-1 に自動設定)

    # --- 目的コスト重み (5.2 式(93)-(94)) ---
    w_slot: float = 5.0
    w_R: float = 1.0
    w_c: float = 1.0
    w_e: float = 1.0
    w_v: float = 0.2
    w_u: float = 1.0e-3
    w_du: float = 1.0e-2
    w_h: float = 3.0
    w_psi: float = 0.1

    # --- 安全制約 (5.3 式(103)-(104)) ---
    d_drone: float = 0.50            # 機体間安全距離 [m]
    d_target: float = 0.35           # 対象物との最小距離 [m]
    d_safe_wall: float = 0.35        # 壁からの安全距離 [m]
    e_h: float = 0.20                # 許容高度誤差 [m]
    a_safe: float = 0.8              # 安全入力範囲 (asafe<1)
    d_i: List[float] = field(default_factory=lambda: [0.02, 0.01, 0.02, 0.02, 0.02])
    m_i: int = 5                     # 制約成分数

    # --- 安全シールド (5.1: 数値実験では既定 OFF = 恒等写像 S_i) ---
    use_safety_shield: bool = False

    # --- 対象物軌道 (8.2-8.4 式(136),(139),(140)) ---
    target_mode: str = "static"      # "static","const_vel","circle","polyline"
    v_tar: float = 0.15              # 等速/折れ線の対象物速度 [m/s]
    circle_A: float = 0.8            # 円軌道半径 [m]
    circle_omega: float = 0.2        # 円軌道角速度 [rad/s]
    poly_K_seg: int = 40             # 折れ線各区間のステップ数

    # --- 終了条件 ---
    terminal_penalty: float = 10.0   # 危険終了時の終端ペナルティ(評価補完用)

    def resolved_n_max_obs(self) -> int:
        return (self.N - 1) if self.n_max_obs < 0 else self.n_max_obs


@dataclass
class TrainConfig:
    """学習(7節 Algorithm 1, 9.4 Algorithm 3)の設定。"""
    Q: int = 2000                    # 学習イテレーション数
    N_roll: int = 8                  # ロールアウト数 (表5: 4-16)
    hidden: int = 128                # MLP 隠れ層 (表5: input-128-128-*)
    # primal/dual ステップサイズ (表5)
    eta_theta0: float = 1.0e-3       # REINFORCE: eta = eta0 / sqrt(q+1)
    eta_mu: float = 2.0e-3           # dual stepsize
    mu_max: float = 100.0            # 乗数上限 (式(115))
    R_X: float = 1.0e3               # 方策パラメータ射影球半径 (式(109))
    # 価値関数 (式(130))
    value_lr: float = 3.0e-3
    value_steps: int = 5
    # エントロピー正則化 (式(128))
    beta_ent0: float = 1.0e-3
    beta_ent_decay: float = 0.999
    # イベントトリガしきい値 (式(122))
    eps_theta0: float = 0.05
    eps_theta_min: float = 0.0
    eps_mu0: float = 0.05
    eps_mu_min: float = 0.0
    rho_trigger: float = 0.999       # 0<rho<1
    # 通信グラフ (2節, 7.1): "cycle" または "complete"
    comm_graph: str = "cycle"
    weight_rule: str = "cycle"       # "cycle"(式119) or "metropolis"(式120)
    # ログ
    log_every: int = 10
    eval_every: int = 50
    seed: int = 0
    device: str = "cpu"


def to_dict(cfg) -> dict:
    return asdict(cfg)
