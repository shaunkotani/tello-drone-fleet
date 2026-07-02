# Tello 協調取り囲み — 事象駆動型分散安全強化学習 実装

原稿「事象駆動型分散安全強化学習に基づく複数ドローンによる協調取り囲み実験」(1〜9章) に
沿った実装。**問題設定・アルゴリズムは原稿に従い**、パラメータ値のみ調整可能としている。

```
tello_enclosing/
├── python/                  # 学習側 (Algorithm 1, 3)
│   ├── train.py             # エントリ (Algorithm 3 メインループ)
│   └── tello_rl/
│       ├── config.py        # 設定 (表5, 式27-29,93-94,103-104,122)
│       ├── formation.py     # 取り囲み制御 (式40-52,88)
│       ├── command_model.py # コマンドレベルモデル (式16-26)
│       ├── nsb.py           # 参照速度・NSB (式53-76,91)
│       ├── costs.py         # 目的/制約コスト (式92,95-99)
│       ├── observation.py   # 局所観測 (式82)
│       ├── target.py        # 対象物軌道 (式136,139,140)
│       ├── mock_env.py      # 純Python環境 (Unity Algorithm 2 のミラー/参照仕様)
│       ├── networks.py      # 方策(tanh-Gaussian)・価値 (式83,84,126,130)
│       ├── agent.py         # エージェント変数 (表4)
│       ├── consensus.py     # 重み・トリガ (式119-122)
│       ├── learner.py       # 勾配推定+primal-dual更新 (式123-130, Algorithm 1)
│       ├── protocol.py      # TCP/JSON フレーミング (9.1)
│       └── unity_env.py     # Unity環境TCPクライアント (MockEnvと同一API)
└── unity/Scripts/           # 環境側 (Algorithm 2)
    ├── Protocol.cs          # protocol.py と対応
    ├── EnvConfig.cs         # config.py の EnvConfig ミラー
    ├── TelloAgent.cs        # command_model.py ミラー
    ├── TargetObject.cs      # target.py ミラー
    ├── FormationManager.cs  # formation.py ミラー
    ├── SafetyEvaluator.cs   # costs.py + nsb 参照速度 ミラー
    ├── ObservationBuilder.cs# observation.py ミラー
    ├── EnvManager.cs        # mock_env.py (Algorithm 2) ミラー
    ├── PythonBridge.cs      # TCPサーバ (config/reset/step/close)
    └── Logger.cs            # CSVログ (9.5)
```

## 設計方針

* **2プロセス構成 (9章)**: Unity = 環境(物理/コスト/観測の計算), Python = 学習。
* **CTDE**: 目的コスト `l_i`・制約コスト `C_i`・局所観測 `o_i` は全状態を知る環境側が計算して返す。
  方策の入力は `o_i` のみ (decentralized execution)。`raw action a~` と `executed action a=S_i(...)` を区別。
* **分散性**: 方策パラメータ `θ_i` と制約乗数 `μ_i` のみが通信グラフ `G_comm` の近傍間で
  差分型コンセンサス + イベントトリガにより更新・通信される。
* **MockEnv = Unity の参照仕様**: `mock_env.py` は `EnvManager.cs` と同じ計算順序のミラー。
  Unity 無しで Python 側を検証でき、同時に Unity 実装の正解になる。両環境は同一 API:
  `reset() -> obs(N,od)`, `step(raw) -> (obs, obj(N,), con(N,5), exe(N,4), done, info)`。

## 主要な式とコードの対応

| 原稿 | 内容 | 実装 |
|---|---|---|
| 式(18),(20) | 行動→世界速度指令 | `command_model.action_to_world_vel_cmd` / `TelloAgent.ActionToWorldVelCmd` |
| 式(21)-(26) | 一次遅れ+積分の前進 | `command_model.step_command_model` / `TelloAgent.Step` |
| 式(40)-(42) | 進行方向角ローパス | `formation.update_heading` / `FormationManager.UpdateHeading` |
| 式(43)-(52) | スロット(Case1/2) | `formation.slots_case1/2` / `FormationManager.ComputeSlots` |
| 式(53),(91) | 参照速度+飽和 | `nsb.reference_velocity/clip_reference` / `SafetyEvaluator` |
| 式(56)-(76) | 優先順位付きNSB(ベースライン) | `nsb.NSBController` |
| 式(82) | 局所観測 | `observation.build_observation` / `ObservationBuilder.Build` |
| 式(92) | 目的コスト | `costs.objective_cost` / `SafetyEvaluator.ObjectiveCost` |
| 式(95)-(99) | 制約コスト C_i∈R^5 | `costs.constraint_cost` / `SafetyEvaluator.ConstraintCost` |
| 式(101) | 正規化係数 κ_{γ,H} | `learner.kappa_gamma_H` |
| 式(119)-(122) | 重み・イベントトリガ | `consensus.py` |
| 式(123)-(130) | ロールアウト勾配推定 | `learner._agent_grad_and_dual` |
| Algorithm 1 | コンセンサス primal-dual | `learner.update` |
| Algorithm 2 | Unity 1ステップ | `mock_env.step` / `EnvManager.Step` |
| Algorithm 3 | 学習メインループ | `train.py` |
| 式(133)-(135) | 取り囲み誤差 | `learner.enclosing_metrics` |
| 式(141)-(144) | 通信(broadcast)率 | `learner.update` の triggers |

## 実行方法

### Python 単体 (MockEnv) — Unity 不要で検証

```bash
cd python
pip install torch numpy
python train.py --env mock --target circle --case 1 --N 3 --H 200 --Q 1000
# 静止/等速/折れ線: --target static|const_vel|polyline,  進行方向依存配置: --case 2
# 完全グラフ: --graph complete
```

### Unity と接続

1. Unity 側: `unity/Scripts/` を Unity プロジェクトへ配置。Package Manager で
   **Newtonsoft Json** (`com.unity.nuget.newtonsoft-json`) を追加。空の GameObject に
   `PythonBridge` をアタッチ (port=5005) して再生。
2. Python 側:
   ```bash
   python train.py --env unity --host 127.0.0.1 --port 5005
   ```
   Python が `config` を送って環境を初期化し、`reset`/`step` で学習を回す。

## 外部計測 (ArUco) — 実機 State Estimator フロントエンド (10.2-10.3)

天井カメラ + ArUco マーカで 3 台の Tello と対象物を計測し、世界座標系 (ENU:
水平 `r=[x,y]`, 高度 `h=z`, `yaw=psi`) の位置・速度・yaw を推定して、
`tello_rl.real.UdpJsonTracker` が受信する UDP JSON を毎フレーム送信する。

```
[天井カメラ] -> run_aruco_tracker.py (検出+姿勢推定+平滑化) --UDP JSON--> UdpJsonTracker -> Agent/Safety
```

| ファイル | 役割 |
|---|---|
| `python/tello_rl/real/aruco_tracker.py` | 検出・姿勢推定・world変換・平滑化・UDP送信のコア |
| `python/run_aruco_tracker.py` | カメラ→UDP のCLI (可視化付き) |
| `python/calibrate_camera.py` | カメラ内部校正 (ChArUco) + マーカ/ボード画像生成 |
| `configs/aruco_tracker_config.example.json` | 設定例 |

**座標系の決め方**: 床に世界原点マーカ (`world_marker_id`) を水平に置き、その
マーカ座標系をそのまま ENU 世界系とする。天井カメラは必ず床より上という拘束から
世界 z 軸を上向きに強制する (高度 `h` は常に正、床上の対象物は `h≈0`)。原点マーカが
毎フレーム見えない構成では、設定の `extrinsics` (R_wc, t_wc) を使う。

**手順**:
```bash
cd python
pip install opencv-contrib-python numpy
# 1) 校正ボード生成→等倍印刷→撮影して内部パラメータ校正
python calibrate_camera.py --make-board --out-image charuco_board.png
python calibrate_camera.py --capture --out configs/camera_calib.npz
# 2) 各 Tello/対象物/原点用マーカ画像を生成→印刷 (印刷後に実寸[m]を測り config に反映)
python calibrate_camera.py --make-markers --ids 0,1,2,5,10
# 3) トラッカ起動 (受信側 UdpJsonTracker は run_real_baseline.py 等が利用)
python run_aruco_tracker.py --config ../configs/aruco_tracker_config.example.json
```

OpenCV 4.7+ の新 ArUco API (`ArucoDetector`) に対応。姿勢は廃止された
`estimatePoseSingleMarkers` の代わりに `solvePnP(IPPE_SQUARE)` で推定。速度は
EMA + 有限差分で平滑化 (10.3、Kalman への差し替え可)。

## 3台取り囲みデモ (ArUco閉ループ, 10.7 段階4)

ArUco外部計測で得た全機・対象物の位置を使い、対象物の周りに半径 R の正三角形を
作る閉ループデモ。低レベル姿勢は使わず Pi ゲートウェイ経由で `rc a b c d` だけを送る。

```
[天井カメラ]->run_aruco_tracker.py --UDP--> UdpJsonTracker --> run_enclosing_demo.py --rc--> PiTelloGroup --> 各Pi --> 各Tello
```

制御は式(30) の PD スロット追従 + 高度保持 + 対象物注視 yaw。`RealSafetyShield`
(機体間/壁/対象物近接・高度・入力の抑制) を通してから送信。取り囲み誤差
`E_slot / E_R / E_edge` (式133-135) と最小機体間距離・高度をライブ表示・CSV保存する。

```bash
cd python
# 端末1: 外部計測 (前節)
python run_aruco_tracker.py --config ../configs/aruco_tracker_config.example.json
# 端末2: 取り囲みデモ。まず固定点(0,0)を囲むのが最も安全 (対象物マーカ不要)
python run_enclosing_demo.py --config configs/pc_real_config.example.json --center 0 0 --takeoff --duration 30
# ArUco対象物マーカを囲む場合は --center を外す
python run_enclosing_demo.py --config configs/pc_real_config.example.json --takeoff --duration 30
```

安全機構: 離陸前にバッテリ確認と対話式の安全確認 (`--yes` で省略)、トラッカ喪失で
hover→land、危険継続で land、Ctrl+C で安全停止。速度は既定 0.20 m/s に制限
(`--v-xy-limit`)。段階的移行 (10.7) では `--center` の固定点→ArUco対象物→低速移動、
と進める。

## インターフェース仕様 (9.1)

長さ前置き (4byte big-endian) + UTF-8 JSON over TCP。Unity=サーバ, Python=クライアント。

```
Python -> Unity : {"cmd":"config","config":{...}}      # 環境設定
                  {"cmd":"reset"}
                  {"cmd":"step","raw_actions":[[a4]xN]}
                  {"cmd":"close"}
Unity  -> Python: (reset) {"obs":[[od]xN],"info":{}}
                  (step)  {"obs":[[od]xN], "objective_costs":[N],
                           "constraint_costs":[[5]xN], "executed_actions":[[4]xN],
                           "done":bool, "info":{...全状態...}}
```

`ML-Agents` / `ZeroMQ` / `ROS-TCP-Connector` への置換も可能 (9.1)。

## 実装メモ・拡張点

* **コスト最小化として直接実装** (7.2 の注意): advantage は「コスト」なので primal は
  `θ ← z − η·g` の降下。報酬最大化ライブラリ流用時の符号反転は不要にしてある。
* **REINFORCE型** を既定とした (式123-130 に直接対応)。PPO 化する場合も Algorithm 1 の
  外側 (コンセンサス・dual・トリガ) はそのままで、`learner._agent_grad_and_dual` の
  primal 勾配計算だけクリップ付き比率に差し替えればよい。
* **安全シールド S_i** は数値実験では恒等写像 (5.1)。`mock_env._safety_shield` /
  `EnvManager.SafetyShield` をフックに、10章の command-level CBF-QP を後から差し込める。
* **乗数スタック** `μ_i ∈ R^{N·m}` は全制約乗数の局所コピー。自分の更新は i ブロック
  (`Agent.own_block`) のみ、コンセンサスは全スタックを混合 (表4, Algorithm 1)。
* パラメータは `config.py` / `EnvConfig.cs` に集約。値は表5を初期値とし調整可能。
