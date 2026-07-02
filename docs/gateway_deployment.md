# Raspberry Pi ゲートウェイ 運用・デプロイ手順書

Tello ゲートウェイ v2（自律セッション管理）の展開・確認・運用手順。
対象は 2026-07-02 に実装した フェーズ1（自動起動 + keepalive + リンク死活）と
フェーズ2（`tello_fleet` 一本化 + `fleet` CLI）。

---

## 1. 構成

```
[母艦PC] ──有線LAN── [ラズパイ×N] ──Wi-Fi── [Tello×N]
   │                     │
   │ TCP(JSON-lines)     │ 各Piは自分のTelloを 192.168.10.1 として見る
   │ :10000              │ UDP 8889(コマンド)/8890(state)
   └ pi_monitor / fleet CLI で各Piの status を読む
```

- **Pi 上で常駐**するのが `tello_gateway.py`（systemd サービス）。電源ONから無人で起動し、
  Tello を SDK モードに保ち、アイドル時も keepalive でつなぎ続ける。
- **PC 上で動かす**のが監視・制御ツール（`pi_monitor.py` / `tello_fleet`）。Pi とは
  独立で、TCP 越しに状態を読むだけ。

### どのファイルがどこで動くか

| ファイル | 実行場所 | 役割 |
|---|---|---|
| `raspberry_pi/tello_gateway.py` | 各ラズパイ | ゲートウェイ本体（常駐） |
| `raspberry_pi/tello-gateway.service` | 各ラズパイ | systemd ユニット（自動起動） |
| `raspberry_pi/install.sh` | 各ラズパイ（1回） | サービス設置スクリプト |
| `python/pi_monitor.py` | 母艦PC | 全機ライブ監視 |
| `python/tello_fleet/` | 母艦PC | 制御ライブラリ + `fleet` CLI |

### Pi 上の固定配置（自動検出しない）

`install.sh` はソースを次の**固定パス**へコピーして登録する。ユーザー名やリポジトリ構造に依存しない。

| 実体 | Pi 上の固定パス |
|---|---|
| ゲートウェイ本体 | `/opt/tello-gateway/tello_gateway.py` |
| systemd ユニット | `/etc/systemd/system/tello-gateway.service` |
| 実行ユーザー | **root**（`iw power_save off` を追加権限なしで実行するため） |

> **重要**: リポジトリを更新しただけでは Pi 上の常駐コードは変わらない。
> 必ず「各 Pi にコピー → `install.sh` 実行」で `/opt/tello-gateway/` に反映すること（→ §2）。

---

## 2. デプロイ手順（各 Pi で1回ずつ）

### 2.1 ソースを Pi に置く
`raspberry_pi/` の3ファイル（`tello_gateway.py` / `tello-gateway.service` / `install.sh`）を
**1つのフォルダにまとめて**、Pi 上の任意の場所に置く。置き場所は何でもよい（`install.sh` が
そこから固定パス `/opt/tello-gateway/` へコピーする）。例（rsync でフォルダごと）:

```bash
# 母艦PCから各Piへ（例: ホーム直下に raspberry_pi/ を置く）
rsync -av raspberry_pi/ inulab@192.168.50.101:~/raspberry_pi/
rsync -av raspberry_pi/ inulab@192.168.50.102:~/raspberry_pi/
rsync -av raspberry_pi/ inulab@192.168.50.103:~/raspberry_pi/
```

### 2.2 サービスを設置・起動（各 Pi 上で）

```bash
ssh inulab@192.168.50.101
sudo bash ~/raspberry_pi/install.sh
```

`install.sh` が行うこと:
1. `tello_gateway.py` を **`/opt/tello-gateway/tello_gateway.py`** にコピー。
2. `tello-gateway.service` を `/etc/systemd/system/` に設置。
3. `systemctl daemon-reload && enable && restart`（失敗カウンタもリセット）。

実行時に設置先が表示される:
```
Installing:
  gateway -> /opt/tello-gateway/tello_gateway.py
  unit    -> /etc/systemd/system/tello-gateway.service
```

再デプロイ（コード更新後）も同じコマンドでよい（冪等。`/opt` のコピーが更新される）。

---

## 3. 動作確認（Pi 上で）

```bash
systemctl status tello-gateway          # 起動しているか
journalctl -u tello-gateway -f          # ログをリアルタイム表示
```

正常時に流れるログ例:

```
[hh:mm:ss] [gateway] starting: host=pi-1 listen=0.0.0.0:10000 tello=192.168.10.1 keepalive=10.0s
[hh:mm:ss] [gateway] listening on 0.0.0.0:10000
[hh:mm:ss] [gateway] TELLO LINK UP            ← Telloと疎通
```

Tello の電源を切る / 電波が切れると:

```
[hh:mm:ss] [gateway] TELLO LINK LOST          ← 切断を検知
[hh:mm:ss] [gateway] TELLO LINK RESTORED      ← 復帰
```

---

## 4. PC 側からの監視・チェック

### 4.1 ライブ監視（常時表示）

```bash
python python/pi_monitor.py --config configs/pc_real_config.example.json
```

- 全機を `UP` / `DOWN` / `UNREACHABLE` + バッテリ + 最終応答経過で1行表示。
- リンクが落ちた/戻った瞬間に `*** [hh:mm:ss] tello-1: TELLO LINK LOST ***` を出力。
- `UNREACHABLE` = Pi/サービス自体に届かない（Pi が落ちている等）。`DOWN` = Pi は生きているが
  Tello との無線が切れている、の区別。

### 4.2 離陸前チェック（ワンショット）

`python/` ディレクトリから実行:

```bash
cd python
python -m tello_fleet doctor --config ../configs/pc_real_config.example.json
python -m tello_fleet status --config ../configs/pc_real_config.example.json
```

- `doctor`: 各機が「到達可能・SDKモード・リンクUP・バッテリ十分」かを PASS/FAIL 判定。
  1機でも未readyなら終了コード非0（スクリプトから前提チェックに使える）。
- `status`: 現在状態を一度だけ表形式で表示。

---

## 5. 仕組み（電源ONから）

```
[電源ON]
 └ systemd が tello-gateway.service を起動 (After=network-online.target, Restart=always)
      ├ ExecStartPre: iw wlan0 set power_save off   … Wi-Fi省電力による切断を防止(root実行)
      └ ExecStart: tello_gateway.py --auto-command --keepalive 10 --link-timeout 4 ...
           ├ 起動時に "command" を送信 → SDKモードへ
           └ スレッド4本:
                ・tello-response : Telloのコマンド応答(UDP)を受信
                ・tello-state    : Telloのstateストリームを受信（生存信号）
                ・watchdog       : 飛行中にrc途絶→hover→land（従来の安全弁）
                ・keepalive      : ★v2で追加（下記）
```

### keepalive スレッド（1秒ごとに評価）
- **電源オフ対策**: アイドル時（非飛行 かつ 直近PCコマンドから10秒以上）だけ、10秒周期で
  `battery?` を送信。これで Tello の SDK 自動離脱・アイドル自動電源オフを防ぐ。
  飛行中は rc ストリームが keepalive を兼ねるので送らない（制御ループに割り込まない）。
- **リンク死活判定**: 「stateストリーム or 応答の最終受信時刻」が `link_timeout`（既定4秒）を
  超えたら `down`、戻れば `up`。遷移を journald に記録。落ちている間は `command` を再送して
  自動復帰を試みる。
- **状態公開**: `status` 応答に `tello_link / battery / uptime_s / last_seen_age_s` を追加。

### 障害時ポリシー（フェーズ2）
飛行実験中に外部トラッカのテレメトリが途絶した場合、`RealTelloEnv` は
**全機着陸**（`Fleet.land_all_safe()`）を実行する。1機でも“目隠し”状態で編隊を続けるより、
安全とデータ整合性を優先する方針（`tello_fleet.Fleet` が全機に land を送る）。
Pi 側 watchdog（rc途絶→hover→land）は最後の安全弁として従来どおり残す。

---

## 6. 運用コマンド

```bash
sudo systemctl restart tello-gateway    # 設定/コード変更後の再起動
sudo systemctl stop    tello-gateway    # 手動制御のため一時停止
sudo systemctl start   tello-gateway    # 再開
sudo systemctl disable tello-gateway    # 起動時の自動起動をやめる
sudo systemctl enable  tello-gateway    # 自動起動を戻す
```

---

## 7. ゲートウェイ起動オプション（`tello_gateway.py`）

| オプション | 既定 | 意味 |
|---|---|---|
| `--listen-host` | `0.0.0.0` | TCP待受アドレス |
| `--listen-port` | `10000` | TCP待受ポート |
| `--tello-ip` | `192.168.10.1` | Tello のIP |
| `--local-cmd-port` | `8889` | コマンド送信/応答用UDP |
| `--state-port` | `8890` | state受信用UDP |
| `--rc-default-limit` | `30` | rc の絶対値上限の既定 |
| `--hover-timeout` | `0.35` | 飛行中この秒数rc途絶で hover |
| `--land-timeout` | `1.50` | 飛行中この秒数rc途絶で land |
| `--keepalive` | `10` | アイドル時 keepalive 周期（0で無効） |
| `--link-timeout` | `4` | この秒数 生存信号なしで link down |
| `--auto-command` | — | 起動時に `command` を1回送る |

サービスの既定引数は `raspberry_pi/tello-gateway.service` の `ExecStart` にある。恒久的に変える場合は
このソースを編集して `install.sh` を再実行（`/opt` とユニットが更新される）。Pi 上で一時的に変える
だけなら `sudo systemctl edit --full tello-gateway` で編集し `restart`。

---

## 8. トラブルシューティング

| 症状 | 確認 |
|---|---|
| PCから `UNREACHABLE` | Pi が起動しているか、有線LAN/IP、`systemctl status tello-gateway` |
| `UNREACHABLE`だがPiは生存 | サービス落ち → `journalctl -u tello-gateway -e` でスタックトレース確認 |
| `DOWN` が続く | wlan0 が Tello SSID に接続済みか、Tello の電源、電池切れ |
| 起動時に `DOWN`/`waiting for Tello link...` | Tello 未起動 or 未接続。電源投入で自動 `UP` に上がる |
| Tello が勝手に電源オフ | `--keepalive` が効いているか。実測より周期が長ければ短くする（→ §9） |
| power_save が offにならない | `iw dev wlan0 get power_save` で確認（サービスは root 実行）。ドライバ非対応なら無視されることがある |
| `No such file or directory` で起動失敗 | `/opt/tello-gateway/tello_gateway.py` が在るか。無ければ `install.sh` を再実行 |

---

## 9. 未確定事項（要実測）

Tello が完全アイドルで自動電源オフになる正確な秒数。keepalive 10秒周期なら十分下回る想定だが、
一度実測して `--keepalive` を最終調整するとよい。

計測手順（1機で安全に）:
1. `--keepalive 0`（keepalive無効）でゲートウェイを起動し `command` で SDK モードに入れる。
2. 以後いっさい命令を送らず放置し、電源が落ちるまでの秒数を測る。
3. 得られた秒数の 1/3〜1/2 程度を `--keepalive` に設定（余裕を持たせる）。

---

## 10. 関連ファイル

- ゲートウェイ本体: `raspberry_pi/tello_gateway.py`
- systemd ユニット: `raspberry_pi/tello-gateway.service`
- 設置スクリプト: `raspberry_pi/install.sh`
- 監視CLI: `python/pi_monitor.py`
- 制御ライブラリ/CLI: `python/tello_fleet/`
- クイックリファレンス: `configs/pi_gateway_commands.txt`
