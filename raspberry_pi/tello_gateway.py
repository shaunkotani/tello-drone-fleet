#!/usr/bin/env python3
"""Raspberry Pi gateway for one normal DJI Tello.

Run one instance on each Raspberry Pi.  The Pi must be connected as follows:
    eth0  -> experiment LAN / switching hub / desktop PC
    wlan0 -> the corresponding Tello Wi-Fi SSID

The desktop PC sends newline-delimited JSON over TCP to this gateway.  This
program converts normalized actions to Tello SDK UDP commands.

Minimal PC message examples:
    {"type":"command"}
    {"type":"takeoff"}
    {"type":"rc", "a":[0.0,0.2,0.0,0.0], "rc_limit":30}
    {"type":"land"}

Safety watchdog:
    If no rc command is received for --hover-timeout seconds, send rc 0 0 0 0.
    If no rc command is received for --land-timeout seconds, send land once.

Keepalive / link health:
    While idle (not flying, no recent PC traffic) the gateway sends ``battery?``
    every --keepalive seconds so the Tello does not leave SDK mode or auto power
    off.  The gateway tracks the Tello link state (up/down) from both the Tello
    state stream and command responses, logs LINK LOST / LINK RESTORED, and
    exposes it through the ``status`` message so a PC-side monitor can display it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import queue
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple


TELLO_ADDR = ("192.168.10.1", 8889)

# tello_link values.
LINK_INIT = "init"
LINK_UP = "up"
LINK_DOWN = "down"


@dataclass
class GatewayState:
    sdk_mode: bool = False
    armed: bool = False
    last_rc_time: float = 0.0
    last_pc_time: float = 0.0
    last_tello_response: str = ""
    last_tello_response_time: float = 0.0
    last_state: Dict[str, str] | None = None
    last_state_time: float = 0.0
    watchdog_landed: bool = False
    tello_link: str = LINK_INIT
    link_since: float = 0.0
    battery: str = ""
    consecutive_keepalive_fail: int = 0


class TelloGateway:
    def __init__(self, listen_host: str, listen_port: int, local_cmd_port: int,
                 state_port: int, rc_default_limit: int, hover_timeout: float,
                 land_timeout: float, tello_ip: str = "192.168.10.1",
                 keepalive_interval: float = 10.0, link_timeout: float = 4.0):
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.local_cmd_port = int(local_cmd_port)
        self.state_port = int(state_port)
        self.rc_default_limit = int(rc_default_limit)
        self.hover_timeout = float(hover_timeout)
        self.land_timeout = float(land_timeout)
        self.tello_addr = (tello_ip, 8889)
        self.keepalive_interval = float(keepalive_interval)
        self.link_timeout = float(link_timeout)
        self.state = GatewayState()
        self._state_lock = threading.Lock()
        self._running = threading.Event()
        self._cmd_sock: Optional[socket.socket] = None
        self._state_sock: Optional[socket.socket] = None
        self._server_sock: Optional[socket.socket] = None
        self._response_queue: "queue.Queue[Tuple[float, str]]" = queue.Queue(maxsize=20)
        self._send_lock = threading.Lock()
        self._last_send_time = 0.0
        self._min_sync_command_gap = 0.10
        self._start_time = time.time()

    # ------------------------------------------------------------------
    @staticmethod
    def _log(msg: str, *, err: bool = False) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] [gateway] {msg}", file=sys.stderr if err else sys.stdout, flush=True)

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running.set()
        self._open_udp_sockets()
        self._launch_threads()
        self._serve_tcp()

    def _launch_threads(self) -> None:
        threading.Thread(target=self._recv_tello_responses, name="tello-response", daemon=True).start()
        threading.Thread(target=self._recv_tello_state, name="tello-state", daemon=True).start()
        threading.Thread(target=self._watchdog_loop, name="watchdog", daemon=True).start()
        threading.Thread(target=self._keepalive_loop, name="keepalive", daemon=True).start()

    def stop(self) -> None:
        self._running.clear()
        for sock in (self._cmd_sock, self._state_sock, self._server_sock):
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass

    def _open_udp_sockets(self) -> None:
        cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binding to 8889 is common for Tello SDK replies. If it fails, choose an ephemeral port.
        try:
            cmd_sock.bind(("", self.local_cmd_port))
        except OSError:
            cmd_sock.bind(("", 0))
        cmd_sock.settimeout(0.2)
        self._cmd_sock = cmd_sock

        state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            state_sock.bind(("", self.state_port))
            state_sock.settimeout(0.2)
            self._state_sock = state_sock
        except OSError:
            # State reception is useful but not mandatory for command relaying.
            self._state_sock = None

    # ------------------------------------------------------------------
    def _clear_response_queue(self) -> None:
        while True:
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break

    def _response_matches(self, cmd: str, resp: str) -> bool:
        """今送った cmd に対して、resp が妥当そうかを判定する。
        Tello SDK にはリクエストIDがないので完全対応はできないが、
        明らかな取り違えを防ぐ。
        """
        c = cmd.strip().lower()
        r = resp.strip().lower()

        if not r:
            return False

        # query 系
        if c.endswith("?"):
            if c == "battery?":
                return r.isdigit()
            # speed?, height?, time? なども数値が多い
            # attitude? などを使うなら必要に応じて広げる
            return r not in ("ok", "error")

        # 通常コマンド。実機は 'error Not joystick' / 'error No valid imu' /
        # 'error Motor stop' など複数語のエラー文字列を返すことがあるので前方一致。
        if c in ("command", "takeoff", "land", "stop"):
            return r == "ok" or r.startswith("error") or r.startswith("unknown")

        # emergency は返信待ちしない想定
        if c == "emergency":
            return True

        # rc は基本 wait_response=False で使う
        if c.startswith("rc "):
            return True

        # その他 primitive command
        return r == "ok" or r.startswith("error") or r.startswith("unknown")


    def send_tello(self, cmd: str, wait_response: bool = False, timeout_s: float = 1.0,
                   try_lock: bool = False) -> Optional[str]:
        """Tello へ SDK コマンドを 1 つ送る。

        戻り値: マッチした返信文字列 (返信なし/待たない場合は "")。
        ``try_lock=True`` のときは、別スレッドが takeoff/land 等の長い同期待ちで
        送信ロックを保持している間はブロックせず即座に ``None`` を返す
        (watchdog と rc ストリームが長時間待たされて誤動作しないため)。
        """
        if self._cmd_sock is None:
            raise RuntimeError("UDP command socket is not open")

        c = cmd.strip().lower()
        log_sync = wait_response and c != "battery?"  # keepalive の battery? はログしない

        if try_lock:
            if not self._send_lock.acquire(blocking=False):
                return None
        else:
            self._send_lock.acquire()
        try:
            # 同期コマンドでは、送信前に古い返信を捨てる
            if wait_response:
                self._clear_response_queue()

                # Tello がコマンド連打で詰まりにくいように少し間隔を空ける
                now = time.time()
                gap = now - self._last_send_time
                if gap < self._min_sync_command_gap:
                    time.sleep(self._min_sync_command_gap - gap)

            send_time = time.time()
            try:
                self._cmd_sock.sendto(cmd.encode("utf-8"), self.tello_addr)
            except OSError as exc:
                # wlan0 is not associated to the Tello yet (Tello powered off, or
                # not connected).  Degrade gracefully instead of crashing: report
                # no response.  The keepalive / link-health loop will mark the
                # link DOWN and keep retrying, and recover automatically once the
                # Tello becomes reachable.
                self._last_send_time = send_time
                if log_sync:
                    self._log(f"cmd {cmd!r} -> send failed ({exc}); wlan0 down?", err=True)
                return ""
            self._last_send_time = send_time

            if not wait_response:
                return ""

            deadline = send_time + timeout_s
            last_unmatched = ""
            result = ""

            while time.time() < deadline:
                try:
                    recv_time, text = self._response_queue.get(
                        timeout=max(0.01, deadline - time.time())
                    )
                except queue.Empty:
                    continue

                # 送信前に受信済みだったものは捨てる
                if recv_time < send_time:
                    continue

                # 今のコマンドに対して妥当な返信だけ採用
                if self._response_matches(cmd, text):
                    result = text
                    break

                # デバッグ用に最後の不一致返信だけ保持
                last_unmatched = text

            if result:
                if log_sync:
                    self._log(f"cmd {cmd!r} -> {result!r} ({time.time() - send_time:.2f}s)")
            else:
                # タイムアウト。取り違え防止のため、怪しい返信は返さない。
                if last_unmatched:
                    self._log(f"ignored unmatched response for cmd={cmd!r}: {last_unmatched!r}",
                              err=True)
                if log_sync:
                    self._log(f"cmd {cmd!r} -> NO RESPONSE ({timeout_s:.1f}s timeout)", err=True)
            return result
        finally:
            # takeoff の arming/rc 鮮度更新はロック解放前に行う。解放後に行うと、
            # 待機していた watchdog が古い last_rc_time を根拠に land を割り込ませ、
            # 離陸直後の機体を着陸させてしまう競合窓ができる。
            if c == "takeoff":
                with self._state_lock:
                    self.state.armed = True
                    self.state.watchdog_landed = False
                    self.state.last_rc_time = time.time()
            self._send_lock.release()
    
    def _recv_tello_responses(self) -> None:
        assert self._cmd_sock is not None
        while self._running.is_set():
            try:
                data, _ = self._cmd_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            recv_time = time.time()
            text = data.decode("utf-8", errors="replace").strip()

            with self._state_lock:
                self.state.last_tello_response = text
                self.state.last_tello_response_time = recv_time

            try:
                self._response_queue.put_nowait((recv_time, text))
            except queue.Full:
                # 古い返信を1つ捨てて、新しい返信を残す
                try:
                    self._response_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._response_queue.put_nowait((recv_time, text))
                except queue.Full:
                    pass

    def _recv_tello_state(self) -> None:
        if self._state_sock is None:
            return
        while self._running.is_set():
            try:
                data, _ = self._state_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            text = data.decode("utf-8", errors="replace").strip()
            parsed = self._parse_state(text)
            with self._state_lock:
                self.state.last_state = parsed
                self.state.last_state_time = time.time()

    @staticmethod
    def _parse_state(text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in text.split(";"):
            if not part or ":" not in part:
                continue
            k, v = part.split(":", 1)
            out[k] = v
        return out

    # ------------------------------------------------------------------
    def _watchdog_loop(self) -> None:
        while self._running.is_set():
            now = time.time()
            with self._state_lock:
                armed = self.state.armed
                last_rc = self.state.last_rc_time
                already_landed = self.state.watchdog_landed
            if armed and last_rc > 0.0:
                age = now - last_rc
                if age > self.land_timeout and not already_landed:
                    # try_lock: takeoff/land の同期待ちがロックを保持している間は
                    # 送らずスキップし、次の tick で新しい last_rc_time を見て
                    # 再判定する (再離陸直後に古い判定で land しないため)。
                    sent = self.send_tello("land", wait_response=False, try_lock=True)
                    if sent is not None:
                        with self._state_lock:
                            self.state.watchdog_landed = True
                            self.state.armed = False
                elif age > self.hover_timeout:
                    self.send_tello("rc 0 0 0 0", wait_response=False, try_lock=True)
            time.sleep(0.05)

    # ------------------------------------------------------------------
    def _keepalive_loop(self) -> None:
        """Keep the Tello awake while idle and track link health.

        Runs a fast health tick (1 s).  Active ``battery?`` keepalives are
        throttled to ``keepalive_interval`` and only sent while idle, because a
        flying / actively controlled Tello is already kept alive by the rc
        stream.  Link state is derived from the most recent sign of life
        (Tello state stream or command response).
        """
        tick = 1.0
        last_keepalive_send = 0.0
        last_heartbeat_log = 0.0
        while self._running.is_set():
            time.sleep(tick)
            now = time.time()
            with self._state_lock:
                armed = self.state.armed
                last_seen = max(self.state.last_state_time, self.state.last_tello_response_time)

            # Idle = we have not sent anything *to the Tello* recently.  We gate on
            # the last Tello send (self._last_send_time), NOT on PC traffic: a PC
            # polling `status` must not suppress keepalive, because a status query
            # never reaches the Tello.  During flight the rc stream keeps
            # _last_send_time fresh and naturally suppresses keepalive.
            idle = (not armed) and (now - self._last_send_time) > self.keepalive_interval

            # Active keepalive: nudge the Tello only when idle and throttled.
            if idle and (now - last_keepalive_send) >= self.keepalive_interval:
                last_keepalive_send = now
                with self._state_lock:
                    link_down = self.state.tello_link != LINK_UP
                if link_down:
                    # Link is not healthy: try to (re)enter SDK mode.
                    resp = self.send_tello("command", wait_response=True, timeout_s=2.0)
                    if resp.strip().lower() == "ok":
                        with self._state_lock:
                            self.state.sdk_mode = True
                        last_seen = time.time()
                else:
                    resp = self.send_tello("battery?", wait_response=True, timeout_s=1.5)
                    if resp.strip().isdigit():
                        with self._state_lock:
                            self.state.battery = resp.strip()
                        last_seen = time.time()

            self._evaluate_link(last_seen, now)

            # Low-rate heartbeat so the journal visibly shows the service is alive
            # and how fresh the Tello link is (helps confirm keepalive is working).
            if now - last_heartbeat_log >= 60.0:
                last_heartbeat_log = now
                with self._state_lock:
                    link = self.state.tello_link
                    batt = self.state.battery or "?"
                age = (now - last_seen) if last_seen > 0 else -1.0
                self._log(f"alive: link={link} battery={batt} last_seen={age:.1f}s")

    def _evaluate_link(self, last_seen: float, now: float) -> None:
        alive = last_seen > 0.0 and (now - last_seen) <= self.link_timeout
        new_link = LINK_UP if alive else LINK_DOWN
        with self._state_lock:
            prev = self.state.tello_link
            if new_link != prev:
                self.state.tello_link = new_link
                self.state.link_since = now
                if new_link == LINK_DOWN:
                    self.state.sdk_mode = False
                    self.state.consecutive_keepalive_fail += 1
            transition = (prev, new_link)
        prev_link, cur_link = transition
        if prev_link != cur_link:
            if cur_link == LINK_DOWN:
                if prev_link == LINK_UP:
                    self._log("TELLO LINK LOST", err=True)
                else:  # INIT -> DOWN: never came up yet, just waiting.
                    self._log("waiting for Tello link...")
            elif prev_link == LINK_DOWN:
                self._log("TELLO LINK RESTORED")
            else:  # INIT -> UP
                self._log("TELLO LINK UP")

    # ------------------------------------------------------------------
    def _serve_tcp(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.listen_host, self.listen_port))
        server.listen(5)
        self._server_sock = server
        self._log(f"listening on {self.listen_host}:{self.listen_port}")
        try:
            while self._running.is_set():
                try:
                    conn, addr = server.accept()
                except OSError:
                    break
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
        finally:
            self.stop()

    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        self._log(f"client connected: {addr}")
        with conn:
            f_in = conn.makefile("r", encoding="utf-8", newline="\n")
            f_out = conn.makefile("w", encoding="utf-8", newline="\n")
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    reply = self._handle_msg(msg)
                except Exception as exc:
                    reply = {"ok": False, "error": str(exc)}
                f_out.write(json.dumps(reply, separators=(",", ":")) + "\n")
                f_out.flush()
        self._log(f"client disconnected: {addr}")

    def _handle_msg(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        typ = str(msg.get("type", ""))
        now = time.time()
        with self._state_lock:
            self.state.last_pc_time = now

        if typ == "command":
            resp = self.send_tello("command", wait_response=True, timeout_s=2.0)
            ok = resp.strip().lower() == "ok"
            with self._state_lock:
                self.state.sdk_mode = ok
            return {"ok": ok, "t": time.time(), "resp": resp}

        if typ == "takeoff":
            # Tello は離陸動作の完了後に 'ok' を返すため、応答まで 10 秒近く
            # かかることがある。arming と last_rc_time の更新は send_tello 内
            # (送信ロック解放前) で行われる。
            resp = self.send_tello("takeoff", wait_response=True, timeout_s=12.0)
            ok = (resp or "").strip().lower() == "ok"

            return {
                "ok": ok,
                "t": time.time(),
                "resp": resp,
                "warning": "" if ok else "takeoff response was not ok; verify by state/visual check",
            }

        if typ == "land":
            resp = self.send_tello("land", wait_response=True, timeout_s=8.0)
            with self._state_lock:
                self.state.armed = False
            return self._ok(resp=resp)

        if typ == "emergency":
            self.send_tello("emergency", wait_response=False)
            with self._state_lock:
                self.state.armed = False
            return self._ok(resp="")

        if typ == "stop":
            resp = self.send_tello("stop", wait_response=True, timeout_s=2.0)
            return self._ok(resp=resp)

        if typ == "query":
            cmd = str(msg["cmd"])
            if not cmd.endswith("?"):
                raise ValueError("query cmd must end with '?' for safety")
            resp = self.send_tello(cmd, wait_response=True, timeout_s=2.0)
            return self._ok(cmd=cmd, resp=resp)

        if typ == "rc":
            a = msg.get("a", None)
            if not isinstance(a, list) or len(a) != 4:
                raise ValueError("rc message requires a list field 'a' of length 4")
            rc_limit = int(msg.get("rc_limit", self.rc_default_limit))
            vals = self._action_to_rc(a, rc_limit)
            cmd = f"rc {vals[0]} {vals[1]} {vals[2]} {vals[3]}"
            # try_lock: 別スレッドの takeoff/land 同期待ち中は rc を 1 発捨てる
            # (10Hz ストリームなので欠落は無害。ブロックすると PC 側の制御
            # ループ全体が最大 12 秒止まり、他機の watchdog 着陸を誘発する)。
            sent = self.send_tello(cmd, wait_response=False, try_lock=True)
            with self._state_lock:
                # PC が生きている証拠なので、Tello へ届かなくても鮮度は更新する
                self.state.last_rc_time = time.time()
                self.state.watchdog_landed = False
            return self._ok(cmd=cmd, rc=vals, dropped=(sent is None))

        if typ == "status":
            with self._state_lock:
                snapshot = asdict(self.state)
                last_seen = max(self.state.last_state_time, self.state.last_tello_response_time)
            snapshot["uptime_s"] = round(time.time() - self._start_time, 1)
            snapshot["last_seen_age_s"] = round(time.time() - last_seen, 2) if last_seen > 0 else None
            return self._ok(status=snapshot)

        raise ValueError(f"unknown message type: {typ}")

    @staticmethod
    def _action_to_rc(a, rc_limit: int):
        limit = max(0, min(100, int(rc_limit)))
        vals = []
        for x in a:
            xf = max(-1.0, min(1.0, float(x)))
            vals.append(int(round(limit * xf)))
        return vals

    def _ok(self, **extra) -> Dict[str, Any]:
        out = {"ok": True, "t": time.time()}
        out.update(extra)
        return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--listen-host", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=10000)
    p.add_argument("--tello-ip", default="192.168.10.1")
    p.add_argument("--local-cmd-port", type=int, default=8889)
    p.add_argument("--state-port", type=int, default=8890)
    p.add_argument("--rc-default-limit", type=int, default=30)
    p.add_argument("--hover-timeout", type=float, default=0.35)
    p.add_argument("--land-timeout", type=float, default=1.50)
    p.add_argument("--keepalive", type=float, default=10.0,
                   help="idle keepalive interval in seconds (0 disables active keepalive)")
    p.add_argument("--link-timeout", type=float, default=4.0,
                   help="seconds without any Tello sign-of-life before link is DOWN")
    p.add_argument("--auto-command", action="store_true", help="send SDK 'command' once at startup")
    args = p.parse_args()

    gw = TelloGateway(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        local_cmd_port=args.local_cmd_port,
        state_port=args.state_port,
        rc_default_limit=args.rc_default_limit,
        hover_timeout=args.hover_timeout,
        land_timeout=args.land_timeout,
        tello_ip=args.tello_ip,
        keepalive_interval=args.keepalive,
        link_timeout=args.link_timeout,
    )

    try:
        gw._running.set()
        gw._open_udp_sockets()
        hostname = socket.gethostname()
        gw._log(f"starting: host={hostname} listen={args.listen_host}:{args.listen_port} "
                f"tello={args.tello_ip} keepalive={args.keepalive}s")
        gw._launch_threads()
        if args.auto_command:
            gw._log("sending SDK command")
            resp = gw.send_tello("command", wait_response=True, timeout_s=5.0)
            gw._log(f"SDK command response: {resp!r}")
            if resp.strip().lower() == "ok":
                with gw._state_lock:
                    gw.state.sdk_mode = True
        gw._serve_tcp()
    except KeyboardInterrupt:
        gw._log("interrupted", err=True)
    finally:
        gw.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
