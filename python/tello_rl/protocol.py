"""
Python <-> Unity インターフェース (9.1)。
長さ前置き (4byte big-endian) + UTF-8 JSON のフレーミングで TCP 上をやり取りする。
ML-Agents / ZeroMQ / ROS-TCP でも置換可能だが、外部依存のない素の TCP を既定とする。

Unity = サーバ (listen), Python = クライアント (connect)。
メッセージ仕様は unity/Scripts/Protocol.cs と一致させること。

  Python -> Unity:
    {"cmd":"config","config":{...}}            # 環境設定の上書き(任意)
    {"cmd":"reset"}                            # エピソード初期化
    {"cmd":"step","raw_actions":[[a4]xN]}      # raw action 送信
    {"cmd":"close"}

  Unity -> Python (reset):
    {"obs":[[od]xN], "info":{...}}
  Unity -> Python (step):
    {"obs":[[od]xN],
     "objective_costs":[N],
     "constraint_costs":[[5]xN],
     "executed_actions":[[4]xN],
     "done":bool,
     "info":{"reason":str,"k":int,"ro":[2],"r":[[2]xN],"slots":[[2]xN],"edge_ref":[N],"phi":float}}
"""
from __future__ import annotations
import json
import socket
import struct


def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    data = _recv_exact(sock, length)
    return json.loads(data.decode("utf-8"))
