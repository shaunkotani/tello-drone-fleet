"""Single Raspberry Pi Tello gateway client.

``DroneClient`` is the one canonical TCP client for a Pi gateway, unifying the
two historical duplicates:
    * no_tracker_demos.no_tracker_common.PiGatewayClient
    * tello_rl.real.pi_client.PiTelloClient

Protocol: newline-delimited JSON over TCP.  The gateway replies with exactly one
JSON line per request.  See raspberry_pi/tello_gateway.py for the server side.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time
from typing import Any, Dict, List, Optional, Sequence, Union


JsonDict = Dict[str, Any]


class FleetError(RuntimeError):
    """Raised when a gateway reports an error or cannot be reached."""


@dataclass
class GatewaySpec:
    name: str
    host: str
    port: int = 10000


@dataclass
class GatewayReply:
    ok: bool
    raw: JsonDict
    elapsed_s: float = 0.0

    @property
    def error(self) -> Optional[str]:
        return self.raw.get("error")

    @property
    def resp(self) -> Any:
        """The Tello SDK response string, when present (query / rc / etc.)."""
        return self.raw.get("resp")


def specs_from_config(cfg: JsonDict) -> List[GatewaySpec]:
    """Build gateway specs from a config dict.

    Accepts either the explicit ``gateways`` list or the ``pi_hosts`` shorthand.
    """
    gateways = cfg.get("gateways")
    if gateways:
        return [
            GatewaySpec(
                name=str(g.get("name", f"tello-{i + 1}")),
                host=str(g["host"]),
                port=int(g.get("port", cfg.get("pi_port", 10000))),
            )
            for i, g in enumerate(gateways)
        ]
    hosts = cfg.get("pi_hosts")
    if not hosts:
        raise FleetError("config requires either 'gateways' or 'pi_hosts'")
    port = int(cfg.get("pi_port", 10000))
    return [GatewaySpec(name=f"tello-{i + 1}", host=str(h), port=port)
            for i, h in enumerate(hosts)]


class DroneClient:
    """Persistent TCP client for one Raspberry Pi Tello gateway.

    Construct either from a :class:`GatewaySpec` or from a host string::

        DroneClient(spec, timeout_s=1.0)
        DroneClient("192.168.50.101", port=10000, timeout_s=1.0, name="tello-1")
    """

    def __init__(self, spec_or_host: Union[GatewaySpec, str], port: int = 10000,
                 timeout_s: float = 1.0, name: Optional[str] = None):
        if isinstance(spec_or_host, GatewaySpec):
            self.spec = spec_or_host
        else:
            self.spec = GatewaySpec(name=name or str(spec_or_host),
                                    host=str(spec_or_host), port=int(port))
        self.timeout_s = float(timeout_s)
        self._sock: Optional[socket.socket] = None
        self._file = None

    # Identity ----------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def host(self) -> str:
        return self.spec.host

    @property
    def port(self) -> int:
        return self.spec.port

    # Connection --------------------------------------------------------------
    def connect(self) -> None:
        self.close()
        sock = socket.create_connection((self.spec.host, self.spec.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._file = sock.makefile("r", encoding="utf-8", newline="\n")

    def close(self) -> None:
        for obj in (self._file, self._sock):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._file = None
        self._sock = None

    def set_timeout(self, timeout_s: float) -> None:
        """Change send/recv timeout, applying it live to an open socket.

        takeoff/land block until the drone finishes moving, so this is used to
        temporarily lengthen the timeout for those commands while keeping a short
        timeout for the control loop.
        """
        self.timeout_s = float(timeout_s)
        if self._sock is not None:
            self._sock.settimeout(self.timeout_s)

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    # Request/reply -----------------------------------------------------------
    def request(self, msg: JsonDict, *, retry: bool = True) -> GatewayReply:
        """Send one JSON message and wait for one JSON reply.

        On any transport error, reconnect once and retry (unless ``retry`` is
        False).  Raises :class:`FleetError` if the gateway reports ``ok=false``.
        """
        try:
            return self._request_once(msg)
        except FleetError:
            raise
        except Exception as exc:
            if not retry:
                raise FleetError(f"{self.name}: request failed: {exc}") from exc
            self.close()
            time.sleep(0.05)
            try:
                return self._request_once(msg)
            except Exception as exc2:
                raise FleetError(f"{self.name}: request failed after reconnect: {exc2}") from exc2

    def _request_once(self, msg: JsonDict) -> GatewayReply:
        self._ensure_connected()
        assert self._sock is not None and self._file is not None
        payload = json.dumps(msg, separators=(",", ":"), ensure_ascii=False) + "\n"
        start = time.time()
        self._sock.sendall(payload.encode("utf-8"))
        line = self._file.readline()
        elapsed = time.time() - start
        if not line:
            raise FleetError(f"{self.name}: gateway closed connection")
        raw = json.loads(line)
        ok = bool(raw.get("ok", False))
        if not ok:
            raise FleetError(f"{self.name}: gateway returned error: {raw}")
        return GatewayReply(ok=ok, raw=raw, elapsed_s=elapsed)

    # Convenience -------------------------------------------------------------
    def command(self) -> GatewayReply:
        return self.request({"type": "command"})

    def query(self, sdk_query: str) -> GatewayReply:
        return self.request({"type": "query", "cmd": sdk_query})

    def status(self) -> GatewayReply:
        return self.request({"type": "status"})

    def takeoff(self) -> GatewayReply:
        return self.request({"type": "takeoff"})

    def land(self) -> GatewayReply:
        return self.request({"type": "land"})

    def stop(self) -> GatewayReply:
        return self.request({"type": "stop"})

    def emergency(self) -> GatewayReply:
        return self.request({"type": "emergency"}, retry=False)

    def hover(self) -> GatewayReply:
        return self.rc([0.0, 0.0, 0.0, 0.0])

    def rc(self, action: Sequence[float], rc_limit: int = 30) -> GatewayReply:
        if len(action) != 4:
            raise ValueError("action must have length 4: [lr, fb, ud, yaw]")
        return self.request({
            "type": "rc",
            "a": [float(x) for x in action],
            "rc_limit": int(rc_limit),
        })

    def __enter__(self) -> "DroneClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
