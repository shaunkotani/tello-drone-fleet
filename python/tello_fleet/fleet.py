"""Fleet: bulk operations over N Raspberry Pi Tello gateways.

``Fleet`` replaces the historical ``PiTelloGroup`` and adds fleet-wide health
inspection (``status_all`` / ``doctor``) plus the agreed safety policy: if any
drone's telemetry is lost, the whole fleet is landed (data-integrity and safety
take priority over continuing an experiment with a partially-blind formation).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .client import DroneClient, GatewayReply, GatewaySpec, FleetError, specs_from_config


JsonDict = Dict[str, Any]


class FleetTelemetryLost(FleetError):
    """Raised after the fleet has been landed because a drone's telemetry dropped."""


class Fleet:
    """Control N Raspberry Pi gateways together."""

    def __init__(self, clients: Sequence[DroneClient]):
        self.clients: List[DroneClient] = list(clients)
        if not self.clients:
            raise ValueError("Fleet requires at least one client")

    # Construction ------------------------------------------------------------
    @classmethod
    def from_specs(cls, specs: Sequence[GatewaySpec], timeout_s: float = 1.0) -> "Fleet":
        return cls([DroneClient(s, timeout_s=timeout_s) for s in specs])

    @classmethod
    def from_hosts(cls, hosts: Sequence[str], port: int = 10000, timeout_s: float = 1.0) -> "Fleet":
        return cls([DroneClient(h, port=port, timeout_s=timeout_s, name=f"tello-{i + 1}")
                    for i, h in enumerate(hosts)])

    @classmethod
    def from_config(cls, cfg_or_path: Union[JsonDict, str, Path],
                    timeout_s: Optional[float] = None) -> "Fleet":
        if isinstance(cfg_or_path, (str, Path)):
            cfg = json.loads(Path(cfg_or_path).read_text(encoding="utf-8"))
        else:
            cfg = cfg_or_path
        specs = specs_from_config(cfg)
        t = float(cfg.get("pi_timeout_s", 1.0)) if timeout_s is None else float(timeout_s)
        return cls.from_specs(specs, timeout_s=t)

    # Connection --------------------------------------------------------------
    def set_timeout(self, timeout_s: float) -> None:
        for c in self.clients:
            c.set_timeout(timeout_s)

    def connect_all(self) -> None:
        for c in self.clients:
            c.connect()

    def close_all(self) -> None:
        for c in self.clients:
            c.close()

    # Bulk commands -----------------------------------------------------------
    def broadcast(self, msg: JsonDict) -> List[GatewayReply]:
        return [c.request(msg) for c in self.clients]

    def takeoff_all(self) -> List[GatewayReply]:
        return [c.takeoff() for c in self.clients]

    def land_all(self) -> List[GatewayReply]:
        return [c.land() for c in self.clients]

    def emergency_all(self) -> List[GatewayReply]:
        replies: List[GatewayReply] = []
        for c in self.clients:
            try:
                replies.append(c.emergency())
            except Exception:
                # Emergency is best effort; keep trying the remaining drones.
                pass
        return replies

    def hover_all(self) -> List[GatewayReply]:
        return [c.hover() for c in self.clients]

    def send_actions(self, actions: Sequence[Sequence[float]], rc_limit: int = 30,
                     land_on_failure: bool = False) -> List[GatewayReply]:
        if len(actions) != len(self.clients):
            raise ValueError(f"actions length {len(actions)} != client count {len(self.clients)}")
        replies: List[GatewayReply] = []
        try:
            for c, a in zip(self.clients, actions):
                replies.append(c.rc(a, rc_limit=rc_limit))
        except Exception as exc:
            if land_on_failure:
                self.land_all_safe()
                raise FleetTelemetryLost(f"rc send failed, fleet landed: {exc}") from exc
            raise
        return replies

    def land_all_safe(self) -> None:
        """Best-effort land of every drone; never raises.

        This is the fleet-wide safety action invoked when any telemetry is lost.
        """
        for c in self.clients:
            try:
                c.land()
            except Exception:
                try:
                    c.emergency()
                except Exception:
                    pass

    # Health ------------------------------------------------------------------
    def _parallel(self, fn) -> List[Tuple[str, Optional[GatewayReply], Optional[str]]]:
        results: List[Tuple[str, Optional[GatewayReply], Optional[str]]] = []
        with ThreadPoolExecutor(max_workers=len(self.clients)) as ex:
            fut_to_client = {ex.submit(fn, c): c for c in self.clients}
            for fut in as_completed(fut_to_client):
                c = fut_to_client[fut]
                try:
                    results.append((c.name, fut.result(), None))
                except Exception as exc:
                    results.append((c.name, None, str(exc)))
        results.sort(key=lambda x: x[0])
        return results

    def status_all(self) -> Dict[str, Optional[JsonDict]]:
        """Return each gateway's status dict, or None if it is unreachable."""
        out: Dict[str, Optional[JsonDict]] = {}
        for name, reply, err in self._parallel(lambda c: c.status()):
            out[name] = reply.raw.get("status") if (reply is not None and err is None) else None
        return out

    def doctor(self) -> List[Dict[str, Any]]:
        """Per-drone diagnostics: reachability, SDK mode, link, battery.

        Returns a list of dicts with an ``ok`` flag and human-readable ``issues``.
        """
        report: List[Dict[str, Any]] = []
        statuses = self.status_all()
        for c in self.clients:
            st = statuses.get(c.name)
            entry: Dict[str, Any] = {"name": c.name, "host": c.host, "issues": []}
            if st is None:
                entry.update(reachable=False, ok=False)
                entry["issues"].append("gateway unreachable")
                report.append(entry)
                continue
            link = str(st.get("tello_link", "init"))
            sdk = bool(st.get("sdk_mode"))
            battery = st.get("battery") or ""
            entry.update(reachable=True, tello_link=link, sdk_mode=sdk,
                         battery=battery, uptime_s=st.get("uptime_s"))
            if link != "up":
                entry["issues"].append(f"tello link {link}")
            if not sdk:
                entry["issues"].append("not in SDK mode")
            try:
                if battery != "" and float(battery) < 20:
                    entry["issues"].append(f"battery low ({battery}%)")
            except ValueError:
                pass
            entry["ok"] = not entry["issues"]
            report.append(entry)
        return report
