"""tello_fleet: the single PC-side library for Raspberry Pi Tello gateways.

Unifies the historical duplicated clients into one implementation:
    * DroneClient  - one gateway (was PiGatewayClient / PiTelloClient)
    * Fleet        - bulk operations over N gateways (was PiTelloGroup)

See raspberry_pi/tello_gateway.py for the gateway (server) side and
``python -m tello_fleet --help`` for the ``fleet status`` / ``fleet doctor`` CLI.
"""
from __future__ import annotations

from .client import (
    DroneClient,
    FleetError,
    GatewayReply,
    GatewaySpec,
    specs_from_config,
)
from .fleet import Fleet, FleetTelemetryLost

__all__ = [
    "DroneClient",
    "Fleet",
    "FleetError",
    "FleetTelemetryLost",
    "GatewayReply",
    "GatewaySpec",
    "specs_from_config",
]
