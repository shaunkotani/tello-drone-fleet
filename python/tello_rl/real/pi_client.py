"""Backward-compatible shim over :mod:`tello_fleet`.

The real client/group implementation now lives in ``tello_fleet``.  This module
keeps the historical names (``PiTelloClient``, ``PiTelloGroup``, ``PiGatewayError``,
``PiGatewayReply``) so existing imports keep working:

    from tello_rl.real.pi_client import PiTelloGroup

Prefer ``from tello_fleet import DroneClient, Fleet`` in new code.
"""
from __future__ import annotations

from tello_fleet import (
    DroneClient as PiTelloClient,
    Fleet as PiTelloGroup,
    FleetError as PiGatewayError,
    GatewayReply as PiGatewayReply,
)

__all__ = ["PiTelloClient", "PiTelloGroup", "PiGatewayError", "PiGatewayReply"]
