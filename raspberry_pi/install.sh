#!/usr/bin/env bash
# Install (or update) the Tello Pi gateway as a systemd service.
#
# Fixed layout (decided here, not auto-detected):
#     runtime code : /opt/tello-gateway/tello_gateway.py
#     systemd unit : /etc/systemd/system/tello-gateway.service
#
# You may keep the source files (tello_gateway.py, tello-gateway.service, this
# script) together in ANY directory on the Pi. This installer copies them into
# the fixed locations above, then enables and starts the service.
#
# Run on each Pi (from wherever the source files are):
#     sudo bash install.sh
#
# Re-running is safe and is also how you deploy code updates.
set -euo pipefail

INSTALL_DIR="/opt/tello-gateway"
UNIT_NAME="tello-gateway.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_SRC="${SCRIPT_DIR}/tello_gateway.py"
UNIT_SRC="${SCRIPT_DIR}/${UNIT_NAME}"
UNIT_DEST="/etc/systemd/system/${UNIT_NAME}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "This installer needs root. Re-run with: sudo bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi

if [[ ! -f "${GATEWAY_SRC}" ]]; then
    echo "Gateway source not found next to this script: ${GATEWAY_SRC}" >&2
    echo "Keep tello_gateway.py, tello-gateway.service and install.sh in one folder." >&2
    exit 1
fi
if [[ ! -f "${UNIT_SRC}" ]]; then
    echo "Unit file not found next to this script: ${UNIT_SRC}" >&2
    exit 1
fi

echo "Installing:"
echo "  gateway -> ${INSTALL_DIR}/tello_gateway.py"
echo "  unit    -> ${UNIT_DEST}"

# Copy the code into the fixed runtime location and the unit into systemd.
install -D -m 0755 "${GATEWAY_SRC}" "${INSTALL_DIR}/tello_gateway.py"
install -m 0644 "${UNIT_SRC}" "${UNIT_DEST}"

systemctl daemon-reload
systemctl enable "${UNIT_NAME}"
systemctl restart "${UNIT_NAME}"
systemctl reset-failed "${UNIT_NAME}" 2>/dev/null || true

echo
echo "Done. tello-gateway is installed and running."
echo "Check status : systemctl status ${UNIT_NAME}"
echo "Follow logs  : journalctl -u ${UNIT_NAME} -f"
