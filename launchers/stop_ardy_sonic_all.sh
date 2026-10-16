#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec "$ARDY_ENV/bin/python" "$MM2_ROOT/simulation/services.py" stop
