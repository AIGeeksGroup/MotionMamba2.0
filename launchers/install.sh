#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3.11}"
SIM_PYTHON="${SIM_PYTHON:-python3.10}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r simulation/requirements.txt -r simulation/requirements.lock.txt
# Keep the production torch arithmetic with its qualified Triton runtime.
.venv/bin/python -m pip install --no-deps triton==3.5.1
.venv/bin/python -m pip install --no-deps -e ardy
"$SIM_PYTHON" -m venv GR00T-WholeBodyControl/.venv_sim
GR00T-WholeBodyControl/.venv_sim/bin/python -m pip install --upgrade pip setuptools wheel
GR00T-WholeBodyControl/.venv_sim/bin/python -m pip install -r simulation/requirements-sim.txt -r simulation/requirements-sim.lock.txt
export CYCLONEDDS_HOME="$ROOT/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty"
GR00T-WholeBodyControl/.venv_sim/bin/python -m pip install --no-deps -e GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python
npm --prefix ardy_sonic_web ci
echo 'Python/frontend installation complete. Next: download assets and build SONIC (README.MD).'
