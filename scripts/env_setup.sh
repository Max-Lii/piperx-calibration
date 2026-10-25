#!/usr/bin/env bash
set -euo pipefail

# Install the standalone runtime without depending on the parent VLA checkout.
# This script never opens CAN, starts a camera, or commands a robot.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep dependency checkout inside this standalone project by default.
DEFAULT_PYAGXARM_DIR="${ROOT_DIR}/third_party/pyAgxArm"
PYAGXARM_DIR="${PYAGXARM_DIR:-${DEFAULT_PYAGXARM_DIR}}"
PYAGXARM_REVISION="799b841"
PROFILE="core"
SKIP_VIRTUAL_CAN=0
GIT_PROXY="${PIPERX_GIT_PROXY:-}"
OFFICIAL_ASSETS=0

usage() {
  cat <<'EOF'
Usage: bash scripts/env_setup.sh [--with-robotwin] [--with-robotwin-task-assets] [--skip-sdk-virtual-can-tests]

core (default): standalone calibration/robotcontrol, mock and offline tests.
--with-robotwin: install SAPIEN and the pinned Piper-X model bundle.
--with-robotwin-task-assets: also download official RoboTwin task assets (~16 GB).
PIPERX_ROBOTWIN_ASSETS: explicitly reuse an existing Piper-X bundle read-only.
PIPERX_ROBOTWIN_TASK_ASSETS: optional official assets destination/reuse path.
HF_ENDPOINT=https://hf-mirror.com: optional Hugging Face mirror.
PIPERX_GIT_PROXY=http://127.0.0.1:19096: optional Git proxy.
--skip-sdk-virtual-can-tests: skip the optional pyAgxArm virtual-CAN tests.

The core profile does not access physical CAN, RealSense, or robot motion.
EOF
}

while (($#)); do
  case "$1" in
    --with-robotwin) PROFILE="robotwin" ;;
    --with-robotwin-task-assets) PROFILE="robotwin"; OFFICIAL_ASSETS=1 ;;
    --skip-sdk-virtual-can-tests) SKIP_VIRTUAL_CAN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"piperx-calibration requires Python 3.12, got {sys.version}")
PY

python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel
python -m pip install -e "${ROOT_DIR}[hardware,test]"

if [[ ! -d "${PYAGXARM_DIR}/.git" ]]; then
  command -v git >/dev/null 2>&1 || { echo "git is required to fetch pyAgxArm" >&2; exit 1; }
  mkdir -p "$(dirname -- "${PYAGXARM_DIR}")"
  if [[ -n "${GIT_PROXY}" ]]; then
    git -c "http.proxy=${GIT_PROXY}" clone https://github.com/agilexrobotics/pyAgxArm.git "${PYAGXARM_DIR}"
  else
    git clone https://github.com/agilexrobotics/pyAgxArm.git "${PYAGXARM_DIR}"
  fi
fi
# An explicitly supplied checkout may belong to another project. Do not
# change its revision or dirty worktree as a side effect of installation.
if [[ "${PYAGXARM_DIR}" == "${DEFAULT_PYAGXARM_DIR}" ]]; then
  git -C "${PYAGXARM_DIR}" checkout --detach "${PYAGXARM_REVISION}"
else
  expected_revision="$(git -C "${PYAGXARM_DIR}" rev-parse "${PYAGXARM_REVISION}^{commit}")"
  actual_revision="$(git -C "${PYAGXARM_DIR}" rev-parse HEAD)"
  [[ "${actual_revision}" == "${expected_revision}" ]] || {
    echo "External PYAGXARM_DIR must already be at ${PYAGXARM_REVISION}; checkout was not modified" >&2
    exit 1
  }
fi
python -m pip install -e "${PYAGXARM_DIR}"

python - <<'PY'
import cv2, fastapi, httpx, numpy, pydantic, scipy, yaml
import pyAgxArm
import pyrealsense2
from piperx_calibration.calibration.geometry import solve_robot_world_hand_eye
from piperx_calibration.robotcontrol.app import create_app
print("piperx-calibration core imports OK")
print(f"numpy={numpy.__version__} opencv={cv2.__version__} scipy={scipy.__version__}")
print(f"pyAgxArm={pyAgxArm.__file__} pyrealsense2={pyrealsense2.__file__}")
PY

if (( SKIP_VIRTUAL_CAN == 0 )); then
  (cd "${PYAGXARM_DIR}" && python -m pytest -q tests/test_piper_driver_virtual_can.py)
fi

python -m pip check

if [[ "${PROFILE}" == "robotwin" ]]; then
  python -m pip install -e "${ROOT_DIR}[simulation]"
  asset_args=()
  if (( OFFICIAL_ASSETS )); then
    asset_args+=(--official-assets "${PIPERX_ROBOTWIN_TASK_ASSETS:-${ROOT_DIR}/third_party/RoboTwin/assets}")
  fi
  python "${ROOT_DIR}/scripts/setup_robotwin_assets.py" "${asset_args[@]}"
  python -c 'import sapien; print("SAPIEN simulation imports OK; no planner or VLA dependency required")'

fi

echo "Standalone core environment is ready; no hardware was accessed."
