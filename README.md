# Piper-X Calibration

Geometric calibration for a dual-arm Piper-X workcell with two wrist-mounted D435 cameras and one fixed third-view D435.

The system combines robot joint feedback with images of a stationary ChArUco board to estimate the robot-base and camera installation transforms. A web interface guides the operator through device checks, sequential arm acquisition, validation, and JSON export.

[Documentation](https://max-lii.github.io/piperx-calibration/)

## Features

- **Workcell geometry:** estimate both robot-base poses, both wrist-camera mounts, and the fixed third-view camera pose relative to the calibration board.
- **Image-based estimation:** detect ChArUco corners, estimate board poses with PnP, and combine them with measured FK using Robot-World/Hand-Eye calibration.
- **Result validation:** inspect observation quality, held-out predictions, solver agreement, and uncertainty. Simulation records also support comparison with known installation transforms after solving.
- **Operator workflow:** check devices and the target, acquire each arm's observations while the other arm remains parked, review the report, and export the results.
- **Offline replay:** run the workflow using the included RoboTwin RGB and robot-state records without connecting physical devices.

The program returns measured transforms and quality metrics for integration into your own environment. It does not automatically modify RoboTwin repositories or task configurations.

## Requirements

- **Python 3.12** for the calibration services and replay.
- **Node.js 22** and npm for the operator interface.
- A compatible Linux GPU/Vulkan environment for new SAPIEN renderings. Replaying the included records does not require a GPU.

## Quick start

The quickest way to try the software is to replay the included simulation records. Run the following commands from the repository root, using Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . 'Pillow>=10,<12'
```

Build the operator interface:

```bash
cd frontend
npm ci
npm run build
cd ..
```

Start the replay with a new session directory:

```bash
PYTHONPATH=src:. python doc/scripts/replay_tutorial.py \
  --port 8020 --session-root /tmp/piperx-replay-session
```

Open **http://127.0.0.1:8020/operator** and follow the stages through Review / Export. World Registration starts recomputation from the recorded observations. The interface identifies this mode as **Offline replay**; it does not move a robot or open a RealSense device.

The included case, `BOARD_002_20261024_tutorial30_default`, contains 30 attempted observations per arm, with 26 accepted on the left and 30 on the right. These are simulation results, not a physical-workcell calibration.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/piperx_calibration/calibration/` | Acquisition workflow, geometric estimation, validation, and calibration API |
| `src/piperx_calibration/robotcontrol/` | Independent robot/camera service, hardware adapters, control lease, and HOLD handling |
| `frontend/` | Web operator interface |
| `sim/` | Synthetic geometry regression and the isolated RoboTwin/SAPIEN calibration scene |
| `assets/` | Printable ChArUco board, target dimensions, and pose library |
| `doc/` | Bilingual tutorial, illustrations, and archived simulation observations |
| `scripts/` | Environment setup and model-download helpers |

## Calibration results

The exported JSON contains the full installation transforms, quality metrics, and session metadata. Matrix translations use meters; robot joint feedback uses radians.

For the archived case, the reference is the OpenCV ChArUco board frame **P**. Legacy fields such as `T_world_base` use P despite the word `world` in their names. Before applying a result, check its reference-frame metadata and end-effector definition, then convert it into the coordinate conventions used by your simulation.

The tutorial covers the geometry, installation, real-device configuration, safety requirements, and concrete RoboTwin integration examples. Physical acquisition uses the independent Robot Control service and requires checks of the actual workcell before motion.
