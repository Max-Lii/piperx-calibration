# robotcontrol (formal path)

This is the canonical robotcontrol package. New service code, tests and host
deployment instructions must use `vla.sim2real.robotcontrol`; the former
`vla.dev.sim2real.robotcontrol` tree is retained only for legacy simulator
launchers and historical comparison.

`robotcontrol` is the long-lived, robot-host control boundary for the Piper-X
workcell. VLA inference and calibration are HTTP clients: they never import
`pyAgxArm`, open CAN, or directly own a D435.

The checked-in default is offline-safe: mock arms, mock cameras, localhost,
and no CAN/D435 access. `piper_x_readonly` can verify dual-arm feedback but
rejects every movement. `piper_x_motion` is the explicit physical-motion
backend: it requires host-local per-joint limits, an explicit `/motions/arm`,
and can run the reviewed calibration pose envelope when enabled in a
host-local config. Physical gripper commands use an explicit normalized
mapping (`0=closed`, `1=gripper_open_width_m`) and remain blocked until that
metric width is supplied in the host-local config.

## Safety contract

The canonical action is `[H, 14]`:

`[left_j1..left_j6, left_gripper, right_j1..right_j6, right_gripper]`.

`POST /v1/motions/execute_joint_waypoints` accepts a single calibration target
or a VLA action chunk. The guardian validates soft limits, interpolates bounded
samples, and serializes execution. Its motion contract is ready, but physical
command mapping is deliberately not enabled until the lab checklist is passed.

- `/v1/control/acquire`, `/alive`, `/release`: exclusive 15-second lease;
  clients should call `alive` every three seconds.
- `/v1/motions/hold`: cancels pending work and holds the measured pose.
- `/v1/motions/arm`: explicit motor-enable preparation; it emits no target
  pose and is available only to the first-motion physical backend.
- `/v1/motions/go_home`: explicit slow recovery to the configured home pose;
  it is not an emergency stop and is never automatic error recovery.
- `/v1/state`, `/v1/health`: current arm/lease/execution state.
- `/v1/cameras/{id}/frame`, `/stream`, `/calibration-frame`: JPEG, MJPEG and
  lossless RGB/depth/intrinsics payloads for the three fixed camera roles.
- `/v1/cameras/{id}/ws`: timestamp metadata followed by a binary JPEG frame.
- `/v1/cameras/ws`: one metadata message followed by three binary JPEG frames;
  use this for continuous three-view capture so processing stays batched.

## Camera image processing

`camera.image_processing` optionally applies one accepted 3D LUT per camera to
JPEG, MJPEG and WebSocket output. The lossless `calibration-frame` endpoint
always returns the original D435 RGB/depth sample so calibration remains
auditable and a changed display LUT never destroys source data.

Supported backends are `cpu`, `cuda` and `auto`. The CPU path expands each
33-cube to a compact 65-cube at startup and then uses direct array lookup. The
CUDA path uses Torch and performs one host-to-device and one device-to-host
copy per new camera frame. Reusable pinned host buffers reduce CUDA staging
overhead. Processed JPEGs are cached by D435 frame number, so
multiple preview clients do not repeat LUT work, PCIe transfers or encoding.
Response headers report the backend, frame number and LUT processing time.

CUDA LUT processing enables `torch.compile(mode="reduce-overhead")` by
default. The configured `(height, width, 3)` shape is compiled and warmed once
before Uvicorn starts, so the first VLA observation does not pay compilation
latency. Set `camera.image_processing.torch_compile: false` only when startup
latency or Torch compiler support is a concern.

CUDA LUT processing enables `torch.compile(mode="reduce-overhead")` by default.
The fixed configured camera shape is compiled and warmed once before Uvicorn
starts, so the first VLA observation does not pay the compilation cost. Set
`camera.image_processing.torch_compile: false` only for environments where
startup latency or Torch compiler support is a concern.

For a VLA observation or capture sample, use `/v1/cameras/frame-batch` to
fetch the three latest views in one request. CUDA processing uses reusable
pinned buffers and one CUDA stream per camera, then JPEG encoding runs in a
small CPU thread pool. A stacked 3D-LUT tensor was measured slower for this
workload than independent streams.

For RGB-only VLA or monitoring, set `camera.enabled_roles` to the required
roles and `camera.depth_enabled: false`. This starts no pipeline for omitted
roles and requests only the RGB stream, so a single D435 can provide
high-resolution RGB without consuming bandwidth for depth. With the current
cabling, `config.robot_host.rgb_three_view.example.yaml` uses the highest
common stable three-view profile measured on the robot host: `1280x720@15`.
The USB-2.1 left-wrist camera limits the common profile; the other two cameras
can run `1920x1080@30`. In RGB-only mode the calibration-frame endpoint
intentionally returns HTTP 409 because no depth sample exists.

The September 10, 2026 robot-host benchmark at 640x480 measured median
LUT-plus-JPEG latency of about 5.2 ms on CPU, 1.9 ms on RTX 4090 and 2.7 ms on
RTX 4080. Reusable pinned buffers reduced the GPU figures to about 1.2 ms and
1.5 ms respectively. The full host example uses GPU 1 (RTX 4080), leaving the RTX 4090 for
VLA inference. GPU JPEG libraries were not added: OpenCV encoding was only
about 0.6-0.7 ms, while the installed Python environment has no nvJPEG binding.
No denoiser is enabled by default because it blurs task-relevant edges and is
not needed to correct the stable cross-camera color shift.

AV1 is reserved for the future continuous recorder and transport path. It is
not a replacement for the random-access JPEG frame endpoint: low-latency AV1
needs a hardware encoder, keyframe and packet framing, plus WebRTC or a video
container. The robot host exposes `av1_nvenc`, `hevc_nvenc` and `h264_nvenc`
through PyAV, but PyAV construction from NumPy still starts in CPU memory.
H.264 and AV1 NVENC should be compared after the recorder can pass a processed
CUDA frame directly to an NVENC surface without downloading it first.

Run an offline benchmark without opening cameras or CAN:

```bash
python -m vla.sim2real.robotcontrol.benchmark_image_processing \
  --backend cpu \
  --asset-root vla/sim2real/robotcontrol/assets/d435_neutral_preset_20260910 \
  --image-root /path/to/three-camera-pngs
```

Lost lease, API errors and driver errors become **HOLD**. This service never
calls pyAgxArm `disable()`, `reset()` or `electronic_emergency_stop()`.
Clients send heartbeat traffic over an independent HTTP connection. Transient
transport failures are retried within the lease window; a rejected or
persistently unavailable heartbeat is surfaced to the motion caller.
`SIGKILL`, host power loss, CAN loss and physical torque behavior remain
mandatory supervised lab tests; Python cannot guarantee those cases.

## Install and offline verification

```bash
conda create -n vla-robotwin-s2r python=3.12 -y
conda activate vla-robotwin-s2r
bash scripts/env_setup/setup_vla_robotwin_s2r.sh

python -m pytest -q vla/sim2real/robotcontrol/test_robotcontrol.py \
  third_party/pyAgxArm/tests/test_piper_driver_virtual_can.py
python -m vla.sim2real.robotcontrol.safety_audit
python -m vla.sim2real.robotcontrol.run
```

The default endpoint is `http://127.0.0.1:8008`.

## Robot-host read-only acceptance

Copy `config.robot_host.example.yaml` to a path **outside the repository**,
fill in confirmed CAN interfaces and firmware, and set `ROBOTCONTROL_CONFIG`.
Then run the static audit and read-only probe:

```bash
export ROBOTCONTROL_CONFIG=/etc/vla/robotcontrol.yaml
python -m vla.sim2real.robotcontrol.safety_audit
python -m vla.sim2real.robotcontrol.probe_piper_readonly \
  --left-can can_left --right-can can_right --firmware v188 \
  --samples 20 --period 0.25
python -m vla.sim2real.robotcontrol.run
```

The probe never calls enable, movement, disable, reset or SDK emergency stop;
it only connects, reads diagnostics, and closes host-side CAN resources. Read
[READY_FOR_LAB.md](READY_FOR_LAB.md) before any later motion implementation.

RoboTwin protocol emulation and calibration UI remain under `vla/dev/` by
design. They are development tools and are not dependencies of this production
robot-host service.

## Supervised first-motion acceptance

After first Home acceptance, use `config.motion_smoke.example.yaml` as the
template for a host-local smoke-test configuration. The harness defaults to
dry-run and prints its plan without contacting robotcontrol:

```bash
python -m vla.sim2real.robotcontrol.motion_smoke \
  --config /etc/vla/motion_smoke.yaml
```

Execution additionally requires both `--execute` and
`--confirm-physical-clearance`. It requires the measured state to already be
near the manually verified Home pose, probes one left joint and one right joint
by at most 0.03 rad, explicitly returns after each probe, then explicitly calls
Go Home. Any fault sends HOLD and does not automatically attempt Go Home.

The optional empty-air gripper cycle is intentionally separate. It opens then
closes the left gripper, returns it to Home, then repeats for the right gripper;
it never moves both grippers together. Fill the physical action convention in
the host-local config, verify clearance and use `--include-gripper-cycle` only
when that phase is explicitly desired.

For camera calibration, `camera.lock_auto_controls_after_warmup: true` freezes
the D435 RGB sensor's converged exposure, gain and white balance. These values
and both auto-control flags are included in every lossless calibration frame.
