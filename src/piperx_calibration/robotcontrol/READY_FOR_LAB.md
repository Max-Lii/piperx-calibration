# Robot-host acceptance checklist

This checklist is intentionally not automated.  Finish every item before
changing `run.py` to permit a real driver or camera backend.

The configured lease TTL is 15 seconds. Clients heartbeat every three seconds
on an independent connection; lease loss must still place the robot in HOLD.

1. Record the left/right CAN interface names, arm serials, firmware profile,
   joint order, joint sign, units, gripper unit/range and the actual safe home
   pose.  Put only host-specific values in an external config file.
2. With both arms physically clear and supervised, run a **read-only** process
   that verifies feedback for both six-joint arms matches manual motion.  Do
   not enable commands yet.
   Use `python -m vla.sim2real.robotcontrol.probe_piper_readonly` and save
   its JSONL output with the host-local configuration record.
3. Verify `is_ok`, CAN disconnect/reconnect behavior, and what the Piper-X
   controller does on a client crash, service `SIGTERM`, service `SIGKILL`, CAN
   loss and host power loss.  In particular, confirm it does not release and
   fall under the site's required safety arrangement.
   The robotcontrol physical adapter is permanently forbidden from calling
   pyAgxArm `disable()`, `reset()` or `electronic_emergency_stop()` during
   HOLD, shutdown, exception recovery, or normal operation. Host-side
   `disconnect()` only closes CAN resources and must leave motor enable state
   unchanged.
4. The pyAgxArm metric gripper mapping is implemented behind the host-local
   `gripper_open_width_m` gate. Test it on one empty arm at a time. Start with a
   single tiny H=1 command from a known clear pose; verify direction, width,
   force, feedback, soft-limit behavior, HOLD, and slow go-home.
5. Repeat for both arms and then validate dual-arm collision-free workspace
   limits.  Software joint limits alone do not protect against self/table
   collisions.
6. Connect each D435 one at a time, map the `pyrealsense2` serial numbers to
   the three stable API roles, confirm resolution/FPS/exposure, and test JPEG
   plus MJPEG clients. The production adapter now provides these endpoints.
   Also validate the lossless calibration-frame payload: RGB color order,
   aligned-depth scale, color intrinsics/distortion convention, frame number,
   hardware timestamp and host monotonic timestamp.
7. Enable the robot-host camera backend only with the host-local serial map,
   rerun the HTTP acceptance suite against the physical service, and begin
   calibration capture. Test the physical robot and camera backends together
   before treating deployment as complete.
