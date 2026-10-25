from __future__ import annotations

import uvicorn

from .app import create_app
from .camera import MockCameraManager, RealSenseCameraManager
from .driver import MockDualPiperDriver
from .guardian import MotionGuardian
from .image_processing import ProcessedCameraManager, create_lut_processor
from .piper_x import PiperXConfig, PiperXDualDriver, PiperXMotionDriver
from .settings import load_settings


def main() -> None:
    settings = load_settings()
    if settings.camera_kind == "mock":
        cameras = MockCameraManager()
    elif settings.camera_kind == "realsense":
        cameras = RealSenseCameraManager(
            settings.camera_serial_by_role,
            enabled_camera_ids=settings.camera_enabled_roles,
            depth_enabled=settings.camera_depth_enabled,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
            warmup_frames=settings.camera_warmup_frames,
            lock_auto_controls_after_warmup=settings.camera_lock_auto_controls_after_warmup,
            manual_exposure=settings.camera_manual_exposure,
        )
    else:
        raise RuntimeError(f"unsupported camera kind: {settings.camera_kind}")
    if settings.image_processing_enabled:
        cameras = ProcessedCameraManager(
            cameras,
            create_lut_processor(
                settings.image_processing_backend,
                settings.image_processing_lut_by_role,
                device=settings.image_processing_device,
                pinned_memory=settings.image_processing_pinned_memory,
                torch_compile=settings.image_processing_torch_compile,
                input_shape=(settings.camera_height, settings.camera_width, 3),
            ),
            jpeg_quality=settings.image_processing_jpeg_quality,
        )
    if settings.driver_kind == "mock":
        driver = MockDualPiperDriver()
    elif settings.driver_kind in {"piper_x_readonly", "piper_x_motion"}:
        if not settings.left_can or not settings.right_can or not settings.firmware_profile:
            raise RuntimeError(f"{settings.driver_kind} requires driver.left_can, driver.right_can and driver.firmware_profile")
        config = PiperXConfig(
            settings.left_can, settings.right_can, settings.firmware_profile, settings.can_interface,
            motion_speed_percent=settings.motion_speed_percent or 5,
            joint_limits_rad=settings.joint_limits_rad,
            first_motion_max_joint_delta_rad=settings.first_motion_max_joint_delta_rad,
            allow_calibration_pose_motion=settings.allow_calibration_pose_motion,
            gripper_open_width_m=settings.gripper_open_width_m,
            gripper_force_n=settings.gripper_force_n,
        )
        if settings.driver_kind == "piper_x_motion":
            if settings.joint_limits_rad is None or settings.motion_speed_percent is None:
                raise RuntimeError("piper_x_motion requires explicit driver.joint_limits_rad and driver.motion_speed_percent")
            driver = PiperXMotionDriver(config)
        else:
            driver = PiperXDualDriver(config)
    else:
        raise RuntimeError(f"unsupported driver kind: {settings.driver_kind}")
    guardian = MotionGuardian(
        driver, safe_home=settings.safe_home,
        lease_ttl_s=settings.lease_ttl_s, idle_disconnect_s=settings.idle_disconnect_s, joint_limit_rad=settings.joint_limit_rad,
    )
    uvicorn.run(create_app(guardian=guardian, cameras=cameras), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
