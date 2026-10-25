"""Record a third-person RoboTwin preview of a wrist-calibration pose library.

This is a visual review tool only.  It uses direct, simulator-only joint
targets to make the selected views understandable; real hardware must replay
the library through robotcontrol's guarded low-speed API instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sim.robotwin.robotwin_rendered import RoboTwinRenderedHandEyeHarness
from sim.virtual_calibration import _inverse


def _load_pose_library(path: Path) -> tuple[list[float], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = tuple([float(value) for value in item["qpos_rad"]] for item in payload["calibration_poses"])
    if len(poses) < 6:
        raise ValueError("pose library needs at least six poses")
    return poses


def _add_board_at_placement(task, harness: RoboTwinRenderedHandEyeHarness, workspace: Path, placement_qpos: list[float]) -> None:
    """Inject and PnP-centre the same board convention as the pose scanner."""
    mesh_path, _ = harness._make_target_mesh(workspace)
    import sapien

    builder = task.scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_visual_from_file(filename=str(mesh_path))
    actor = builder.build(name="dev_recording_charuco")
    camera = task.cameras.get_config()["left_camera"]
    camera_world = np.eye(4)
    camera_world[:3, :] = np.asarray(camera["extrinsic_cv"], dtype=np.float64)
    world_camera = _inverse(camera_world)
    forward, normal, x_axis = world_camera[:3, 2], -world_camera[:3, 2], world_camera[:3, 0]
    y_axis = np.cross(normal, x_axis); y_axis /= np.linalg.norm(y_axis)
    quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, normal))).as_quat()
    actor.set_pose(sapien.Pose(world_camera[:3, 3] + 0.48 * forward, [quat[3], *quat[:3]]))
    desired = np.array([320.0, 240.0])
    for _ in range(5):
        harness._drive_left(task, placement_qpos)
        image = np.asarray(task.get_obs()["observation"]["left_camera"]["rgb"])
        camera = task.cameras.get_config()["left_camera"]
        intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
        corners, ids, _, _ = harness.detector.detectBoard(image)
        if corners is None or ids is None or len(ids) < 12:
            raise RuntimeError("placement pose cannot see the injected ChArUco board")
        object_points = harness.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
        ok, _, translation = cv2.solvePnP(object_points, corners, intrinsic, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("PnP failed while centring recording board")
        centre = np.mean(corners.reshape(-1, 2), axis=0)
        error = desired - centre
        if float(np.linalg.norm(error)) < 1.0:
            break
        delta_camera = np.array([error[0] * translation[2, 0] / intrinsic[0, 0], error[1] * translation[2, 0] / intrinsic[1, 1], 0.0])
        pose = actor.get_pose()
        actor.set_pose(sapien.Pose(np.asarray(pose.p) + world_camera[:3, :3] @ delta_camera, pose.q))


def record(pose_library: Path, output: Path, *, fps: int = 15, capture_stride: int = 8, settle_steps: int = 220) -> dict[str, int | str]:
    poses = _load_pose_library(pose_library)
    workspace = output.parent
    workspace.mkdir(parents=True, exist_ok=True)
    harness = RoboTwinRenderedHandEyeHarness(output_dir=None, pose_pool=poses, center_target=True)
    env = harness._make_env()
    writer = None
    frames = 0
    try:
        env.reset(init_id=0)
        task = env._task
        assert task is not None
        _add_board_at_placement(task, harness, workspace, poses[0])
        velocity = np.full(6, 0.16, dtype=np.float64)
        for pose_index, target in enumerate(poses, start=1):
            for step in range(settle_steps):
                task.robot.set_arm_joints(np.asarray(target, dtype=np.float64), velocity, "left")
                task.scene.step()
                if step % capture_stride != 0:
                    continue
                task._update_render()
                frame = task.cameras.get_observer_rgb()
                annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(annotated, f"Calibration pose {pose_index:02d}/{len(poses)}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 230, 30), 2, cv2.LINE_AA)
                cv2.putText(annotated, "Simulator preview - not a hardware trajectory", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 210, 255), 1, cv2.LINE_AA)
                if writer is None:
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (annotated.shape[1], annotated.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer for {output}")
                writer.write(annotated)
                frames += 1
    finally:
        if writer is not None:
            writer.release()
        env.close()
    return {"video": str(output), "frames": frames, "fps": fps, "poses": len(poses)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--capture-stride", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(record(args.pose_library, args.output, fps=args.fps, capture_stride=args.capture_stride), indent=2))


if __name__ == "__main__":
    main()
