"""Record a high-resolution world-view preview of dual tabletop calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sim.generate_a4_charuco_pdf import MARKER_MM, SQUARE_MM, SQUARES
from sim.robotwin.robotwin_dual_tabletop import BOARD_CENTER_W, make_printable_target_mesh
from sim.robotwin.robotwin_rendered import RoboTwinRenderedHandEyeHarness


def _world_camera(task, width: int = 1280, height: int = 960):
    import sapien
    camera = task.scene.add_camera("calibration_world_view", width, height, np.deg2rad(70), 0.1, 100.0)
    position = np.array([0.0, 0.34, 1.42]); target = np.array([0.0, -0.08, 0.80])
    forward = target - position; forward /= np.linalg.norm(forward)
    left = np.array([1.0, 0.0, 0.0]); up = np.cross(forward, left); up /= np.linalg.norm(up)
    matrix = np.eye(4); matrix[:3, :3] = np.column_stack((forward, left, up)); matrix[:3, 3] = position
    camera.entity.set_pose(sapien.Pose(matrix)); return camera


def _frame(camera) -> np.ndarray:
    camera.take_picture(); rgba = camera.get_picture("Color")
    return cv2.cvtColor((rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def record(libraries: Path, output: Path, arm_only: str | None = None) -> dict:
    poses = json.loads(libraries.read_text(encoding="utf-8"))
    poses_per_arm = len(poses["left"])
    target_id = poses.get("target", {}).get("target_id", "BOARD_001")
    h = RoboTwinRenderedHandEyeHarness(output_dir=None)
    h.square_length_m, h.marker_length_m = SQUARE_MM / 1000.0, MARKER_MM / 1000.0
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    h.board = cv2.aruco.CharucoBoard(SQUARES, h.square_length_m, h.marker_length_m, dictionary)
    env = h._make_env(); writer = None; frames = 0
    try:
        env.reset(init_id=0); task = env._task; assert task is not None
        if hasattr(task, "bell"): task.scene.remove_actor(task.bell.actor)
        import sapien
        output.parent.mkdir(parents=True, exist_ok=True)
        asset_dir=Path("/tmp/robotwin_dual_video_assets"); asset_dir.mkdir(parents=True,exist_ok=True)
        mesh, _ = make_printable_target_mesh(asset_dir)
        builder=task.scene.create_actor_builder(); builder.set_physx_body_type("static"); builder.add_visual_from_file(filename=str(mesh))
        board=builder.build(name="flat_a4_charuco"); board.set_pose(sapien.Pose(BOARD_CENTER_W,[1,0,0,0]))
        camera=_world_camera(task); writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),15,(1280,960))
        if not writer.isOpened(): raise RuntimeError(f"cannot open {output}")
        park=[0,.05,-.05,0,0,0]; velocity=np.full(6,.16)
        arms=(arm_only,) if arm_only else ("left","right")
        for arm in arms:
            other="right" if arm=="left" else "left"
            for _ in range(120): task.robot.set_arm_joints(np.asarray(park),velocity,other); task.scene.step()
            for index,target in enumerate(poses[arm],start=1):
                for step in range(120):
                    task.robot.set_arm_joints(np.asarray(target),velocity,arm); task.scene.step()
                    if step % 40: continue
                    task._update_render(); image=_frame(camera)
                    cv2.putText(image,f"{arm.upper()} wrist calibration  {index:02d}/{poses_per_arm}",(28,52),cv2.FONT_HERSHEY_SIMPLEX,1.15,(20,230,20),3,cv2.LINE_AA)
                    cv2.putText(image,f"{target_id} flat A4 ChArUco on table - simulator motion preview",(28,90),cv2.FONT_HERSHEY_SIMPLEX,.68,(20,210,255),2,cv2.LINE_AA)
                    writer.write(image); frames += 1
            for _ in range(120): task.robot.set_arm_joints(np.asarray(park),velocity,arm); task.scene.step()
    finally:
        if writer is not None: writer.release()
        env.close()
    return {"video":str(output),"resolution":"1280x960","fps":15,"frames":frames,"arms":list(arms),"poses_per_arm":poses_per_arm,"target_id":target_id}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--pose-libraries",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--arm",choices=("left","right")); a=p.parse_args()
    print(json.dumps(record(a.pose_libraries,a.output,a.arm),indent=2))
if __name__=="__main__": main()
