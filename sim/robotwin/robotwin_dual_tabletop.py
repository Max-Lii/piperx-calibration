"""Dual-Piper calibration regression with one flat A4 ChArUco target."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sim.generate_a4_charuco_pdf import (
    A4_MM, BOARD_TOP_LEFT_MM, MARKER_MM, SQUARE_MM, SQUARES, render_page,
)
from sim.robotwin.robotwin_rendered import RoboTwinRenderedHandEyeHarness
from sim.virtual_calibration import _inverse, _pose_error, _transform


TABLE_Z = 0.741
BOARD_CENTER_W = np.array([0.0, -0.08, TABLE_Z + 0.001], dtype=np.float64)
# The printed A4 alignment X and ChArUco board X are defined in this page
# frame.  The simulator places that page with +X exactly along world +X.
BOARD_X_W = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def make_printable_target_mesh(directory: Path) -> tuple[Path, np.ndarray]:
    """Create the exact printable A4 page, including margins and alignment X."""
    directory.mkdir(parents=True, exist_ok=True)
    # BOARD_002-style outer fiducials are part of the simulated print as well;
    # the existing A4 dimensions and ALIGN X geometry remain unchanged.
    render_page(metric_fiducials=True).save(directory / "a4_target.png")
    (directory / "a4_target.mtl").write_text(
        "newmtl target\nmap_Kd a4_target.png\n", encoding="utf-8",
    )
    width, height = (value / 1000.0 for value in A4_MM)
    half_x, half_y = width / 2.0, height / 2.0
    (directory / "a4_target.obj").write_text(
        "mtllib a4_target.mtl\nusemtl target\n"
        f"v {-half_x} {-half_y} 0\nv {half_x} {-half_y} 0\n"
        f"v {half_x} {half_y} 0\nv {-half_x} {half_y} 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n",
        encoding="utf-8",
    )
    board_origin = np.asarray([
        -half_x + BOARD_TOP_LEFT_MM[0] / 1000.0,
        -half_y + BOARD_TOP_LEFT_MM[1] / 1000.0,
        0.0,
    ])
    return directory / "a4_target.obj", board_origin


@dataclass(frozen=True)
class DualTabletopResult:
    target_id: str
    square_length_mm: float
    marker_length_mm: float
    requested_poses_per_arm: int
    left_accepted_poses: int
    right_accepted_poses: int
    left_base_translation_error_mm: float
    right_base_translation_error_mm: float
    left_handeye_translation_error_mm: float
    right_handeye_translation_error_mm: float
    dual_base_distance_truth_mm: float
    dual_base_distance_estimated_mm: float
    dual_base_distance_error_mm: float
    head_height_truth_mm: float
    head_height_estimated_mm: float
    head_height_error_mm: float
    head_pitch_truth_deg: float
    head_pitch_estimated_deg: float
    head_pitch_error_deg: float
    head_pnp_translation_error_mm: float
    head_pnp_rotation_error_deg: float
    passed: bool


def _select_poses(paths: list[Path], arm: str, wanted: int = 36) -> tuple[list[float], ...]:
    payload = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Pose scanners emit one arm per file; retain compatibility with the
        # older combined {left: [...], right: [...]} format as well.
        payload.extend(data.get("calibration_poses", data.get(arm, [])))
    eligible = [item for item in payload if item["charuco_corners"] >= 8]
    if len(eligible) < wanted:
        raise RuntimeError(f"{arm}: only {len(eligible)} PnP-capable candidates")
    q = np.asarray([item["qpos_rad"] for item in eligible], dtype=np.float64)
    scale = np.maximum(np.ptp(q, axis=0), 1e-6)
    feature = (q - np.mean(q, axis=0)) / scale
    quality = np.asarray([min(item["charuco_corners"], 24) / 24 for item in eligible])
    chosen = [int(np.argmax(quality))]
    minimum = np.linalg.norm(feature - feature[chosen[0]], axis=1)
    while len(chosen) < wanted:
        score = minimum * (0.65 + 0.35 * quality)
        score[chosen] = -1
        index = int(np.argmax(score)); chosen.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(feature - feature[index], axis=1))
    return tuple([float(v) for v in eligible[i]["qpos_rad"]] for i in chosen)


def _camera_matrix(config: dict) -> np.ndarray:
    value = np.eye(4); value[:3, :] = np.asarray(config["extrinsic_cv"], dtype=np.float64)
    return value


def _pitch_deg(world_camera: np.ndarray) -> float:
    forward = world_camera[:3, 2]
    return float(np.degrees(np.arctan2(-forward[2], np.linalg.norm(forward[:2]))))


class DualTabletopHarness:
    def __init__(self, left_candidates: list[Path], right_candidates: list[Path], output_dir: Path, *, wanted: int = 36, pose_library: Path | None = None) -> None:
        self.wanted = wanted
        if pose_library is not None:
            payload = json.loads(pose_library.read_text(encoding="utf-8"))
            self.left = tuple([float(value) for value in pose] for pose in payload["left"][:wanted])
            self.right = tuple([float(value) for value in pose] for pose in payload["right"][:wanted])
            if len(self.left) != wanted or len(self.right) != wanted:
                raise ValueError(f"pose library must contain at least {wanted} poses per arm")
        else:
            self.left = _select_poses(left_candidates, "left", wanted)
            self.right = _select_poses(right_candidates, "right", wanted)
        self.output_dir = output_dir
        self.h = RoboTwinRenderedHandEyeHarness(output_dir=None)
        self.h.square_length_m, self.h.marker_length_m = SQUARE_MM / 1000.0, MARKER_MM / 1000.0
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.h.board = cv2.aruco.CharucoBoard(SQUARES, self.h.square_length_m, self.h.marker_length_m, dictionary)
        self.h.detector = cv2.aruco.CharucoDetector(self.h.board)

    @staticmethod
    def _drive(task, target: list[float], arm: str, steps: int = 120) -> None:
        for _ in range(steps):
            task.robot.set_arm_joints(np.asarray(target), np.full(6, 0.18), arm); task.scene.step()
        task._update_render()

    def _inject_board(self, task):
        import sapien
        mesh, origin = make_printable_target_mesh(self.output_dir)
        builder = task.scene.create_actor_builder(); builder.set_physx_body_type("static")
        builder.add_visual_from_file(filename=str(mesh)); actor = builder.build(name="flat_a4_charuco")
        # The shared target is fixed flat at the tabletop centre.  Each arm is
        # moved into its own downward-facing calibration poses before capture.
        actor.set_pose(sapien.Pose(BOARD_CENTER_W, [1, 0, 0, 0]))
        width, height = SQUARES[0] * self.h.square_length_m, SQUARES[1] * self.h.square_length_m
        actor_world = self.h._pose_matrix(actor.get_pose())
        local = [
            _transform(np.eye(3), origin),
            _transform(Rotation.from_euler("x", np.pi).as_matrix(), origin + [0, height, 0]),
            _transform(Rotation.from_euler("y", np.pi).as_matrix(), origin + [width, 0, 0]),
            _transform(Rotation.from_euler("z", np.pi).as_matrix(), origin + [width, height, 0]),
        ]
        return actor, [actor_world @ item for item in local]

    @staticmethod
    def _camera_config(task, arm: str) -> dict:
        return task.cameras.get_config()[f"{arm}_camera"]

    @staticmethod
    def _save_highres_sample(task, source, config: dict | None, path: Path, *, fovy_deg: float = 60.0) -> None:
        """Save a 4:3 4K view with the source camera's exact projection."""
        import sapien
        width, height = 3840, 2880
        source_width, source_height = 640.0, 480.0
        scale_x, scale_y = width / source_width, height / source_height
        camera = task.scene.add_camera(
            name=f"doc_{path.stem}", width=width, height=height,
            fovy=np.deg2rad(fovy_deg), near=0.01, far=100.0,
        )
        if config is not None:
            matrix = np.asarray(config["intrinsic_cv"], dtype=np.float64)
            camera.set_perspective_parameters(
                0.01, 100.0,
                float(matrix[0, 0] * scale_x), float(matrix[1, 1] * scale_y),
                float(matrix[0, 2] * scale_x), float(matrix[1, 2] * scale_y),
                float(matrix[0, 1] * scale_x),
            )
        camera.entity.set_pose(source.entity.get_pose())
        task._update_render()
        camera.take_picture()
        rgba = np.asarray(camera.get_picture("Color"))
        rgb = rgba[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def _solve_arm(self, task, arm: str, poses: tuple[list[float], ...], world_targets: list[np.ndarray]):
        cw_r=[]; cw_t=[]; bg_r=[]; bg_t=[]; mapping_errors=[[] for _ in world_targets]
        accepted_qposes=[]; rejected_qposes=[]
        for index, pose in enumerate(poses):
            self._drive(task, pose, arm)
            raw=task.get_obs(); image=np.asarray(raw["observation"][f"{arm}_camera"]["rgb"])
            cv2.imwrite(str(self.output_dir / f"{arm}_{index:02d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if index == 12:
                self._save_highres_sample(
                    task, getattr(task.cameras, f"{arm}_camera"),
                    self._camera_config(task, arm), self.output_dir / f"{arm}_wrist_4k.png",
                )
            corners, ids, _, _ = self.h.detector.detectBoard(image)
            if corners is None or ids is None or len(ids) < 4:
                rejected_qposes.append(pose); continue
            cfg=task.cameras.get_config()[f"{arm}_camera"]; k=np.asarray(cfg["intrinsic_cv"])
            obj=self.h.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
            pnp_method = cv2.SOLVEPNP_ITERATIVE if len(ids) >= 6 else cv2.SOLVEPNP_SQPNP
            ok,rvec,tvec=cv2.solvePnP(obj,corners,k,None,flags=pnp_method)
            if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
                rejected_qposes.append(pose); continue
            rot,_=cv2.Rodrigues(rvec); ct=_transform(rot,tvec.reshape(3)); camera_world=_camera_matrix(cfg)
            for errors,wt in zip(mapping_errors,world_targets,strict=True): errors.append(_pose_error(ct,camera_world@wt))
            world_base=self.h._pose_matrix(getattr(task.robot,f"{arm}_entity").get_pose())
            world_ee=self.h._list_pose_matrix(task.get_arm_pose(arm)); ee_base=_inverse(_inverse(world_base)@world_ee)
            cw_r.append(rot); cw_t.append(tvec); bg_r.append(ee_base[:3,:3]); bg_t.append(ee_base[:3,3:4])
            accepted_qposes.append(pose)
        if len(cw_r)<12: raise RuntimeError(f"{arm}: only {len(cw_r)} frames survived replay")
        rb,tb,rc,tc=cv2.calibrateRobotWorldHandEye(cw_r,cw_t,bg_r,bg_t,method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
        chosen=int(np.argmin([np.median(np.asarray(x)[:,0]) for x in mapping_errors]))
        target_base=_transform(rb,tb.reshape(3)); camera_ee=_transform(rc,tc.reshape(3))
        world_base_truth=self.h._pose_matrix(getattr(task.robot,f"{arm}_entity").get_pose())
        world_ee=self.h._list_pose_matrix(task.get_arm_pose(arm)); camera_world=_camera_matrix(task.cameras.get_config()[f"{arm}_camera"])
        truth_tb=_inverse(world_targets[chosen])@world_base_truth; truth_ce=camera_world@world_ee
        world_base_est=world_targets[chosen]@target_base
        return len(cw_r),world_base_est,world_base_truth,_pose_error(target_base,truth_tb),_pose_error(camera_ee,truth_ce),chosen,accepted_qposes,rejected_qposes

    def run(self) -> DualTabletopResult:
        self.output_dir.mkdir(parents=True,exist_ok=True); env=self.h._make_env()
        try:
            env.reset(init_id=0); task=env._task; assert task is not None
            self.h._clear_task_objects(task)
            _,world_targets=self._inject_board(task)
            left=self._solve_arm(task,"left",self.left,world_targets)
            self._drive(task,[0,.05,-.05,0,0,0],"left")
            right=self._solve_arm(task,"right",self.right,world_targets)
            raw=task.get_obs(); cfg=task.cameras.get_config()["head_camera"]; image=np.asarray(raw["observation"]["head_camera"]["rgb"])
            cv2.imwrite(str(self.output_dir / "world_view.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            head_index = task.cameras.static_camera_name.index("head_camera")
            self._save_highres_sample(
                task, task.cameras.static_camera_list[head_index], cfg,
                self.output_dir / "head_camera_view_4k.png",
            )
            self._save_highres_sample(
                task, task.cameras.observer_camera, None,
                self.output_dir / "world_view_4k.png", fovy_deg=93.0,
            )
            corners,ids,_,_=self.h.detector.detectBoard(image)
            if corners is None or ids is None or len(ids)<8: raise RuntimeError("head camera cannot detect flat A4")
            k=np.asarray(cfg["intrinsic_cv"]); obj=self.h.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
            ok,rvec,tvec=cv2.solvePnP(obj,corners,k,None,flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: raise RuntimeError("head camera PnP failed")
            rot,_=cv2.Rodrigues(rvec); camera_target=_transform(rot,tvec.reshape(3)); camera_world_truth=_camera_matrix(cfg)
            head_errors=[_pose_error(camera_target,camera_world_truth@wt) for wt in world_targets]
            mapping=int(np.argmin([x[0] for x in head_errors])); world_camera_est=world_targets[mapping]@_inverse(camera_target); world_camera_truth=_inverse(camera_world_truth)
            distance_truth=np.linalg.norm(left[2][:3,3]-right[2][:3,3]); distance_est=np.linalg.norm(left[1][:3,3]-right[1][:3,3])
            height_truth=world_camera_truth[2,3]-TABLE_Z; height_est=world_camera_est[2,3]-TABLE_Z
            pitch_truth=_pitch_deg(world_camera_truth); pitch_est=_pitch_deg(world_camera_est)
            result=DualTabletopResult("BOARD_002",SQUARE_MM,MARKER_MM,self.wanted,left[0],right[0],left[3][0]*1000,right[3][0]*1000,left[4][0]*1000,right[4][0]*1000,distance_truth*1000,distance_est*1000,abs(distance_est-distance_truth)*1000,height_truth*1000,height_est*1000,abs(height_est-height_truth)*1000,pitch_truth,pitch_est,abs(pitch_est-pitch_truth),head_errors[mapping][0]*1000,head_errors[mapping][1],False)
            passed=result.left_accepted_poses>=30 and result.right_accepted_poses>=30 and result.dual_base_distance_error_mm<10 and result.head_height_error_mm<10 and result.head_pitch_error_deg<1
            result=DualTabletopResult(**{**asdict(result),"passed":passed})
            (self.output_dir/"dual_tabletop_result.json").write_text(json.dumps(asdict(result),indent=2)+"\n")
            (self.output_dir/"pose_libraries.json").write_text(json.dumps({
                "schema_version": 2,
                "target": {
                    "target_id": "BOARD_002", "square_length_mm": SQUARE_MM,
                    "marker_length_mm": MARKER_MM, "page_x_axis_world": BOARD_X_W.tolist(),
                    "placement": "flat A4; ALIGN X and ChArUco page +X align with world +X",
                },
                "simulation_only_not_motion_approved": True,
                "left": self.left,
                "right": self.right,
            },indent=2)+"\n")
            (self.output_dir / "pose_audit.json").write_text(json.dumps({
                "left": {"accepted_qpos_rad": left[6], "rejected_qpos_rad": left[7]},
                "right": {"accepted_qpos_rad": right[6], "rejected_qpos_rad": right[7]},
            }, indent=2) + "\n")
            return result
        finally: env.close()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--left-candidates",type=Path,nargs="+",default=[]); p.add_argument("--right-candidates",type=Path,nargs="+",default=[]); p.add_argument("--pose-library",type=Path); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--wanted",type=int,default=36); a=p.parse_args()
    if a.pose_library is None and (not a.left_candidates or not a.right_candidates):
        p.error("provide --pose-library or both --left-candidates and --right-candidates")
    result=DualTabletopHarness(a.left_candidates,a.right_candidates,a.output_dir,wanted=a.wanted,pose_library=a.pose_library).run(); print(json.dumps(asdict(result),indent=2))
    if not result.passed: raise SystemExit("dual tabletop calibration thresholds failed")
if __name__=="__main__": main()
