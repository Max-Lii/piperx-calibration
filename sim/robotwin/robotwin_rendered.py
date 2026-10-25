"""Dev-only RoboTwin/SAPIEN rendered ChArUco hand-eye validation.

Unlike :mod:`virtual_calibration`, this harness uses the actual Piper-X URDF,
SAPIEN scene, wrist camera entity and simulator joint control.  It injects a
static textured ChArUco board into one throwaway task scene, commands a small
safe pose pool on the left arm, then recovers transforms from rendered RGB.

No upstream RoboTwin source is patched and no real robot service is involved.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sim.virtual_calibration import _inverse, _pose_error, _transform
from sim.robotwin.adapter import load_robotwin_env


@dataclass(frozen=True)
class RoboTwinRenderedResult:
    attempted_poses: int
    accepted_poses: int
    base_translation_error_mm: float
    base_rotation_error_deg: float
    handeye_translation_error_mm: float
    handeye_rotation_error_deg: float
    pnp_translation_median_mm: float
    pnp_rotation_median_deg: float
    passed: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return self.__dict__.copy()


class RoboTwinRenderedHandEyeHarness:
    """Actual RoboTwin wrist-camera regression with hidden SAPIEN truth."""

    squares_x, squares_y = 5, 7
    square_length_m, marker_length_m = 0.035, 0.026
    # Conservatively small, simulation-only pose pool around the verified home.
    default_pose_pool = (
        [0.00, 0.05, -0.05, 0.00, 0.00, 0.00],
        [0.14, 0.16, -0.18, 0.10, -0.08, 0.12],
        [-0.14, 0.17, -0.20, -0.10, 0.10, -0.11],
        [0.22, 0.07, -0.12, 0.18, -0.14, 0.08],
        [-0.22, 0.08, -0.13, -0.17, 0.14, -0.08],
        [0.10, 0.25, -0.30, 0.12, 0.10, -0.15],
        [-0.10, 0.24, -0.29, -0.12, -0.10, 0.15],
        [0.28, 0.20, -0.24, 0.05, -0.18, 0.20],
    )

    def __init__(
        self,
        *,
        output_dir: Path | None = None,
        gpu: int | None = None,
        pose_pool: tuple[list[float], ...] | None = None,
        center_target: bool = False,
        arm: str = "left",
    ) -> None:
        self.output_dir = output_dir
        self.gpu = gpu
        if arm not in {"left", "right"}:
            raise ValueError("arm must be left or right")
        self.arm = arm
        self.center_target = center_target
        if pose_pool is not None:
            if len(pose_pool) < 6:
                raise ValueError("a hand-eye pose library needs at least six joint poses")
            self.pose_pool = pose_pool
        else:
            self.pose_pool = self.default_pose_pool
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_length_m, self.marker_length_m, dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def _make_target_mesh(self, directory: Path) -> tuple[Path, np.ndarray]:
        """Create a one-sided UV board mesh; no assets are written into RoboTwin."""
        width, height = self.squares_x * self.square_length_m, self.squares_y * self.square_length_m
        image = self.board.generateImage((1000, 1400), marginSize=0)
        cv2.imwrite(str(directory / "board.png"), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        (directory / "board.mtl").write_text("newmtl board\nmap_Kd board.png\n", encoding="utf-8")
        half_x, half_y = width / 2, height / 2
        (directory / "board.obj").write_text(
            "mtllib board.mtl\nusemtl board\n"
            f"v {-half_x} {-half_y} 0\nv {half_x} {-half_y} 0\nv {half_x} {half_y} 0\nv {-half_x} {half_y} 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n",
            encoding="utf-8",
        )
        # The target's ChArUco origin is its top-left texture corner, not the
        # actor origin at its geometric centre.
        return directory / "board.obj", np.array([-half_x, -half_y, 0.0])

    @staticmethod
    def _pose_matrix(pose) -> np.ndarray:
        return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)

    @staticmethod
    def _list_pose_matrix(values: list[float]) -> np.ndarray:
        import sapien

        return np.asarray(sapien.Pose(values[:3], values[3:]).to_transformation_matrix(), dtype=np.float64)

    def _make_env(self):
        return load_robotwin_env()({})

    @staticmethod
    def _clear_task_objects(task) -> None:
        """Leave only the workcell geometry for calibration captures."""
        keep = {"ground", "wall", "table"}
        for actor in list(task.scene.get_all_actors()):
            if str(actor.get_name()) not in keep:
                task.scene.remove_actor(actor)

    @staticmethod
    def _drive_arm(task, target: list[float], arm: str) -> None:
        velocity = np.full(6, 0.25, dtype=np.float64)
        for _ in range(220):
            task.robot.set_arm_joints(np.asarray(target, dtype=np.float64), velocity, arm)
            task.scene.step()
        task._update_render()

    def run(self) -> RoboTwinRenderedResult:
        # GPU selection is deliberately delegated to CUDA_VISIBLE_DEVICES so
        # this harness obeys the caller's normal RoboTwin resource policy.
        del self.gpu
        env = self._make_env()
        workspace = self.output_dir or Path(tempfile.mkdtemp(prefix="robotwin_calibration_"))
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            env.reset(init_id=0)
            task = env._task
            assert task is not None
            self._clear_task_objects(task)
            mesh_path, target_origin_local = self._make_target_mesh(workspace)
            import sapien

            builder = task.scene.create_actor_builder()
            builder.set_physx_body_type("static")
            builder.add_visual_from_file(filename=str(mesh_path))
            target_actor = builder.build(name="dev_calibration_charuco")
            # Derive the board pose from the *actual OpenCV* wrist-camera
            # extrinsic at home.  Do not use SAPIEN's GL camera forward axis:
            # its optical direction is the opposite convention.  This gives a
            # fixed, front-facing board centred at the home optical ray.
            home_cw = np.eye(4)
            home_cw[:3, :] = np.asarray(task.cameras.get_config()[f"{self.arm}_camera"]["extrinsic_cv"], dtype=np.float64)
            home_wc = _inverse(home_cw)
            forward_w = home_wc[:3, 2]
            normal_w = -forward_w
            x_w = home_wc[:3, 0]
            y_w = np.cross(normal_w, x_w)
            y_w /= np.linalg.norm(y_w)
            board_rotation = np.column_stack((x_w, y_w, normal_w))
            board_center = home_wc[:3, 3] + 0.48 * forward_w
            quat_xyzw = Rotation.from_matrix(board_rotation).as_quat()
            target_actor.set_pose(sapien.Pose(board_center, [quat_xyzw[3], *quat_xyzw[:3]]))
            if self.center_target:
                # Match robotwin_pose_scan's operator-placement convention:
                # choose the target position from rendered PnP, not an assumed
                # mesh coordinate convention, until its visual centre is the
                # image centre at the designated first pose.
                desired_centre = np.array([320.0, 240.0])
                for _ in range(5):
                    self._drive_arm(task, self.pose_pool[0], self.arm)
                    image = np.asarray(task.get_obs()["observation"][f"{self.arm}_camera"]["rgb"])
                    camera = task.cameras.get_config()[f"{self.arm}_camera"]
                    intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
                    corners, ids, _, _ = self.detector.detectBoard(image)
                    if corners is None or ids is None or len(ids) < 12:
                        raise RuntimeError("cannot centre rendered target: placement pose failed ChArUco gate")
                    object_points = self.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
                    ok, rvec, translation = cv2.solvePnP(object_points, corners, intrinsic, None, flags=cv2.SOLVEPNP_ITERATIVE)
                    if not ok:
                        raise RuntimeError("cannot centre rendered target: PnP failed")
                    centre = np.mean(corners.reshape(-1, 2), axis=0)
                    error = desired_centre - centre
                    if float(np.linalg.norm(error)) < 1.0:
                        break
                    camera_delta = np.array([error[0] * translation[2, 0] / intrinsic[0, 0], error[1] * translation[2, 0] / intrinsic[1, 1], 0.0])
                    pose = target_actor.get_pose()
                    target_actor.set_pose(sapien.Pose(np.asarray(pose.p) + home_wc[:3, :3] @ camera_delta, pose.q))
            board_w, board_h = self.squares_x * self.square_length_m, self.squares_y * self.square_length_m
            actor_world = self._pose_matrix(target_actor.get_pose())
            # Candidate mesh-to-printed-board mappings are evaluator metadata.
            # The selected one is not passed to PnP or to the hand-eye solver.
            local_targets = [
                _transform(np.eye(3), target_origin_local),
                _transform(Rotation.from_euler("x", np.pi).as_matrix(), target_origin_local + np.array([0, board_h, 0])),
                _transform(Rotation.from_euler("y", np.pi).as_matrix(), target_origin_local + np.array([board_w, 0, 0])),
                _transform(Rotation.from_euler("z", np.pi).as_matrix(), target_origin_local + np.array([board_w, board_h, 0])),
            ]
            world_targets = [actor_world @ value for value in local_targets]

            ctw_rot, ctw_trans, e2b_rot, e2b_trans = [], [], [], []
            pnp_errors_by_mapping: list[list[tuple[float, float]]] = [[] for _ in world_targets]
            for index, target in enumerate(self.pose_pool):
                self._drive_arm(task, target, self.arm)
                raw = task.get_obs()
                image = np.asarray(raw["observation"][f"{self.arm}_camera"]["rgb"])
                if self.output_dir is not None:
                    cv2.imwrite(str(workspace / f"{self.arm}_{index:02d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                corners, ids, _, _ = self.detector.detectBoard(image)
                if corners is None or ids is None or len(ids) < 8:
                    continue
                camera = task.cameras.get_config()[f"{self.arm}_camera"]
                intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
                object_points = self.board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
                ok, rvec, translation = cv2.solvePnP(object_points, corners, intrinsic, None, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                rotation, _ = cv2.Rodrigues(rvec)
                estimated_camera_target = _transform(rotation, translation.reshape(3))
                camera_world = np.eye(4)
                camera_world[:3, :] = np.asarray(camera["extrinsic_cv"], dtype=np.float64)
                for errors, world_target in zip(pnp_errors_by_mapping, world_targets, strict=True):
                    errors.append(_pose_error(estimated_camera_target, camera_world @ world_target))
                world_base = self._pose_matrix(getattr(task.robot, f"{self.arm}_entity").get_pose())
                world_ee = self._list_pose_matrix(task.get_arm_pose(self.arm))
                base_ee = _inverse(world_base) @ world_ee
                ee_base = _inverse(base_ee)
                ctw_rot.append(rotation); ctw_trans.append(translation.reshape(3, 1))
                e2b_rot.append(ee_base[:3, :3]); e2b_trans.append(ee_base[:3, 3:4])

            if len(ctw_rot) < 6:
                raise RuntimeError(f"only {len(ctw_rot)} simulated wrist frames passed the ChArUco gate")
            r_tb, t_tb, r_ce, t_ce = cv2.calibrateRobotWorldHandEye(ctw_rot, ctw_trans, e2b_rot, e2b_trans, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
            estimated_target_base = _transform(r_tb, t_tb.reshape(3))
            estimated_camera_ee = _transform(r_ce, t_ce.reshape(3))
            mapping_scores = [float(np.median(np.asarray(errors)[:, 0])) for errors in pnp_errors_by_mapping]
            world_target = world_targets[int(np.argmin(mapping_scores))]
            pnp_errors = pnp_errors_by_mapping[int(np.argmin(mapping_scores))]
            # Truth remains evaluator-only.  Use the final scene state only to
            # calculate errors after OpenCV has finished solving.
            world_base = self._pose_matrix(getattr(task.robot, f"{self.arm}_entity").get_pose())
            world_ee = self._list_pose_matrix(task.get_arm_pose(self.arm))
            camera_world = np.eye(4); camera_world[:3, :] = np.asarray(task.cameras.get_config()[f"{self.arm}_camera"]["extrinsic_cv"], dtype=np.float64)
            truth_target_base = _inverse(world_target) @ world_base
            truth_camera_ee = camera_world @ world_ee
            base_error = _pose_error(estimated_target_base, truth_target_base)
            handeye_error = _pose_error(estimated_camera_ee, truth_camera_ee)
            pnp = np.asarray(pnp_errors)
            return RoboTwinRenderedResult(
                attempted_poses=len(self.pose_pool), accepted_poses=len(ctw_rot),
                base_translation_error_mm=base_error[0] * 1000, base_rotation_error_deg=base_error[1],
                handeye_translation_error_mm=handeye_error[0] * 1000, handeye_rotation_error_deg=handeye_error[1],
                pnp_translation_median_mm=float(np.median(pnp[:, 0]) * 1000), pnp_rotation_median_deg=float(np.median(pnp[:, 1]),),
                passed=base_error[0] < 0.010 and base_error[1] < 1.0 and handeye_error[0] < 0.010 and handeye_error[1] < 1.0,
            )
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pose-library",
        type=Path,
        help="Pose-library JSON produced by robotwin_pose_scan; defaults to the small built-in regression pool.",
    )
    parser.add_argument("--center-target", action="store_true", help="Use the same image-centred board placement convention as robotwin_pose_scan.")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    args = parser.parse_args()
    pose_pool = None
    if args.pose_library is not None:
        payload = json.loads(args.pose_library.read_text(encoding="utf-8"))
        pose_pool = tuple([float(value) for value in item["qpos_rad"]] for item in payload["calibration_poses"])
    result = RoboTwinRenderedHandEyeHarness(output_dir=args.output_dir, pose_pool=pose_pool, center_target=args.center_target, arm=args.arm).run()
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    if not result.passed:
        raise SystemExit("RoboTwin rendered hand-eye validation did not meet thresholds")


if __name__ == "__main__":
    main()
