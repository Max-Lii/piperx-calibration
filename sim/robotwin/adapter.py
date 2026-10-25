"""Small SAPIEN workcell using the RoboTwin Piper-X assets and conventions.

Only joint replay, FK, cameras and a clean table are needed for calibration.
No VLA environment, policy, task objects or motion planner is imported.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def asset_root() -> Path:
    root = Path(os.environ.get("PIPERX_ROBOTWIN_ASSETS", ROOT / "third_party/robotwin_piperx_dual_assets")).expanduser().resolve()
    for arm in ("left", "right"):
        if not (root / f"embodiments/piper_x_{arm}/piper_x.urdf").is_file():
            raise FileNotFoundError(f"Missing Piper-X assets at {root}; run scripts/env_setup.sh --with-robotwin or set PIPERX_ROBOTWIN_ASSETS explicitly")
    return root


def _look_pose(position, forward, left):
    import sapien
    forward, left = np.asarray(forward, float), np.asarray(left, float)
    forward /= np.linalg.norm(forward)
    left -= np.dot(left, forward) * forward
    left /= np.linalg.norm(left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((forward, left, np.cross(forward, left)))
    matrix[:3, 3] = position
    return sapien.Pose(matrix)


class Robot:
    """The RoboTwin joint-drive and end-pose conventions, without planners."""

    def __init__(self, scene, root):
        import sapien
        self.configs = {}
        for arm, side in (("left", -1), ("right", 1)):
            directory = root / f"embodiments/piper_x_{arm}"
            config = yaml.safe_load((directory / "config.yml").read_text())
            self.configs[arm] = config
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            entity = loader.load(str(directory / config["urdf_path"]))
            if entity is None:
                raise RuntimeError(f"Unable to load {directory}")
            values = config["robot_pose"][0]
            position = np.asarray(values[:3], float) + [side * 0.5870514170838435 / 2, 0, 0]
            entity.set_root_pose(sapien.Pose(position, values[3:]))
            setattr(self, f"{arm}_entity", entity)
            joints = [entity.find_joint_by_name(name) for name in config["arm_joints_name"][0]]
            setattr(self, f"{arm}_arm_joints", joints)
            setattr(self, f"{arm}_camera", entity.find_link_by_name("camera"))
            for link in entity.get_links():
                link.set_mass(1)
            for joint in entity.get_active_joints():
                joint.set_drive_property(config["joint_stiffness"], config["joint_damping"])
            qpos = entity.get_qpos()
            for joint, value in zip(joints, config["homestate"][0], strict=True):
                qpos[entity.get_active_joints().index(joint)] = value
                joint.set_drive_target(value)
            entity.set_qpos(qpos)
            self.set_gripper(0, arm)

    def set_arm_joints(self, target_position, target_velocity, arm_tag):
        for arm in ("left", "right"):
            entity = getattr(self, f"{arm}_entity")
            entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        for joint, position, velocity in zip(getattr(self, f"{arm_tag}_arm_joints"), target_position, target_velocity, strict=True):
            joint.set_drive_target(position)
            joint.set_drive_velocity_target(velocity)

    def set_gripper(self, gripper_val, arm_tag, gripper_eps=0.1):
        config = self.configs[arm_tag]
        entity = getattr(self, f"{arm_tag}_entity")
        lo, hi = config["gripper_scale"]
        value = lo + np.clip(gripper_val, 0, 1) * (hi - lo)
        spec = config["gripper_name"][0]
        for name, multiplier, offset in [(spec["base"], 1, 0), *spec["mimic"]]:
            joint = entity.find_joint_by_name(name)
            joint.set_drive_target(value * multiplier + offset)
            joint.set_drive_velocity_target(0)

    def ee_pose(self, arm):
        config = self.configs[arm]
        entity = getattr(self, f"{arm}_entity")
        pose = entity.find_joint_by_name(config["ee_joints"][0]).global_pose
        matrix = pose.to_transformation_matrix().astype(float)
        matrix[:3, :3] = matrix[:3, :3] @ config["global_trans_matrix"] @ np.asarray(config["delta_matrix"])
        matrix[:3, 3] += matrix[:3, :3] @ [config["gripper_bias"] - 0.12, 0, 0]
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return [*matrix[:3, 3], q[3], *q[:3]]


class Cameras:
    # Match RoboTwin-PiperX: wrist-mounted grippers can be closer than 10 cm.
    # A 10 cm near plane cuts through their visible surfaces.
    NEAR_M = 0.01
    FAR_M = 100.0

    def __init__(self, scene, configs):
        import sapien
        self.all = {}
        for arm in ("left", "right"):
            self.all[f"{arm}_camera"] = self._create(scene, f"{arm}_camera", configs[arm].get("wrist_camera_calibration"))
            setattr(self, f"{arm}_camera", self.all[f"{arm}_camera"])
        config = configs["left"]
        head = self._create(scene, "head_camera", config.get("head_camera_calibration"))
        info = config["static_camera_list"][0]
        head.entity.set_pose(_look_pose(info["position"], info["forward"], info["left"]))
        self.all["head_camera"] = head
        self.static_camera_name, self.static_camera_list = ["head_camera"], [head]
        self.observer_camera = scene.add_camera("observer_camera", 640, 480, np.deg2rad(93), self.NEAR_M, self.FAR_M)
        # Preserve the RoboTwin observer's original pose construction.
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack(([0, -1, -1.02], [1, 0, 0], [0, -1.02, 1]))
        matrix[:3, 3] = [0, .23, 1.33]
        self.observer_camera.entity.set_pose(sapien.Pose(matrix))

    @classmethod
    def _create(cls, scene, name, calibration):
        camera = scene.add_camera(name, 640, 480, np.deg2rad(37), cls.NEAR_M, cls.FAR_M)
        if calibration:
            k = np.asarray(calibration["camera_matrix"], float).copy()
            width, height = calibration["image_size"]
            k[0] *= 640 / width
            k[1] *= 480 / height
            camera.set_perspective_parameters(cls.NEAR_M, cls.FAR_M, k[0, 0], k[1, 1], k[0, 2], k[1, 2], k[0, 1])
        return camera

    def get_config(self):
        return {name: {"intrinsic_cv": c.get_intrinsic_matrix(), "extrinsic_cv": c.get_extrinsic_matrix(), "cam2world_gl": c.get_model_matrix()} for name, c in self.all.items()}

    @staticmethod
    def rgb(camera):
        camera.take_picture()
        return np.clip(camera.get_picture("Color")[..., :3] * 255, 0, 255).astype(np.uint8)

    def get_observer_rgb(self):
        return self.rgb(self.observer_camera)


class Workcell:
    def __init__(self):
        import sapien
        sapien.render.set_global_config(max_num_materials=2048, max_num_textures=2048)
        # Use RoboTwin's raster renderer; no ray-tracing denoiser is required.
        sapien.render.set_camera_shader_dir("default")
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 250)
        self.scene.add_ground(0)
        self.scene.set_ambient_light([.6, .6, .6])
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1], shadow=True)
        for name, position, half in (("table", [0, 0, .715], [.6, .35, .025]), ("wall", [0, 1, 1.5], [3, .6, 1.5])):
            builder = self.scene.create_actor_builder()
            builder.set_physx_body_type("static")
            builder.add_box_collision(half_size=half)
            builder.add_box_visual(half_size=half, material=[.9, .9, .9, 1])
            actor = builder.build(name=name)
            actor.set_pose(sapien.Pose(position))
        self.robot = Robot(self.scene, asset_root())
        self.cameras = Cameras(self.scene, self.robot.configs)
        for _ in range(100):
            for arm in ("left", "right"):
                self.robot.set_arm_joints(self.robot.configs[arm]["homestate"][0], np.zeros(6), arm)
            self.scene.step()
        self._update_render()

    def _update_render(self):
        for arm in ("left", "right"):
            getattr(self.cameras, f"{arm}_camera").entity.set_pose(getattr(self.robot, f"{arm}_camera").get_pose())
        self.scene.update_render()

    def get_arm_pose(self, arm):
        return self.robot.ee_pose(arm)

    def get_obs(self):
        self._update_render()
        return {"observation": {name: {"rgb": self.cameras.rgb(camera)} for name, camera in self.cameras.all.items()}}

    def close(self):
        self.scene = None
        self.renderer = None
        self.engine = None


class Environment:
    def __init__(self, config):
        self._task = None

    def reset(self, init_id=0):
        if self._task is not None:
            raise RuntimeError("Create a new calibration environment for each replay")
        self._task = Workcell()

    def close(self):
        self._task = None


def load_robotwin_env():
    """Keep the existing capture tools' narrow factory API."""
    return Environment
