"""Render cover material without changing calibration code or observations.

Run from the repository root in the simulation environment. Supply --camera-mesh
with the official realsense-ros d435.dae. Outputs stay in ignored temp/ by default.
The camera housings and support stand are decorative visuals, without collisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from sim.robotwin.robotwin_dual_tabletop import DualTabletopHarness, BOARD_CENTER_W


def look_at(position, target):
    forward = np.asarray(target) - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((forward, left, np.cross(forward, left)))
    matrix[:3, 3] = position
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('temp/banner_render'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import sapien

    harness = DualTabletopHarness([], [], args.output, wanted=30,
                                  pose_library=Path('doc/data/BOARD_002_pose_libraries.json'))
    env = harness.h._make_env()
    try:
        env.reset(init_id=0)
        task = env._task
        harness._inject_board(task)
        # Cover-only mounting risers bridge the asset's 20 mm base clearance.
        for arm in ('left', 'right'):
            base = getattr(task.robot, arm + '_entity').get_pose().p
            builder = task.scene.create_actor_builder()
            builder.set_physx_body_type('static')
            builder.add_box_visual(half_size=[.07, .07, .01], material=[.22,.25,.29,1])
            actor = builder.build(name='cover_mount_' + arm)
            actor.set_pose(sapien.Pose([base[0], base[1]+.025, .75]))
        for arm in ('left', 'right'):
            task.robot.set_gripper(0.65, arm)
            harness._drive(task, list(getattr(harness, arm)[5]), arm)
        task._update_render()
        points = {'board': BOARD_CENTER_W.tolist()}
        # Official xacro: mesh rotation Rx(pi/2), Rz(pi/2), front-plate
        # displacement +4.3 mm; camera_link origin is the depth optical origin.
        local = np.eye(4)
        local[:3, :3] = Rotation.from_euler('xyz', [np.pi/2, 0, np.pi/2]).as_matrix()
        local[:3, 3] = [0.0043, -0.0175, 0]
        for name, camera in task.cameras.all.items():
            builder = task.scene.create_actor_builder()
            builder.set_physx_body_type('static')
            builder.add_visual_from_file(filename=str(args.camera_mesh.resolve()), pose=sapien.Pose(local))
            # The upstream CAD mesh is uniformly silver. Add the dark optical
            # face and lens windows as cover-only styling, at nominal offsets.
            builder.add_box_visual(pose=sapien.Pose([.0048,-.0175,0]),
                                   half_size=[.0006,.040,.008], material=[.025,.032,.04,1])
            for y in (-.05, -.027, 0, .015):
                builder.add_sphere_visual(pose=sapien.Pose([.0055,y,0]),
                                          radius=.0038, material=[.12,.19,.24,1])
            actor = builder.build(name='cover_' + name)
            actor.set_pose(camera.entity.get_pose())
            points[name] = camera.entity.get_pose().p.tolist()
            if name != 'head_camera':
                arm = name.split('_')[0]
                start = getattr(task.robot, arm+'_entity').find_link_by_name('link6').get_pose().p
                end = camera.entity.get_pose().p
                support = task.scene.create_actor_builder()
                support.set_physx_body_type('static')
                support.add_box_visual(half_size=[float(np.linalg.norm(end-start))/2,.004,.004],
                                       material=[.18,.20,.23,1])
                bracket = support.build(name='cover_wrist_bracket_'+arm)
                bracket.set_pose(sapien.Pose(look_at((start+end)/2, end)))
        # Simple stand beneath the fixed camera; separate from the robots.
        head = np.asarray(points['head_camera'])
        builder = task.scene.create_actor_builder()
        builder.set_physx_body_type('static')
        builder.add_box_visual(half_size=[0.008, 0.008, (head[2]-.74)/2], material=[.16,.20,.25,1])
        stand = builder.build(name='cover_camera_stand')
        stand.set_pose(sapien.Pose([head[0], head[1]-.035, (.74+head[2])/2]))
        views = {}
        for i, position in enumerate(([1.30, 1.55, 2.15], [-1.25, 1.60, 2.10], [.95, 1.9, 2.15])):
            target = np.array([0, -.12, 1.0])
            position = target + .8 * (np.asarray(position, float) - target)
            matrix = look_at(position, target)
            width, height, fovy = 3840, 2560, np.deg2rad(38)
            camera = task.scene.add_camera(f'cover_view_{i}', width, height, fovy, .01, 100)
            camera.entity.set_pose(sapien.Pose(matrix))
            task._update_render()
            camera.take_picture()
            rgb = np.clip(camera.get_picture('Color')[...,:3]*255, 0, 255).astype('uint8')
            Image.fromarray(rgb).save(args.output / f'view_{i}.png')
            k = np.asarray(camera.get_intrinsic_matrix())
            extrinsic = np.asarray(camera.get_extrinsic_matrix())
            projected = {}
            for name, point in points.items():
                p = extrinsic @ np.r_[point, 1]
                uv = k @ p[:3]
                projected[name] = (uv[:2]/uv[2]).tolist()
            views[str(i)] = {'pose': matrix.tolist(), 'points_px': projected}
        (args.output/'views.json').write_text(json.dumps(views, indent=2)+'\n')
        print('Rendered three cover views to', args.output, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
