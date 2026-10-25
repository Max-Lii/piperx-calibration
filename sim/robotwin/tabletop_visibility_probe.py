"""Probe left/right wrist visibility of a flat A4 ChArUco target on the table."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sim.generate_a4_charuco_pdf import MARKER_MM, SQUARE_MM, SQUARES
from sim.robotwin.robotwin_dual_tabletop import make_printable_target_mesh
from sim.robotwin.robotwin_rendered import RoboTwinRenderedHandEyeHarness


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--arm', choices=('left','right'), default='left'); parser.add_argument('--seed',type=int); args=parser.parse_args()
    workspace = Path('/tmp/tabletop_visibility_probe'); workspace.mkdir(exist_ok=True)
    h = RoboTwinRenderedHandEyeHarness(output_dir=None)
    h.square_length_m, h.marker_length_m = SQUARE_MM / 1000., MARKER_MM / 1000.
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    h.board = cv2.aruco.CharucoBoard(SQUARES, h.square_length_m, h.marker_length_m, d)
    h.detector = cv2.aruco.CharucoDetector(h.board)
    env = h._make_env()
    # Broad but deliberately raised simulator-only candidates.  They are not
    # authorised hardware poses.
    # The Piper wrist optical frame points upward in the old near-zero pool;
    # probe substantial wrist rotations before deciding the tabletop envelope.
    anchor=np.array([-.45,.25,-.55,1.,0.,0.]) if args.arm=='left' else np.array([.45,.25,-.55,1.,0.,0.])
    seed=args.seed if args.seed is not None else (43 if args.arm=='left' else 47)
    rng=np.random.default_rng(seed)
    scale=np.array([.20,.22,.24,.42,.42,.55])
    candidates=[anchor.tolist(), *((anchor+rng.uniform(-1,1,6)*scale).tolist() for _ in range(95))]
    out={}
    try:
        env.reset(init_id=0); task=env._task; assert task is not None
        if hasattr(task, 'bell'): task.scene.remove_actor(task.bell.actor)
        import sapien
        mesh,_=make_printable_target_mesh(workspace)
        b=task.scene.create_actor_builder(); b.set_physx_body_type('static'); b.add_visual_from_file(filename=str(mesh)); a=b.build(name='flat_a4_charuco')
        # The printed face is horizontal, flush with the RoboTwin table.
        a.set_pose(sapien.Pose([0., -.08, .742], [1.,0.,0.,0.]))
        for arm in (args.arm,):
            accepted=[]
            for i,q in enumerate(candidates):
                for _ in range(120):
                    task.robot.set_arm_joints(np.asarray(q), np.full(6,.20), arm); task.scene.step()
                raw=task.get_obs(); image=np.asarray(raw['observation'][f'{arm}_camera']['rgb'])
                corners,ids,_,_=h.detector.detectBoard(image); n=0 if ids is None else len(ids)
                accepted.append({'qpos_rad':[float(v) for v in q], 'charuco_corners':int(n)})
                cv2.imwrite(str(workspace/f'{arm}_{i:02d}_{n}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            out[arm]=accepted
    finally: env.close()
    (workspace/f'{args.arm}_candidates_seed{seed}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({args.arm:{'total':len(out[args.arm]), 'pnp_min4':sum(x['charuco_corners']>=4 for x in out[args.arm]), 'quality_min12':sum(x['charuco_corners']>=12 for x in out[args.arm])}},indent=2))
if __name__=='__main__': main()
