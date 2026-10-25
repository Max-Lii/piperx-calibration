from sim.virtual_calibration import RenderedCharucoRobotWorldHarness


def test_rendered_charuco_robot_world_handeye_recovers_hidden_truth(tmp_path):
    result = RenderedCharucoRobotWorldHarness(seed=17, output_dir=tmp_path).run(observations=14)
    assert result.detections >= 12
    assert result.passed, result.as_dict()
    assert len(list(tmp_path.glob("render_*.png"))) == 14
