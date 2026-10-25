from sim.run_standalone_simulation import run


def test_standalone_simulation_is_self_contained(tmp_path):
    report = run(tmp_path / "simulation", observations=14)
    assert report["external_runtime_required"] is False
    assert report["passed"] is True
    assert report["pose_library"]["poses_per_arm"] == 14
