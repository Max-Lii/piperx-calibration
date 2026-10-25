"""Check concurrent preview clients without requiring a GPU."""
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from .adapter import RobotwinPreviewAdapter


def test_preview_clients_share_frames_and_renderer_thread(monkeypatch):
    from sim.robotwin import adapter as simulation

    owner = []
    rendered = []
    released = []

    class Workcell:
        def __init__(self):
            owner.append(threading.get_ident())

        def get_obs(self):
            assert threading.get_ident() == owner[0]
            rendered.append(True)
            return {"observation": {
                name: {"rgb": np.full((12, 16, 3), value, np.uint8)}
                for name, value in (("left_camera", 20), ("head_camera", 100), ("right_camera", 220))
            }}

        def close(self):
            assert threading.get_ident() == owner[0]
            released.append(True)

    monkeypatch.setattr(simulation, "Workcell", Workcell)
    preview = RobotwinPreviewAdapter(preview_fps=0.01)
    try:
        cameras = ["left_wrist", "third_view", "right_wrist"]
        with ThreadPoolExecutor(max_workers=3) as clients:
            frames = list(clients.map(preview.frame_jpeg, cameras))
        assert len(rendered) == 1
        assert len({timestamp for _, timestamp in frames}) == 1
        for (payload, _), expected in zip(frames, (20, 100, 220), strict=True):
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            assert image.shape == (12, 16, 3)
            assert abs(float(image.mean()) - expected) < 2
    finally:
        preview.close()
    preview.close()
    assert released == [True]
    with pytest.raises(RuntimeError, match="closed"):
        preview.frame_jpeg("left_wrist")


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan")])
def test_preview_rejects_invalid_frame_rate(fps):
    with pytest.raises(ValueError, match="preview_fps"):
        RobotwinPreviewAdapter(preview_fps=fps)
