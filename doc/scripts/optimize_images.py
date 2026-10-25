"""Publish full-resolution, pixel-identical WebP figures without PNG duplicates.

Original technical figures live in doc/source-images; recorded frames and UI
screenshots stay in the unchanged experiment archive. No resizing or lossy
encoding is used. Run again after regenerating the technical figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

DOC = Path(__file__).resolve().parents[1]
ROOT = DOC.parent
RUN = DOC / "experiments/BOARD_002_20261024_tutorial30_default"
MANIFEST = DOC / "data/image_assets.json"
SOURCES = {
    name: DOC / "source-images" / f"{name}.png"
    for name in (
        "piper_coordinate_frames_figure", "piper_shared_world_and_charuco_65_35",
        "board002_pose_diversity", "board002_right05_correspondences",
        "board002_right04_heldout",
    )
}
SOURCES.update({name: RUN / f"{name}.png" for name in (
    "home_right_camera", "right_05_right_wrist", "right_00_observer",
)})
SOURCES.update({
    "01-device": RUN / "ui/01-device.png",
    "board002_ui_placement": RUN / "ui/04-placement.png",
    "board002_ui_acquisition": RUN / "ui/06-capture-right.png",
    "board002_ui_review": RUN / "ui/08-review.png",
})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_assets() -> dict:
    assets = json.loads(MANIFEST.read_text())
    assert set(assets) == {f"/images/{name}.webp" for name in SOURCES}
    for url, entry in assets.items():
        source = ROOT / entry["source"]
        target = DOC / "public" / url.lstrip("/")
        assert sha(source) == entry["source_sha256"], source
        assert sha(target) == entry["webp_sha256"], target
        with Image.open(source) as original, Image.open(target) as encoded:
            assert original.size == encoded.size == (entry["width"], entry["height"])
            assert original.convert("RGBA").tobytes() == encoded.convert("RGBA").tobytes(), target
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing assets without writing")
    args = parser.parse_args()
    if not args.check:
        assets = {}
        for name, source in SOURCES.items():
            url = f"/images/{name}.webp"
            target = DOC / "public" / url.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.save(target, format="WEBP", lossless=True, exact=True, method=6)
                width, height = image.size
            assets[url] = {
                "width": width, "height": height,
                "source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
                "webp_sha256": sha(target), "source_bytes": source.stat().st_size,
                "webp_bytes": target.stat().st_size,
            }
        MANIFEST.write_text(json.dumps(assets, indent=2) + "\n")
    assets = check_assets()
    before = sum(entry["source_bytes"] for entry in assets.values())
    after = sum(entry["webp_bytes"] for entry in assets.values())
    print(json.dumps({"images": len(assets), "source_bytes": before,
                      "webp_bytes": after, "reduction_percent": round(100 * (1 - after / before), 1),
                      "full_resolution": True, "all_rgba_pixels_identical": True}, indent=2))


if __name__ == "__main__":
    main()
