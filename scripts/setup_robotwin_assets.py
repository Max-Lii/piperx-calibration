"""Prepare pinned Piper-X assets; optionally download the official task assets.

An explicit PIPERX_ROBOTWIN_ASSETS reuses an existing bundle read-only.
HF_ENDPOINT=https://hf-mirror.com can be set before invoking this script.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PIPER_REPO = "maxlii/Robotwin-piper-x-dual"
PIPER_REVISION = "c3156b179aab0722dd2228850b62768ca2bb1332"


def validate_piper(root: Path) -> None:
    for arm in ("left", "right"):
        directory = root / f"embodiments/piper_x_{arm}"
        for name in ("config.yml", "piper_x.urdf"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
    meshes = list((root / "embodiments/piper_x_shared").rglob("*.dae"))
    if not meshes:
        raise FileNotFoundError(f"Missing Piper-X shared meshes in {root}")
    for path in meshes:
        if path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs"):
            raise RuntimeError(f"Git LFS pointer is not a mesh: {path}; download actual assets")


def prepare_piper() -> Path:
    external = os.environ.get("PIPERX_ROBOTWIN_ASSETS")
    root = Path(external).expanduser().resolve() if external else ROOT / "third_party/robotwin_piperx_dual_assets"
    if external:
        validate_piper(root)
        print(f"Reusing external Piper-X bundle (unmodified): {root}")
        return root
    try:
        validate_piper(root)
    except (FileNotFoundError, RuntimeError):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=PIPER_REPO, repo_type="model", revision=PIPER_REVISION,
                          local_dir=root, allow_patterns=["embodiments/**", "LICENSE*", "README*", "*manifest*"])
        validate_piper(root)
    print(f"Piper-X simulation bundle ready: {root}")
    return root


def prepare_official_assets(root: Path) -> None:
    """Same official dataset and archives as RoboTwin's assets/_download.py.

    Not needed for the clean calibration scene. Opt in when developing full
    RoboTwin tasks. An existing complete directory is reused without download.
    """
    from huggingface_hub import hf_hub_download
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("background_texture", "embodiments", "objects"):
        if (root / name).is_dir() and any((root / name).iterdir()):
            print(f"Reusing official assets: {root / name}")
            continue
        archive = Path(hf_hub_download(repo_id="TianxingChen/RoboTwin2.0", repo_type="dataset",
                                       filename=f"{name}.zip", local_dir=root))
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"Archive member outside asset directory: {member.filename}")
            source.extractall(root)
        print(f"Official assets unpacked: {root / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-assets", type=Path, help="Optional official RoboTwin assets directory (large download)")
    args = parser.parse_args()
    prepare_piper()
    if args.official_assets:
        prepare_official_assets(args.official_assets)


if __name__ == "__main__":
    main()
