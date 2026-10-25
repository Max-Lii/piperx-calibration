"""List or copy the standalone repository files without running Git.

Default: report only. --output copies to a new directory outside this checkout.
Local history, dependencies, sessions and build outputs stay in the source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (".github", "assets", "src", "sim", "scripts", "frontend", "doc")
ROOT_FILES = ("README.md", "RELEASING.md", ".gitignore", "pyproject.toml")
EXCLUDED_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "node_modules", ".next",
    ".source", "out", "dist", "build", "coverage", "runtime", "validation",
}
EXCLUDED_FILES = {
    "next-env.d.ts", "quality-check.json", "REVISION_2026-10-24.md",
    "SINGLE_PAGE_RESTORATION_2026-10-24.md", "TEACHING_REFINEMENT_2026-10-25.md",
}


def selected_files():
    for name in (*ROOT_FILES, *(p.name for pattern in ("LICENSE*", "NOTICE*", "COPYING*") for p in ROOT.glob(pattern))):
        path = ROOT / name
        if path.is_file():
            yield path
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Review symlink before publication: {relative}")
            if not path.is_file() or path.name in EXCLUDED_FILES:
                continue
            if path.suffix in {".pyc", ".pyo", ".tsbuildinfo", ".pem", ".log"}:
                continue
            if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
                continue
            if path.name.endswith((".local.yaml", ".local.yml")) or path.name == ".DS_Store":
                continue
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a NEW directory; never overwrites an existing directory")
    args = parser.parse_args()
    files = sorted(set(selected_files()))
    total = sum(p.stat().st_size for p in files)
    report = {
        "file_count": len(files), "total_bytes": total,
        "top_level": sorted({str(p.relative_to(ROOT)).split("/")[0] for p in files}),
        "largest_files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size}
                          for p in sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:8]],
        "license_present": any(ROOT.glob("LICENSE*")),
        "git_operations_performed": False,
    }
    if args.output:
        destination = args.output.expanduser().resolve()
        if destination.is_relative_to(ROOT) or ROOT.is_relative_to(destination):
            raise ValueError("Choose a destination outside the source checkout, not an ancestor")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        manifest_path = destination.with_name(destination.name + ".manifest.json")
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite {manifest_path}")
        destination.mkdir(parents=True)
        manifest = []
        for source in files:
            relative = source.relative_to(ROOT)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Copy checksum mismatch: {relative}")
            manifest.append({"path": str(relative), "sha256": digest})
        # Keep the packaging report beside the export, outside the repository.
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        report.update(output=str(destination), manifest=str(manifest_path))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
