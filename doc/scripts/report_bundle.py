"""Report static deployment size separately from one route's script tags.

Gzip sizes are local estimates, not measurements of GitHub Pages responses.
No source files or exports are modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlsplit


class Scripts(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.paths = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("src"):
            path = urlsplit(attrs["src"]).path
            if "/_next/" in path:
                self.paths.append("_next/" + path.split("/_next/", 1)[1])


def report(output):
    groups = defaultdict(int)
    files = [p for p in output.rglob("*") if p.is_file()]
    for path in files:
        if path.suffix == ".html":
            category = "html"
        elif path.suffix == ".txt":
            category = "next_navigation_data"
        elif path.suffix == ".js":
            category = "javascript_all_chunks"
        elif path.suffix == ".css":
            category = "css"
        elif path.suffix in {".woff2", ".woff", ".ttf"}:
            category = "fonts"
        elif path.suffix in {".webp", ".png", ".jpg"}:
            category = "images"
        else:
            category = "other"
        groups[category] += path.stat().st_size
    routes = {}
    for name in ("index.html", "zh/index.html"):
        page = output / name
        paths = sorted(set(Scripts(page).paths))
        routes[name] = {
            "script_tag_files": len(paths),
            "script_tag_bytes": sum((output / path).stat().st_size for path in paths),
            "script_tag_gzip_estimate_bytes": sum(len(gzip.compress((output / path).read_bytes())) for path in paths),
            "html_bytes": page.stat().st_size,
            "html_gzip_estimate_bytes": len(gzip.compress(page.read_bytes())),
        }
    return {"deployment_bytes": sum(groups.values()), "deployment_files": len(files),
            "categories": dict(groups), "routes": routes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "out")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = report(args.output_dir)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")
