"""Check local links, anchors, images and scripts in a Next.js static export.

Uses only Python's standard library. Run after npm run build, with the same
base path as the build, e.g. --base-path /piperx-calibration.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.errors: list[str] = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        key = "href" if tag in {"a", "link"} else "src" if tag in {"img", "script"} else None
        if key and attrs.get(key):
            self.links.append(attrs[key])
        if "katex-error" in attrs.get("class", ""):
            self.errors.append("KaTeX rendering error")


def detect_base_path(output: Path) -> str:
    page = Page(output / "index.html")
    for href in page.links:
        path = urlsplit(href).path
        if "/_next/" in path:
            return path.split("/_next/", 1)[0]
    raise ValueError("Cannot infer export base path from index.html")


def check_export(output: Path, base_path: str | None = None) -> int:
    output = output.resolve()
    if base_path is None:
        base_path = detect_base_path(output)
    if base_path and (not base_path.startswith("/") or base_path.endswith("/")):
        raise ValueError("base path must start with / and have no trailing slash")
    pages = {path: Page(path) for path in output.rglob("*.html")}
    for route in ("index.html", "zh/index.html"):
        if output / route not in pages:
            raise ValueError(f"Missing tutorial: {route}")
    failures = []
    checked = 0
    for path, page in pages.items():
        failures.extend(f"{path.relative_to(output)}: {e}" for e in page.errors)
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                if url.netloc == "github.com" and "/blob/" in url.path:
                    relative = url.path.split("/blob/", 1)[1].split("/", 1)
                    repository = Path(__file__).resolve().parents[2]
                    source = repository / unquote(relative[1]) if len(relative) == 2 else None
                    if source and relative[1].startswith("doc/experiments/") and not source.is_file():
                        failures.append(f"{path.relative_to(output)}: missing repository target: {href}")
                continue
            local = unquote(url.path)
            if local.startswith("/"):
                if base_path:
                    if local == base_path:
                        local = "/"
                    elif local.startswith(base_path + "/"):
                        local = local[len(base_path):]
                    else:
                        failures.append(f"{path.relative_to(output)}: missing base path: {href}")
                        continue
                target = output / local.lstrip("/")
            else:
                target = path.parent / local if local else path
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(output) or not target.is_file():
                failures.append(f"{path.relative_to(output)}: missing target: {href}")
            elif url.fragment and target.suffix == ".html":
                if unquote(url.fragment) not in pages[target].ids:
                    failures.append(f"{path.relative_to(output)}: missing anchor: {href}")
            checked += 1
    if failures:
        raise ValueError("\n".join(failures))
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "out")
    parser.add_argument("--base-path", default=None, help="Default: infer from exported HTML")
    args = parser.parse_args()
    base_path = args.base_path if args.base_path is not None else detect_base_path(args.output)
    count = check_export(args.output, base_path)
    print(f"Static export OK: English and Chinese routes; {count} local links/assets; base path {base_path or '/'}")
