# Calibration documentation

The Fumadocs site contains one complete tutorial per language: English `/` and Chinese `/zh/`. Both pages use the existing table of contents, theme switch, code copying, and image lightbox.

## Develop and export

Use Node.js 22 and npm in this directory:

```bash
npm ci
npm run dev
```

Build and check the static site:

```bash
npm run types:check
npm run build
python3 scripts/check_static_export.py
npm run start
```

`out/` is generated. Do not commit it. The link checker uses only Python's standard library; it checks both tutorial routes, local links and anchors, images, scripts, and KaTeX errors in the exported HTML.

For a project GitHub Pages site, use the same base path for the build and check:

```bash
NEXT_PUBLIC_BASE_PATH=/piperx-calibration npm run build
python3 scripts/check_static_export.py --base-path /piperx-calibration
```

`lib/site-config.mjs` defaults to the root path for local builds. In GitHub Actions it derives the Pages prefix and repository URL from `GITHUB_REPOSITORY`; no repository name is embedded in the workflow. `site.config.json` supplies the local repository-link fallback and default branch. `NEXT_PUBLIC_BASE_PATH`, `NEXT_PUBLIC_REPOSITORY_URL`, and `NEXT_PUBLIC_REPOSITORY_REF` can override these values. Language navigation uses Next.js links; MDX links and figures use `lib/paths.ts`.

## Source map

| Path | Purpose |
| --- | --- |
| `content/docs/index.mdx`, `content/docs/zh/index.mdx` | The two tutorials: ten chapters and three appendices each |
| `app/page.tsx`, `app/zh/page.tsx` | English and Chinese routes |
| `components/tutorial.tsx` | Fumadocs article, TOC, language and theme controls |
| `components/mdx.tsx` | Code, math, figures, and base-path-aware links |
| `experiments/BOARD_002_20261024_tutorial30_default/` | Repository-only experiment archive, RGB, feedback, results, and original UI captures; not deployed to Pages |
| `public/images/` | Full-resolution lossless WebP images used by the tutorial |
| `source-images/` | Original PNG technical figures, reference image, and provenance; excluded from the static site |
| `data/image_assets.json` | Image dimensions and original/WebP checksums |
| `data/` | Historical pose-library inputs and the figure checksum baseline |
| `scripts/` | Content checks, replay, recording, and figure tools |

Keep the technical facts, equations, numerical values, figure paths, and concept order consistent between languages. Use native h2/h3 headings for TOC entries. Put long installation and interface material in the existing appendices or disclosures. Python examples marked `runnable` are executed by the tutorial checker; `excerpt` examples are compiled but not run.

## Check tutorial data and rendering

From the repository root, with the calibration dependencies and Pillow installed:

```bash
PYTHONPATH=src:. python doc/scripts/check_tutorial.py
```

The checker detects the built base path from the exported HTML, so the same command works for a local or Pages build. It executes the marked examples, compares both languages, checks image and record checksums, redetects the right_05 corners, and refits the held-out example using its training poses only. The original frame illustrations in `source-images/` are checked against `data/figure_checksums.json`; every published WebP is also compared pixel-for-pixel with its source. No local backup directory is needed. The generated report is `quality-check.json`.

For browser checks, install Playwright in a separate environment if needed:

```bash
python -m pip install playwright
python -m playwright install chromium
python -m http.server 18781 --bind 127.0.0.1 --directory doc/out
```

In another terminal, from the repository root:

```bash
python doc/scripts/check_browser.py
```

The helper uses Playwright's Chromium by default. `PIPERX_CHROME` can select an installed Chromium executable. `PIPERX_DOCS_URL` can select a different origin or project-site prefix. The checks cover both languages, image loading, TOC navigation, desktop/mobile overflow, disclosures, lightbox, clipboard, and theme/language controls. Reports and screenshots go to `validation/`, which is excluded from the release. These checks do not use a GPU or physical devices.

## Replay and figures

Build `frontend/` with `npm ci` and `npm run build`. With Python 3.12 and the base package plus Pillow installed, start the replay from the repository root:

```bash
PYTHONPATH=src:. python doc/scripts/replay_tutorial.py \
  --port 8020 --session-root /tmp/piperx-replay-session
```

Open `http://127.0.0.1:8020/operator`. Existing archived data are sufficient; a new simulation is not needed. Installation details are in tutorial Appendix A; runtime and hardware setup are in Chapter 8.

The current derived figures can be regenerated from the unchanged archive:

```bash
python doc/scripts/compose_tutorial_figures.py
python doc/scripts/compose_analysis_figures.py
python doc/scripts/optimize_images.py
```

The composition tools write PNG sources and provenance into `source-images/`. The optimizer writes only the twelve displayed images into `public/images/`, using full-resolution lossless WebP, and records dimensions and checksums in `data/image_assets.json`. It verifies every decoded RGBA pixel against the source. UI images use the original archived browser captures directly; no UI elements are rearranged and no values are changed. Original experiment files remain unchanged. Run `python doc/scripts/optimize_images.py --check` to verify without regenerating. `prepare_figures.py` retains the earlier corner-illustration generator used as a style reference.

`capture_tutorial_experiment.py` records a new simulation to a new output directory. `capture_replay_ui.py` runs a browser walkthrough and writes UI captures into the selected archive location defined in that script. Neither tool is part of a routine documentation build; do not rerun them over the published case merely to validate prose changes.

The main 2026-10-24 experiment and the historical integration example are separate data sources. Preserve that distinction and the P/W/S frame conventions when editing. Known simulation transforms are used for scoring after estimation; replay is not a physical calibration.

## Deployment size

Only `public/images/` is deployed for tutorial figures. The experiment archive and PNG sources stay in the repository, outside `public/`; MDX evidence links use repository-relative targets such as `repo:/doc/experiments/.../result.json`; `lib/paths.ts` generates GitHub URLs using the shared repository configuration. The link checker verifies those targets against the local checkout before publication. The GitHub links become available after these files are pushed to `main`.

Code highlighting and KaTeX run during the build. `lib/rehype-static-content.mjs` serializes their static subtrees once as HTML instead of one React element per token; `components/static-content.tsx` renders that trusted build output. Copy controls and all formula MathML remain available. Highlight colors share CSS classes, and the font plugin publishes only KaTeX WOFF2 files. Language links keep client navigation but disable automatic prefetch of the other long article.

Inspect deployment bytes separately from the script tags requested by each route:

```bash
python3 scripts/report_bundle.py
```

The report includes local gzip estimates. These are not measured GitHub response sizes. Keep Next.js navigation payloads and runtime chunks: they support language navigation, hydration, and the error page and should not be removed merely because they are not all fetched on the first visit.

### Previewing a local export

The Pages build contains its deployment prefix. Serving that build directly at `/` does not reproduce the Pages mount. For an ordinary root preview, rebuild locally (with the base-path override empty if it was exported in the shell):

```bash
NEXT_PUBLIC_BASE_PATH= npm run build
python -m http.server 3010 --bind 0.0.0.0 --directory out
```

The GitHub Actions build automatically derives the project prefix again. These are two builds from the same source; a Next.js base path is compiled into navigation as well as asset URLs, so a single export is not intended to be moved between arbitrary URL prefixes without rebuilding.

The first push only builds and checks the documentation. After enabling GitHub Pages, run the Documentation workflow manually on `main` to deploy. Set the repository Actions variable `PAGES_ENABLED` to `true` when automatic deployment on subsequent documentation pushes is desired. Pull requests never deploy.
