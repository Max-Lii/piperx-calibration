The cover is a SAPIEN rendering of this repository's isolated simulation scene,
with the title and schematic camera sight lines added using Pillow. It is an illustration of the
workcell, not a physical calibration result. Camera housings, lens styling,
mounting risers, wrist brackets, and the fixed-camera stand are cover visuals.
Robot poses and camera frames come from the existing simulation assets; these
visual additions are never used by the calibration solver.

The D435 housing uses the official RealSense ROS CAD mesh:
[realsense-ros, revision 7a87097](https://github.com/realsenseai/realsense-ros/blob/7a87097c499e3b0b69518d54a8e7a9ac1924e87c/realsense2_description/meshes/d435.dae).
The mesh's nominal placement follows the adjacent `_d435.urdf.xacro`.
The upstream mesh is uniformly silver; the dark face and optical windows are
illustrative additions. The original CAD file stays in ignored `temp/`.

To reproduce, download that mesh to `temp/banner_render/realsense/d435.dae` and
run in the simulation environment from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/render_readme_banner.py \
  --camera-mesh temp/banner_render/realsense/d435.dae
python scripts/compose_readme_banner.py
```

Set `PIPERX_ROBOTWIN_ASSETS` if the robot assets are outside `third_party/`.
The render script produces three 3840 × 2560 observer views in `temp/banner_render/`.
Only the compressed `banner.webp` is needed by README; it is outside the
documentation website's published resources.
