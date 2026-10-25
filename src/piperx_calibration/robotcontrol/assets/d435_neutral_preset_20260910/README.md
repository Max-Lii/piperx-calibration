# D435 LUTs using the iPhone neutral color preset

This is the current candidate set. It replaces the earlier `recommended` directory, whose output shifted the neutral tabletop toward yellow/green and must not be used.

## Reference

- iPhone image: `neutral_color_preset/IMG_1880 2_srgb.png`
- The iPhone neutral-color rendering is used directly.
- The tabletop surrounding the card is treated as the white-point reference.
- The same physical printed card is present in the iPhone and D435 images.

## Robust sampling

- The card orientation is detected before assigning patch IDs.
- Most of each patch interior is sampled instead of a small center point.
- Printed labels, patch borders, high-gradient pixels, reflections, and median-absolute-deviation outliers are rejected.
- The tabletop is sampled outside a dilated card mask, using millions of low-saturation, non-clipped, low-gradient pixels.

## Correction model

1. Match each D435 tabletop chromaticity to the iPhone neutral-preset tabletop.
2. Fit a regularized color matrix to the 24 robust patch medians.
3. Constrain the matrix to preserve the reference white axis.
4. Preserve each input pixel's luminance to avoid forcing iPhone exposure/tone mapping onto D435 output.

## Results

| Camera | Table RGB RMSE before | After | Patch chroma RMSE before | After |
| --- | ---: | ---: | ---: | ---: |
| Left wrist | 0.0833 | 0.0357 | 0.2727 | 0.1588 |
| Third view | 0.0722 | 0.0348 | 0.3184 | 0.1367 |
| Right wrist | 0.0680 | 0.0623 | 0.2329 | 0.1713 |

Cross-camera patch chroma dispersion falls by 58.8%. The right wrist uses only 25% residual color-matrix strength because stronger correction worsened held-out patches.

Use `all_cameras_AB.png` for visual review. The `.cube` and `.npz` files are offline candidates and have not been enabled in `robotcontrol`.

This remains a relative appearance correction, not a colorimeter-traceable camera profile. It is valid only for lighting and camera controls comparable to the capture session.
