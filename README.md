# lytro-py

A Python toolkit for working with Lytro camera captures. It can:
1. Read Lytro `.RAW` files and metadata.
2. Build calibration data from calibration captures.
3. Render standard demosaiced PNGs.
4. Synthesize light‑field subaperture views.

This README is intentionally detailed because the file formats and light‑field steps are non‑obvious.

**What’s In This Repo**
1. `main.py`: CLI entry point.
2. `lib/raw_image.py`: Packed 12‑bit RAW decoding and demosaic.
3. `lib/lyli_metadata.py`: Metadata parsing (from `.TXT` files).
4. `lib/captured_picture.py`: `.128` thumbnail decoding.
5. `lib/lightfield_pipeline.py`: Calibration loading, view synthesis, and export helpers.
6. `lib/calibration/`: Calibration pipeline.

**Key File Types**
1. `.RAW` (camera capture): Packed 12‑bit Bayer data, 2 pixels per 3 bytes. This is the actual raw sensor mosaic.
2. `.TXT` (metadata): Capture metadata (dimensions, black level, white balance, CFA pattern, etc.).
3. `.128` (thumbnail): 128x128, 16‑bit little‑endian grayscale. The code uses the low 8 bits.
4. `calibration.json`: Per‑camera calibration describing microlens grid geometry and lens intrinsics at multiple zoom/focus settings.

**RAW Format Details**
1. Packed 12‑bit layout: every 3 bytes encode 2 pixels.
2. CFA (color filter array) pattern is read from metadata (example: `r,gr:gb,b`).
3. The demosaic step uses the CFA pattern to generate RGB output.

**Thumbnail (.128) Details**
1. Exactly 128x128 16‑bit unsigned values (little‑endian).
2. The implementation reads the low byte to get an 8‑bit thumbnail.

**Calibration File (calibration.json)**
1. `serial`: Camera serial number.
2. `array.grid.horizontal` and `array.grid.vertical`: Microlens grid line positions, tagged by `subgrid`.
3. `array.translation` and `array.rotation`: Global alignment corrections used before synthesis.
4. `lens[]`: Per zoom/focus settings with `cameraMatrix` (intrinsics) and `distCoeffs`.

These grid line lists are how the pipeline knows where each microlens center lies, which directly drives subaperture sampling.

**View Synthesis (Subaperture Rendering)**
This is the most important conceptual path:
1. Read `.RAW` + `.TXT` metadata.
2. Decode packed 12‑bit Bayer and demosaic to RGB.
3. Apply calibration transform (`translation` + `rotation`) to align the microlens grid.
4. Compute mean microlens spacing (pitch) from grid lines.
5. Choose a subaperture grid size (`--grid`, default `9`, must be odd).
6. For each view (grid cell), offset sampling locations by a fraction of pitch (`--offset-scale`, default `0.18`).
7. Sample RGB at those offsets using bilinear interpolation.
8. Optionally apply:
   - White balance from metadata.
   - Per‑view normalization (default on).
   - Aspect correction using pitch ratio (default on).
   - Row color balance if enabled.

**Important: Output Size Of Subaperture Views**
Subaperture view resolution is determined by the microlens grid, not the RAW resolution. The number of grid lines (for subgrid 0) defines the output height and width. Aspect correction then rescales width based on pitch ratio.

**Export Behavior**
`export_subaperture_tiled_png` supports two modes:
1. If output path ends with `.png`, a single tiled PNG is produced.
2. Otherwise, a directory is created and each view is written as `view_rXX_cYY.png`.

**CLI Usage (Main Entry Points)**
These are the high‑level commands exposed by `main.py`:
1. Calibrate a directory:
   - `python main.py calibrate <calibration_dir> [calibration.json]`
2. Process a directory of RAWs:
   - `python main.py process <input_dir> [calibration.json] [--raw-png]`
3. Export a single RAW to PNG:
   - `python main.py export-raw-png <raw_path> <metadata_path> <output.png>`
4. Export subaperture views:
   - `python main.py export-subaperture <raw_path> <metadata_path> <output.png or output_dir>`
   - Options: `--grid`, `--offset-scale`, `--white-balance`, `--no-per-view-normalize`, `--no-aspect-correction`, `--row-color-balance`, `--row-color-balance-strength`

**Common Confusions Answered**
1. `--grid 9` is a default convenience, not “optimal.” It defines how many subaperture views you synthesize.
2. Subaperture view resolution is smaller than RAW because it’s driven by microlens grid density.
3. The `.RAW` file is packed 12‑bit Bayer, not a standard image format.
4. The `.128` thumbnail is a separate low‑res preview, not derived from `.RAW` here.

If you want a shorter “quickstart” version or a diagram of the pipeline, say the word and I’ll add it.
