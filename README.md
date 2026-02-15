# Master Raster Packager (Playground)

This directory contains high-performance utilities for preparing large GeoTIFF deliverables using GDAL.

## 🚀 Utilities

### 1. `package.py`
A parallelized pipeline for scaling, shifting, downsizing, tiling, and cleaning rasters.

**Key Features:**
- **Adaptive Downsizing**: Automatically applies 30% downsizing for input files > 20GB (unless a manual scale is specified).
- **Smart Tiling**: Parallelized tile generation with automatic detection and removal of empty (alpha=0) tiles.
- **Zero-Padded Naming**: Sequential numbering with padding (e.g., `_01.tif`, `_001.tif`) for consistent sorting.
- **macOS Metadata Safety**: Automatically ignores `._*` files on macOS-formatted volumes.
- **Robustness**: Interrupts and cleans up partial output if disk space is exhausted or a write error occurs.

**Usage:**
```bash
conda run -n gdal --no-capture-output python package.py <input.tiff> [options]
```
- `--output-dir`: Output directory (default: same parent as input file).
- `--clobber`: Overwrites output directory if it exists.
- `--scale`: Scale factor (instant VRT).
- `--downsize-percent`: Downsize resolution (e.g., 25 for 0.25x pixels).
- `--shift-x` / `--shift-y`: Easting/Northing translation.
- `--tile-size`: Pixel size of output tiles (default 20,000).

### 2. `app.py` (GUI)
A Flask-based graphical interface for `package.py` with real-time log streaming and native file picking.

**Usage:**
```bash
conda run -n gdal --no-capture-output python app.py
```
Open your browser to `http://127.0.0.1:5001`.

### 3. `compare.py`
A metadata verification tool (sub-second) to confirm transformations.

**Usage:**
```bash
conda run -n gdal --no-capture-output python compare.py <original.tif> <processed_dir_or_vrt>
```
- **Relative Transformation Analysis**: Calculate "Origin Shift" and "Edge Drift" as residuals *after* accounting for the measured scale factor. 
- **Anchor Logic**: Correctly identifies the scaling anchor point assuming zero residual shift.
- **Reporting**: Prints "(No Shift)" and "(No Drift)" if the transformation is a mathematically pure scale about the coordinate origin (0,0).

---

## 🛠 Setup & Workflow

1.  **Environment**: Run `bash setup_env.sh` to create the `gdal` environment (installs GDAL, Flask, and Tkinter).
2.  **Process**: Use `package.py` to create your tiled deliverable.
3.  **Verify**: Use `compare.py` to perform Quality Control on the output.

### Example Workflow
```bash
# Package with 0.9996 scale and +100ft shift, downsizing to 25% resolution
conda run -n gdal --no-capture-output python package.py data.tif --scale 0.9996 --shift-x 100 --downsize-percent 25 --clobber

# Verify results
conda run -n gdal --no-capture-output python compare.py data.tif data_tiles/
```
