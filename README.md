# Geo Packager

High-performance GDAL wrappers for fast and easy preparation of large-scale GeoTIFF deliverables.

## 🛠 Setup & Quick Start

1.  **Environment**: Run `bash setup_env.sh` to create the `gdal` environment.
2.  **Activate**: Run `conda activate gdal`.
3.  **Launch GUI**: Run `./app.py` and open `http://127.0.0.1:5001` in your browser.

---

## 🚀 Components

### 1. `app.py` (GUI Dashboard)
A Flask-based graphical interface for `package.py` featuring real-time log streaming, logic grouping, and native macOS file/directory pickers.

**Usage:**
```bash
python app.py
```

### 2. `package.py` (The Engine)
A parallelized pipeline for scaling, shifting, downsizing, tiling, and cleaning rasters.

**Key Features:**
- **Adaptive Downsizing**: Automatically applies 30% downsizing for input files > 20GB (unless overridden).
- **Smart Tiling**: Parallel generation with automatic detection and removal of empty (alpha=0) tiles.
- **Sequential Naming**: Zero-padded numbering (e.g., `_001.tif`) for consistent file sorting.
- **macOS Safety**: Ignores `._*` metadata files on Apple-formatted volumes.
- **Error Handling**: Gracefully cleans up partial output if a process is interrupted.

**Usage:**
```bash
python package.py <input.tiff> [options]
```
- `--output-dir`: Output directory (default: `<input_dir>/<name>_tiles`).
- `--clobber`: Overwrites output directory if it exists.
- `--scale`: Grid-to-ground scale factor.
- `--downsize-percent`: Resolution reduction (e.g., 25 for 0.25x pixels).
- `--shift-x` / `--shift-y`: Easting/Northing translation shift.
- `--tile-size`: Pixel size of output tiles (default 20,000).

### 3. `compare.py` (Validation)
A metadata verification tool to confirm transformations and analyze residuals.

**Usage:**
```bash
python compare.py <original.tif> <processed_dir_or_vrt>
```
- **Residual Analysis**: Calculates "Origin Shift" and "Edge Drift" *after* accounting for the measured scale factor. 
- **Anchor Logic**: Identifies the scaling anchor point assuming zero residual shift.

---

## 🏁 Example CLI Workflow

Assuming the `gdal` environment is active:

```bash
# Package with 0.9996 scale and +100ft shift, downsizing to 25% resolution
python package.py data.tif --scale 0.9996 --shift-x 100 --downsize-percent 25 --clobber

# Verify the results against the original
python compare.py data.tif data_tiles/
```


