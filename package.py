#!/usr/bin/env python3
"""
package.py

A high-performance parallel raster packager that:
1. Scales/Shifts (Optional): Using metadata-only VRT georeferencing adjustments.
2. Tiles (Parallel): Using parallelized gdal_translate.
3. Cleans (Parallel): Removes empty tiles using a parallel worker pool with progress.
4. Indexes: Builds a final VRT index.

Usage:
  python package.py input.tiff [options]
"""

import os
import sys
import argparse
import subprocess
import shutil
from multiprocessing import Pool
from osgeo import gdal

gdal.UseExceptions()

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def get_dir_size_gb(directory):
    """Returns total size of files in directory in GB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)

# ------------------------------------------------------------------------------
# Transformation Logic (VRT-based)
# ------------------------------------------------------------------------------

def create_transformed_vrt(input_path, scale, anchor_x, anchor_y, shift_x, shift_y, resample_factor):
    """Creates a temporary VRT with scaling, translation, and resampling applied."""
    src_ds = gdal.Open(input_path)
    if src_ds is None:
        raise RuntimeError(f"Could not open {input_path}")

    # Step 1: Resampling (Resolution Reduction)
    # We use gdal.Translate to create a VRT with a specific outsize
    # This also handles the geotransform update for the pixel size automatically
    vrt_path = f"{os.path.splitext(input_path)[0]}_transformed.vrt"
    
    translate_options = gdal.TranslateOptions(
        format="VRT",
        widthPct=resample_factor * 100,
        heightPct=resample_factor * 100,
        resampleAlg=gdal.GRIORA_Bilinear  # Smoothing for downsampling
    )
    vrt_ds = gdal.Translate(vrt_path, src_ds, options=translate_options)
    
    # Step 2: Scale/Shift Adjustments (Top of resampling)
    gt = vrt_ds.GetGeoTransform()
    
    # GT indices: (0: OriginX, 1: PxWidth, 2: SkewX, 3: OriginY, 4: SkewY, 5: PxHeight)
    
    # Scaling Logic (relative to anchor)
    new_px_width = gt[1] * scale
    new_skew_x   = gt[2] * scale
    new_origin_x = anchor_x + (gt[0] - anchor_x) * scale
    
    new_skew_y   = gt[4] * scale
    new_px_height = gt[5] * scale
    new_origin_y = anchor_y + (gt[3] - anchor_y) * scale
    
    # Translation Logic (applied after scaling)
    new_origin_x += shift_x
    new_origin_y += shift_y
    
    new_gt = (new_origin_x, new_px_width, new_skew_x, new_origin_y, new_skew_y, new_px_height)
    vrt_ds.SetGeoTransform(new_gt)
    
    vrt_ds.FlushCache()
    vrt_ds = None
    src_ds = None
    
    return vrt_path

# ------------------------------------------------------------------------------
# Tiling Task
# ------------------------------------------------------------------------------

def create_tile(args):
    x, y, w, h, out_path, input_file, keep_xml = args
    cmd = [
        "gdal_translate", "-q",
        "-srcwin", str(x), str(y), str(w), str(h),
        "-co", "TILED=YES", "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2",
        "--config", "GDAL_PAM_ENABLED", "YES" if keep_xml else "NO",
        input_file, out_path
    ]
    subprocess.run(cmd, check=True)
    return out_path

# ------------------------------------------------------------------------------
# Cleaning Task (Parallelized)
# ------------------------------------------------------------------------------

def is_tile_empty(file_path):
    """Checks if a tile is empty (all bands 0 or fully transparent alpha)."""
    try:
        ds = gdal.Open(file_path)
        if ds is None: return False
        
        band_count = ds.RasterCount
        alpha_band = None
        for i in range(1, band_count + 1):
            band = ds.GetRasterBand(i)
            if band.GetColorInterpretation() == gdal.GCI_AlphaBand:
                alpha_band = band
                break
        
        if alpha_band:
            stats = alpha_band.GetStatistics(0, 1)
            return stats[1] == 0 
        else:
            for i in range(1, band_count + 1):
                band = ds.GetRasterBand(i)
                stats = band.GetStatistics(0, 1)
                if stats[1] > 0: return False
            return True
    except Exception:
        return False

def clean_task(tile_path):
    """Parallel worker for empty tile removal."""
    if is_tile_empty(tile_path):
        try:
            os.remove(tile_path)
            # Always remove XML sidecar if the tile is removed
            for ext in [".aux.xml", ".xml"]:
                sidecar = tile_path + ext
                if os.path.exists(sidecar):
                    os.remove(sidecar)
        except Exception as e:
            print(f"Warning: Could not remove {tile_path} or its sidecar: {e}")
        return True
    return False

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel Raster Packager (Scale, Shift, Downsize, Tile, Clean).")
    parser.add_argument("input_file", help="Input GeoTIFF file.")
    parser.add_argument("--output-dir", help="Output directory (default: <input_dir>/<input_basename>_tiles)")
    parser.add_argument("--clobber", action="store_true", help="Overwrite output directory if it exists")
    
    # transformation / georeferencing options
    parser.add_argument("--scale", type=float, default=1.0, help="Grid-to-ground scale factor (default 1.0)")
    parser.add_argument("--anchor-x", type=float, default=0.0, help="Anchor X for scaling (default 0.0)")
    parser.add_argument("--anchor-y", type=float, default=0.0, help="Anchor Y for scaling (default 0.0)")
    parser.add_argument("--shift-x", type=float, default=0.0, help="X translation shift (default 0.0)")
    parser.add_argument("--shift-y", type=float, default=0.0, help="Y translation shift (default 0.0)")

    # Downsampling / Resolution options
    parser.add_argument("--downsize-percent", type=float, help="Downsize resolution by percentage (e.g. 50 = 0.5x pixels)")
    parser.add_argument("--downsize-GB", type=float, help="Target total deliverable size in GB (estimate)")
    parser.add_argument("--downsize-gsd", type=float, help="Target Ground Sample Distance in map units (must be larger than source)")
    
    # Tiling options
    parser.add_argument("--tile-size", type=int, default=20000, help="Tile size in pixels")
    parser.add_argument("--load", type=int, default=100, help="CPU load percentage (default 100)")
    parser.add_argument("--include_empty_tiles", action="store_true", help="Do not delete empty tiles")
    parser.add_argument("--keep-xml", action="store_true", help="Keep PAM XML sidecar files for valid tiles")

    args = parser.parse_args()

    # Step 0: Validation
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        sys.exit(1)

    ds = gdal.Open(args.input_file)
    original_gsd = abs(ds.GetGeoTransform()[1])
    original_size_gb = os.path.getsize(args.input_file) / (1024**3)
    ds = None

    # Calculate Resampling Factor
    resample_factor = 1.0
    input_size_gb = os.path.getsize(args.input_file) / (1024**3)
    
    if args.downsize_percent:
        resample_factor = args.downsize_percent / 100.0
    elif args.downsize_gsd:
        if args.downsize_gsd < original_gsd:
            print(f"Error: Target GSD ({args.downsize_gsd}) is smaller than original GSD ({original_gsd:.4f}). Resolution cannot be increased.")
            sys.exit(1)
        resample_factor = original_gsd / args.downsize_gsd
    elif args.downsize_GB:
        # Factor^2 = Target/Current
        resample_factor = (args.downsize_GB / original_size_gb)**0.5
    elif input_size_gb > 20:
        print(f"🐘 Input file is large ({input_size_gb:.1f}GB). Defaulting to 30% downsizing to preserve disk space...")
        resample_factor = 0.30

    input_filename = os.path.basename(args.input_file)

    # Step 1: Transformation (VRT)
    input_to_tile = args.input_file
    temp_vrt = None
    
    # Check if any transform is needed
    needs_vrt = args.scale != 1.0 or args.shift_x != 0.0 or args.shift_y != 0.0 or resample_factor != 1.0
    
    if needs_vrt:
        msg = "⚖️  Applying virtual transformations: "
        if resample_factor != 1.0:
            msg += f"Resample={resample_factor:.4f} (GSD -> {original_gsd/resample_factor:.3f}) "
        if args.scale != 1.0:
            msg += f"Scale={args.scale} @ ({args.anchor_x}, {args.anchor_y}) "
        if args.shift_x != 0.0 or args.shift_y != 0.0:
            msg += f"Shift=({args.shift_x}, {args.shift_y})"
        print(msg)
        
        temp_vrt = create_transformed_vrt(args.input_file, args.scale, args.anchor_x, args.anchor_y, args.shift_x, args.shift_y, resample_factor)
        input_to_tile = temp_vrt

    # Step 2: Directories
    if args.output_dir:
        output_dir = args.output_dir
    else:
        input_abs = os.path.abspath(args.input_file)
        input_dir = os.path.dirname(input_abs)
        base_name = os.path.splitext(input_filename)[0]
        output_dir = os.path.join(input_dir, f"{base_name}_tiles")
    
    if os.path.exists(output_dir):
        if args.clobber:
            print(f"⚠️  Output directory {output_dir} exists. Overwriting...")
            for f in os.listdir(output_dir):
                fp = os.path.join(output_dir, f)
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
        else:
            print(f"Error: Output directory {output_dir} already exists. Use --clobber to replace it.")
            sys.exit(1)
    else:
        os.makedirs(output_dir)

    # Step 3: Parallel Tiling
    ds = gdal.Open(input_to_tile)
    width, height = ds.RasterXSize, ds.RasterYSize
    ds = None
    
    num_cores = max(1, int(os.cpu_count() * (args.load / 100.0)))
    print(f"🚀 Tiling {input_filename} using {num_cores} cores...")
    
    tasks = []
    tile_paths = []
    for y in range(0, height, args.tile_size):
        for x in range(0, width, args.tile_size):
            tile_w = min(args.tile_size, width - x)
            tile_h = min(args.tile_size, height - y)
            out_name = os.path.join(output_dir, f"tile_{x}_{y}.tif")
            tasks.append((x, y, tile_w, tile_h, out_name, input_to_tile, args.keep_xml))
            tile_paths.append(out_name)

    total_tasks = len(tasks)
    with Pool(num_cores) as p:
        try:
            completed = 0
            for _ in p.imap_unordered(create_tile, tasks):
                completed += 1
                if completed % max(1, total_tasks // 10) == 0 or completed == total_tasks:
                    print(f"   Tiling Progress: {(completed/total_tasks)*100:6.1f}% ({completed}/{total_tasks} tiles completed)")
        except Exception as e:
            print(f"\n\033[91m❌ [FATAL] Error during tiling (possibly disk full): {e}\033[0m")
            print(f"🧹 Cleaning up partial output directory: {output_dir}")
            p.terminate()
            p.join()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            sys.exit(1)

    # Step 4: Parallel Cleaning
    if not args.include_empty_tiles:
        print(f"🧹 Removing empty tiles using {num_cores} cores...")
        total_tiles_to_scan = len(tile_paths)
        deleted_count = 0
        completed_scans = 0
        with Pool(num_cores) as p:
            for result in p.imap_unordered(clean_task, tile_paths):
                if result:
                    deleted_count += 1
                completed_scans += 1
                if completed_scans % max(1, total_tiles_to_scan // 10) == 0 or completed_scans == total_tiles_to_scan:
                    print(f"   Cleaning Progress: {(completed_scans/total_tiles_to_scan)*100:6.1f}% ({completed_scans}/{total_tiles_to_scan} tiles scanned)")
        print(f"   Removed {deleted_count} empty tiles.")

    # Step 4.5: XML Cleanup Pass (Remove sidecars unless requested)
    if not args.keep_xml:
        for f in os.listdir(output_dir):
            if f.endswith(".xml") and not f.startswith("._"):
                try:
                    os.remove(os.path.join(output_dir, f))
                except Exception as e:
                    print(f"Warning: Could not remove XML {f}: {e}")

    # Step 4.6: Sequential Renaming
    print(f"🏷️  Renaming remaining tiles using prefix: {os.path.splitext(input_filename)[0]}...")
    current_tiles = [f for f in os.listdir(output_dir) if f.lower().endswith('.tif') and not f.startswith('._')]
    # Sort to maintain spatial order. Filenames are tile_x_y.tif
    def sort_key(f):
        try:
            parts = f.replace('.tif', '').split('_')
            # Expecting ['tile', x, y]
            if len(parts) >= 3:
                return (int(parts[-1]), int(parts[-2]))
        except (ValueError, IndexError):
            pass
        return f
    current_tiles.sort(key=sort_key)
    
    base_prefix = os.path.splitext(input_filename)[0]
    final_tiles = []
    
    # Calculate padding based on total tiles
    total_tiles = len(current_tiles)
    padding = len(str(total_tiles)) if total_tiles >= 10 else 1
    
    for i, old_name in enumerate(current_tiles, 1):
        new_name = f"{base_prefix}_{i:0{padding}d}.tif"
        old_path = os.path.join(output_dir, old_name)
        new_path = os.path.join(output_dir, new_name)
        
        # Also handle XML if it exists
        for ext in [".aux.xml", ".xml"]:
            old_xml = old_path + ext
            if os.path.exists(old_xml):
                try:
                    os.rename(old_xml, new_path + ext)
                except Exception as e:
                    print(f"Warning: Could not rename XML {old_xml}: {e}")
        
        try:
            os.rename(old_path, new_path)
            final_tiles.append(new_path)
        except Exception as e:
            print(f"Error: Could not rename tile {old_path}: {e}")
            final_tiles.append(old_path)

    # Step 5: Indexing
    print("📋 Building VRT index...")
    vrt_index_path = os.path.join(output_dir, "index.vrt")
    subprocess.run(["gdalbuildvrt", vrt_index_path] + final_tiles, stdout=subprocess.DEVNULL)

    # Cleanup temp scale VRT
    if temp_vrt and os.path.exists(temp_vrt):
        os.remove(temp_vrt)

    # Final summary metrics
    total_tiles_count = len(final_tiles)
    dir_size_gb = get_dir_size_gb(output_dir)
    print(f"✨ Done! Deliverable ready in: {output_dir} ({total_tiles_count} tiles, {dir_size_gb:.2f}GB)")

if __name__ == "__main__":
    main()
