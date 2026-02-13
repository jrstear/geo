#!/usr/bin/env python3
"""
compare.py

A Quality Control utility to compare an original GeoTIFF with its 
processed/scaled/shifted counterpart (VRT or TIFF).

Determines:
- Scale Factor
- Grid-to-Ground Shift
- Bounding Box Footprint
- Deliverable Statistics (Tile Count, Total Size)
"""

import os
import sys
import argparse
from osgeo import gdal

gdal.UseExceptions()

def get_envelope(gt, width, height):
    """Returns (minX, maxX, minY, maxY) based on geotransform."""
    minX = gt[0]
    maxX = gt[0] + width * gt[1] + height * gt[2]
    minY = gt[3] + width * gt[4] + height * gt[5]
    maxY = gt[3]
    return minX, maxX, minY, maxY

def get_dir_stats(path):
    """Scan directory for tile count and total size in MB/GB."""
    count = 0
    total_size = 0
    for f in os.listdir(path):
        if f.lower().endswith(('.tif', '.tiff')):
            count += 1
            total_size += os.path.getsize(os.path.join(path, f))
    return count, total_size

def analyze_ds(path):
    """Extract key metadata from a GDAL-compatible file."""
    ds = gdal.Open(path)
    if ds is None:
        return None
    
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    env = get_envelope(gt, width, height)
    
    return {
        "path": path,
        "width": width,
        "height": height,
        "gt": gt,
        "envelope": env,
        "px_size_x": abs(gt[1]),
        "px_size_y": abs(gt[5])
    }

def main():
    parser = argparse.ArgumentParser(description="Compare two rasters to determine georeferencing differences.")
    parser.add_argument("original", help="Path to original GeoTIFF")
    parser.add_argument("processed", help="Path to processed VRT or Output Directory")
    args = parser.parse_args()

    # 1. Resolve Processed Paths
    processed_path = args.processed
    out_dir = None
    
    if os.path.isdir(args.processed):
        out_dir = args.processed
        # Look for index.vrt in the directory
        index_vrt = os.path.join(args.processed, "index.vrt")
        if os.path.exists(index_vrt):
            processed_path = index_vrt
        else:
            # Look for any .tif if index doesn't exist
            tifs = [f for f in os.listdir(args.processed) if f.lower().endswith('.tif')]
            if tifs:
                processed_path = os.path.join(args.processed, tifs[0])
                print(f"Note: No index.vrt found, using sample tile: {tifs[0]}")
            else:
                print(f"Error: Directory contains no valid rasters.")
                sys.exit(1)
    elif os.path.isfile(args.processed) and args.processed.lower().endswith('.vrt'):
        # If user pointed directly to index.vrt, the container is the out_dir
        out_dir = os.path.dirname(args.processed) or "."

    # 2. Analyze Datasets
    orig = analyze_ds(args.original)
    proc = analyze_ds(processed_path)

    if not orig or not proc:
        print("Error: Could not open one or both datasets.")
        sys.exit(1)

    # 3. Calculate Differences (Relative to 0,0 Origin)
    scale_x = proc['px_size_x'] / orig['px_size_x']
    scale_y = proc['px_size_y'] / orig['px_size_y']
    avg_scale = (scale_x + scale_y) / 2.0
    
    # Translation Residual (Shift at 0,0)
    # If it was a pure scale about 0,0 then ProcOrg = OrigOrg * Scale
    shift_x = proc['gt'][0] - orig['gt'][0] * avg_scale
    shift_y = proc['gt'][3] - orig['gt'][3] * avg_scale

    # Anchor calculation (Theoretical point of zero displacement)
    # Anchor = (NewOrg - OldOrg * Scale) / (1 - Scale)
    try:
        anchor_x = (proc['gt'][0] - orig['gt'][0] * avg_scale) / (1 - avg_scale)
        anchor_y = (proc['gt'][3] - orig['gt'][3] * avg_scale) / (1 - avg_scale)
    except ZeroDivisionError:
        anchor_x, anchor_y = 0, 0

    # 4. Reporting
    orig_count = 1
    orig_size = os.path.getsize(args.original)
    
    proc_count = 1
    proc_size = os.path.getsize(processed_path)
    if out_dir:
        proc_count, proc_size = get_dir_stats(out_dir)

    def fmt_size(size_bytes):
        return f"{size_bytes / (1024**3):.2f} GB"

    print("-" * 105)
    print(f"{'📊 RASTER COMPARISON REPORT':^105}")
    print("-" * 105)
    print(f"{'Source':<30} {'TIFS':<8} {'SIZE':<12} {'GSD':<10} {'DIMENSIONS':<20}")
    print(f"{os.path.basename(args.original):<30} {orig_count:<8} {fmt_size(orig_size):<12} {orig['px_size_x']:<10.4f} {orig['width']}x{orig['height']}")
    print(f"{os.path.basename(processed_path):<30} {proc_count:<8} {fmt_size(proc_size):<12} {proc['px_size_x']:<10.4f} {proc['width']}x{proc['height']}")
    print("-" * 105)
    
    print("\n[TRANSFORMATION]")
    if abs(avg_scale - 1.0) < 1e-9:
        print("Scale:      1.0 (No Scaling)")
    else:
        print(f"Scale:      {avg_scale:.8f}")
    
    if abs(shift_x) < 1e-3 and abs(shift_y) < 1e-3:
        print(f"Origin Shift: (No Shift)")
    else:
        print(f"Origin Shift: X={shift_x:+.4f}, Y={shift_y:+.4f}")
    
    if abs(avg_scale - 1.0) > 1e-9:
        # Round the probable anchor if it's very close to zero
        anchor_x = 0.0 if abs(anchor_x) < 1e-4 else anchor_x
        anchor_y = 0.0 if abs(anchor_y) < 1e-4 else anchor_y
        print(f"Probable Anchor (if shift was 0): ({anchor_x:.3f}, {anchor_y:.3f})")

    print("\n[BOUNDING BOX - FOOTPRINT]")
    print(f"Original: {orig['envelope'][0]:.2f}, {orig['envelope'][2]:.2f} -> {orig['envelope'][1]:.2f}, {orig['envelope'][3]:.2f}")
    print(f"Scaled:   {proc['envelope'][0]:.2f}, {proc['envelope'][2]:.2f} -> {proc['envelope'][1]:.2f}, {proc['envelope'][3]:.2f}")
    
    # Drift residuals after scaling
    drift_w = proc['envelope'][1] - orig['envelope'][1] * avg_scale
    drift_h = proc['envelope'][2] - orig['envelope'][2] * avg_scale
    
    if abs(drift_w) < 1e-2 and abs(drift_h) < 1e-2:
        print(f"Edge Drift:   (No Drift)")
    else:
        print(f"Edge Drift:   Easting={drift_w:+.3f}, Northing={drift_h:+.3f}")
    print("-" * 80)

if __name__ == "__main__":
    main()
