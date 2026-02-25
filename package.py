#!/usr/bin/env python3
"""
package.py

A high-performance parallel raster packager that:
1. Scales/Shifts (Optional): Using metadata-only VRT georeferencing adjustments.
2. Tiles (Parallel): Using parallelized gdal_translate.
3. Cleans (Parallel): Removes empty tiles using a parallel worker pool with progress.
4. Indexes: Builds a final VRT index.

Also supports:
- Contour .dxf transformation (scale/shift X,Y; preserve Z elevation)
- TIN LandXML .xml transformation (scale/shift northing/easting; preserve elevation)

Usage:
  python package.py input.tiff [options]
  python package.py --contour-file contour.dxf [options]
  python package.py --tin-file tin.xml [options]
"""

import os
import sys
import argparse
import subprocess
import shutil
import math
import re
from collections import defaultdict
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
# Contour DXF Transformation
# ------------------------------------------------------------------------------

def transform_dxf(input_path, output_path, scale, anchor_x, anchor_y, shift_x, shift_y, log_fn):
    """Transform X,Y coordinates in a DXF file while preserving Z (elevation).

    DXF files use alternating code/value line pairs. Group codes:
      10 = X coordinate  →  transform
      20 = Y coordinate  →  transform
      38 = elevation (Z) →  leave unchanged
    All other codes pass through unchanged.
    """
    log_fn(f"  Reading DXF: {input_path}")

    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    out_lines = []
    i = 0
    transformed = 0

    while i < len(lines) - 1:
        code_line = lines[i]
        value_line = lines[i + 1]

        try:
            code = int(code_line.strip())
        except ValueError:
            # Not a valid integer code line — output as-is and advance one line
            out_lines.append(code_line)
            i += 1
            continue

        if code == 10:
            try:
                x = float(value_line.strip())
                new_x = anchor_x + (x - anchor_x) * scale + shift_x
                orig = value_line.strip()
                n_dec = len(orig.split('.')[-1]) if '.' in orig else 3
                n_dec = max(n_dec, 3)
                out_lines.append(code_line)
                out_lines.append(f"{new_x:.{n_dec}f}\n")
                transformed += 1
                i += 2
                continue
            except ValueError:
                pass
        elif code == 20:
            try:
                y = float(value_line.strip())
                new_y = anchor_y + (y - anchor_y) * scale + shift_y
                orig = value_line.strip()
                n_dec = len(orig.split('.')[-1]) if '.' in orig else 3
                n_dec = max(n_dec, 3)
                out_lines.append(code_line)
                out_lines.append(f"{new_y:.{n_dec}f}\n")
                transformed += 1
                i += 2
                continue
            except ValueError:
                pass

        # Pass through both code and value lines unchanged
        out_lines.append(code_line)
        out_lines.append(value_line)
        i += 2

    # Append any remaining odd line
    if i < len(lines):
        out_lines.append(lines[i])

    log_fn(f"  Writing transformed DXF: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

    log_fn(f"  DXF done. Transformed {transformed} coordinate values.")

# ------------------------------------------------------------------------------
# TIN LandXML Processing
# ------------------------------------------------------------------------------

def process_tin(input_path, scale, anchor_x, anchor_y, shift_x, shift_y,
                max_size_mb, output_dir, suffix, log_fn):
    """Process a LandXML TIN file — transform northing/easting, preserve elevation.

    LandXML point format: <P id="N">northing easting elevation</P>
      northing = Y  →  transform with anchor_y / shift_y
      easting  = X  →  transform with anchor_x / shift_x
      elevation     →  leave unchanged

    If max_size_mb > 0 and file exceeds that size, the output is spatially tiled.
    """
    log_fn(f"  Reading TIN XML: {input_path}")
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    log_fn(f"  File size: {file_size_mb:.1f} MB")

    input_abs = os.path.abspath(input_path)
    input_dir = os.path.dirname(input_abs)
    basename = os.path.splitext(os.path.basename(input_abs))[0]

    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    do_tiling = (max_size_mb > 0) and (file_size_mb > max_size_mb)

    # ── Non-tiled: simple regex substitution ──────────────────────────────────
    if not do_tiling:
        def replace_point(m):
            pid = m.group(1)
            parts = m.group(2).split()
            if len(parts) < 3:
                return m.group(0)
            northing_str, easting_str, elevation = parts[0], parts[1], parts[2]
            northing = float(northing_str)
            easting  = float(easting_str)
            new_northing = anchor_y + (northing - anchor_y) * scale + shift_y
            new_easting  = anchor_x + (easting  - anchor_x) * scale + shift_x
            n_dec_n = max(len(northing_str.split('.')[-1]) if '.' in northing_str else 3, 3)
            n_dec_e = max(len(easting_str.split('.')[-1])  if '.' in easting_str  else 3, 3)
            return f'<P id="{pid}">{new_northing:.{n_dec_n}f} {new_easting:.{n_dec_e}f} {elevation}</P>'

        new_content = re.sub(r'<P id="([^"]+)">([^<]+)</P>', replace_point, content)

        out_path = os.path.join(input_dir, f"{basename}{suffix}.xml")
        log_fn(f"  Writing: {out_path}")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        log_fn(f"  TIN transformation complete.")
        return

    # ── Tiled case ─────────────────────────────────────────────────────────────
    log_fn(f"  File exceeds {max_size_mb} MB limit — tiling output...")

    # Parse and transform all P elements
    points = {}
    for m in re.finditer(r'<P id="([^"]+)">([^<]+)</P>', content):
        pid = m.group(1)
        parts = m.group(2).split()
        if len(parts) >= 3:
            northing  = float(parts[0])
            easting   = float(parts[1])
            elevation = float(parts[2])
            new_northing = anchor_y + (northing - anchor_y) * scale + shift_y
            new_easting  = anchor_x + (easting  - anchor_x) * scale + shift_x
            points[pid] = (new_northing, new_easting, elevation)

    log_fn(f"  Parsed {len(points)} points.")

    # Parse all F elements
    faces = []
    for m in re.finditer(r'<F>([^<]+)</F>', content):
        parts = m.group(1).split()
        if len(parts) >= 3:
            faces.append((parts[0], parts[1], parts[2]))

    log_fn(f"  Parsed {len(faces)} faces.")

    if not points or not faces:
        log_fn("  No points/faces found — skipping tiling.")
        return

    # Extract LandXML header (everything before <Surfaces)
    surfaces_match = re.search(r'<Surfaces[\s>]', content)
    header = content[:surfaces_match.start()] if surfaces_match else '<?xml version="1.0" encoding="utf-8"?>\n<LandXML>\n'

    # Compute grid dimensions
    n_tiles_needed = math.ceil(file_size_mb / max_size_mb)
    n_per_side = math.ceil(math.sqrt(n_tiles_needed))
    log_fn(f"  Grid: {n_per_side}x{n_per_side} ({n_per_side**2} cells max)")

    # Coordinate bounds for grid partitioning
    all_eastings  = [p[1] for p in points.values()]
    all_northings = [p[0] for p in points.values()]
    min_e, max_e = min(all_eastings),  max(all_eastings)
    min_n, max_n = min(all_northings), max(all_northings)
    e_range = (max_e - min_e) or 1.0
    n_range = (max_n - min_n) or 1.0

    # Assign each face to a grid cell by centroid
    tile_faces = defaultdict(list)
    for p1, p2, p3 in faces:
        if p1 not in points or p2 not in points or p3 not in points:
            continue
        ce = (points[p1][1] + points[p2][1] + points[p3][1]) / 3
        cn = (points[p1][0] + points[p2][0] + points[p3][0]) / 3
        gi = min(int((ce - min_e) / e_range * n_per_side), n_per_side - 1)
        gj = min(int((cn - min_n) / n_range * n_per_side), n_per_side - 1)
        tile_faces[(gj, gi)].append((p1, p2, p3))

    # Output directory
    tile_dir = output_dir if output_dir else os.path.join(input_dir, f"{basename}_tiles")
    os.makedirs(tile_dir, exist_ok=True)

    # Write each non-empty tile
    tile_num = 1
    for cell_faces in tile_faces.values():
        if not cell_faces:
            continue

        # Collect unique point IDs and build local 1-based mapping
        used_pids = set()
        for p1, p2, p3 in cell_faces:
            used_pids.update([p1, p2, p3])

        def sort_pid(pid):
            try:
                return int(pid)
            except ValueError:
                return pid

        sorted_pids = sorted(used_pids, key=sort_pid)
        pid_map = {old: str(new_id) for new_id, old in enumerate(sorted_pids, 1)}

        # Build tile XML
        parts = [header]
        parts.append('  <Surfaces>\n')
        parts.append(f'    <Surface name="{basename}_tile_{tile_num}">\n')
        parts.append('      <Definition surfType="TIN">\n')
        parts.append('        <Pnts>\n')
        for old_pid in sorted_pids:
            n, e, z = points[old_pid]
            parts.append(f'          <P id="{pid_map[old_pid]}">{n:.6f} {e:.6f} {z:.6f}</P>\n')
        parts.append('        </Pnts>\n')
        parts.append('        <Faces>\n')
        for p1, p2, p3 in cell_faces:
            parts.append(f'          <F>{pid_map[p1]} {pid_map[p2]} {pid_map[p3]}</F>\n')
        parts.append('        </Faces>\n')
        parts.append('      </Definition>\n')
        parts.append('    </Surface>\n')
        parts.append('  </Surfaces>\n')
        parts.append('</LandXML>\n')

        out_path = os.path.join(tile_dir, f"{basename}_{tile_num}.xml")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(parts)

        log_fn(f"  Tile {tile_num}: {len(cell_faces)} faces, {len(used_pids)} points → {os.path.basename(out_path)}")
        tile_num += 1

    log_fn(f"  TIN tiling complete: {tile_num - 1} tiles in {tile_dir}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel Raster Packager (Scale, Shift, Downsize, Tile, Clean).")
    parser.add_argument("input_file", nargs='?', help="Input GeoTIFF file (optional if --contour-file or --tin-file provided)")
    parser.add_argument("--output-dir", help="Output directory (default: <input_dir>/<input_basename>_tiles)")
    parser.add_argument("--clobber", action="store_true", help="Overwrite conflicting files in output directory")

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

    # Contour DXF options
    parser.add_argument("--contour-file", help="Input .dxf contour file")
    parser.add_argument("--contour-suffix", default="_geo", help="Output suffix for contour DXF (default: _geo)")

    # TIN XML options
    parser.add_argument("--tin-file", help="Input LandXML .xml TIN file")
    parser.add_argument("--tin-suffix", default="_geo", help="Output suffix for single TIN file (default: _geo)")
    parser.add_argument("--tin-max-mb", type=float, default=0.0, help="Max output XML size in MB; if exceeded, tile (default: 0 = no limit)")
    parser.add_argument("--tin-output-dir", help="Output directory for TIN tiles")

    args = parser.parse_args()

    # Validate: at least one input must be provided
    if not args.input_file and not args.contour_file and not args.tin_file:
        parser.error("At least one of: input_file, --contour-file, --tin-file must be provided.")

    # ── Contour DXF processing ──────────────────────────────────────────────────
    if args.contour_file:
        contour_in = os.path.abspath(args.contour_file)
        if not os.path.exists(contour_in):
            print(f"Error: Contour file not found: {contour_in}")
            sys.exit(1)
        contour_base = os.path.splitext(contour_in)[0]
        contour_out = f"{contour_base}{args.contour_suffix}.dxf"
        print(f"📐 Processing contour DXF: {os.path.basename(contour_in)}")
        transform_dxf(contour_in, contour_out,
                      args.scale, args.anchor_x, args.anchor_y,
                      args.shift_x, args.shift_y, print)
        print(f"✨ Contour done: {contour_out}")

    # ── TIN XML processing ──────────────────────────────────────────────────────
    if args.tin_file:
        tin_in = os.path.abspath(args.tin_file)
        if not os.path.exists(tin_in):
            print(f"Error: TIN file not found: {tin_in}")
            sys.exit(1)
        print(f"🗺️  Processing TIN XML: {os.path.basename(tin_in)}")
        process_tin(tin_in,
                    args.scale, args.anchor_x, args.anchor_y,
                    args.shift_x, args.shift_y,
                    args.tin_max_mb, args.tin_output_dir,
                    args.tin_suffix, print)
        print(f"✨ TIN done.")

    # ── TIF processing ──────────────────────────────────────────────────────────
    if args.input_file:
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

        os.makedirs(output_dir, exist_ok=True)

        # Conflict check
        vrt_index_path = os.path.join(output_dir, "index.vrt")
        base_prefix = os.path.splitext(input_filename)[0]

        # We check if index.vrt exists OR if any final numbered tiles exist
        # (Since total tile count depends on cleaning, we just check for the first few)
        potentially_conflicting = [vrt_index_path]
        for i in range(1, 10): # Check first few as a proxy
            potentially_conflicting.append(os.path.join(output_dir, f"{base_prefix}_{i}.tif"))
            potentially_conflicting.append(os.path.join(output_dir, f"{base_prefix}_{i:02d}.tif"))
            potentially_conflicting.append(os.path.join(output_dir, f"{base_prefix}_{i:03d}.tif"))

        conflicts = [f for f in potentially_conflicting if os.path.exists(f)]

        if conflicts and not args.clobber:
            print(f"Error: Conflicting files found in {output_dir}:")
            for c in conflicts[:5]:
                print(f"  - {os.path.basename(c)}")
            if len(conflicts) > 5:
                print(f"  ... and {len(conflicts)-5} more.")
            print("Use --clobber to overwrite them.")
            sys.exit(1)
        elif conflicts:
             print(f"⚠️  Conflicts detected in {output_dir}. Overwriting relevant files...")

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
        print(f"🏷️  Renaming remaining tiles using prefix: {base_prefix}...")

        # CRITICAL: We only rename the tiles WE created in THIS run
        current_run_tiles = [f for f in tile_paths if os.path.exists(f)]

        # Sort to maintain spatial order. Filenames are tile_x_y.tif
        def sort_key(f_path):
            f = os.path.basename(f_path)
            try:
                parts = f.replace('.tif', '').split('_')
                if len(parts) >= 3:
                    return (int(parts[-1]), int(parts[-2]))
            except (ValueError, IndexError):
                pass
            return f
        current_run_tiles.sort(key=sort_key)

        base_prefix = os.path.splitext(input_filename)[0]
        final_tiles = []

        # Calculate padding based on total tiles
        total_tiles = len(current_run_tiles)
        padding = len(str(total_tiles)) if total_tiles >= 10 else 1

        for i, old_path in enumerate(current_run_tiles, 1):
            new_name = f"{base_prefix}_{i:0{padding}d}.tif"
            old_path = os.path.abspath(old_path)
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
