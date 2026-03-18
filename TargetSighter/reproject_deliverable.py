#!/usr/bin/env python3
"""Reproject an ODM deliverable raster to a field CRS for QGIS review or customer delivery.

Relationship to package.py
---------------------------
This script handles the CRS reprojection step (map projection change, pixel resampling).
package.py handles the subsequent design-grid offset step (geotransform shift, no
resampling) and tiling/delivery formatting.

Full customer delivery pipeline:
  1. reproject_deliverable.py  --dst-crs EPSG:3618   (EPSG:32613 → state plane, gdalwarp)
  2. package.py --shift-x +1546702.929 --shift-y -3567.471  (state plane → design grid, VRT)

GeoTIFF cannot embed an arbitrary linear offset in its CRS definition, so raster
delivery stays in state plane (EPSG:3618) and the design-grid offset is documented
for the client's CAD/GIS software. The offset only needs to be applied to vector
layers (contours, TIN) via package.py, which already supports --shift-x/y.

When transform.yaml (per-job config, see geo-eta) is available, a wrapper script
can read src_crs/dst_crs/design_grid_offset and chain both steps automatically.

Usage examples
--------------
# Aztec3 review in QGIS (state plane feet, overlays F100340_AZTEC_points.csv):
python TargetSighter/reproject_deliverable.py \\
    ~/stratus/aztec3/odm_orthophoto/odm_orthophoto.tif \\
    ~/stratus/aztec3/odm_orthophoto/odm_orthophoto_3618.tif \\
    --src-crs EPSG:32613 --dst-crs EPSG:3618

# Explicit COG output (for large files):
python TargetSighter/reproject_deliverable.py \\
    ~/stratus/aztec3/odm_orthophoto/odm_orthophoto.tif \\
    ~/stratus/aztec3/odm_orthophoto/odm_orthophoto_3618_cog.tif \\
    --src-crs EPSG:32613 --dst-crs EPSG:3618 --cog

QGIS instructions
-----------------
Load both layers in QGIS with project CRS = EPSG:3618:
  - Raster: odm_orthophoto_3618.tif  (CRS embedded in file)
  - Vector: F100340_AZTEC_points.csv as Delimited Text layer
            X = easting_ft, Y = northing_ft, CRS = EPSG:3618

Design-grid offset (for customer delivery via package.py, not applied here):
  Design E = state_plane_easting  + 1,546,702.929 ft
  Design N = state_plane_northing - 3,567.471 ft
"""

import argparse
import subprocess
import sys
from pathlib import Path


def reproject(src: Path, dst: Path, src_crs: str, dst_crs: str, cog: bool) -> None:
    cmd = [
        "gdalwarp",
        "-s_srs", src_crs,
        "-t_srs", dst_crs,
        "-r", "bilinear",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
    ]
    if cog:
        cmd += ["-of", "COG"]
    cmd += [str(src), str(dst)]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"Output: {dst}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input",  type=Path, help="Input raster (GeoTIFF)")
    p.add_argument("output", type=Path, help="Output raster path")
    p.add_argument("--src-crs", default="EPSG:32613",
                   help="Source CRS (default: EPSG:32613 — ODM UTM 13N)")
    p.add_argument("--dst-crs", default="EPSG:3618",
                   help="Destination CRS (default: EPSG:3618 — NAD83 NM Central ftUS)")
    p.add_argument("--cog", action="store_true",
                   help="Write Cloud Optimized GeoTIFF (for large files)")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    if args.output.exists():
        print(f"Warning: overwriting {args.output}")

    reproject(args.input, args.output, args.src_crs, args.dst_crs, args.cog)


if __name__ == "__main__":
    main()
