#!/usr/bin/env python3
"""
convert_coords.py — Split a GCPEditorPro confirmed file into ODM control and check files.

Reads a GCPEditorPro confirmed file (tab-separated ODM format, first line = CRS),
splits observations by label prefix:
  • GCP-* → gcp_list.txt   (used as ODM --gcp input, or auto-detected if in project root)
  • CHK-* → chk_list.txt   (used by rmse_calc.py for accuracy verification)

Reprojects XY to EPSG:32613 (WGS 84 / UTM zone 13N, metres) and converts Z
from US survey feet to metres when the input CRS is a feet-based state plane
(e.g. EPSG:3618, EPSG:6529).  Only "confirmed" observations are written
(rows whose 8th column is "confirmed", or all rows if column 8 is absent).

Usage:
    conda run -n geo python TargetSighter/convert_coords.py \\
        <confirmed_file> \\
        --out-dir <output_dir> \\
        [--target-crs EPSG:32613]

Outputs (written to --out-dir):
    gcp_list.txt   GCP-* only, EPSG:32613
    chk_list.txt   CHK-* only, EPSG:32613
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

FT_TO_M = 0.3048006096012192   # US survey foot (exact 1200/3937)
TARGET_CRS_DEFAULT = "EPSG:32613"


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def _crs_axis_unit(epsg: str) -> str:
    """Return 'foot' or 'metre' for the horizontal axis of an EPSG code."""
    try:
        from pyproj import CRS
        crs = CRS.from_epsg(int(epsg.split(":")[-1]))
        unit = crs.axis_info[0].unit_name.lower()
        return unit
    except Exception:
        return "metre"


def _is_feet_crs(epsg: str) -> bool:
    unit = _crs_axis_unit(epsg)
    return "foot" in unit or "feet" in unit


def _make_transformer(src_epsg: str, dst_epsg: str):
    from pyproj import Transformer
    return Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)


# ---------------------------------------------------------------------------
# Parse confirmed file
# ---------------------------------------------------------------------------

def parse_confirmed(path: str):
    """
    Returns (crs_header, rows) where each row is a list of string fields.
    Only returns rows with >= 7 fields and (if col 8 is present) col 8 == 'confirmed'.
    """
    with open(path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"File is empty: {path}")

    crs = lines[0].strip()
    rows = []
    for raw in lines[1:]:
        line = raw.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        # Filter to confirmed observations only (if the column exists)
        if len(fields) >= 8 and fields[7] != "confirmed":
            continue
        rows.append(fields)

    return crs, rows


# ---------------------------------------------------------------------------
# Write output file
# ---------------------------------------------------------------------------

def write_gcp_file(path: Path, crs: str, rows: List[List[str]]):
    with open(path, "w") as f:
        f.write(crs + "\n")
        for fields in rows:
            # Write only the 7 standard ODM columns: geo_x geo_y geo_z px py image label
            f.write("\t".join(fields[:7]) + "\n")
    print(f"  wrote {path}  ({len(rows)} observations)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Split a GCPEditorPro confirmed file into ODM control + check files."
    )
    ap.add_argument("confirmed", help="Input confirmed file (GCPEditorPro format)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: same dir as input)")
    ap.add_argument("--target-crs", default=TARGET_CRS_DEFAULT,
                    help=f"Output CRS (default: {TARGET_CRS_DEFAULT})")
    args = ap.parse_args()

    in_path = Path(args.confirmed)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse ---
    src_crs, rows = parse_confirmed(str(in_path))
    print(f"Input:  {in_path}  ({len(rows)} confirmed observations, CRS: {src_crs})")

    # --- Coordinate conversion ---
    needs_xy_reproject = src_crs.upper() != args.target_crs.upper()
    needs_z_convert = _is_feet_crs(src_crs)

    xfm = None
    if needs_xy_reproject:
        try:
            xfm = _make_transformer(src_crs, args.target_crs)
            print(f"Reprojecting XY: {src_crs} → {args.target_crs}")
        except Exception as e:
            print(f"ERROR: cannot build transformer {src_crs} → {args.target_crs}: {e}",
                  file=sys.stderr)
            sys.exit(1)

    if needs_z_convert:
        print(f"Converting Z: US survey feet → metres (× {FT_TO_M})")

    converted: List[List[str]] = []
    for fields in rows:
        try:
            x = float(fields[0])
            y = float(fields[1])
            z = float(fields[2])
        except ValueError:
            continue

        if xfm is not None:
            x, y = xfm.transform(x, y)
        if needs_z_convert:
            z *= FT_TO_M

        new_fields = [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"] + fields[3:]
        converted.append(new_fields)

    # --- Split by label prefix ---
    control_rows: List[List[str]] = []
    check_rows: List[List[str]] = []
    skipped = 0

    for fields in converted:
        label = fields[6] if len(fields) > 6 else ""
        if label.startswith("GCP-"):
            control_rows.append(fields)
        elif label.startswith("CHK-"):
            check_rows.append(fields)
        else:
            skipped += 1

    if skipped:
        labels = {f[6] for f in converted if len(f) > 6
                  and not f[6].startswith("GCP-") and not f[6].startswith("CHK-")}
        print(f"WARNING: skipped {skipped} observations with unrecognised label prefix: "
              f"{sorted(labels)}", file=sys.stderr)

    unique_gcp = len({f[6] for f in control_rows})
    unique_chk = len({f[6] for f in check_rows})
    print(f"\n  GCP- points: {unique_gcp} unique, {len(control_rows)} observations")
    print(f"  CHK- points: {unique_chk} unique, {len(check_rows)} observations")

    # --- Write ---
    control_path = out_dir / "gcp_list.txt"
    check_path   = out_dir / "chk_list.txt"

    print()
    write_gcp_file(control_path, args.target_crs, control_rows)
    write_gcp_file(check_path,   args.target_crs, check_rows)

    print(f"\nDone.  Run ODM with:  --gcp {control_path}")
    print(f"       Run RMSE with: {check_path}")


if __name__ == "__main__":
    main()
