#!/usr/bin/env python3
"""
accuracy_report.py — CLI tool for comparing survey CSV elevations against a TIN model.

The survey CSV coordinates are transformed into TIN space using shift-e / shift-n / shift-z
before interpolation.  Feature types that are known to have systematic offsets unrelated
to TIN accuracy (e.g. building corners measured to a different surface) can be flagged
with --flag-types so they are reported separately and excluded from the headline statistics.

Usage example (Red Rocks, March 2026 survey vs. Dec 2025 TIN):

  python accuracy_report.py \\
      ~/stratus/redrocks/tin.xml \\
      ~/stratus/redrocks/260303.csv \\
      --shift-e 2222911.070 \\
      --shift-n 58.733 \\
      --shift-z 4.169 \\
      --flag-types "BUILDING CORNER" "CONCRETE PAD" "BTM CULVERT" \\
      --out-csv ~/stratus/redrocks/accuracy_260303.csv
"""

import argparse
import csv
import math
import os
import sys
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from tin_processor import parse_landxml_tin, compare_elevations, calculate_statistics


# Feature types whose vertical offsets are systematic (different measurement point on
# the structure, not TIN error).  These are flagged in the report but not excluded from
# the comparison — callers can add more via --flag-types.
_DEFAULT_FLAG_TYPES_REDROCKS = [
    "BUILDING CORNER",
    "CONCRETE PAD",
    "BTM CULVERT",
]


def _detect_note_field(fieldnames):
    """Return the name of the feature-type column, tolerating 'Note' or 'Type'."""
    for name in fieldnames:
        if name.strip().lower() in ("note", "type"):
            return name
    return None


def load_survey_csv(path: str, note_field: Optional[str] = None,
                    shift_e: float = 0.0, shift_n: float = 0.0,
                    shift_z: float = 0.0) -> pd.DataFrame:
    """Read survey CSV, apply 3D shift, return DataFrame expected by tin_processor."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        nf = note_field or _detect_note_field(reader.fieldnames or [])
        for row in reader:
            rows.append({
                "id": row["ID"],
                "easting": float(row["Easting"]) - shift_e,
                "northing": float(row["Northing"]) - shift_n,
                "elevation": float(row["Elevation"]) - shift_z,
                "type": row[nf].strip() if nf and nf in row else "",
            })
    return pd.DataFrame(rows)


def print_stats_block(label: str, results, unit: str, file=sys.stdout):
    """Print a compact statistics block for a subset of results."""
    valid = [r for r in results if r["status"] == "success"]
    outside = [r for r in results if r["status"] == "outside_coverage"]
    n = len(valid)
    print(f"\n{label}", file=file)
    print(f"  Points in comparison : {len(results)}", file=file)
    print(f"  Inside TIN coverage  : {n}", file=file)
    if outside:
        print(f"  Outside TIN coverage : {len(outside)}", file=file)
    if n == 0:
        print("  (no valid points)", file=file)
        return
    errs = [r["discrepancy"] for r in valid]
    abs_errs = [abs(e) for e in errs]
    rmse = math.sqrt(sum(e**2 for e in errs) / n)
    mean_e = sum(errs) / n
    std_e = math.sqrt(sum((e - mean_e)**2 for e in errs) / n)
    u = f" {unit}" if unit else ""
    print(f"  RMSE                 : {rmse:.4f}{u}", file=file)
    print(f"  Mean error (TIN-obs) : {mean_e:+.4f}{u}", file=file)
    print(f"  Std deviation        : {std_e:.4f}{u}", file=file)
    print(f"  Max abs error        : {max(abs_errs):.4f}{u}", file=file)
    print(f"  Min abs error        : {min(abs_errs):.4f}{u}", file=file)


def print_type_breakdown(results, unit: str, file=sys.stdout):
    """Print per-feature-type error stats, sorted by RMSE descending."""
    by_type = {}
    for r in results:
        if r["status"] != "success":
            continue
        t = r.get("gcp_type") or "Unknown"
        by_type.setdefault(t, []).append(r["discrepancy"])

    if not by_type:
        return

    u = f" {unit}" if unit else ""
    print(f"\n  {'Feature type':<25}  {'N':>4}  {'RMSE':>8}  {'Mean':>8}  {'MaxAbs':>8}", file=file)
    print(f"  {'-'*25}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}", file=file)
    rows = []
    for t, errs in by_type.items():
        n = len(errs)
        rmse = math.sqrt(sum(e**2 for e in errs) / n)
        mean_e = sum(errs) / n
        max_abs = max(abs(e) for e in errs)
        rows.append((rmse, t, n, mean_e, max_abs))
    for rmse, t, n, mean_e, max_abs in sorted(rows, key=lambda x: -x[0]):
        print(f"  {t:<25}  {n:>4}  {rmse:>7.3f}{u}  {mean_e:>+7.3f}{u}  {max_abs:>7.3f}{u}", file=file)


def write_csv(path: str, results, shift_e: float, shift_n: float, shift_z: float):
    """Write full per-point results to a CSV file."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "ID", "Feature_Type",
            "Easting_orig", "Northing_orig", "Elevation_orig",
            "Easting_shifted", "Northing_shifted", "Elevation_shifted",
            "TIN_Elevation", "Residual_TIN_minus_Obs", "Status",
        ])
        for r in results:
            writer.writerow([
                r["gcp_id"],
                r.get("gcp_type", ""),
                f"{r['easting'] + shift_e:.4f}",
                f"{r['northing'] + shift_n:.4f}",
                f"{r['gcp_elevation'] + shift_z:.4f}",
                f"{r['easting']:.4f}",
                f"{r['northing']:.4f}",
                f"{r['gcp_elevation']:.4f}",
                f"{r['tin_elevation']:.4f}" if r["tin_elevation"] is not None else "",
                f"{r['discrepancy']:.4f}" if r["discrepancy"] is not None else "",
                r["status"],
            ])


def main():
    parser = argparse.ArgumentParser(
        description="Compare a survey CSV against a LandXML TIN and report elevation residuals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tin",    help="LandXML TIN file (.xml)")
    parser.add_argument("survey", help="Survey CSV (columns: ID, Northing, Easting, Elevation, Note|Type)")
    parser.add_argument("--shift-e", type=float, default=0.0,
                        help="Subtract from Easting before comparing (default 0)")
    parser.add_argument("--shift-n", type=float, default=0.0,
                        help="Subtract from Northing before comparing (default 0)")
    parser.add_argument("--shift-z", type=float, default=0.0,
                        help="Subtract from Elevation before comparing (default 0)")
    parser.add_argument("--flag-types", nargs="+", default=[],
                        metavar="TYPE",
                        help="Feature types to report separately (excluded from headline stats)")
    parser.add_argument("--note-field", default=None,
                        help="Column name for feature type (auto-detected: 'Note' or 'Type')")
    parser.add_argument("--out-csv", default=None,
                        help="Write per-point results to this CSV file")
    parser.add_argument("--no-default-flags", action="store_true",
                        help="Do not apply the built-in list of flagged types")
    parser.add_argument("--unit", default=None,
                        help="Override unit label when auto-detection fails (e.g. 'ft' or 'm')")
    args = parser.parse_args()

    # Build complete flag list
    flag_types = set(t.upper() for t in args.flag_types)
    if not args.no_default_flags:
        flag_types.update(t.upper() for t in _DEFAULT_FLAG_TYPES_REDROCKS)

    # --- Parse TIN ---
    print(f"Loading TIN: {args.tin}", file=sys.stderr)
    triangles, unit = parse_landxml_tin(args.tin)
    if args.unit:
        unit = args.unit
    print(f"  {len(triangles)} triangles, unit='{unit}'", file=sys.stderr)

    # --- Load survey CSV ---
    print(f"Loading survey: {args.survey}", file=sys.stderr)
    df = load_survey_csv(args.survey, args.note_field,
                         shift_e=args.shift_e,
                         shift_n=args.shift_n,
                         shift_z=args.shift_z)
    print(f"  {len(df)} points", file=sys.stderr)

    # --- Compare elevations ---
    print("Comparing elevations...", file=sys.stderr)
    results = compare_elevations(triangles, df, debug=False)

    # --- Split into primary / flagged ---
    primary = [r for r in results if r.get("gcp_type", "").upper() not in flag_types]
    flagged = [r for r in results if r.get("gcp_type", "").upper() in flag_types]

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  ELEVATION ACCURACY REPORT")
    print(f"  TIN    : {os.path.basename(args.tin)}")
    print(f"  Survey : {os.path.basename(args.survey)}")
    print(f"  Transform applied to survey points:")
    print(f"    shift-e = -{args.shift_e:.3f}")
    print(f"    shift-n = -{args.shift_n:.3f}")
    print(f"    shift-z = -{args.shift_z:.3f}")
    print(f"{'='*60}")

    print_stats_block("PRIMARY STATISTICS (anomalous types excluded)", primary, unit)
    print_type_breakdown(primary, unit)

    if flagged:
        print_stats_block(
            f"FLAGGED TYPES ({', '.join(sorted(flag_types))})\n"
            f"  Note: these have systematic offsets unrelated to TIN accuracy.",
            flagged, unit)
        print_type_breakdown(flagged, unit)

    # --- Optional CSV export ---
    if args.out_csv:
        write_csv(args.out_csv, results, args.shift_e, args.shift_n, args.shift_z)
        print(f"\nPer-point results written to: {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
