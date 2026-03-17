#!/usr/bin/env python3
"""
test_projection.py — Validate Mode A and Mode B pixel projection accuracy.

Measures pixel error of project_pixel_mode_a() and project_pixel_mode_b()
against GCPEditorPro-confirmed pixel observations from gcp_confirmed.txt.

Usage:
    conda run -n geo python test_projection.py \\
        <reconstruction.json> \\
        <gcp_confirmed.txt> \\
        <emlid.csv> \\
        <image_dir> \\
        [--crs EPSG:6529]

Mode A requires exiftool on PATH.
Mode B requires scipy + numpy (conda geo env).
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import from TargetSighter
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TargetSighter"))
from csv2gcp import parse_survey_csv, project_pixel_mode_a, project_pixel_mode_b


# ---------------------------------------------------------------------------
# Parse gcp_confirmed.txt
# ---------------------------------------------------------------------------


def parse_confirmed(path: str) -> Tuple[str, Dict[str, List[Tuple[str, float, float]]]]:
    """
    Parse gcp_confirmed.txt (tab-separated ODM format).

    Returns:
        crs_header  : str  (line 1, e.g. 'EPSG:6529')
        obs         : dict  label → [(image_name, px, py), ...]
    """
    with open(path) as f:
        lines = f.readlines()

    crs = lines[0].strip()
    obs: Dict[str, List[Tuple[str, float, float]]] = {}

    for line in lines[1:]:
        line = line.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        try:
            px = float(fields[3])
            py = float(fields[4])
        except ValueError:
            continue
        image = fields[5]
        label = fields[6]
        obs.setdefault(label, []).append((image, px, py))

    return crs, obs


# ---------------------------------------------------------------------------
# EXIF reader for a specific list of image paths
# ---------------------------------------------------------------------------

_EXIF_TAGS = [
    "-GPSLatitude", "-GPSLongitude",
    "-AbsoluteAltitude",
    "-FocalLength", "-FocalLengthIn35mmFormat",
    "-ImageWidth", "-ImageHeight",
    "-GimbalPitchDegree", "-GimbalYawDegree", "-GimbalRollDegree",
]


def read_exif_for_files(paths: List[Path]) -> Dict[str, dict]:
    """
    Run exiftool on a specific list of image files.
    Returns {filename: exif_dict} compatible with project_pixel_mode_a().
    Returns {} if exiftool is not found.
    """
    try:
        cmd = ["exiftool", "-json", "-n"] + _EXIF_TAGS + [str(p) for p in paths]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        records = json.loads(result.stdout)
    except FileNotFoundError:
        print("WARNING: exiftool not found — skipping Mode A", file=sys.stderr)
        return {}
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"WARNING: exiftool failed: {e} — skipping Mode A", file=sys.stderr)
        return {}

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    exif_map = {}
    for rec in records:
        lat = rec.get("GPSLatitude")
        lon = rec.get("GPSLongitude")
        if lat is None or lon is None:
            continue
        fname = Path(rec.get("SourceFile", "")).name
        exif_map[fname] = {
            "lat":          float(lat),
            "lon":          float(lon),
            "abs_alt":      _f(rec.get("AbsoluteAltitude")),
            "focal_mm":     _f(rec.get("FocalLength")),
            "focal35_mm":   _f(rec.get("FocalLengthIn35mmFormat")),
            "img_w":        int(rec["ImageWidth"])  if "ImageWidth"  in rec else None,
            "img_h":        int(rec["ImageHeight"]) if "ImageHeight" in rec else None,
            "gimbal_pitch": _f(rec.get("GimbalPitchDegree")),
            "gimbal_yaw":   _f(rec.get("GimbalYawDegree")),
            "gimbal_roll":  _f(rec.get("GimbalRollDegree")),
        }
    return exif_map


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _rms(vals):
    return math.sqrt(sum(v**2 for v in vals) / len(vals)) if vals else float("nan")


def _px_err(px1, py1, px2, py2) -> float:
    return math.sqrt((px1 - px2)**2 + (py1 - py2)**2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Validate Mode A / Mode B pixel projection vs confirmed observations."
    )
    ap.add_argument("reconstruction", help="opensfm/reconstruction.json")
    ap.add_argument("gcp_confirmed",  help="gcp_confirmed.txt (GCPEditorPro export)")
    ap.add_argument("emlid_csv",      help="Emlid rover survey CSV")
    ap.add_argument("image_dir",      help="Directory containing raw drone images")
    ap.add_argument("--crs", default=None, help="CRS override (e.g. EPSG:6529)")
    args = ap.parse_args()

    # --- Load reconstruction ---
    with open(args.reconstruction) as f:
        recon = json.load(f)[0]
    shots   = recon["shots"]
    cameras = recon["cameras"]
    ref_lla = recon["reference_lla"]
    print(f"Reconstruction: {len(shots)} shots, {len(cameras)} camera(s)")

    # --- Parse confirmed observations ---
    crs_header, obs_by_label = parse_confirmed(args.gcp_confirmed)
    crs = args.crs or crs_header
    n_obs_total = sum(len(v) for v in obs_by_label.items())
    print(f"Confirmed observations: {sum(len(v) for v in obs_by_label.values())} "
          f"across {len(obs_by_label)} GCPs  (CRS: {crs})")

    # --- Parse survey CSV ---
    gcps = parse_survey_csv(args.emlid_csv, fallback_crs=crs)
    gcp_by_label = {g["label"]: g for g in gcps}
    print(f"Survey CSV: {len(gcps)} points\n")

    # --- Collect image paths referenced in confirmed file ---
    image_dir = Path(args.image_dir)
    all_images = {img for obs_list in obs_by_label.values() for img, _, _ in obs_list}
    image_paths = []
    missing_images = []
    for fname in sorted(all_images):
        p = image_dir / fname
        if p.exists():
            image_paths.append(p)
        else:
            missing_images.append(fname)

    if missing_images:
        print(f"WARNING: {len(missing_images)} images not found in {image_dir}", file=sys.stderr)

    # --- Run exiftool for Mode A ---
    print(f"Reading EXIF from {len(image_paths)} images for Mode A...")
    exif_map = read_exif_for_files(image_paths)
    mode_a_available = bool(exif_map)
    if mode_a_available:
        print(f"  EXIF loaded for {len(exif_map)} images")
    print()

    # --- Per-observation comparison ---
    # Rows: label, image, conf_px, conf_py, err_a, err_b
    results_a: Dict[str, List[float]] = {}   # label → [pixel errors]
    results_b: Dict[str, List[float]] = {}

    skipped_no_shot = 0
    skipped_no_gcp  = 0
    skipped_no_exif = 0
    skipped_b_none  = 0
    skipped_a_none  = 0

    for label, obs_list in sorted(obs_by_label.items()):
        # gcp_confirmed.txt labels may have a prefix (e.g. "GCP-104" or "CHK-104")
        # that csv2gcp adds; strip it to look up in the raw survey CSV.
        survey_label = label.split("-", 1)[1] if "-" in label else label
        gcp = gcp_by_label.get(survey_label) or gcp_by_label.get(label)
        if gcp is None:
            skipped_no_gcp += len(obs_list)
            continue

        for image, conf_px, conf_py in obs_list:
            # --- Mode B ---
            if image not in shots:
                skipped_no_shot += 1
            else:
                shot = shots[image]
                cam  = cameras[shot["camera"]]
                proj_b = project_pixel_mode_b(gcp, shot, cam, ref_lla)
                if proj_b is None:
                    skipped_b_none += 1
                else:
                    err_b = _px_err(proj_b[0], proj_b[1], conf_px, conf_py)
                    results_b.setdefault(label, []).append(err_b)

            # --- Mode A ---
            if mode_a_available:
                exif = exif_map.get(image)
                if exif is None:
                    skipped_no_exif += 1
                else:
                    proj_a = project_pixel_mode_a(exif, gcp)
                    if proj_a is None:
                        skipped_a_none += 1
                    else:
                        err_a = _px_err(proj_a[0], proj_a[1], conf_px, conf_py)
                        results_a.setdefault(label, []).append(err_a)

    # --- Report ---
    print(f"{'GCP':<14} {'N_B':>4} {'MeanB':>8} {'MaxB':>8} {'N_A':>4} {'MeanA':>8} {'MaxA':>8}")
    print("-" * 64)

    all_err_a: List[float] = []
    all_err_b: List[float] = []

    for label in sorted(set(list(results_a.keys()) + list(results_b.keys()))):
        errs_b = results_b.get(label, [])
        errs_a = results_a.get(label, [])
        mean_b = _mean(errs_b)
        max_b  = max(errs_b) if errs_b else float("nan")
        mean_a = _mean(errs_a)
        max_a  = max(errs_a) if errs_a else float("nan")

        print(f"{label:<14} {len(errs_b):>4} {mean_b:>8.1f} {max_b:>8.1f} "
              f"{len(errs_a):>4} {mean_a:>8.1f} {max_a:>8.1f}")

        all_err_b.extend(errs_b)
        all_err_a.extend(errs_a)

    print("-" * 64)
    print(f"{'OVERALL':<14} {len(all_err_b):>4} {_mean(all_err_b):>8.1f} "
          f"{(max(all_err_b) if all_err_b else float('nan')):>8.1f} "
          f"{len(all_err_a):>4} {_mean(all_err_a):>8.1f} "
          f"{(max(all_err_a) if all_err_a else float('nan')):>8.1f}")
    print(f"\n  RMS Mode B: {_rms(all_err_b):.1f} px   RMS Mode A: {_rms(all_err_a):.1f} px")

    if skipped_no_shot:
        print(f"\n  Skipped (image not in reconstruction): {skipped_no_shot}")
    if skipped_no_gcp:
        print(f"  Skipped (label not in survey CSV):     {skipped_no_gcp}")
    if skipped_b_none:
        print(f"  Skipped (Mode B returned None):        {skipped_b_none}")
    if skipped_no_exif:
        print(f"  Skipped Mode A (no EXIF):              {skipped_no_exif}")
    if skipped_a_none:
        print(f"  Skipped Mode A (returned None):        {skipped_a_none}")


if __name__ == "__main__":
    main()
