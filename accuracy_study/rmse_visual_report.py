#!/usr/bin/env python3
"""
rmse_visual_report.py — Annotated orthophoto crops for each GCP/CHK target.

Generates an HTML report with 4x-upscaled orthophoto crops for each target, showing:
  - Green crosshair: survey coordinate (ground truth)
  - Cyan circle: 0.5 ft radius (scale reference)
  - Yellow circle: 1 ft radius (scale reference)
  - Label + rmse.py residuals (dH, dZ, d3D)

Note: rmse.py's dH measures triangulation residual, not orthophoto positioning
error.  The ortho positions targets via DSM-based orthorectification which
introduces additional lateral offsets not captured by dH.

Sorted worst-first by dH so the most suspect targets are at the top.

Usage:
    conda run -n geo python accuracy_study/rmse_visual_report.py \\
        reconstruction.topocentric.json gcp_list.txt chk_list.txt \\
        targets.csv orthophoto.tif [-o report.html] [--crop-radius 3.0]

All coordinate inputs must be in the same CRS as the orthophoto (typically EPSG:32613).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal

# Suppress GDAL warnings about PAM
gdal.PushErrorHandler("CPLQuietErrorHandler")

RMSE_PY = Path(__file__).resolve().parent.parent / "rmse.py"
M_TO_FT = 3.28084
FT_TO_M = 1.0 / M_TO_FT


def run_rmse(reconstruction: str, gcp_list: str, chk_list: str) -> dict:
    """Run rmse.py and return the parsed JSON result."""
    cmd = [sys.executable, str(RMSE_PY), reconstruction, gcp_list, chk_list]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # rmse.py prints human-readable to stderr, JSON to stdout
    stdout = result.stdout.strip()
    if not stdout:
        print("rmse.py produced no JSON output", file=sys.stderr)
        print("stderr:", result.stderr, file=sys.stderr)
        sys.exit(1)
    return json.loads(stdout)


def read_targets(targets_csv: str) -> dict[str, tuple[float, float, float]]:
    """Read {job}_targets.csv → {label: (X, Y, Z)}."""
    targets = {}
    with open(targets_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets[row["label"]] = (
                float(row["X"]),
                float(row["Y"]),
                float(row["Z"]),
            )
    return targets


def crop_ortho(ds: gdal.Dataset, cx: float, cy: float,
               crop_radius_m: float) -> tuple[np.ndarray, float]:
    """Extract a square RGB crop from the orthophoto centered on (cx, cy).

    Returns (bgr_image, pixel_size_m).
    """
    gt = ds.GetGeoTransform()
    px_size = gt[1]  # metres per pixel (assumes square pixels, north-up)

    half_px = int(math.ceil(crop_radius_m / abs(px_size)))
    center_col = int((cx - gt[0]) / gt[1])
    center_row = int((cy - gt[3]) / gt[5])

    x_off = max(0, center_col - half_px)
    y_off = max(0, center_row - half_px)
    x_size = min(2 * half_px, ds.RasterXSize - x_off)
    y_size = min(2 * half_px, ds.RasterYSize - y_off)

    if x_size <= 0 or y_size <= 0:
        return None, px_size

    bands = min(ds.RasterCount, 3)
    img = np.zeros((y_size, x_size, 3), dtype=np.uint8)
    for b in range(bands):
        band_data = ds.GetRasterBand(b + 1).ReadAsArray(x_off, y_off, x_size, y_size)
        if band_data is not None:
            img[:, :, b] = band_data

    # GDAL reads as RGB; cv2 uses BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Actual pixel of the target center within the crop
    cx_px = center_col - x_off
    cy_px = center_row - y_off

    return img, px_size, cx_px, cy_px


def annotate_crop(img: np.ndarray, cx_px: int, cy_px: int,
                  px_size: float, label: str,
                  dh_m: float, dz_m: float, d3d_m: float,
                  group: str, upscale: int = 4) -> np.ndarray:
    """Upscale crop and draw survey coordinate + fixed scale circles.

    The green crosshair marks the survey coordinate.  Cyan and yellow
    circles show 0.5 ft and 1 ft radii for visual offset assessment.
    No rmse.py-derived annotations (dH circle, reconstructed dot) are
    drawn — the orthophoto positioning depends on DSM-based
    orthorectification, which rmse.py's triangulation does not model.
    """
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * upscale, h * upscale),
                     interpolation=cv2.INTER_NEAREST)
    cxs = cx_px * upscale + upscale // 2
    cys = cy_px * upscale + upscale // 2

    # --- Green crosshair: survey coordinate ---
    cross_len = 20
    cv2.line(out, (cxs - cross_len, cys), (cxs + cross_len, cys),
             (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(out, (cxs, cys - cross_len), (cxs, cys + cross_len),
             (0, 255, 0), 1, cv2.LINE_AA)

    # --- Scale circles: 0.5 ft (cyan), 1 ft (yellow) ---
    ft_to_scaled_px = lambda ft: int(round(ft * FT_TO_M / px_size * upscale))
    half_ft = ft_to_scaled_px(0.5)
    one_ft = ft_to_scaled_px(1.0)
    if half_ft > 2:
        cv2.circle(out, (cxs, cys), half_ft, (255, 255, 0), 1, cv2.LINE_AA)
    if one_ft > 2:
        cv2.circle(out, (cxs, cys), one_ft, (0, 255, 255), 1, cv2.LINE_AA)

    # --- Text overlay ---
    dh_ft = dh_m * M_TO_FT
    dz_ft = dz_m * M_TO_FT
    d3d_ft = d3d_m * M_TO_FT
    text_lines = [
        f"{label} ({group})",
        f"dH={dh_ft:+.3f} ft  dZ={dz_ft:+.3f} ft  d3D={d3d_ft:.3f} ft",
        "cyan=0.5ft  yellow=1ft",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    y0 = 18
    for i, line in enumerate(text_lines):
        y = y0 + i * 18
        cv2.putText(out, line, (6, y), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(out, line, (6, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def img_to_data_uri(img: np.ndarray) -> str:
    """Encode a BGR image as a PNG data URI."""
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_html(entries: list[dict]) -> str:
    """Build the HTML report from annotated entries."""
    rows = []
    for e in entries:
        dh_ft = e["dH"] * M_TO_FT
        dz_ft = e["dZ"] * M_TO_FT
        d3d_ft = e["d3D"] * M_TO_FT
        suspect = " ⚠" if e.get("suspect") else ""
        rows.append(f"""
        <div class="card">
            <div class="info">
                <b>{e['label']}</b> <span class="group">{e['group']}</span>{suspect}<br/>
                dH={dh_ft:+.4f} ft &nbsp; dZ={dz_ft:+.4f} ft &nbsp; d3D={d3d_ft:.4f} ft
            </div>
            <img src="{e['data_uri']}" />
        </div>""")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>RMSE Visual Report</title>
<style>
    body {{ font-family: monospace; background: #1a1a1a; color: #eee; margin: 20px; }}
    h1 {{ font-size: 18px; }}
    .legend {{ font-size: 13px; margin-bottom: 16px; color: #aaa; }}
    .legend span {{ padding: 2px 6px; border-radius: 3px; margin-right: 8px; }}
    .green {{ color: #0f0; }}
    .red {{ color: #f44; }}
    .yellow {{ color: #ff0; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .card {{ background: #2a2a2a; border-radius: 6px; overflow: hidden; }}
    .card img {{ display: block; }}
    .info {{ padding: 6px 10px; font-size: 12px; line-height: 1.5; }}
    .group {{ color: #888; font-size: 11px; }}
</style>
</head>
<body>
<h1>RMSE Visual Report — sorted by dH (worst first)</h1>
<div class="legend">
    <span class="green">+ survey coordinate</span>
    <span style="color:#0ff">○ 0.5 ft radius</span>
    <span class="yellow">○ 1 ft radius</span>
    &nbsp; | &nbsp; dH = rmse.py triangulation residual (not orthophoto positioning error)
</div>
<div class="grid">
{''.join(rows)}
</div>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reconstruction", help="reconstruction.topocentric.json")
    ap.add_argument("gcp_list", help="gcp_list.txt (GCP control file)")
    ap.add_argument("chk_list", help="chk_list.txt (CHK accuracy file)")
    ap.add_argument("targets_csv", help="{job}_targets.csv (label,X,Y,Z)")
    ap.add_argument("orthophoto", help="Orthophoto GeoTIFF (same CRS as targets)")
    ap.add_argument("-o", "--output", default="rmse_report.html",
                    help="Output HTML file (default: rmse_report.html)")
    ap.add_argument("--crop-radius", type=float, default=5.0,
                    help="Crop radius in metres (default: 5.0)")
    ap.add_argument("--upscale", type=int, default=4,
                    help="Nearest-neighbor upscale factor (default: 4)")
    args = ap.parse_args()

    # 1. Run rmse.py
    print("Running rmse.py...", file=sys.stderr)
    rmse_data = run_rmse(args.reconstruction, args.gcp_list, args.chk_list)

    # 2. Read targets
    targets = read_targets(args.targets_csv)

    # 3. Collect all points with residuals
    all_points = []
    for group_key in ("gcp", "chk"):
        group = rmse_data.get(group_key, {})
        for p in group.get("points", []):
            label = p["label"]
            if label in targets:
                all_points.append({**p, "group": group_key.upper(),
                                   "survey": targets[label]})

    if not all_points:
        print("No matching targets found between rmse output and targets CSV",
              file=sys.stderr)
        sys.exit(1)

    # Sort by dH descending (worst first)
    all_points.sort(key=lambda p: p["dH"], reverse=True)

    # Flag suspects (same 5× median logic as rmse.py)
    dh_vals = sorted(p["dH"] for p in all_points)
    mid = len(dh_vals) // 2
    median_dh = (dh_vals[mid] if len(dh_vals) % 2 else
                 (dh_vals[mid - 1] + dh_vals[mid]) / 2)
    suspect_thresh = max(5.0 * median_dh, 0.5 * 0.3048)  # 0.5 ft floor
    for p in all_points:
        p["suspect"] = p["dH"] > suspect_thresh

    # 4. Open orthophoto
    print(f"Opening {args.orthophoto}...", file=sys.stderr)
    ds = gdal.Open(args.orthophoto, gdal.GA_ReadOnly)
    if ds is None:
        print(f"Cannot open {args.orthophoto}", file=sys.stderr)
        sys.exit(1)

    gt = ds.GetGeoTransform()
    px_size = abs(gt[1])

    # 5. Generate annotated crops
    entries = []
    for i, p in enumerate(all_points):
        label = p["label"]
        sx, sy, sz = p["survey"]
        print(f"  [{i+1}/{len(all_points)}] {label}  dH={p['dH']*M_TO_FT:.4f} ft",
              file=sys.stderr)

        result = crop_ortho(ds, sx, sy, args.crop_radius)
        if result is None or result[0] is None:
            print(f"    skipped (outside orthophoto extent)", file=sys.stderr)
            continue
        img, px_sz, cx_px, cy_px = result

        annotated = annotate_crop(
            img, cx_px, cy_px,
            px_size=px_sz, label=label,
            dh_m=p["dH"], dz_m=p["dZ"], d3d_m=p["d3D"],
            group=p["group"], upscale=args.upscale,
        )

        entries.append({
            "label": label,
            "group": p["group"],
            "dH": p["dH"],
            "dZ": p["dZ"],
            "d3D": p["d3D"],
            "suspect": p.get("suspect", False),
            "data_uri": img_to_data_uri(annotated),
        })

    ds = None  # close

    # 6. Write HTML
    html = build_html(entries)
    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nWrote {out_path} ({len(entries)} targets)", file=sys.stderr)


if __name__ == "__main__":
    main()
