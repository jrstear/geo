#!/usr/bin/env python3
"""
rmse.py — 3D RMSE from ODM reconstruction + GCP/CHK point files.

Usage:
    conda run -n geo python rmse.py \\
        reconstruction.topocentric.json gcp_list.txt chk_list.txt

    # Synthetic self-test:
    conda run -n geo python rmse.py --test

REQUIRED: reconstruction.topocentric.json, NOT reconstruction.json.

ODM's GCP bundle adjustment refines per-shot camera orientations and stores
the result in reconstruction.topocentric.json.  reconstruction.json holds the
pre-GCP-alignment poses; similarity_transform.json maps from that early frame
to GPS-ENU (not GCP-ENU).  Using reconstruction.json + similarity_transform.json
produces ~20m+ errors because:
  1. The per-shot rotations in reconstruction.json are GPS-aligned (~2.7° off from
     the GCP-refined orientations stored in reconstruction.topocentric.json).
  2. A global similarity transform cannot correct per-shot orientation errors.

A Umeyama similarity transform is auto-fitted from the GCP triangulation
results, correcting the ~25m GPS translation offset and ~1.75° UTM grid
convergence between the reconstruction's topocentric ENU frame and the
surveyed CRS.  Without this correction horizontal errors are ~58m RMS.
Both GCP residuals and CHK residuals are reported after applying this
correction, so gcp_list.txt is always required.

gcp_list.txt — GCP-* rows only (transform.py split output); used to fit
               the georeferencing similarity and to report control residuals.
chk_list.txt — CHK-* rows only (transform.py split output); assessed
               independently (withheld from ODM BA) for accuracy QC.

For each point group:
  1. Gathers tagged (image, px, py) observations.
  2. Builds viewing rays through SfM-optimised camera poses (Rodrigues rotation).
  3. Triangulates 3D position via linear DLT (least-squares on projection-error system).
  4. Converts ENU reconstruction coords → projected world CRS (ref_UTM + local).
  5. Auto-fits similarity from GCP triangulations → surveyed GCP positions.
  6. Computes dX/dY/dZ vs surveyed positions; reports RMS_H and RMS_Z in m and ft.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    raise RuntimeError("scipy is required: conda install scipy")

try:
    from pyproj import CRS as ProjCRS, Transformer
except ImportError:
    raise RuntimeError("pyproj is required: conda install pyproj")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FT_TO_M = 0.3048006096012192   # US survey foot
M_TO_FT = 1.0 / FT_TO_M


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def enu_to_projected(
    p_enu: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
    epsg: str,
) -> Tuple[float, float, float]:
    """
    Convert a point in ODM's topocentric ENU frame to projected CRS coordinates.

    Uses the proper geodetic conversion chain: ENU → ECEF → lat/lon → projected,
    matching OpenSFM's TopocentricConverter.to_lla() + pyproj projection.

    The topocentric frame has X=east, Y=north, Z=up at the reference point.
    This is NOT aligned with the projected (e.g. UTM) grid — the UTM grid
    convergence angle and scale factor vary with position.  A simple
    ref_UTM + offset approach gives ~20 m/km error at sites far from the
    UTM central meridian.

    Returns (x_proj, y_proj, z_ellip_m).
    """
    lat, lon, alt = _enu_to_lla(
        p_enu[0], p_enu[1], p_enu[2],
        ref_lat_deg, ref_lon_deg, ref_alt_m,
    )
    xfm = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x, y = xfm.transform(lon, lat)
    return x, y, alt


# WGS84 ellipsoid (matching OpenSFM opensfm/geo.py)
_WGS84_a = 6378137.0
_WGS84_b = 6356752.314245


def _ecef_from_lla(lat_deg: float, lon_deg: float, alt: float) -> Tuple[float, float, float]:
    """WGS84 geodetic to ECEF (matches OpenSFM geo.ecef_from_lla)."""
    a2 = _WGS84_a ** 2
    b2 = _WGS84_b ** 2
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    L = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    return (
        (a2 * L + alt) * np.cos(lat) * np.cos(lon),
        (a2 * L + alt) * np.cos(lat) * np.sin(lon),
        (b2 * L + alt) * np.sin(lat),
    )


def _enu_to_lla(
    x: float, y: float, z: float,
    ref_lat: float, ref_lon: float, ref_alt: float,
) -> Tuple[float, float, float]:
    """
    Convert topocentric ENU to geodetic lat/lon/alt.

    Matches OpenSFM geo.lla_from_topocentric: builds the 4×4
    ecef_from_topocentric_transform and applies it, then converts ECEF → LLA.
    """
    rx, ry, rz = _ecef_from_lla(ref_lat, ref_lon, ref_alt)
    sa = np.sin(np.radians(ref_lat))
    ca = np.cos(np.radians(ref_lat))
    so = np.sin(np.radians(ref_lon))
    co = np.cos(np.radians(ref_lon))

    # ENU → ECEF rotation (OpenSFM ecef_from_topocentric_transform columns 0-2)
    ex = -so * x + (-sa * co) * y + (ca * co) * z + rx
    ey =  co * x + (-sa * so) * y + (ca * so) * z + ry
    ez =             ca       * y +  sa       * z + rz

    # ECEF → LLA (OpenSFM geo.lla_from_ecef)
    a = _WGS84_a
    b = _WGS84_b
    ea2 = (a ** 2 - b ** 2) / a ** 2
    eb2 = (a ** 2 - b ** 2) / b ** 2
    p = np.sqrt(ex ** 2 + ey ** 2)
    theta = np.arctan2(ez * a, p * b)
    lon = np.arctan2(ey, ex)
    lat = np.arctan2(
        ez + eb2 * b * np.sin(theta) ** 3,
        p - ea2 * a * np.cos(theta) ** 3,
    )
    N = a / np.sqrt(1 - ea2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def crs_axis_unit(epsg: str) -> str:
    """Return the unit name for axis 0 of the projected CRS (e.g. 'US survey foot')."""
    try:
        crs_obj = ProjCRS.from_user_input(epsg)
        return crs_obj.axis_info[0].unit_name
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# reconstruction.json parsing
# ---------------------------------------------------------------------------


def parse_reconstruction(recon_path: str) -> dict:
    """Load reconstruction.json and return the first reconstruction element."""
    with open(recon_path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"reconstruction.json must be a non-empty JSON array: {recon_path}")
    return data[0]


# ---------------------------------------------------------------------------
# ODM point file parsing
# ---------------------------------------------------------------------------


def parse_points(
    path: str,
) -> Tuple[str,
           Dict[str, Tuple[float, float, float]],
           Dict[str, List[Tuple[str, float, float]]]]:
    """
    Parse an ODM-format point file (tab-separated, CRS header on line 1).

    Works for gcp_list.txt (GCP-* labels) or chk_list.txt (CHK-* labels);
    no label-prefix filtering is applied.

    Returns:
        crs_header : str
        coords     : dict — label → (geo_x, geo_y, geo_z)
        obs        : dict — label → [(image_name, px, py), ...]
    """
    with open(path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"file is empty: {path}")

    crs_header = lines[0].rstrip("\n").strip()
    coords: Dict[str, Tuple[float, float, float]] = {}
    obs: Dict[str, List[Tuple[str, float, float]]] = {}

    for raw_line in lines[1:]:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) < 7:
            continue
        try:
            geo_x = float(fields[0])
            geo_y = float(fields[1])
            geo_z = float(fields[2])
            px    = float(fields[3])
            py    = float(fields[4])
        except ValueError:
            continue
        image_name = fields[5]
        label      = fields[6]

        if label not in coords:
            coords[label] = (geo_x, geo_y, geo_z)
        obs.setdefault(label, []).append((image_name, px, py))

    return crs_header, coords, obs


# ---------------------------------------------------------------------------
# Ray building + triangulation
# ---------------------------------------------------------------------------


def build_rays(
    observations: List[Tuple[str, float, float]],
    shots: dict,
    cameras: dict,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert pixel observations to (camera_centre, unit_direction) rays in ENU.

    observations : [(image_name, px, py), ...]
    Returns list of (C, d) pairs where C and d are 3-element numpy arrays.
    Missing shot names are silently skipped (caller logs the warning).
    """
    rays: List[Tuple[np.ndarray, np.ndarray]] = []

    for image_name, px, py in observations:
        if image_name not in shots:
            continue

        shot = shots[image_name]
        cam = cameras[shot["camera"]]

        R = Rotation.from_rotvec(shot["rotation"]).as_matrix()  # world → camera
        t = np.array(shot["translation"], dtype=float)
        C = -R.T @ t  # camera centre in world (ENU)

        w, h = cam["width"], cam["height"]
        mwh = max(w, h)

        # Support both OpenSfM camera model conventions:
        #   perspective: 'focal'           (one value, normalised by max(w,h))
        #   brown:       'focal_x'/'focal_y' + principal point 'c_x'/'c_y'
        if "focal" in cam:
            fx = fy = cam["focal"] * mwh
            cx_off = cy_off = 0.0
            k1 = cam.get("k1", 0.0)
            k2 = cam.get("k2", 0.0)
            k3 = p1 = p2 = 0.0
        else:
            fx = cam["focal_x"] * mwh
            fy = cam["focal_y"] * mwh
            cx_off = cam.get("c_x", 0.0) * mwh
            cy_off = cam.get("c_y", 0.0) * mwh
            k1 = cam.get("k1", 0.0)
            k2 = cam.get("k2", 0.0)
            k3 = cam.get("k3", 0.0)
            p1 = cam.get("p1", 0.0)
            p2 = cam.get("p2", 0.0)

        cx, cy = w / 2.0, h / 2.0

        # Unproject pixel to normalised image coordinates (undo principal point)
        xn = (px - cx - cx_off) / fx
        yn = (py - cy - cy_off) / fy

        # Inverse distortion: 3 fixed-point iterations (handles k1,k2,k3,p1,p2)
        xnd, ynd = xn, yn
        for _ in range(3):
            r2 = xnd ** 2 + ynd ** 2
            radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            tang_x = 2 * p1 * xnd * ynd + p2 * (r2 + 2 * xnd ** 2)
            tang_y = p1 * (r2 + 2 * ynd ** 2) + 2 * p2 * xnd * ynd
            xnd = (xn - tang_x) / radial
            ynd = (yn - tang_y) / radial

        # Ray direction in camera frame → world frame
        d_cam = np.array([xnd, ynd, 1.0])
        d_world = R.T @ d_cam
        d_world /= np.linalg.norm(d_world)

        rays.append((C, d_world))

    return rays


def triangulate_dlt(rays: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    """
    Linear DLT triangulation from N >= 2 rays.

    Uses the (I - d d^T)(X - C) = 0 formulation, which gives a well-conditioned
    linear system: (I - d d^T) X = (I - d d^T) C, solved via least squares.

    Returns the 3D point X in ENU, or None if the system is rank-deficient.
    """
    A_rows = []
    b_rows = []
    for C_i, d_i in rays:
        M = np.eye(3) - np.outer(d_i, d_i)  # projects out the ray direction
        A_rows.append(M)
        b_rows.append(M @ C_i)

    A = np.vstack(A_rows)   # (3N, 3)
    b = np.concatenate(b_rows)

    result, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return None
    return result


# ---------------------------------------------------------------------------
# Similarity transform (Umeyama algorithm)
# ---------------------------------------------------------------------------


def fit_similarity(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Fit a 7-parameter similarity transform: dst = scale * R @ src + t.

    Uses the Umeyama algorithm (closed-form, handles reflection).

    Parameters:
        src : (N, 3) array of source points (triangulated GCP positions)
        dst : (N, 3) array of target points (surveyed GCP positions)

    Returns:
        scale : float
        R     : (3, 3) rotation matrix
        t     : (3,)   translation vector
    """
    n = len(src)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    Sc = src - mu_s
    Dc = dst - mu_d

    H = Sc.T @ Dc / n
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, np.sign(d)])

    R = Vt.T @ D @ U.T
    sigma = np.sum(Sc ** 2) / n
    scale = np.trace(D @ np.diag(S)) / sigma
    t = mu_d - scale * R @ mu_s

    return scale, R, t


# ---------------------------------------------------------------------------
# Triangulation helper
# ---------------------------------------------------------------------------


def _triangulate_labels(
    obs_by_label: Dict[str, List[Tuple[str, float, float]]],
    coords_by_label: Dict[str, Tuple[float, float, float]],
    shots: dict,
    cameras: dict,
    ref_lla: dict,
    crs: str,
    uses_feet: bool,
) -> Tuple[List[dict], List[str]]:
    """
    Triangulate all labels and return (results, skipped_labels).

    Each result dict has: label, tri_m (3-array in metres), survey_m (3-array in metres).
    """
    results = []
    skipped: List[str] = []

    for label, obs in sorted(obs_by_label.items()):
        missing = [img for img, _, _ in obs if img not in shots]
        for img in missing:
            print(f"WARNING: shot {img!r} for {label!r} not in reconstruction — skipped",
                  file=sys.stderr)

        rays = build_rays(obs, shots, cameras)
        if len(rays) < 2:
            print(f"WARNING: {label!r} has {len(rays)} rays — skipped", file=sys.stderr)
            skipped.append(label)
            continue

        X_enu = triangulate_dlt(rays)
        if X_enu is None:
            print(f"WARNING: {label!r} rank-deficient — skipped", file=sys.stderr)
            skipped.append(label)
            continue

        x_p, y_p, z_m = enu_to_projected(
            X_enu, ref_lla["latitude"], ref_lla["longitude"], ref_lla["altitude"], crs
        )

        coords = coords_by_label.get(label)
        if coords is None:
            print(f"WARNING: {label!r} not in coords — skipped", file=sys.stderr)
            skipped.append(label)
            continue

        sx, sy, sz = coords

        # Convert to metres for comparison
        tri_m = np.array([
            x_p * FT_TO_M if uses_feet else x_p,
            y_p * FT_TO_M if uses_feet else y_p,
            z_m,
        ])
        survey_m = np.array([
            sx * FT_TO_M if uses_feet else sx,
            sy * FT_TO_M if uses_feet else sy,
            sz,
        ])

        results.append({"label": label, "tri_m": tri_m, "survey_m": survey_m})

    return results, skipped


# ---------------------------------------------------------------------------
# Residuals helper
# ---------------------------------------------------------------------------


def _compute_residuals(
    results: List[dict],
    sim_scale: Optional[float],
    sim_R: Optional[np.ndarray],
    sim_t: Optional[np.ndarray],
    apply_sim: bool,
) -> List[dict]:
    points = []
    for r in results:
        tri = r["tri_m"].copy()
        if apply_sim and sim_R is not None:
            tri = sim_scale * sim_R @ tri + sim_t
        surv = r["survey_m"]

        dX = tri[0] - surv[0]
        dY = tri[1] - surv[1]
        dZ = tri[2] - surv[2]
        dH = math.sqrt(dX ** 2 + dY ** 2)
        d3D = math.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)

        points.append({
            "label": r["label"],
            "dX": round(dX, 6),
            "dY": round(dY, 6),
            "dZ": round(dZ, 6),
            "dH": round(dH, 6),
            "d3D": round(d3D, 6),
            "survey_x": round(surv[0], 6),
            "survey_y": round(surv[1], 6),
            "survey_z": round(surv[2], 6),
        })
    return points


def _group_stats(points: List[dict]) -> dict:
    n = len(points)
    if n == 0:
        return {"n": 0, "points": [], "rms_x": None, "rms_y": None,
                "rms_z": None, "rms_h": None, "rms_3d": None,
                "mean_dz": None, "std_dz": None}
    rms_x  = math.sqrt(sum(p["dX"] ** 2 for p in points) / n)
    rms_y  = math.sqrt(sum(p["dY"] ** 2 for p in points) / n)
    rms_z  = math.sqrt(sum(p["dZ"] ** 2 for p in points) / n)
    rms_h  = math.sqrt(sum(p["dH"] ** 2 for p in points) / n)
    rms_3d = math.sqrt(sum(p["d3D"] ** 2 for p in points) / n)
    mean_dz = sum(p["dZ"] for p in points) / n
    std_dz = math.sqrt(sum((p["dZ"] - mean_dz) ** 2 for p in points) / n) if n > 1 else 0.0
    return {
        "n": n,
        "points": points,
        "rms_x":   round(rms_x,   6),
        "rms_y":   round(rms_y,   6),
        "rms_z":   round(rms_z,   6),
        "rms_h":   round(rms_h,   6),
        "rms_3d":  round(rms_3d,  6),
        "mean_dz": round(mean_dz, 6),
        "std_dz":  round(std_dz,  6),
    }


# ---------------------------------------------------------------------------
# Main RMSE computation
# ---------------------------------------------------------------------------


def compute_rmse(
    recon_path: str,
    gcp_list_path: str,
    chk_list_path: str,
    crs_override: Optional[str] = None,
) -> dict:
    """
    Full pipeline: parse inputs, triangulate both GCP and CHK groups,
    convert to survey CRS via proper geodetic conversion, compute RMSE.

    recon_path    : path to reconstruction.topocentric.json (NOT reconstruction.json)
    gcp_list_path : GCP-* rows only (from transform.py split); control residuals.
    chk_list_path : CHK-* rows only (from transform.py split); independent
                    accuracy assessment (withheld from ODM BA).

    Coordinate conversion uses the proper geodetic chain (ENU → ECEF → lat/lon
    → projected CRS) matching OpenSFM's TopocentricConverter.  No similarity
    transform is fitted — the topocentric frame from reconstruction.topocentric.json
    already incorporates ODM's GCP-constrained bundle adjustment.

    Returns a dict with 'gcp' and 'chk' sub-dicts, each with rms_h, rms_z, etc.
    """
    # --- Step 1: parse reconstruction ---
    recon = parse_reconstruction(recon_path)
    cameras = recon["cameras"]
    shots = recon["shots"]
    ref_lla = recon["reference_lla"]

    # --- Step 2: parse gcp_list.txt and chk_list.txt ---
    crs_header, gcp_coords, gcp_obs = parse_points(gcp_list_path)
    _, chk_coords, chk_obs = parse_points(chk_list_path)
    crs = crs_override or crs_header
    if not crs:
        raise ValueError("CRS not found in gcp_list header and --crs not provided.")

    unit = crs_axis_unit(crs)
    uses_feet = "foot" in unit.lower()

    # --- Step 3: triangulate GCP points ---
    gcp_results, gcp_skipped = _triangulate_labels(
        gcp_obs, gcp_coords, shots, cameras, ref_lla, crs, uses_feet,
    )

    print(f"\nGeodetic conversion: ENU → ECEF → lat/lon → {crs}", file=sys.stderr)
    print(f"  reference: lat={ref_lla['latitude']:.10f}  lon={ref_lla['longitude']:.10f}  "
          f"alt={ref_lla['altitude']}", file=sys.stderr)

    # --- Step 4: triangulate CHK points ---
    chk_results, chk_skipped = _triangulate_labels(
        chk_obs, chk_coords, shots, cameras, ref_lla, crs, uses_feet,
    )

    # --- Step 5: compute residuals (no similarity transform) ---
    gcp_points = _compute_residuals(gcp_results, None, None, None, apply_sim=False)
    chk_points = _compute_residuals(chk_results, None, None, None, apply_sim=False)

    return {
        "gcp": _group_stats(gcp_points),
        "chk": _group_stats(chk_points),
        "geotransform_source": "geodetic (ENU→ECEF→LLA→proj)",
    }


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def _fmt_group(label: str, g: dict) -> None:
    n = g["n"]
    print(f"\n{label} — N={n}", file=sys.stderr)
    if n == 0:
        print("  (no valid points computed)", file=sys.stderr)
        return

    def _row(name: str, val_m: float) -> None:
        print(f"  {name:<10} = {val_m:8.4f} m   {val_m * M_TO_FT:8.4f} ft", file=sys.stderr)

    _row("RMS_H",    g["rms_h"])
    _row("RMS_Z",    g["rms_z"])
    _row("RMS_3D",   g["rms_3d"])
    _row("mean_dZ",  g["mean_dz"])
    _row("std_dZ",   g["std_dz"])
    print("\n  Per-point:", file=sys.stderr)
    for p in g["points"]:
        print(
            f"    {p['label']:<12}  dH={p['dH'] * M_TO_FT:+.4f} ft  "
            f"dZ={p['dZ'] * M_TO_FT:+.4f} ft  d3D={p['d3D'] * M_TO_FT:.4f} ft",
            file=sys.stderr,
        )

    # Outlier detection: flag points whose dH is suspiciously large relative
    # to the group median.  A 5× multiplier catches gross mis-tags (e.g. tagging
    # the base-station pole instead of the target) while ignoring normal spread.
    # The 0.5 ft absolute floor prevents false positives in very accurate sets.
    dh_vals = sorted(p["dH"] for p in g["points"])
    mid = len(dh_vals) // 2
    median_dh = (dh_vals[mid] if len(dh_vals) % 2 else
                 (dh_vals[mid - 1] + dh_vals[mid]) / 2)
    OUTLIER_RATIO = 5.0
    OUTLIER_FLOOR_M = 0.5 * FT_TO_M  # 0.5 ft minimum to trigger
    suspects = [
        p for p in g["points"]
        if p["dH"] > max(OUTLIER_RATIO * median_dh, OUTLIER_FLOOR_M)
    ]
    if suspects:
        print("\n  ⚠  SUSPECT TAGGING — verify pixel placement for:", file=sys.stderr)
        for p in suspects:
            print(
                f"    {p['label']:<12}  dH={p['dH'] * M_TO_FT:.2f} ft"
                f"  ({p['dH'] / median_dh:.0f}× median)",
                file=sys.stderr,
            )


def print_summary(result: dict) -> None:
    """Print human-readable summary to stderr."""
    src = result.get("geotransform_source", "none")
    print(f"\nRMSE report  (geotransform: {src})", file=sys.stderr)
    _fmt_group("GCP residuals (control fit)", result["gcp"])
    _fmt_group("CHK accuracy  (independent)", result["chk"])
    print("", file=sys.stderr)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def _read_targets(targets_csv: str) -> Dict[str, Tuple[float, float, float]]:
    """Read {job}_targets.csv → {label: (X, Y, Z)}."""
    import csv
    targets = {}
    with open(targets_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets[row["label"]] = (float(row["X"]), float(row["Y"]), float(row["Z"]))
    return targets


def _crop_ortho(ds, cx: float, cy: float, crop_radius_m: float = 5.0):
    """Extract a square RGB crop from the orthophoto centered on (cx, cy)."""
    gt = ds.GetGeoTransform()
    px_size = gt[1]
    half_px = int(math.ceil(crop_radius_m / abs(px_size)))
    center_col = int((cx - gt[0]) / gt[1])
    center_row = int((cy - gt[3]) / gt[5])
    x_off = max(0, center_col - half_px)
    y_off = max(0, center_row - half_px)
    x_size = min(2 * half_px, ds.RasterXSize - x_off)
    y_size = min(2 * half_px, ds.RasterYSize - y_off)
    if x_size <= 0 or y_size <= 0:
        return None
    bands = min(ds.RasterCount, 3)
    img = np.zeros((y_size, x_size, 3), dtype=np.uint8)
    for b in range(bands):
        band_data = ds.GetRasterBand(b + 1).ReadAsArray(x_off, y_off, x_size, y_size)
        if band_data is not None:
            img[:, :, b] = band_data
    try:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except ImportError:
        img = img[:, :, ::-1]  # RGB→BGR without cv2
    cx_px = center_col - x_off
    cy_px = center_row - y_off
    return img, px_size, cx_px, cy_px


def _annotate_crop(img: np.ndarray, cx_px: int, cy_px: int,
                   dx_m: float, dy_m: float,
                   px_size: float, label: str,
                   dh_m: float, dz_m: float, d3d_m: float,
                   group: str, upscale: int = 4) -> np.ndarray:
    """Upscale crop and annotate with survey + projected markers."""
    import cv2
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
    cxs = cx_px * upscale + upscale // 2
    cys = cy_px * upscale + upscale // 2

    # Green X: survey coordinate
    xlen = 14
    cv2.line(out, (cxs - xlen, cys - xlen), (cxs + xlen, cys + xlen), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(out, (cxs - xlen, cys + xlen), (cxs + xlen, cys - xlen), (0, 255, 0), 2, cv2.LINE_AA)

    # Yellow +: projected/triangulated position
    proj_x = int(cxs + dx_m / px_size * upscale)
    proj_y = int(cys - dy_m / px_size * upscale)
    cl = 14
    cv2.line(out, (proj_x - cl, proj_y), (proj_x + cl, proj_y), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(out, (proj_x, proj_y - cl), (proj_x, proj_y + cl), (0, 255, 255), 2, cv2.LINE_AA)

    # Yellow circle: 1 ft radius centered on projected
    one_ft_px = int(round(FT_TO_M / px_size * upscale))
    if one_ft_px > 2:
        cv2.circle(out, (proj_x, proj_y), one_ft_px, (0, 255, 255), 1, cv2.LINE_AA)

    # Text overlay
    dh_ft = dh_m * M_TO_FT
    dz_ft = dz_m * M_TO_FT
    d3d_ft = d3d_m * M_TO_FT
    for i, line in enumerate([f"{label} ({group})",
                              f"dH={dh_ft:+.3f} ft  dZ={dz_ft:+.3f} ft  d3D={d3d_ft:.3f} ft"]):
        y = 18 + i * 18
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _img_to_data_uri(img: np.ndarray) -> str:
    """Encode a BGR image as a PNG data URI."""
    import cv2, base64
    _, buf = cv2.imencode(".png", img)
    return f"data:image/png;base64,{base64.b64encode(buf).decode('ascii')}"


def generate_html_report(
    result: dict,
    output_path: str,
    ortho_path: Optional[str] = None,
    crop_radius: float = 5.0,
    upscale: int = 4,
    suspect_ratio: float = 5.0,
    suspect_floor_ft: float = 0.5,
) -> None:
    """Generate HTML accuracy report with optional ortho crop images."""

    # Collect all points sorted worst-first
    all_points = []
    for group_key in ("gcp", "chk"):
        group = result.get(group_key, {})
        for p in group.get("points", []):
            all_points.append({**p, "group": group_key.upper()})
    all_points.sort(key=lambda p: p["dH"], reverse=True)

    # Suspect flagging
    suspect_floor_m = suspect_floor_ft * FT_TO_M
    median_dh = 0.0
    suspect_thresh = suspect_floor_m
    if all_points:
        dh_vals = sorted(p["dH"] for p in all_points)
        mid = len(dh_vals) // 2
        median_dh = dh_vals[mid] if len(dh_vals) % 2 else (dh_vals[mid - 1] + dh_vals[mid]) / 2
        suspect_thresh = max(suspect_ratio * median_dh, suspect_floor_m)
        for p in all_points:
            p["suspect"] = p["dH"] > suspect_thresh

    # --- Combined summary table (GCP + CHK side by side) ---
    gcp = result["gcp"]
    chk = result["chk"]
    summary_rows = ""
    for name, key in [("RMS_H", "rms_h"), ("RMS_Z", "rms_z"), ("RMS_3D", "rms_3d"),
                      ("mean_dZ", "mean_dz"), ("std_dZ", "std_dz")]:
        gv = gcp.get(key)
        cv = chk.get(key)
        gm = f"{gv:.4f}" if gv is not None else "—"
        gf = f"{gv * M_TO_FT:.4f}" if gv is not None else "—"
        cm = f"{cv:.4f}" if cv is not None else "—"
        cf = f"{cv * M_TO_FT:.4f}" if cv is not None else "—"
        summary_rows += f"<tr><td>{name}</td><td>{gm}</td><td>{gf}</td><td>{cm}</td><td>{cf}</td></tr>\n"

    summary_table = f"""<h3>Summary — GCP (N={gcp['n']}) &amp; CHK (N={chk['n']})</h3>
<table class="summary">
<tr><th></th><th colspan="2">GCP (control fit)</th><th colspan="2">CHK (independent)</th></tr>
<tr><th>Metric</th><th>m</th><th>ft</th><th>m</th><th>ft</th></tr>
{summary_rows}</table>"""

    # --- Overview map: low-res ortho with point locations ---
    has_ortho = ortho_path is not None
    overview_html = ""
    if has_ortho:
        try:
            from osgeo import gdal
            gdal.UseExceptions()
        except ImportError:
            print("WARNING: GDAL not available — skipping ortho crops", file=sys.stderr)
            has_ortho = False

    if has_ortho:
        try:
            import cv2
            ds = gdal.Open(ortho_path, gdal.GA_ReadOnly)
            if ds is not None:
                gt = ds.GetGeoTransform()
                ow, oh = ds.RasterXSize, ds.RasterYSize
                # Read a low-res overview (~800px wide) using GDAL's
                # buf_xsize/buf_ysize for efficient subsampled read.
                # For COGs with overviews this is fast; for raw TIFFs
                # GDAL must read the full raster — accept the cost.
                scale = max(1, ow // 800)
                sw, sh = ow // scale, oh // scale
                t_ov = time.time()
                overview = np.zeros((sh, sw, 3), dtype=np.uint8)
                for b in range(min(3, ds.RasterCount)):
                    overview[:, :, b] = ds.GetRasterBand(b + 1).ReadAsArray(
                        0, 0, ow, oh, buf_xsize=sw, buf_ysize=sh)
                overview = cv2.cvtColor(overview, cv2.COLOR_RGB2BGR)
                print(f"  Overview map ({sw}x{sh}) in {time.time()-t_ov:.1f}s"
                      f" (use --ortho with a COG for faster overview)",
                      file=sys.stderr)

                # Draw each point as a colored dot + label with collision avoidance
                placed_labels = []  # [(x, y, w, h), ...] for collision detection
                font_scale = 0.28
                font = cv2.FONT_HERSHEY_SIMPLEX

                for p in all_points:
                    label = p["label"]
                    sx, sy = p["survey_x"], p["survey_y"]
                    px = int((sx - gt[0]) / gt[1] / scale)
                    py = int((sy - gt[3]) / gt[5] / scale)
                    if not (0 <= px < sw and 0 <= py < sh):
                        continue

                    # Color: green for GCP, yellow for CHK, red for suspect
                    if p.get("suspect"):
                        color = (0, 0, 255)
                    elif p["group"] == "GCP":
                        color = (0, 255, 0)
                    else:
                        color = (0, 255, 255)
                    cv2.circle(overview, (px, py), 4, color, -1, cv2.LINE_AA)
                    cv2.circle(overview, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)

                    # Label placement with collision + boundary avoidance
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
                    lx, ly = px + 6, py + 3
                    # Try offsets: right, left, below, above
                    for dx, dy in [(6, 3), (-tw - 6, 3), (6, th + 8), (6, -th - 2),
                                   (-tw - 6, th + 8), (-tw - 6, -th - 2)]:
                        lx, ly = px + dx, py + dy
                        # Check image bounds
                        if lx < 0 or lx + tw > sw or ly - th < 0 or ly > sh:
                            continue
                        collision = False
                        for ox, oy, ow, oh in placed_labels:
                            if (lx < ox + ow and lx + tw > ox and
                                    ly - th < oy and ly > oy - oh):
                                collision = True
                                break
                        if not collision:
                            break
                    # Final clamp to image bounds
                    lx = max(0, min(lx, sw - tw))
                    ly = max(th, min(ly, sh))

                    placed_labels.append((lx, ly, tw, th))
                    cv2.putText(overview, label, (lx, ly), font, font_scale,
                                (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(overview, label, (lx, ly), font, font_scale,
                                color, 1, cv2.LINE_AA)
                ds = None  # will reopen for crops
                overview_html = (f'<img src="{_img_to_data_uri(overview)}" '
                                 f'style="max-height:600px; border:1px solid #444; border-radius:4px;" />')
        except Exception as e:
            print(f"WARNING: overview map failed: {e}", file=sys.stderr)

    # --- Per-point detail table (with overview map to the right) ---
    detail_rows = ""
    for p in all_points:
        dh_ft = p["dH"] * M_TO_FT
        dz_ft = p["dZ"] * M_TO_FT
        d3d_ft = p["d3D"] * M_TO_FT
        suspect = " ⚠" if p.get("suspect") else ""
        label_cell = f'<a href="#img-{p["label"]}">{p["label"]}</a>' if has_ortho else p["label"]
        suspect_mark = ' <span style="color:#f44">⚠</span>' if p.get("suspect") else ""
        detail_rows += (f'<tr><td>{label_cell}{suspect_mark}</td><td>{p["group"]}</td>'
                        f'<td>{dh_ft:+.4f}</td><td>{dz_ft:+.4f}</td>'
                        f'<td>{d3d_ft:.4f}</td></tr>\n')

    detail_table_html = f"""<table class="detail">
<tr><th>Label</th><th>Group</th><th>dH (ft)</th><th>dZ (ft)</th><th>d3D (ft)</th></tr>
{detail_rows}</table>"""

    # --- Suspect / outlier check ---
    suspects = [p for p in all_points if p.get("suspect")]
    criteria_desc = (
        f"A point is flagged when its horizontal residual (dH) exceeds both "
        f"{suspect_ratio:.0f}× the median dH <b>and</b> {suspect_floor_ft:.1f} ft. "
        f"Median dH = {median_dh * M_TO_FT:.4f} ft, "
        f"threshold = {suspect_thresh * M_TO_FT:.4f} ft."
    )
    if suspects:
        suspect_rows = ""
        for p in sorted(suspects, key=lambda x: x["dH"], reverse=True):
            ratio = p["dH"] / median_dh if median_dh > 0 else 0
            suspect_rows += (
                f'<tr><td>{p["label"]}</td><td>{p["group"]}</td>'
                f'<td>{p["dH"] * M_TO_FT:.3f}</td>'
                f'<td>{ratio:.0f}×</td>'
                f'<td>Verify pixel tagging — possible mis-tag, wrong target, '
                f'or base station confusion.</td></tr>\n')
        suspect_html = f"""<h3>⚠ Suspect tagging — {len(suspects)} point{'s' if len(suspects) != 1 else ''} flagged</h3>
<p style="color:#aaa; font-size:12px;">{criteria_desc}</p>
<table class="detail">
<tr><th>Label</th><th>Group</th><th>dH (ft)</th><th>× median</th><th>Recommendation</th></tr>
{suspect_rows}</table>"""
    else:
        suspect_html = f"""<h3>✓ Outlier check — no suspect points</h3>
<p style="color:#aaa; font-size:12px;">{criteria_desc}
All points are within the threshold.</p>"""

    detail_section = f"""<h3>Per-point residuals (all, sorted worst-first by dH)</h3>
<div class="detail-with-map">
{detail_table_html}
{overview_html}
</div>
{suspect_html}"""

    # --- Ortho crop images (threaded for speed) ---
    image_html = ""
    if has_ortho:
        from osgeo import gdal
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Each thread opens its own GDAL handle (thread-local)
        _tls = threading.local()

        def _process_crop(p):
            if not hasattr(_tls, "ds"):
                _tls.ds = gdal.Open(ortho_path, gdal.GA_ReadOnly)
            label = p["label"]
            sx, sy = p["survey_x"], p["survey_y"]
            crop_result = _crop_ortho(_tls.ds, sx, sy, crop_radius)
            if crop_result is None:
                return None
            img, px_sz, cx_px, cy_px = crop_result
            annotated = _annotate_crop(
                img, cx_px, cy_px,
                dx_m=p["dX"], dy_m=p["dY"],
                px_size=px_sz, label=label,
                dh_m=p["dH"], dz_m=p["dZ"], d3d_m=p["d3D"],
                group=p["group"], upscale=upscale,
            )
            return f"""
        <div class="card" id="img-{label}">
            <div class="info">
                <b>{label}</b> <span class="group">{p['group']}</span><br/>
                dH={p['dH']*M_TO_FT:+.4f} ft &nbsp; dZ={p['dZ']*M_TO_FT:+.4f} ft
            </div>
            <img src="{_img_to_data_uri(annotated)}" />
        </div>"""

        print(f"  Generating {len(all_points)} ortho crops (threaded)...",
              file=sys.stderr)
        t_crops = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            # map preserves order (worst-first) matching the detail table
            results = list(pool.map(_process_crop, all_points))
        cards = [r for r in results if r is not None]
        skipped = sum(1 for r in results if r is None)
        print(f"  {len(cards)} crops in {time.time()-t_crops:.1f}s "
              f"({skipped} skipped)", file=sys.stderr)
        if cards:
                image_html = f"""
<h3>Ortho crops</h3>
<div class="legend">
    <span class="green">X survey coordinate</span>
    <span class="yellow">+ projected (triangulated)</span>
    <span class="yellow">○ 1 ft radius</span>
    &nbsp; | &nbsp; gap between markers and visible target = ortho positioning error
</div>
<div class="grid">
{''.join(cards)}
</div>"""

    # --- Assemble HTML ---
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>RMSE Accuracy Report</title>
<style>
    body {{ font-family: monospace; background: #1a1a1a; color: #eee; margin: 20px; }}
    h1 {{ font-size: 18px; }}
    h3 {{ font-size: 14px; margin-top: 24px; color: #ccc; }}
    table {{ border-collapse: collapse; margin: 8px 0 16px 0; }}
    th, td {{ padding: 4px 12px; text-align: right; border: 1px solid #444; }}
    th {{ background: #333; }}
    td:first-child {{ text-align: left; }}
    .summary td:first-child {{ font-weight: bold; }}
    .summary th[colspan] {{ text-align: center; }}
    .detail a {{ color: #6cf; }}
    .detail-with-map {{ display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }}
    .detail-with-map table {{ flex-shrink: 0; }}
    .legend {{ font-size: 13px; margin-bottom: 16px; color: #aaa; }}
    .green {{ color: #0f0; }}
    .yellow {{ color: #ff0; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .card {{ background: #2a2a2a; border-radius: 6px; overflow: hidden; }}
    .card img {{ display: block; }}
    .info {{ padding: 6px 10px; font-size: 12px; line-height: 1.5; }}
    .group {{ color: #888; font-size: 11px; }}
</style>
</head>
<body>
<h1>RMSE Accuracy Report</h1>
{summary_table}
{detail_section}
{image_html}
</body>
</html>"""

    from pathlib import Path
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\nWrote HTML report: {output_path} ({len(all_points)} points)",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Synthetic self-contained test
# ---------------------------------------------------------------------------


def run_synthetic_test() -> bool:
    """
    Construct a known camera setup, project known 3D points, run triangulation,
    verify RMS_3D < 0.001 m for CHK points.  No external files required.

    Returns True if the test passes, False otherwise.
    """
    import tempfile
    import os

    print("Running synthetic test...", file=sys.stderr)

    # ---- Known 3D points in ENU (metres) ----
    gcp_points_enu = {
        "GCP-1": np.array([0.0,  0.0, 0.0]),
        "GCP-2": np.array([50.0, 0.0, 0.0]),
        "GCP-3": np.array([0.0, 50.0, 0.0]),
    }
    chk_points_enu = {
        "CHK-1": np.array([10.0, 20.0, -5.0]),
        "CHK-2": np.array([-8.0, 15.0, -6.5]),
    }

    # ---- Synthetic reference_lla ----
    ref_lat = 34.123
    ref_lon = -106.456
    ref_alt = 1850.0

    # ---- Camera intrinsics ----
    width, height = 4000, 3000
    focal = 0.85
    k1, k2 = -0.05, 0.01
    focal_px = focal * max(width, height)
    cx, cy = width / 2.0, height / 2.0

    # ---- Three synthetic camera poses ----
    camera_centres = [
        np.array([0.0, 0.0, 100.0]),
        np.array([30.0, 5.0, 100.0]),
        np.array([-5.0, 25.0, 100.0]),
    ]

    def make_R(C: np.ndarray) -> np.ndarray:
        forward = np.array([0.0, 0.0, 0.0]) - C
        forward /= np.linalg.norm(forward)
        world_north = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_north)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        return np.vstack([right, down, forward])

    shots_dict: Dict[str, dict] = {}
    cam_key = "synthetic_cam 4000 3000 perspective 0"
    for i, C in enumerate(camera_centres):
        R = make_R(C)
        t = -R @ C
        rotvec = Rotation.from_matrix(R).as_rotvec().tolist()
        img_name = f"IMG_{i:04d}.JPG"
        shots_dict[img_name] = {
            "camera": cam_key,
            "rotation": rotvec,
            "translation": t.tolist(),
        }

    cameras_dict = {
        cam_key: {
            "width": width, "height": height,
            "focal": focal, "k1": k1, "k2": k2,
        }
    }

    reconstruction_data = [{
        "cameras": cameras_dict,
        "shots": shots_dict,
        "reference_lla": {"latitude": ref_lat, "longitude": ref_lon, "altitude": ref_alt},
        "points": {},
    }]

    def project_point(P_world, R, t):
        p_cam = R @ P_world + t
        if p_cam[2] <= 0:
            return None
        xn = p_cam[0] / p_cam[2]
        yn = p_cam[1] / p_cam[2]
        r2 = xn ** 2 + yn ** 2
        distort = 1.0 + k1 * r2 + k2 * r2 ** 2
        px = focal_px * xn * distort + cx
        py = focal_px * yn * distort + cy
        if 0 <= px < width and 0 <= py < height:
            return (px, py)
        return None

    crs_header = "EPSG:32613"

    def _make_lines(points_enu):
        lines = [crs_header]
        for label, P_enu in points_enu.items():
            x_p, y_p, z_e = enu_to_projected(P_enu, ref_lat, ref_lon, ref_alt, crs_header)
            for i, C in enumerate(camera_centres):
                R = make_R(C)
                t = -R @ C
                img_name = f"IMG_{i:04d}.JPG"
                proj = project_point(P_enu, R, t)
                if proj is None:
                    continue
                px, py = proj
                lines.append(
                    f"{x_p:.3f}\t{y_p:.3f}\t{z_e:.3f}\t{px:.3f}\t{py:.3f}\t{img_name}\t{label}"
                )
        return "\n".join(lines) + "\n"

    gcp_content = _make_lines(gcp_points_enu)
    chk_content = _make_lines(chk_points_enu)

    with tempfile.TemporaryDirectory() as tmpdir:
        recon_path = os.path.join(tmpdir, "synthetic_recon.json")
        gcp_path   = os.path.join(tmpdir, "gcp_list.txt")
        chk_path   = os.path.join(tmpdir, "chk_list.txt")

        with open(recon_path, "w") as f:
            json.dump(reconstruction_data, f)
        with open(gcp_path, "w") as f:
            f.write(gcp_content)
        with open(chk_path, "w") as f:
            f.write(chk_content)

        result = compute_rmse(recon_path, gcp_path, chk_path)

    print_summary(result)

    chk = result["chk"]
    if chk["n"] == 0:
        print("FAIL: no CHK points triangulated", file=sys.stderr)
        return False

    rms_3d = chk["rms_3d"]
    threshold = 0.001  # metres
    if rms_3d < threshold:
        print(f"PASS: CHK RMS_3D = {rms_3d:.6f} m < {threshold} m", file=sys.stderr)
        return True
    else:
        print(f"FAIL: CHK RMS_3D = {rms_3d:.6f} m >= {threshold} m", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RMSE for GCP and CHK points from ODM reconstruction.",
    )
    parser.add_argument(
        "reconstruction", nargs="?",
        metavar="reconstruction.topocentric.json",
        help="Path to reconstruction.topocentric.json (NOT reconstruction.json)",
    )
    parser.add_argument(
        "gcp_list", nargs="?",
        metavar="gcp_list.txt",
        help="GCP-* rows (from transform.py split); used to fit georeferencing similarity",
    )
    parser.add_argument(
        "chk_list", nargs="?",
        metavar="chk_list.txt",
        help="CHK-* rows (from transform.py split); independent accuracy assessment",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="CRS override (e.g. EPSG:32613). Falls back to header in gcp_list.txt.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run synthetic self-contained test (no external files required).",
    )
    parser.add_argument(
        "--html",
        default=None,
        metavar="report.html",
        help="Generate HTML accuracy report (tables + optional ortho crops).",
    )
    parser.add_argument(
        "--ortho",
        default=None,
        metavar="orthophoto.tif",
        help="Orthophoto for annotated crop images in HTML report.",
    )
    parser.add_argument(
        "--crop-radius",
        type=float,
        default=5.0,
        help="Crop radius in metres for ortho images (default: 5.0).",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
        help="Upscale factor for ortho crop images (default: 4).",
    )
    parser.add_argument(
        "--suspect-ratio",
        type=float,
        default=5.0,
        help="Flag points with dH > N× median dH (default: 5.0).",
    )
    parser.add_argument(
        "--suspect-floor",
        type=float,
        default=0.5,
        help="Minimum dH in feet to trigger suspect flag (default: 0.5).",
    )
    args = parser.parse_args()

    if args.test:
        ok = run_synthetic_test()
        sys.exit(0 if ok else 1)

    if not args.reconstruction or not args.gcp_list or not args.chk_list:
        parser.error("reconstruction, gcp_list, and chk_list are required unless --test is used.")

    result = compute_rmse(
        args.reconstruction, args.gcp_list, args.chk_list, args.crs,
    )
    print_summary(result)
    json.dump(result, sys.stdout, indent=2)
    print()  # trailing newline

    if args.html:
        generate_html_report(
            result,
            output_path=args.html,
            ortho_path=args.ortho,
            crop_radius=args.crop_radius,
            upscale=args.upscale,
            suspect_ratio=args.suspect_ratio,
            suspect_floor_ft=args.suspect_floor,
        )


if __name__ == "__main__":
    main()
