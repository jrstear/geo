#!/usr/bin/env python3
"""
rmse_calc.py — 3D RMSE from WebODM reconstruction.json + tagged check pixels.

Usage:
    conda run -n geo python GCPSighter/rmse_calc.py \\
        reconstruction.json chk_confirmed.txt emlid.csv [--crs EPSG:6529]

    conda run -n geo python GCPSighter/rmse_calc.py --test

For each CHK-* label in chk_confirmed.txt:
  1. Gathers tagged (image, px, py) pairs.
  2. Builds viewing rays through SfM-optimised camera poses (Rodrigues rotation).
  3. Triangulates 3D position via linear DLT (least-squares on projection-error system).
  4. Converts ENU reconstruction coords → projected world CRS (pyproj).
  5. Computes dX/dY/dZ vs surveyed position from emlid.csv.

Outputs per-point table + RMS_X/RMS_Y/RMS_Z/RMS_3D to stdout as JSON.
Human-readable summary to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
# Import parse_survey_csv from sibling module
# ---------------------------------------------------------------------------

try:
    from .csv2gcp import parse_survey_csv  # type: ignore[import]
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from csv2gcp import parse_survey_csv  # type: ignore[import]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METERS_PER_DEG_LAT = 111319.9
FT_TO_M = 0.3048006096012192   # US survey foot (more precise than 0.3048)


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
    Convert a point in local ENU (metres) to projected CRS coordinates.

    Uses flat-earth ENU approximation (valid to <1 mm at typical survey scales
    of a few km). Returns (x_proj, y_proj, z_ellip_m) where z_ellip_m is the
    ellipsoidal altitude in metres.
    """
    mid_lat_rad = math.radians(ref_lat_deg)

    dlon = p_enu[0] / (METERS_PER_DEG_LAT * math.cos(mid_lat_rad))
    dlat = p_enu[1] / METERS_PER_DEG_LAT
    lon = ref_lon_deg + dlon
    lat = ref_lat_deg + dlat
    alt = ref_alt_m + p_enu[2]

    xfm = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x, y = xfm.transform(lon, lat)
    return x, y, alt


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
# chk_confirmed.txt parsing
# ---------------------------------------------------------------------------


def parse_chk_confirmed(chk_path: str) -> Tuple[str, Dict[str, List[Tuple[str, float, float]]]]:
    """
    Parse chk_confirmed.txt (same tab-separated format as gcp_list.txt).

    Returns:
        crs_header  : str  — CRS string from line 1 (e.g. 'EPSG:6529')
        obs_by_label: dict — label → [(image_name, px, py), ...]
    """
    with open(chk_path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"chk_confirmed.txt is empty: {chk_path}")

    crs_header = lines[0].rstrip("\n").strip()
    obs_by_label: Dict[str, List[Tuple[str, float, float]]] = {}

    for raw_line in lines[1:]:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        # geo_x geo_y geo_z px py image_name gcp_label [confidence [marker_bbox]]
        try:
            px = float(fields[3])
            py = float(fields[4])
        except ValueError:
            continue
        image_name = fields[5]
        label = fields[6]
        obs_by_label.setdefault(label, []).append((image_name, px, py))

    return crs_header, obs_by_label


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
# Main RMSE computation
# ---------------------------------------------------------------------------


def compute_rmse(
    recon_path: str,
    chk_path: str,
    emlid_csv: str,
    crs_override: Optional[str] = None,
) -> dict:
    """
    Full pipeline: parse inputs, triangulate, convert coords, compute RMSE.

    Returns a dict suitable for JSON output.
    """
    # --- Step 1: parse reconstruction.json ---
    recon = parse_reconstruction(recon_path)
    cameras = recon["cameras"]
    shots = recon["shots"]
    ref_lla = recon["reference_lla"]

    # --- Step 2: parse chk_confirmed.txt ---
    crs_header, obs_by_label = parse_chk_confirmed(chk_path)
    crs = crs_override or crs_header
    if not crs:
        raise ValueError("CRS not found in chk_confirmed.txt header and --crs not provided.")

    # --- Step 3: parse emlid.csv ---
    survey_gcps = parse_survey_csv(emlid_csv, fallback_crs=crs)
    survey_by_label: Dict[str, dict] = {g["label"]: g for g in survey_gcps}

    # --- Determine CRS units ---
    unit = crs_axis_unit(crs)
    uses_feet = "foot" in unit.lower()

    # --- Steps 4-5: triangulate + compare ---
    points = []
    skipped_labels: List[str] = []

    for label, obs in sorted(obs_by_label.items()):
        # Labels in chk_confirmed.txt may carry a csv2gcp prefix (e.g. "CHK-101",
        # "GCP-102"). Strip it for lookup against raw survey CSV labels ("101").
        survey_label = label.split("-", 1)[1] if "-" in label else label
        # Check all shots exist; warn about missing ones
        missing_shots = [img for img, _, _ in obs if img not in shots]
        for img in missing_shots:
            print(f"WARNING: shot {img!r} for label {label!r} not in reconstruction — skipped",
                  file=sys.stderr)

        rays = build_rays(obs, shots, cameras)

        if len(rays) < 2:
            print(
                f"WARNING: label {label!r} has fewer than 2 valid rays "
                f"({len(rays)} found) — skipped",
                file=sys.stderr,
            )
            skipped_labels.append(label)
            continue

        X_enu = triangulate_dlt(rays)
        if X_enu is None:
            print(
                f"WARNING: triangulation for {label!r} produced rank-deficient system — skipped",
                file=sys.stderr,
            )
            skipped_labels.append(label)
            continue

        # Convert ENU → projected CRS
        x_proj, y_proj, z_ellip_m = enu_to_projected(
            X_enu, ref_lla["latitude"], ref_lla["longitude"], ref_lla["altitude"], crs
        )

        # Survey ground truth
        g = survey_by_label.get(survey_label) or survey_by_label.get(label)
        if g is None:
            print(f"WARNING: label {label!r} not found in emlid.csv — skipped", file=sys.stderr)
            skipped_labels.append(label)
            continue
        survey_x = g["easting"]    # projected, CRS units
        survey_y = g["northing"]   # projected, CRS units
        survey_z = g["elevation"]  # orthometric, CRS units

        if survey_x is None or survey_y is None:
            print(f"WARNING: label {label!r} has no easting/northing — skipped", file=sys.stderr)
            skipped_labels.append(label)
            continue

        if survey_z is None:
            # Fall back to ellipsoidal height if available
            if g["ellip_alt_m"] is not None:
                survey_z_m = g["ellip_alt_m"]
                print(
                    f"WARNING: label {label!r} has no orthometric elevation; "
                    "falling back to ellipsoidal height (metres)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"WARNING: label {label!r} has no elevation data — Z residual set to NaN",
                    file=sys.stderr,
                )
                survey_z_m = float("nan")
            survey_x_m = survey_x * FT_TO_M if uses_feet else survey_x
            survey_y_m = survey_y * FT_TO_M if uses_feet else survey_y
        else:
            # Convert to metres for consistent residual computation
            if uses_feet:
                survey_x_m = survey_x * FT_TO_M
                survey_y_m = survey_y * FT_TO_M
                survey_z_m = survey_z * FT_TO_M
            else:
                survey_x_m = survey_x
                survey_y_m = survey_y
                survey_z_m = survey_z

        # Projected coords to metres
        if uses_feet:
            x_proj_m = x_proj * FT_TO_M
            y_proj_m = y_proj * FT_TO_M
        else:
            x_proj_m = x_proj
            y_proj_m = y_proj

        dX = x_proj_m - survey_x_m
        dY = y_proj_m - survey_y_m
        dZ = z_ellip_m - survey_z_m  # geoid offset not corrected; systematic bias expected
        d3D = math.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)

        points.append({
            "label": label,
            "dX": round(dX, 6),
            "dY": round(dY, 6),
            "dZ": round(dZ, 6),
            "d3D": round(d3D, 6),
        })

    n = len(points)
    if n == 0:
        result: dict = {
            "rms_x": None, "rms_y": None, "rms_z": None, "rms_3d": None,
            "mean_dz": None, "std_dz": None,
            "n": 0, "points": [],
        }
    else:
        rms_x = math.sqrt(sum(p["dX"] ** 2 for p in points) / n)
        rms_y = math.sqrt(sum(p["dY"] ** 2 for p in points) / n)
        rms_z = math.sqrt(sum(p["dZ"] ** 2 for p in points) / n)
        rms_3d = math.sqrt(sum(p["d3D"] ** 2 for p in points) / n)
        mean_dz = sum(p["dZ"] for p in points) / n
        std_dz = math.sqrt(sum((p["dZ"] - mean_dz) ** 2 for p in points) / n) if n > 1 else 0.0

        result = {
            "rms_x": round(rms_x, 6),
            "rms_y": round(rms_y, 6),
            "rms_z": round(rms_z, 6),
            "rms_3d": round(rms_3d, 6),
            "mean_dz": round(mean_dz, 6),
            "std_dz": round(std_dz, 6),
            "n": n,
            "points": points,
        }

    return result


def print_summary(result: dict) -> None:
    """Print human-readable summary to stderr."""
    n = result["n"]
    print(f"\nCHK RMSE report — N={n}", file=sys.stderr)
    if n == 0:
        print("  (no valid check points computed)", file=sys.stderr)
        return

    print(f"  RMS_X  = {result['rms_x']:8.4f} m", file=sys.stderr)
    print(f"  RMS_Y  = {result['rms_y']:8.4f} m", file=sys.stderr)
    print(f"  RMS_Z  = {result['rms_z']:8.4f} m", file=sys.stderr)
    print(f"  RMS_3D = {result['rms_3d']:8.4f} m", file=sys.stderr)
    print(f"  mean_Z = {result['mean_dz']:8.4f} m  std_Z = {result['std_dz']:.4f} m",
          file=sys.stderr)
    print("\nPer-point:", file=sys.stderr)
    for p in result["points"]:
        print(
            f"  {p['label']:<10}  dX={p['dX']:+.4f}  dY={p['dY']:+.4f}  "
            f"dZ={p['dZ']:+.4f}  d3D={p['d3D']:.4f}",
            file=sys.stderr,
        )
    print("", file=sys.stderr)


# ---------------------------------------------------------------------------
# Synthetic self-contained test
# ---------------------------------------------------------------------------


def run_synthetic_test() -> bool:
    """
    Construct a known camera setup, project known 3D points, run triangulation,
    verify RMS_3D < 0.001 m.  No external files required.

    Returns True if the test passes, False otherwise.
    """
    import tempfile
    import io

    print("Running synthetic test...", file=sys.stderr)

    # ---- Known 3D points in ENU (metres) ----
    # Place two check points at known positions within a simulated survey site.
    chk_points_enu = {
        "CHK-1": np.array([10.0, 20.0, -5.0]),
        "CHK-2": np.array([-8.0, 15.0, -6.5]),
    }

    # ---- Synthetic reference_lla (central New Mexico, arbitrary) ----
    ref_lat = 34.123
    ref_lon = -106.456
    ref_alt = 1850.0  # metres above WGS-84 ellipsoid

    # ---- Camera intrinsics ----
    width, height = 4000, 3000
    focal = 0.85          # normalised (focal_px = focal * max(w, h))
    k1, k2 = -0.05, 0.01
    focal_px = focal * max(width, height)
    cx, cy = width / 2.0, height / 2.0

    # ---- Three synthetic camera poses ----
    # Cameras hovering above the scene looking straight down (nadir),
    # offset in XY so each gives a different viewing angle on the CHK points.
    # pose: camera centre in ENU (world), then derive R and t.
    #
    # For a nadir camera at position P looking straight down:
    #   - camera Z axis ≈ -world-Z, camera X ≈ world-X, camera Y ≈ -world-Y
    #   - Rotation R (world→camera) has rows [X_cam_in_world, Y_cam_in_world, Z_cam_in_world]
    # We use small pitch/yaw perturbations so triangulation is well-conditioned.

    camera_centres = [
        np.array([0.0, 0.0, 100.0]),    # directly above origin
        np.array([30.0, 5.0, 100.0]),   # offset east
        np.array([-5.0, 25.0, 100.0]),  # offset north
    ]

    # Build R for a camera pointing from C toward the scene centroid (0,0,0).
    # Camera convention: +Z forward (into scene), +X right, +Y down.
    def make_R(C: np.ndarray) -> np.ndarray:
        """Simple nadir-ish rotation: camera -Z axis points from C toward origin."""
        forward = np.array([0.0, 0.0, 0.0]) - C
        forward /= np.linalg.norm(forward)
        # right = forward × world_up... use a stable Gram-Schmidt
        world_north = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_north)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        # R rows: [right, down, forward] — world→camera
        R = np.vstack([right, down, forward])
        return R

    shots_dict: Dict[str, dict] = {}
    cam_key = "synthetic_cam 4000 3000 perspective 0"
    for i, C in enumerate(camera_centres):
        R = make_R(C)
        t = -R @ C          # world→camera translation
        rotvec = Rotation.from_matrix(R).as_rotvec().tolist()
        img_name = f"IMG_{i:04d}.JPG"
        shots_dict[img_name] = {
            "camera": cam_key,
            "rotation": rotvec,
            "translation": t.tolist(),
        }

    cameras_dict = {
        cam_key: {
            "width": width,
            "height": height,
            "focal": focal,
            "k1": k1,
            "k2": k2,
        }
    }

    reconstruction_data = [{
        "cameras": cameras_dict,
        "shots": shots_dict,
        "reference_lla": {
            "latitude": ref_lat,
            "longitude": ref_lon,
            "altitude": ref_alt,
        },
        "points": {},
    }]

    # ---- Forward-project CHK points through each camera ----
    def project_point(P_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> Optional[Tuple[float, float]]:
        p_cam = R @ P_world + t
        if p_cam[2] <= 0:
            return None
        xn = p_cam[0] / p_cam[2]
        yn = p_cam[1] / p_cam[2]
        r2 = xn ** 2 + yn ** 2
        distort = 1.0 + k1 * r2 + k2 * r2 ** 2
        xd, yd = xn * distort, yn * distort
        px = focal_px * xd + cx
        py = focal_px * yd + cy
        if 0 <= px < width and 0 <= py < height:
            return (px, py)
        return None

    # Build chk_confirmed.txt content in memory
    crs_header = "EPSG:6529"
    chk_lines = [crs_header]

    # Build a lookup of surveyed positions — for the synthetic test, we use the
    # ENU → projected conversion to generate "surveyed" coordinates.
    # The RMSE should be near zero because the triangulated result should match.

    for label, P_enu in chk_points_enu.items():
        for i, C in enumerate(camera_centres):
            R = make_R(C)
            t = -R @ C
            img_name = f"IMG_{i:04d}.JPG"
            proj = project_point(P_enu, R, t)
            if proj is None:
                continue
            px, py = proj
            # geo_x, geo_y, geo_z from ENU→projected (as a synthetic survey position)
            x_p, y_p, z_e = enu_to_projected(P_enu, ref_lat, ref_lon, ref_alt, crs_header)
            chk_lines.append(
                f"{x_p:.3f}\t{y_p:.3f}\t{z_e:.3f}\t{px:.3f}\t{py:.3f}\t{img_name}\t{label}\tprojection"
            )

    chk_content = "\n".join(chk_lines) + "\n"

    # Build emlid.csv content in memory — synthetic survey file with easting/northing/elevation
    # Use the exact projected coords derived from ENU positions.
    emlid_rows = ["Name,Easting,Northing,Elevation,CS name"]
    for label, P_enu in chk_points_enu.items():
        x_p, y_p, z_e = enu_to_projected(P_enu, ref_lat, ref_lon, ref_alt, crs_header)
        # x_p, y_p are in feet (EPSG:6529 ftUS); z_e is in metres.
        # compute_rmse applies FT_TO_M to elevation when uses_feet=True,
        # so store elevation in feet here to match real Emlid CSV behaviour.
        z_e_ft = z_e / FT_TO_M
        emlid_rows.append(f"{label},{x_p:.6f},{y_p:.6f},{z_e_ft:.6f},")
    emlid_content = "\n".join(emlid_rows) + "\n"

    # Write temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        recon_path = os.path.join(tmpdir, "synthetic_recon.json")
        chk_path = os.path.join(tmpdir, "synthetic_chk.txt")
        emlid_path = os.path.join(tmpdir, "synthetic_emlid.csv")

        with open(recon_path, "w") as f:
            json.dump(reconstruction_data, f)
        with open(chk_path, "w") as f:
            f.write(chk_content)
        with open(emlid_path, "w") as f:
            f.write(emlid_content)

        result = compute_rmse(recon_path, chk_path, emlid_path, crs_override="EPSG:6529")

    print_summary(result)

    n = result["n"]
    if n == 0:
        print("FAIL: no check points triangulated", file=sys.stderr)
        return False

    rms_3d = result["rms_3d"]
    threshold = 0.001  # metres
    if rms_3d < threshold:
        print(f"PASS: RMS_3D = {rms_3d:.6f} m < {threshold} m", file=sys.stderr)
        return True
    else:
        print(f"FAIL: RMS_3D = {rms_3d:.6f} m >= {threshold} m", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 3D RMSE for CHK-* check points from ODM reconstruction.json."
    )
    parser.add_argument("reconstruction", nargs="?", help="Path to reconstruction.json")
    parser.add_argument("chk_confirmed", nargs="?", help="Path to chk_confirmed.txt")
    parser.add_argument("emlid_csv", nargs="?", help="Path to emlid.csv (rover survey)")
    parser.add_argument(
        "--crs",
        default=None,
        help="CRS override (e.g. EPSG:6529). Falls back to header in chk_confirmed.txt.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run synthetic self-contained test (no external files required).",
    )
    args = parser.parse_args()

    if args.test:
        ok = run_synthetic_test()
        sys.exit(0 if ok else 1)

    # Normal run
    if not args.reconstruction or not args.chk_confirmed or not args.emlid_csv:
        parser.error("reconstruction, chk_confirmed, and emlid_csv are required unless --test is used.")

    result = compute_rmse(args.reconstruction, args.chk_confirmed, args.emlid_csv, args.crs)
    print_summary(result)
    json.dump(result, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
