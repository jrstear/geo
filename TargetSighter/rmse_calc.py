#!/usr/bin/env python3
"""
rmse_calc.py — 3D RMSE from ODM reconstruction + tagged check pixels.

Usage:
    # Basic (pre-geotransform accuracy):
    conda run -n geo python TargetSighter/rmse_calc.py \\
        reconstruction.topocentric.json chk_list.txt

    # With georeferencing transform (deliverable accuracy, matches Pix4D):
    conda run -n geo python TargetSighter/rmse_calc.py \\
        reconstruction.topocentric.json chk_list.txt --gcp gcp_list.txt

    # Synthetic self-test:
    conda run -n geo python TargetSighter/rmse_calc.py --test

IMPORTANT: Use reconstruction.topocentric.json, NOT reconstruction.json.
The .topocentric file stores camera poses in ODM's local ENU frame where
ref_UTM + local_offset = exact UTM.  The plain reconstruction.json has a
linearised projected-CRS rotation baked in by export_geocoords, and
ref_UTM + those offsets gives ~1m/km error due to the linearisation.

For each CHK-* label in chk_list.txt:
  1. Gathers tagged (image, px, py) pairs.
  2. Builds viewing rays through SfM-optimised camera poses (Rodrigues rotation).
  3. Triangulates 3D position via linear DLT (least-squares on projection-error system).
  4. Converts ENU reconstruction coords → projected world CRS (ref_UTM + local).
  5. Optionally fits a 7-parameter similarity transform from GCP triangulated
     positions → GCP ground truth (--gcp), then applies it to CHK positions.
     This replicates ODM's odm_georeferencing stage and produces RMSE that
     matches the delivered orthophoto accuracy.
  6. Computes dX/dY/dZ vs surveyed position from chk_list.txt cols 1-3.

Outputs per-point table + RMS_X/RMS_Y/RMS_Z/RMS_3D to stdout as JSON.
Human-readable summary to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
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
# Constants
# ---------------------------------------------------------------------------

FT_TO_M = 0.3048006096012192   # US survey foot


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
    Convert a point in ODM's topocentric local frame to projected CRS coordinates.

    IMPORTANT: This function assumes the input is from reconstruction.topocentric.json,
    where coordinates are in a true ENU frame aligned with the UTM grid at the
    reference point.  For this frame, ref_UTM + local = exact UTM.

    Do NOT use with reconstruction.json (the geocoords version), which has a
    linearised topocentric→projected rotation baked in.  Using ref_UTM + geocoords
    gives ~1 m/km error due to the linearisation (e.g. ~88m at 3.4km from ref
    for a site 3° from the UTM central meridian).

    Returns (x_proj, y_proj, z_ellip_m).
    """
    xfm = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    ref_x, ref_y = xfm.transform(ref_lon_deg, ref_lat_deg)
    x = ref_x + p_enu[0]
    y = ref_y + p_enu[1]
    z = ref_alt_m + p_enu[2]
    return x, y, z


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


def parse_chk_confirmed(
    chk_path: str,
) -> Tuple[str, Dict[str, Tuple[float, float, float]], Dict[str, List[Tuple[str, float, float]]]]:
    """
    Parse chk_confirmed.txt (tab-separated ODM GCP format produced by convert_coords.py).

    Returns:
        crs_header   : str  — CRS string from line 1 (e.g. 'EPSG:32613')
        coords_by_label : dict — label → (geo_x, geo_y, geo_z) in the output CRS
        obs_by_label : dict — label → [(image_name, px, py), ...]

    cols: geo_x geo_y geo_z px py image_name label [confidence ...]
    """
    with open(chk_path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"chk_confirmed.txt is empty: {chk_path}")

    crs_header = lines[0].rstrip("\n").strip()
    coords_by_label: Dict[str, Tuple[float, float, float]] = {}
    obs_by_label: Dict[str, List[Tuple[str, float, float]]] = {}

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
            px = float(fields[3])
            py = float(fields[4])
        except ValueError:
            continue
        image_name = fields[5]
        label = fields[6]

        if label not in coords_by_label:
            coords_by_label[label] = (geo_x, geo_y, geo_z)
        obs_by_label.setdefault(label, []).append((image_name, px, py))

    return crs_header, coords_by_label, obs_by_label


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
# Triangulation helper (shared by compute_rmse)
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
# Main RMSE computation
# ---------------------------------------------------------------------------


def compute_rmse(
    recon_path: str,
    chk_path: str,
    crs_override: Optional[str] = None,
    gcp_path: Optional[str] = None,
) -> dict:
    """
    Full pipeline: parse inputs, triangulate, optionally geotransform, compute RMSE.

    If gcp_path is provided, a 7-parameter similarity transform is fitted from
    triangulated GCP positions → surveyed GCP positions and applied to the CHK
    triangulated positions before computing residuals.  This replicates ODM's
    odm_georeferencing stage and produces RMSE matching the delivered orthophoto.

    Returns a dict suitable for JSON output.
    """
    # --- Step 1: parse reconstruction ---
    recon = parse_reconstruction(recon_path)
    cameras = recon["cameras"]
    shots = recon["shots"]
    ref_lla = recon["reference_lla"]

    # --- Step 2: parse check file ---
    crs_header, chk_coords, chk_obs = parse_chk_confirmed(chk_path)
    crs = crs_override or crs_header
    if not crs:
        raise ValueError("CRS not found in chk file header and --crs not provided.")

    unit = crs_axis_unit(crs)
    uses_feet = "foot" in unit.lower()

    # --- Step 3: triangulate CHK points ---
    chk_results, chk_skipped = _triangulate_labels(
        chk_obs, chk_coords, shots, cameras, ref_lla, crs, uses_feet
    )

    # --- Step 4: optionally fit similarity from GCPs ---
    sim_scale, sim_R, sim_t = None, None, None
    gcp_pre_rms = None
    gcp_post_rms = None

    if gcp_path:
        _, gcp_coords, gcp_obs = parse_chk_confirmed(gcp_path)
        gcp_results, _ = _triangulate_labels(
            gcp_obs, gcp_coords, shots, cameras, ref_lla, crs, uses_feet
        )

        if len(gcp_results) < 3:
            print(f"WARNING: only {len(gcp_results)} GCPs triangulated, need ≥3 for similarity — "
                  f"skipping geotransform", file=sys.stderr)
        else:
            src = np.array([r["tri_m"] for r in gcp_results])
            dst = np.array([r["survey_m"] for r in gcp_results])

            # Pre-transform GCP residuals
            gcp_pre_diffs = src - dst
            gcp_pre_rms = np.sqrt((gcp_pre_diffs ** 2).mean(axis=0)).tolist()

            sim_scale, sim_R, sim_t = fit_similarity(src, dst)

            # Post-transform GCP residuals (validation)
            gcp_post = sim_scale * (sim_R @ src.T).T + sim_t - dst
            gcp_post_rms = np.sqrt((gcp_post ** 2).mean(axis=0)).tolist()

            print(f"\nSimilarity transform fitted from {len(gcp_results)} GCPs:", file=sys.stderr)
            print(f"  scale = {sim_scale:.10f}", file=sys.stderr)
            rot_deg = Rotation.from_matrix(sim_R).as_euler('xyz', degrees=True)
            print(f"  rotation (deg) = X:{rot_deg[0]:.4f}  Y:{rot_deg[1]:.4f}  Z:{rot_deg[2]:.4f}",
                  file=sys.stderr)
            print(f"  GCP pre-transform  RMS: X={gcp_pre_rms[0]:.4f}  Y={gcp_pre_rms[1]:.4f}  "
                  f"Z={gcp_pre_rms[2]:.4f}", file=sys.stderr)
            print(f"  GCP post-transform RMS: X={gcp_post_rms[0]:.6f}  Y={gcp_post_rms[1]:.6f}  "
                  f"Z={gcp_post_rms[2]:.6f}", file=sys.stderr)

    # --- Step 5: compute residuals ---
    points = []
    for r in chk_results:
        tri = r["tri_m"]
        if sim_R is not None:
            tri = sim_scale * sim_R @ tri + sim_t
        surv = r["survey_m"]

        dX = tri[0] - surv[0]
        dY = tri[1] - surv[1]
        dZ = tri[2] - surv[2]
        d3D = math.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)

        points.append({
            "label": r["label"],
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
            "geotransform_applied": gcp_path is not None and sim_R is not None,
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
            "geotransform_applied": gcp_path is not None and sim_R is not None,
        }

        if gcp_pre_rms:
            result["gcp_pre_transform_rms"] = {
                "x": round(gcp_pre_rms[0], 6),
                "y": round(gcp_pre_rms[1], 6),
                "z": round(gcp_pre_rms[2], 6),
            }
        if gcp_post_rms:
            result["gcp_post_transform_rms"] = {
                "x": round(gcp_post_rms[0], 6),
                "y": round(gcp_post_rms[1], 6),
                "z": round(gcp_post_rms[2], 6),
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

    print("Running synthetic test...", file=sys.stderr)

    # ---- Known 3D points in ENU (metres) ----
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

    def project_point(
        P_world: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> Optional[Tuple[float, float]]:
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

    # Build chk_confirmed.txt — cols 0-2 are the surveyed position in the output CRS.
    # Use EPSG:32613 (metres) to match convert_coords.py output.
    crs_header = "EPSG:32613"
    chk_lines = [crs_header]

    for label, P_enu in chk_points_enu.items():
        x_p, y_p, z_e = enu_to_projected(P_enu, ref_lat, ref_lon, ref_alt, crs_header)
        for i, C in enumerate(camera_centres):
            R = make_R(C)
            t = -R @ C
            img_name = f"IMG_{i:04d}.JPG"
            proj = project_point(P_enu, R, t)
            if proj is None:
                continue
            px, py = proj
            chk_lines.append(
                f"{x_p:.3f}\t{y_p:.3f}\t{z_e:.3f}\t{px:.3f}\t{py:.3f}\t{img_name}\t{label}\tprojection"
            )

    chk_content = "\n".join(chk_lines) + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        recon_path = os.path.join(tmpdir, "synthetic_recon.json")
        chk_path = os.path.join(tmpdir, "synthetic_chk.txt")

        with open(recon_path, "w") as f:
            json.dump(reconstruction_data, f)
        with open(chk_path, "w") as f:
            f.write(chk_content)

        result = compute_rmse(recon_path, chk_path)

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
        description="Compute 3D RMSE for CHK-* check points from ODM reconstruction.",
    )
    parser.add_argument(
        "reconstruction", nargs="?",
        help="Path to reconstruction.topocentric.json (NOT reconstruction.json)",
    )
    parser.add_argument(
        "chk_confirmed", nargs="?",
        help="Path to chk_list.txt (tab-separated: geo_x geo_y geo_z px py image label)",
    )
    parser.add_argument(
        "--gcp",
        default=None,
        metavar="GCP_FILE",
        help="Path to gcp_list.txt. If provided, fits a 7-parameter similarity "
             "transform from GCP triangulations → ground truth, then applies it "
             "to CHK triangulations before computing RMSE.  This replicates ODM's "
             "odm_georeferencing and gives deliverable-accurate numbers.",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="CRS override (e.g. EPSG:32613). Falls back to header in chk file.",
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

    if not args.reconstruction or not args.chk_confirmed:
        parser.error("reconstruction and chk_confirmed are required unless --test is used.")

    result = compute_rmse(args.reconstruction, args.chk_confirmed, args.crs, args.gcp)
    print_summary(result)
    json.dump(result, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
