#!/usr/bin/env python3
"""
rmse_calc.py — 3D RMSE from ODM reconstruction + GCP/CHK point files.

Usage:
    conda run -n geo python accuracy_study/rmse_calc.py \\
        reconstruction.topocentric.json gcp_list.txt chk_list.txt

    # Synthetic self-test:
    conda run -n geo python accuracy_study/rmse_calc.py --test

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
    apply georeferencing similarity, compute and return RMSE for both.

    recon_path    : path to reconstruction.topocentric.json (NOT reconstruction.json)
    gcp_list_path : GCP-* rows only (from transform.py split); used to fit the
                    georeferencing similarity and report control residuals.
    chk_list_path : CHK-* rows only (from transform.py split); independent
                    accuracy assessment (withheld from ODM BA).

    The georeferencing similarity is auto-fitted from GCP triangulation results
    → surveyed GCP positions, correcting the ~25m GPS translation offset and
    ~1.75° UTM grid convergence inherent in reconstruction.topocentric.json.

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

    sim_scale, sim_R, sim_t = None, None, None

    # --- Step 3: triangulate GCP points ---
    gcp_results, gcp_skipped = _triangulate_labels(
        gcp_obs, gcp_coords, shots, cameras, ref_lla, crs, uses_feet,
    )

    # --- Step 4: auto-fit similarity from GCP triangulations → surveyed positions ---
    if gcp_results:
        if len(gcp_results) < 3:
            print(f"WARNING: only {len(gcp_results)} GCPs triangulated, need ≥3 for "
                  f"similarity — skipping geotransform", file=sys.stderr)
        else:
            src = np.array([r["tri_m"] for r in gcp_results])
            dst = np.array([r["survey_m"] for r in gcp_results])
            sim_scale, sim_R, sim_t = fit_similarity(src, dst)
            rot_deg = Rotation.from_matrix(sim_R).as_euler('xyz', degrees=True)
            print(f"\nSimilarity transform fitted from {len(gcp_results)} GCPs:",
                  file=sys.stderr)
            print(f"  scale = {sim_scale:.10f}", file=sys.stderr)
            print(f"  rotation (deg) = X:{rot_deg[0]:.4f}  Y:{rot_deg[1]:.4f}  "
                  f"Z:{rot_deg[2]:.4f}", file=sys.stderr)

    # --- Step 5: triangulate CHK points ---
    chk_results, chk_skipped = _triangulate_labels(
        chk_obs, chk_coords, shots, cameras, ref_lla, crs, uses_feet,
    )

    # --- Step 6: compute residuals (apply auto-fitted similarity) ---
    apply_sim = sim_R is not None

    gcp_points = _compute_residuals(gcp_results, sim_scale, sim_R, sim_t,
                                    apply_sim=apply_sim)
    chk_points = _compute_residuals(chk_results, sim_scale, sim_R, sim_t,
                                    apply_sim=apply_sim)

    return {
        "gcp": _group_stats(gcp_points),
        "chk": _group_stats(chk_points),
        "geotransform_source": "fitted from GCPs",
        "similarity_scale": round(sim_scale, 10) if sim_scale is not None else None,
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


def print_summary(result: dict) -> None:
    """Print human-readable summary to stderr."""
    src = result.get("geotransform_source", "none")
    print(f"\nRMSE report  (geotransform: {src})", file=sys.stderr)
    _fmt_group("GCP residuals (control fit)", result["gcp"])
    _fmt_group("CHK accuracy  (independent)", result["chk"])
    print("", file=sys.stderr)


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


if __name__ == "__main__":
    main()
