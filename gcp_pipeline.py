#!/usr/bin/env python3
"""
GCP estimation pipeline — replaces gcp.py.

Stages:
  B1  parse_emlid_csv()           Parse Emlid Reach CSV (all solution statuses).
  B1  read_image_exif_batch()     Batch exiftool read for all images in parallel.
  B1  match_images_to_gcps()      Footprint-based image↔GCP association.
  B2  project_pixel_mode_a()      EXIF-based pinhole projection (nadir + oblique).
  B2  project_pixel_mode_b()      reconstruction.json-based projection (optional).
  B3  run_pipeline()              Full pipeline: B1 → B2 → write outputs.
"""

import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METERS_PER_DEG_LAT = 111319.9
FULL_FRAME_DIAG_MM = math.sqrt(36**2 + 24**2)   # 43.267 mm
FT_TO_M = 0.3048
NADIR_TOL_DEG = 10.0   # pitch within 10° of -90° is treated as nadir

# ---------------------------------------------------------------------------
# B1 — Emlid CSV Parser
# ---------------------------------------------------------------------------

def parse_emlid_csv(csv_path: str) -> List[dict]:
    """
    Parse an Emlid Reach CSV and return a list of GCP dicts.

    All points are returned regardless of Solution status (FIX, FLOAT, SINGLE).
    The status is preserved in the 'solution_status' key so callers can filter
    or annotate as needed. Each dict has:
        label           : str   (Name column)
        lat             : float (WGS84 degrees)
        lon             : float (WGS84 degrees)
        ellip_alt_m     : float (WGS84 ellipsoidal height, metres)
        easting         : float (projected X in CRS units)
        northing        : float (projected Y in CRS units)
        elevation       : float (projected Z / geoid height in CRS units)
        cs_name         : str   (CRS description from CS name column)
        solution_status : str   (e.g. 'FIX', 'FLOAT', 'SINGLE', or '')

    Raises ValueError if no valid rows are found.
    """
    gcps = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames or []]
        h_lower = [h.lower() for h in headers]

        def _col(keywords):
            """Return the first header that contains any keyword (case-insensitive)."""
            for kw in keywords:
                for i, h in enumerate(h_lower):
                    if kw in h:
                        return headers[i]
            return None

        col_name   = _col(['name', 'point'])
        col_lat    = _col(['latitude'])
        col_lon    = _col(['longitude'])
        col_ellip  = _col(['ellipsoidal'])
        col_east   = _col(['easting', ' east'])
        col_north  = _col(['northing', ' north'])
        col_elev   = _col(['elevation'])
        col_status = _col(['solution status', 'status'])
        col_cs     = _col(['cs name', 'coordinate system'])

        if not col_lat or not col_lon:
            raise ValueError(
                f"Cannot find Latitude/Longitude columns in {csv_path}. "
                f"Headers: {headers}"
            )

        for row in reader:
            try:
                gcp = {
                    'label':           (row.get(col_name) or '').strip(),
                    'lat':             float(row[col_lat]),
                    'lon':             float(row[col_lon]),
                    'ellip_alt_m':     float(row[col_ellip]) * FT_TO_M if col_ellip else None,
                    'easting':         float(row[col_east])  if col_east  else None,
                    'northing':        float(row[col_north]) if col_north else None,
                    'elevation':       float(row[col_elev])  if col_elev  else None,
                    'cs_name':         (row.get(col_cs) or '').strip(),
                    'solution_status': (row.get(col_status) or '').strip().upper() if col_status else '',
                }
            except (ValueError, KeyError):
                continue   # skip malformed rows

            if not gcp['label']:
                gcp['label'] = f"gcp_{len(gcps)}"

            gcps.append(gcp)

    if not gcps:
        raise ValueError(
            f"No valid GCP points found in {csv_path}. "
            f"Headers found: {headers}"
        )
    return gcps


# ---------------------------------------------------------------------------
# B1 — Batch EXIF Read
# ---------------------------------------------------------------------------

_EXIF_TAGS = [
    '-GPSLatitude', '-GPSLongitude',
    '-AbsoluteAltitude', '-RelativeAltitude',
    '-FocalLength', '-FocalLengthIn35mmFormat',
    '-ImageWidth', '-ImageHeight',
    '-GimbalPitchDegree', '-GimbalYawDegree', '-GimbalRollDegree',
]


def read_image_exif_batch(image_dir: str,
                          extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.tif', '.tiff')
                          ) -> Dict[str, dict]:
    """
    Run a single exiftool batch call over all images in image_dir.

    Returns dict mapping filename → EXIF dict with keys:
        lat, lon, abs_alt, rel_alt, focal_mm, focal35_mm,
        img_w, img_h, gimbal_pitch, gimbal_yaw, gimbal_roll

    Images missing lat/lon are omitted.
    """
    image_dir = Path(image_dir)
    files = [f for f in image_dir.iterdir()
             if f.suffix.lower() in extensions]
    if not files:
        return {}

    cmd = ['exiftool', '-json', '-n'] + _EXIF_TAGS + [str(image_dir)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        records = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise RuntimeError(f"exiftool failed on {image_dir}: {e}")

    exif_map = {}
    for rec in records:
        lat = rec.get('GPSLatitude')
        lon = rec.get('GPSLongitude')
        if lat is None or lon is None:
            continue

        src = Path(rec.get('SourceFile', ''))
        fname = src.name

        exif_map[fname] = {
            'path':         str(src),
            'lat':          float(lat),
            'lon':          float(lon),
            'abs_alt':      _float(rec.get('AbsoluteAltitude')),
            'rel_alt':      _float(rec.get('RelativeAltitude')),
            'focal_mm':     _float(rec.get('FocalLength')),
            'focal35_mm':   _float(rec.get('FocalLengthIn35mmFormat')),
            'img_w':        int(rec['ImageWidth'])  if 'ImageWidth'  in rec else None,
            'img_h':        int(rec['ImageHeight']) if 'ImageHeight' in rec else None,
            'gimbal_pitch': _float(rec.get('GimbalPitchDegree')),
            'gimbal_yaw':   _float(rec.get('GimbalYawDegree')),
            'gimbal_roll':  _float(rec.get('GimbalRollDegree')),
        }
    return exif_map


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# B1 — Parallel Footprint Matching
# ---------------------------------------------------------------------------

def _footprint_worker(args):
    """
    Worker: check which GCPs fall within this image's footprint.

    args: (fname, exif_dict, gcp_list, fallback_radius_m, site_ellip_alt_m)
    Returns: (fname, [gcp_labels]) or None
    """
    fname, exif, gcps, fallback_radius, site_ellip_alt_m = args

    lat_c = exif['lat']
    lon_c = exif['lon']
    focal_mm  = exif.get('focal_mm')
    focal35   = exif.get('focal35_mm')
    img_w     = exif.get('img_w')
    img_h     = exif.get('img_h')
    abs_alt   = exif.get('abs_alt')
    rel_alt   = exif.get('rel_alt')

    # AGL altitude for footprint sizing.
    # rel_alt (AGL above takeoff) is unreliable when the takeoff point differs
    # significantly from the GCP site elevation. Use abs_alt (WGS84 ellipsoidal)
    # minus the mean GCP site elevation as a robust AGL estimate.
    # Fall back to rel_alt if abs_alt or site reference is unavailable.
    if abs_alt is not None and site_ellip_alt_m is not None:
        agl = abs_alt - site_ellip_alt_m
    elif rel_alt is not None:
        agl = rel_alt
    else:
        agl = None

    # Compute footprint radius
    if agl is not None and agl > 1.0 and focal_mm and focal35 and img_w and img_h:
        # Derive diagonal FOV from sensor geometry
        scale = focal_mm / focal35
        sensor_diag = FULL_FRAME_DIAG_MM * scale
        aspect = img_w / img_h
        half_diag_fov = math.atan(sensor_diag / (2 * focal_mm))
        radius = agl * math.tan(half_diag_fov)
    else:
        radius = fallback_radius

    # Check each GCP
    hits = []
    mid_lat_rad = math.radians(lat_c)
    for gcp in gcps:
        dE = (gcp['lon'] - lon_c) * METERS_PER_DEG_LAT * math.cos(mid_lat_rad)
        dN = (gcp['lat'] - lat_c) * METERS_PER_DEG_LAT
        dist = math.sqrt(dE**2 + dN**2)
        if dist <= radius:
            hits.append(gcp['label'])

    return (fname, hits) if hits else None


def match_images_to_gcps(exif_map: Dict[str, dict],
                         gcps: List[dict],
                         fallback_radius_m: float = 50.0,
                         threads: int = 0) -> Dict[str, List[str]]:
    """
    For each image, determine which GCPs fall within its ground footprint.

    Returns dict: {filename: [gcp_label, ...]}
    Only images that contain at least one GCP are included.
    """
    if not threads:
        threads = cpu_count()

    # Compute mean GCP ellipsoidal altitude as site ground reference.
    # Used by workers to compute AGL from abs_alt when rel_alt is unreliable
    # (e.g. drone launched from a hilltop far above the GCP survey site).
    gcp_alts = [g['ellip_alt_m'] for g in gcps if g.get('ellip_alt_m') is not None]
    site_ellip_alt_m = sum(gcp_alts) / len(gcp_alts) if gcp_alts else None

    task_args = [
        (fname, exif, gcps, fallback_radius_m, site_ellip_alt_m)
        for fname, exif in exif_map.items()
    ]

    image_to_gcps: Dict[str, List[str]] = {}
    total = len(task_args)
    completed = 0

    if threads == 1 or total <= 1:
        # Sequential path: no pool overhead, fork-safe for gunicorn/web contexts
        for args in task_args:
            result = _footprint_worker(args)
            completed += 1
            if result is not None:
                fname, labels = result
                image_to_gcps[fname] = labels
            if completed % max(1, total // 10) == 0 or completed == total:
                pct = 100 * completed / total
                print(f"  footprint match {pct:5.1f}%  ({completed}/{total})",
                      end='\r', flush=True)
    else:
        # ThreadPoolExecutor: threads are fork-safe (no fork-of-fork risk), and
        # avoid multiprocessing.Pool deadlocks inside gunicorn/preloaded workers.
        # For CPU-bound footprint math the GIL limits true parallelism, but for
        # typical dataset sizes this is negligible.
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for result in executor.map(_footprint_worker, task_args):
                completed += 1
                if result is not None:
                    fname, labels = result
                    image_to_gcps[fname] = labels
                if completed % max(1, total // 10) == 0 or completed == total:
                    pct = 100 * completed / total
                    print(f"  footprint match {pct:5.1f}%  ({completed}/{total})",
                          end='\r', flush=True)

    print()  # newline after progress
    return image_to_gcps


# ---------------------------------------------------------------------------
# B2 — Pixel Projection
# ---------------------------------------------------------------------------

def is_nadir(exif: dict) -> bool:
    """Return True if the image pitch is within NADIR_TOL_DEG of straight down (-90°)."""
    pitch = exif.get('gimbal_pitch')
    if pitch is None:
        return True   # no pitch data → assume nadir (legacy behaviour)
    return abs(pitch + 90.0) <= NADIR_TOL_DEG


def project_pixel_mode_a(exif: dict, gcp: dict) -> Optional[Tuple[float, float]]:
    """
    Project GCP world coordinates to pixel (px, py) using EXIF-only camera model.

    Handles both nadir (pitch ≈ -90°) and oblique cameras via a full 3D rotation
    matrix derived from gimbal pitch (θ) and yaw (ψ):

        X_cam (image right) = ( cos(ψ),        -sin(ψ),         0     )
        Z_cam (optical axis) = ( cos(θ)·sin(ψ),  cos(θ)·cos(ψ),  sin(θ))
        Y_cam (image down)   = ( sin(θ)·sin(ψ),  sin(θ)·cos(ψ), -cos(θ))
                             = Z_cam × X_cam

    At θ = -90° this degenerates to the original nadir formula.
    GimbalRoll = 180° flips both X and Y axes (opposite-facing mount).

    Validated (nadir): mean 73.6 px error at 7–99 m AGL
    (see docs/dji_m3e_camera_model.md).

    Returns (px, py) in image pixel space, or None if GCP is behind/out of frame.
    """
    cam_lat   = exif['lat']
    cam_lon   = exif['lon']
    cam_alt   = exif.get('abs_alt')        # WGS84 ellipsoidal, metres
    pitch_deg = exif.get('gimbal_pitch', -90.0) or -90.0   # default nadir
    yaw_deg   = exif.get('gimbal_yaw')
    roll_deg  = exif.get('gimbal_roll')
    focal_mm  = exif.get('focal_mm')
    focal35   = exif.get('focal35_mm')
    img_w     = exif.get('img_w')
    img_h     = exif.get('img_h')

    gcp_lat   = gcp['lat']
    gcp_lon   = gcp['lon']
    gcp_alt   = gcp.get('ellip_alt_m')    # WGS84 ellipsoidal, metres

    if any(v is None for v in [cam_alt, gcp_alt, yaw_deg, roll_deg,
                                focal_mm, focal35, img_w, img_h]):
        return None

    # --- Sensor geometry ---
    scale = focal_mm / focal35
    sensor_diag = FULL_FRAME_DIAG_MM * scale
    aspect = img_w / img_h
    sensor_h = sensor_diag / math.sqrt(1 + aspect**2)
    sensor_w = sensor_h * aspect
    fx = focal_mm * img_w / sensor_w
    fy = focal_mm * img_h / sensor_h
    cx, cy = img_w / 2.0, img_h / 2.0

    # --- ENU displacement (camera → GCP) ---
    mid_lat = math.radians((cam_lat + gcp_lat) / 2)
    dE = (gcp_lon - cam_lon) * METERS_PER_DEG_LAT * math.cos(mid_lat)
    dN = (gcp_lat - cam_lat) * METERS_PER_DEG_LAT
    dU = gcp_alt - cam_alt

    # --- 3D camera axes in ENU ---
    psi   = math.radians(yaw_deg)
    theta = math.radians(pitch_deg)
    cp, sp = math.cos(psi),   math.sin(psi)
    ct, st = math.cos(theta), math.sin(theta)

    # X_cam: image right — always horizontal
    Xc = ( cp, -sp, 0.0)
    # Z_cam: optical axis (look direction, positive into scene)
    Zc = (ct * sp, ct * cp,  st)
    # Y_cam: image down = Z_cam × X_cam
    Yc = (st * sp, st * cp, -ct)

    # Roll=180°: camera mounted flipped 180° around optical axis
    flip = -1.0 if abs(roll_deg - 180.0) < 1.0 else 1.0

    # --- Camera-frame coordinates ---
    cam_x = flip * (Xc[0]*dE + Xc[1]*dN + Xc[2]*dU)
    cam_y = flip * (Yc[0]*dE + Yc[1]*dN + Yc[2]*dU)
    cam_z =         Zc[0]*dE + Zc[1]*dN + Zc[2]*dU   # depth; not flipped

    if cam_z <= 0:
        return None  # GCP behind or at camera plane

    # --- Pinhole projection ---
    px = fx * cam_x / cam_z + cx
    py = fy * cam_y / cam_z + cy

    if 0 <= px < img_w and 0 <= py < img_h:
        return (px, py)
    return None


def project_pixel_mode_b(gcp: dict,
                          shot: dict,
                          camera: dict,
                          reference_lla: dict) -> Optional[Tuple[float, float]]:
    """
    Project GCP to pixel using SfM-refined camera pose from reconstruction.json.

    shot         : dict from reconstruction.json shots[filename]
    camera       : dict from reconstruction.json cameras[shot['camera']]
    reference_lla: dict with 'latitude', 'longitude', 'altitude'

    Returns (px, py) or None if GCP is behind/out-of-frame.
    Requires scipy (for Rotation.from_rotvec).
    """
    try:
        from scipy.spatial.transform import Rotation
        import numpy as np
    except ImportError:
        raise RuntimeError("Mode B requires scipy and numpy")

    gcp_lat = gcp['lat']
    gcp_lon = gcp['lon']
    gcp_alt = gcp.get('ellip_alt_m')
    if gcp_alt is None:
        return None

    ref_lat = reference_lla['latitude']
    ref_lon = reference_lla['longitude']
    ref_alt = reference_lla['altitude']

    # GCP in ENU topocentric (flat-earth approx, good to <1 mm at site scales)
    mid_lat = math.radians((gcp_lat + ref_lat) / 2)
    p_world = np.array([
        (gcp_lon - ref_lon) * METERS_PER_DEG_LAT * math.cos(mid_lat),
        (gcp_lat - ref_lat) * METERS_PER_DEG_LAT,
        gcp_alt - ref_alt,
    ])

    # Camera pose: world → camera
    R = Rotation.from_rotvec(shot['rotation']).as_matrix()
    t = np.array(shot['translation'])
    p_cam = R @ p_world + t

    if p_cam[2] <= 0:
        return None  # behind camera

    # Intrinsics (OpenSfM normalized focal: focal_px / max(w, h))
    w, h = camera['width'], camera['height']
    focal_px = camera['focal'] * max(w, h)
    k1 = camera.get('k1', 0.0)
    k2 = camera.get('k2', 0.0)

    # Normalized image coords + radial distortion
    xn = p_cam[0] / p_cam[2]
    yn = p_cam[1] / p_cam[2]
    r2 = xn**2 + yn**2
    distort = 1 + k1 * r2 + k2 * r2**2
    xd, yd = xn * distort, yn * distort

    px = focal_px * xd + w / 2.0
    py = focal_px * yd + h / 2.0

    if 0 <= px < w and 0 <= py < h:
        return (px, py)
    return None


# ---------------------------------------------------------------------------
# B3 — Pipeline Runner + Output Writers
# ---------------------------------------------------------------------------

def _cs_name_to_epsg(cs_name: str) -> Optional[str]:
    """
    Convert an Emlid CS name string to an EPSG code string (e.g. 'EPSG:6531').

    For compound CRS like:
      "NAD83(2011) / New Mexico Central (ftUS) + NAVD88(GEOID18) height (ftUS)"
    splits on ' + ' and uses only the horizontal (first) component.

    Returns None if pyproj is unavailable or the name cannot be resolved.
    """
    if not cs_name:
        return None
    try:
        from pyproj import CRS
    except ImportError:
        return None

    # Try horizontal component first (compound CRS split on ' + ')
    candidates = [cs_name.split(' + ')[0].strip(), cs_name]
    for candidate in candidates:
        try:
            crs = CRS.from_user_input(candidate)
            epsg = crs.to_epsg()
            if epsg:
                return f'EPSG:{epsg}'
        except Exception:
            continue
    return None


def _write_gcp_list(gcps: List[dict],
                    estimates: Dict[str, Dict[str, dict]]) -> str:
    """
    Build gcp_list.txt content for GCPEditorPro / OpenDroneMap.

    Prefers projected coordinates (easting, northing, elevation) from the Emlid
    CSV over WGS-84 lat/lon, because GCPEditorPro expects the same projected
    coordinate system that the surveyor used.  The PROJ/EPSG header is derived
    from the Emlid CS name column via pyproj; falls back to the raw cs_name
    string, then to WGS-84 if projected coords are absent.

    Line 1: EPSG:xxxx or PROJ string
    Lines 2+: geo_x geo_y geo_z px py image_name gcp_label  (one per image)
    """
    gcp_by_label = {g['label']: g for g in gcps}

    # Prefer projected easting/northing/elevation when available
    use_projected = any(
        g.get('easting') is not None
        and g.get('northing') is not None
        and g.get('elevation') is not None
        for g in gcps
    )

    rows = []
    for gcp_label, img_map in estimates.items():
        gcp = gcp_by_label.get(gcp_label)
        if not gcp:
            continue
        if use_projected:
            x, y, z = gcp.get('easting'), gcp.get('northing'), gcp.get('elevation')
        else:
            x, y, z = gcp.get('lon'), gcp.get('lat'), gcp.get('ellip_alt_m')
        if None in (x, y, z):
            continue
        for img_name, est in img_map.items():
            rows.append(
                f"{x:.4f} {y:.4f} {z:.4f} "
                f"{est['px']:.2f} {est['py']:.2f} "
                f"{img_name} {gcp_label}"
            )

    if not rows:
        return ''

    if use_projected:
        cs_name = gcps[0].get('cs_name', '')
        proj = _cs_name_to_epsg(cs_name) or cs_name or '+proj=longlat +datum=WGS84 +no_defs'
    else:
        proj = '+proj=longlat +datum=WGS84 +no_defs'

    return proj + '\n' + '\n'.join(rows) + '\n'


def _write_pix4d(estimates: Dict[str, Dict[str, dict]]) -> str:
    """
    Build pix4d.txt content: GCP image position file for Pix4D.

    Comma-separated with header; one row per (image, GCP) pair.
    Columns: Filename,Label,PixelX,PixelY
    """
    rows = ['Filename,Label,PixelX,PixelY']
    for gcp_label, img_map in estimates.items():
        for img_name, est in img_map.items():
            rows.append(f"{img_name},{gcp_label},{est['px']:.2f},{est['py']:.2f}")
    return '\n'.join(rows) + '\n' if len(rows) > 1 else ''


def _compute_estimates_mode_a(
        image_to_gcps: Dict[str, List[str]],
        exif_map: Dict[str, dict],
        gcp_by_label: Dict[str, dict],
        nadir_only: bool = False) -> Dict[str, Dict[str, dict]]:
    """
    Compute pixel estimates for all (image, GCP) pairs using Mode A (EXIF).

    nadir_only: if True, skip images whose gimbal pitch is not near -90°.

    Returns {gcpLabel: {imgFilename: {px, py, mode}}}
    """
    estimates: Dict[str, Dict[str, dict]] = {}
    skipped_oblique = 0
    for fname, gcp_labels in image_to_gcps.items():
        exif = exif_map.get(fname)
        if exif is None:
            continue
        if nadir_only and not is_nadir(exif):
            skipped_oblique += 1
            continue
        for label in gcp_labels:
            gcp = gcp_by_label.get(label)
            if gcp is None:
                continue
            result = project_pixel_mode_a(exif, gcp)
            if result is None:
                continue
            px, py = result
            if label not in estimates:
                estimates[label] = {}
            estimates[label][fname] = {'px': px, 'py': py, 'mode': 'exif'}
    if nadir_only and skipped_oblique:
        print(f"  --nadir-only: skipped {skipped_oblique} oblique image(s)")
    return estimates


def run_pipeline(images_dir: str,
                 emlid_csv_path: str,
                 reconstruction_path: Optional[str] = None,
                 fallback_radius_m: float = 50.0,
                 threads: int = 0,
                 nadir_only: bool = False) -> Tuple[str, str]:
    """
    Full pipeline: B1 → B2 → B3.

    Returns (gcp_txt_content, estimates_json_content).

    If reconstruction_path is provided and valid, Mode B projection is used
    for images that have a matching shot in the reconstruction. All remaining
    images use Mode A (EXIF-based).

    nadir_only: skip images whose gimbal pitch is not near -90°.
    """
    # B1 — Parse inputs
    print("Parsing Emlid CSV...")
    gcps = parse_emlid_csv(emlid_csv_path)
    print(f"  {len(gcps)} GCPs")

    print(f"Reading EXIF from {images_dir}...")
    exif_map = read_image_exif_batch(images_dir)
    print(f"  {len(exif_map)} images with GPS data")

    print("Matching images to GCPs...")
    image_to_gcps = match_images_to_gcps(
        exif_map, gcps,
        fallback_radius_m=fallback_radius_m,
        threads=threads,
    )
    print(f"  {len(image_to_gcps)} images contain at least one GCP")

    gcp_by_label = {g['label']: g for g in gcps}

    # B2 — Pixel projection
    # Optionally load reconstruction for Mode B
    reconstruction = None
    if reconstruction_path and Path(reconstruction_path).exists():
        print(f"Loading reconstruction from {reconstruction_path}...")
        try:
            with open(reconstruction_path) as f:
                reconstruction = json.load(f)[0]   # first reconstruction
            print("  Mode B (reconstruction) available")
        except Exception as e:
            print(f"  WARNING: failed to load reconstruction: {e}. Using Mode A only.")

    # B3 — Compute pixel estimates
    print("Projecting pixels and writing outputs...")
    if reconstruction:
        ref = reconstruction['reference_lla']
        shots = reconstruction.get('shots', {})
        cameras = reconstruction.get('cameras', {})
        # Mode B where available, Mode A fallback
        estimates: Dict[str, Dict[str, dict]] = {}
        skipped_oblique = 0
        for fname, gcp_labels in image_to_gcps.items():
            exif = exif_map.get(fname)
            if nadir_only and exif and not is_nadir(exif):
                skipped_oblique += 1
                continue
            shot = shots.get(fname)
            for label in gcp_labels:
                gcp = gcp_by_label.get(label)
                if gcp is None:
                    continue
                result = None
                mode_used = 'exif'
                if shot:
                    cam_key = shot.get('camera', '')
                    cam = cameras.get(cam_key, {})
                    result = project_pixel_mode_b(gcp, shot, cam, ref)
                    mode_used = 'reconstruction'
                if result is None and exif:
                    result = project_pixel_mode_a(exif, gcp)
                    mode_used = 'exif'
                if result is not None:
                    px, py = result
                    if label not in estimates:
                        estimates[label] = {}
                    estimates[label][fname] = {'px': px, 'py': py, 'mode': mode_used}
        if nadir_only and skipped_oblique:
            print(f"  --nadir-only: skipped {skipped_oblique} oblique image(s)")
    else:
        estimates = _compute_estimates_mode_a(image_to_gcps, exif_map, gcp_by_label,
                                              nadir_only=nadir_only)

    # B3 — Write outputs
    gcp_txt = _write_gcp_list(gcps, estimates)
    estimates_json = json.dumps(estimates, indent=2)

    return gcp_txt, estimates_json


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='GCP estimation pipeline: Emlid CSV + drone images → gcpeditpro.txt + gcpeditpro.json + pix4d.txt'
    )
    parser.add_argument('emlid_csv',  help='Emlid CSV file path')
    parser.add_argument('image_dir',  help='Directory of drone images')
    parser.add_argument('--reconstruction', default=None,
                        help='Path to opensfm/reconstruction.json (enables Mode B)')
    parser.add_argument('--out-dir',  default='.',
                        help='Output directory for gcpeditpro.txt, gcpeditpro.json, and pix4d.txt (default: .)')
    parser.add_argument('--radius',   type=float, default=50.0,
                        help='Fallback footprint radius in metres (default 50)')
    parser.add_argument('--threads',  type=int,   default=0,
                        help='Worker threads (default: all CPUs)')
    parser.add_argument('--nadir-only', action='store_true',
                        help=f'Skip oblique images (gimbal pitch not within {NADIR_TOL_DEG}° of -90°)')
    parser.add_argument('--b1-only',  action='store_true',
                        help='Run B1 only (footprint match) and print results without writing files')
    args = parser.parse_args()

    if args.b1_only:
        print(f'Parsing {args.emlid_csv}...')
        gcps = parse_emlid_csv(args.emlid_csv)
        print(f'  {len(gcps)} FIX GCPs: {[g["label"] for g in gcps]}')

        print(f'\nReading EXIF from {args.image_dir}...')
        exif_map = read_image_exif_batch(args.image_dir)
        print(f'  {len(exif_map)} images with GPS data')

        print(f'\nMatching images to GCPs (fallback radius={args.radius}m)...')
        image_to_gcps = match_images_to_gcps(
            exif_map, gcps,
            fallback_radius_m=args.radius,
            threads=args.threads,
        )

        print(f'\nResult: {len(image_to_gcps)} images contain at least one GCP')
        for fname, labels in sorted(image_to_gcps.items()):
            print(f'  {fname}: {labels}')
    else:
        gcp_txt, estimates_json = run_pipeline(
            images_dir=args.image_dir,
            emlid_csv_path=args.emlid_csv,
            reconstruction_path=args.reconstruction,
            fallback_radius_m=args.radius,
            threads=args.threads,
            nadir_only=args.nadir_only,
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gcp_out    = out_dir / 'gcpeditpro.txt'
        est_out    = out_dir / 'gcpeditpro.json'
        pix4d_out  = out_dir / 'pix4d.txt'

        gcp_out.write_text(gcp_txt)
        est_out.write_text(estimates_json)
        pix4d_out.write_text(_write_pix4d(json.loads(estimates_json)))

        print(f'\nWrote {gcp_out}')
        print(f'Wrote {est_out}')
        print(f'Wrote {pix4d_out}')
