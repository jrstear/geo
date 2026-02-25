#!/usr/bin/env python3
"""
GCP estimation pipeline — replaces gcp.py.

Stages:
  B1  parse_emlid_csv()           Parse Emlid Reach CSV, filter FIX-only points.
  B1  read_image_exif_batch()     Batch exiftool read for all images in parallel.
  B1  match_images_to_gcps()      Footprint-based image↔GCP association.
  B2  project_pixel_mode_a()      EXIF-based nadir pinhole projection.
  B2  project_pixel_mode_b()      reconstruction.json-based projection (optional).
  B3  run_pipeline()              Full pipeline: B1 → B2 → write outputs.
"""

import csv
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METERS_PER_DEG_LAT = 111319.9
FULL_FRAME_DIAG_MM = math.sqrt(36**2 + 24**2)   # 43.267 mm
FT_TO_M = 0.3048

# ---------------------------------------------------------------------------
# B1 — Emlid CSV Parser
# ---------------------------------------------------------------------------

def parse_emlid_csv(csv_path: str) -> List[dict]:
    """
    Parse an Emlid Reach CSV and return a list of GCP dicts.

    Returns only FIX-quality points. Each dict has:
        label          : str   (Name column)
        lat            : float (WGS84 degrees)
        lon            : float (WGS84 degrees)
        ellip_alt_m    : float (WGS84 ellipsoidal height, metres)
        easting        : float (projected X in CRS units)
        northing       : float (projected Y in CRS units)
        elevation      : float (projected Z / geoid height in CRS units)
        cs_name        : str   (CRS description from CS name column)

    Raises ValueError if no FIX points are found.
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
            # Filter: FIX only
            if col_status and row.get(col_status, '').strip().upper() != 'FIX':
                continue

            try:
                gcp = {
                    'label':       (row.get(col_name) or '').strip(),
                    'lat':         float(row[col_lat]),
                    'lon':         float(row[col_lon]),
                    'ellip_alt_m': float(row[col_ellip]) * FT_TO_M if col_ellip else None,
                    'easting':     float(row[col_east])  if col_east  else None,
                    'northing':    float(row[col_north]) if col_north else None,
                    'elevation':   float(row[col_elev])  if col_elev  else None,
                    'cs_name':     (row.get(col_cs) or '').strip(),
                }
            except (ValueError, KeyError) as e:
                continue   # skip malformed rows

            if not gcp['label']:
                gcp['label'] = f"gcp_{len(gcps)}"

            gcps.append(gcp)

    if not gcps:
        raise ValueError(
            f"No FIX-quality GCP points found in {csv_path}. "
            "Check that the Solution status column contains 'FIX' rows."
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

    with Pool(threads) as pool:
        for result in pool.imap_unordered(_footprint_worker, task_args):
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

def project_pixel_mode_a(exif: dict, gcp: dict) -> Optional[Tuple[float, float]]:
    """
    Project GCP world coordinates to pixel (px, py) using EXIF-only camera model.

    Uses nadir pinhole projection with GimbalYaw rotation and GimbalRoll flip.
    Validated: mean 73.6 px error at 7–99 m AGL (see docs/dji_m3e_camera_model.md).

    Returns (px, py) in image pixel space, or None if GCP is behind/out of frame.
    """
    cam_lat   = exif['lat']
    cam_lon   = exif['lon']
    cam_alt   = exif.get('abs_alt')       # WGS84 ellipsoidal, metres
    yaw_deg   = exif.get('gimbal_yaw')
    roll_deg  = exif.get('gimbal_roll')
    focal_mm  = exif.get('focal_mm')
    focal35   = exif.get('focal35_mm')
    img_w     = exif.get('img_w')
    img_h     = exif.get('img_h')

    gcp_lat      = gcp['lat']
    gcp_lon      = gcp['lon']
    gcp_alt      = gcp.get('ellip_alt_m')  # WGS84 ellipsoidal, metres

    # Require all essential parameters
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
    dU = gcp_alt - cam_alt   # negative when GCP is below camera

    if dU >= 0:
        return None  # GCP at or above camera

    # --- Camera axes in ENU (nadir, pitch = −90°) ---
    psi = math.radians(yaw_deg)
    Xx, Xy =  math.cos(psi), -math.sin(psi)   # X_cam (image right)
    Yx, Yy = -math.sin(psi), -math.cos(psi)   # Y_cam (image down)

    if abs(roll_deg - 180.0) < 1.0:
        # Roll=180: camera rotated 180° around optical axis; flip both axes
        Xx, Xy, Yx, Yy = -Xx, -Xy, -Yx, -Yy

    # --- Camera-frame coordinates ---
    cam_x = Xx * dE + Xy * dN
    cam_y = Yx * dE + Yy * dN
    cam_z = -dU   # positive into scene

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

def _write_gcpeditpro(gcps: List[dict]) -> str:
    """
    Write gcpeditpro.txt content from GCP list.

    Format (space-separated):
        easting northing elevation gcp_label
    """
    lines = []
    for gcp in gcps:
        e = gcp.get('easting')
        n = gcp.get('northing')
        z = gcp.get('elevation')
        label = gcp['label']
        if e is None or n is None or z is None:
            continue
        lines.append(f"{e} {n} {z} {label}")
    return '\n'.join(lines) + '\n' if lines else ''


def _write_estimates_json(
        image_to_gcps: Dict[str, List[str]],
        exif_map: Dict[str, dict],
        gcp_by_label: Dict[str, dict],
        mode: str = 'exif') -> str:
    """
    Build estimates JSON: {gcpLabel: {imgFilename: {px, py, mode}}}

    Only includes entries where projection succeeded (px/py not None).
    """
    estimates: Dict[str, Dict[str, dict]] = {}

    for fname, gcp_labels in image_to_gcps.items():
        exif = exif_map.get(fname)
        if exif is None:
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
            estimates[label][fname] = {'px': px, 'py': py, 'mode': mode}

    return json.dumps(estimates, indent=2)


def run_pipeline(images_dir: str,
                 emlid_csv_path: str,
                 reconstruction_path: Optional[str] = None,
                 fallback_radius_m: float = 50.0,
                 threads: int = 0) -> Tuple[str, str]:
    """
    Full pipeline: B1 → B2 → B3.

    Returns (gcpeditpro_txt_content, estimates_json_content).

    If reconstruction_path is provided and valid, Mode B projection is used
    for images that have a matching shot in the reconstruction. All remaining
    images use Mode A (EXIF-based).
    """
    # B1 — Parse inputs
    print("Parsing Emlid CSV...")
    gcps = parse_emlid_csv(emlid_csv_path)
    print(f"  {len(gcps)} FIX GCPs")

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

    # B3 — Build outputs
    print("Projecting pixels and writing outputs...")
    gcpeditpro_txt = _write_gcpeditpro(gcps)

    if reconstruction:
        ref = reconstruction['reference_lla']
        shots = reconstruction.get('shots', {})
        cameras = reconstruction.get('cameras', {})
        # Use Mode B where available, fall back to Mode A
        estimates_b: Dict[str, Dict[str, dict]] = {}
        for fname, gcp_labels in image_to_gcps.items():
            exif = exif_map.get(fname)
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
                    if label not in estimates_b:
                        estimates_b[label] = {}
                    estimates_b[label][fname] = {'px': px, 'py': py, 'mode': mode_used}
        estimates_json = json.dumps(estimates_b, indent=2)
    else:
        estimates_json = _write_estimates_json(
            image_to_gcps, exif_map, gcp_by_label, mode='exif'
        )

    return gcpeditpro_txt, estimates_json


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='GCP estimation pipeline: Emlid CSV + drone images → gcpeditpro.txt + estimates.json'
    )
    parser.add_argument('emlid_csv',  help='Emlid CSV file path')
    parser.add_argument('image_dir',  help='Directory of drone images')
    parser.add_argument('--reconstruction', default=None,
                        help='Path to opensfm/reconstruction.json (enables Mode B)')
    parser.add_argument('--out-dir',  default='.',
                        help='Output directory for gcpeditpro.txt and estimates.json (default: .)')
    parser.add_argument('--radius',   type=float, default=50.0,
                        help='Fallback footprint radius in metres (default 50)')
    parser.add_argument('--threads',  type=int,   default=0,
                        help='Worker threads (default: all CPUs)')
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
        gcpeditpro_txt, estimates_json = run_pipeline(
            images_dir=args.image_dir,
            emlid_csv_path=args.emlid_csv,
            reconstruction_path=args.reconstruction,
            fallback_radius_m=args.radius,
            threads=args.threads,
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gcp_out = out_dir / 'gcpeditpro.txt'
        est_out = out_dir / 'gcpeditpro.estimates.json'

        gcp_out.write_text(gcpeditpro_txt)
        est_out.write_text(estimates_json)

        print(f'\nWrote {gcp_out}')
        print(f'Wrote {est_out}')
