#!/usr/bin/env python3
"""
emlid2gcp — Emlid CSV + drone images → GCP pixel estimates.

Stages:
  B1  parse_emlid_csv()           Parse Emlid Reach CSV (all solution statuses).
  B1  read_image_exif_batch()     Batch exiftool read for all images in parallel.
  B1  match_images_to_gcps()      Footprint-based image↔GCP association.
  B2  project_pixel_mode_a()      EXIF-based pinhole projection (nadir + oblique).
  B2  project_pixel_mode_b()      reconstruction.json-based projection (optional).
  B3  refine_all_estimates()      Color-based pixel refinement + marker bbox (optional).
                                   (lives in coloredX.py)
  B3  run_pipeline()              Full pipeline: B1 → B2 → B3 → write outputs.
"""

import csv
import json
import math
import subprocess
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
    # Pre-process: a lone " as a field value (e.g. from Emlid Flow note entry)
    # breaks csv.DictReader by starting an unclosed quoted field.  Strip them.
    import re as _re
    raw = open(csv_path, encoding='utf-8-sig').read()
    raw = _re.sub(r',"\n', ',\n', raw)   # lone " at end of field before newline
    import io as _io

    gcps = []
    with _io.StringIO(raw) as f:
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
            except (ValueError, KeyError, TypeError):
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

    if threads == 1 or len(task_args) <= 1:
        # Sequential path: no pool overhead, fork-safe for gunicorn/web contexts
        for args in task_args:
            result = _footprint_worker(args)
            if result is not None:
                fname, labels = result
                image_to_gcps[fname] = labels
    else:
        # ThreadPoolExecutor: threads are fork-safe (no fork-of-fork risk), and
        # avoid multiprocessing.Pool deadlocks inside gunicorn/preloaded workers.
        # For CPU-bound footprint math the GIL limits true parallelism, but for
        # typical dataset sizes this is negligible.
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for result in executor.map(_footprint_worker, task_args):
                if result is not None:
                    fname, labels = result
                    image_to_gcps[fname] = labels

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
    # Emlid BSN CSVs append a revision suffix like "(5)" — strip it before lookup.
    import re as _re
    def _strip_revision(s):
        return _re.sub(r'\s*\(\d+\)\s*$', '', s).strip()

    horiz = _strip_revision(cs_name.split(' + ')[0].strip())
    candidates = [horiz, cs_name.split(' + ')[0].strip(), cs_name]
    for candidate in candidates:
        try:
            crs = CRS.from_user_input(candidate)
            epsg = crs.to_epsg()
            if epsg:
                return f'EPSG:{epsg}'
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Output ordering helpers
# ---------------------------------------------------------------------------

def _gcp_distance_m(g1: dict, g2: dict) -> float:
    """Flat-earth distance between two GCPs in metres, using lat/lon."""
    mid_lat = math.radians((g1['lat'] + g2['lat']) / 2)
    dE = (g1['lon'] - g2['lon']) * METERS_PER_DEG_LAT * math.cos(mid_lat)
    dN = (g1['lat'] - g2['lat']) * METERS_PER_DEG_LAT
    return math.sqrt(dE**2 + dN**2)


def _separate_near_duplicates(
        gcps: List[dict],
        tolerance_m: float) -> Tuple[List[dict], List[Tuple[dict, dict, float]]]:
    """Split GCPs into (main, demoted) based on spatial proximity.

    Iterates in original order.  The first GCP of each near-duplicate cluster
    is kept in *main*; any subsequent GCP within *tolerance_m* of ANY accepted
    main GCP is moved to *demoted*.

    Returns:
        main    : GCPs to be passed to the structural sorter.
        demoted : List of (dup_gcp, closest_main_gcp, distance_m) triples, in
                  original-file order — appended at the end of the output.
    """
    main: List[dict] = []
    demoted: List[Tuple[dict, dict, float]] = []
    for gcp in gcps:
        closest_m: Optional[dict] = None
        closest_d = float('inf')
        for accepted in main:
            d = _gcp_distance_m(gcp, accepted)
            if d < closest_d:
                closest_d = d
                closest_m = accepted
        if closest_d < tolerance_m:
            demoted.append((gcp, closest_m, closest_d))
        else:
            main.append(gcp)
    return main, demoted


def _sort_gcps(gcps: List[dict], z_threshold: float) -> Tuple[List[dict], set]:
    """Return (priority_ordered_gcps, z_critical_labels).

    Structural priority:
      Slot 1 — most distal from centroid (sets one anchor of the bounding box)
      Slot 2 — most distal from slot 1   (defines global scale + orientation)
      Slot 3 — max elevation             (only when z_range > z_threshold * xy_diagonal)
      Slot 4 — min elevation             (same condition)
      Next   — closest to centroid       (anti-doming centre pin)
      Rest   — greedy farthest-from-any-selected (maximises spatial coverage)

    z_critical_labels: labels of the max/min elevation GCPs (used for image ordering).
    """
    if len(gcps) <= 1:
        return gcps[:], set()

    def _xy(g):
        if g.get('easting') is not None and g.get('northing') is not None:
            return g['easting'], g['northing']
        return g['lon'], g['lat']

    def _z(g):
        if g.get('elevation') is not None:
            return g['elevation']
        return g.get('ellip_alt_m') or 0.0

    xs = [_xy(g)[0] for g in gcps]
    ys = [_xy(g)[1] for g in gcps]
    zs = [_z(g) for g in gcps]

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    xy_diag = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2) or 1.0
    z_lo, z_hi = min(zs), max(zs)
    z_range = z_hi - z_lo
    z_significant = z_range > z_threshold * xy_diag

    # Z-critical: global elevation extremes (for image ordering, regardless of GCP slot).
    # Computed from the full set before any slot assignment so that a GCP placed early
    # (e.g. as the most-distal corner) is still identified as z_critical.
    z_critical_labels: set = set()
    if z_significant:
        z_critical_labels.add(max(gcps, key=lambda g: _z(g))['label'])
        z_critical_labels.add(min(gcps, key=lambda g: _z(g))['label'])

    def dist2d(ax, ay, bx, by):
        return math.sqrt((ax - bx)**2 + (ay - by)**2)

    # Pool entries: (gcp_dict, x, y, z, dist_from_centroid)
    pool = [
        (g, _xy(g)[0], _xy(g)[1], _z(g), dist2d(_xy(g)[0], _xy(g)[1], cx, cy))
        for g in gcps
    ]

    result = []

    def pick(pool, key_fn):
        """Pop and return the pool entry with the highest key_fn value."""
        idx = max(range(len(pool)), key=lambda i: key_fn(pool[i]))
        return pool.pop(idx)

    # Slot 1: most distal from centroid
    s1 = pick(pool, lambda item: item[4])
    result.append(s1)

    # Slot 2: most distal from slot 1
    if pool:
        s1x, s1y = s1[1], s1[2]
        s2 = pick(pool, lambda item: dist2d(item[1], item[2], s1x, s1y))
        result.append(s2)

    # Slots 3+4: Z extremes (only when Z variation is significant, and only if not
    # already placed in slots 1-2 as a distal point).
    # Composite score = z_extremity_norm + 0.5 * spatial_separation_norm so that
    # among GCPs near the elevation extreme the most spatially distinct one wins,
    # avoiding a wasted slot when the absolute Z-extreme is clustered with an
    # already-selected spatial extreme.
    _Z_SPATIAL_W = 0.5
    if z_significant and pool:
        def _zmax_score(item):
            z_norm = (item[3] - z_lo) / z_range
            d_norm = min(dist2d(item[1], item[2], r[1], r[2]) for r in result) / xy_diag
            return z_norm + _Z_SPATIAL_W * d_norm
        result.append(pick(pool, _zmax_score))
    if z_significant and pool:
        def _zmin_score(item):
            z_norm_inv = (z_hi - item[3]) / z_range
            d_norm = min(dist2d(item[1], item[2], r[1], r[2]) for r in result) / xy_diag
            return z_norm_inv + _Z_SPATIAL_W * d_norm
        result.append(pick(pool, _zmin_score))

    # Next: closest to centroid (anti-doming centre pin)
    if pool:
        centre = pick(pool, lambda item: -item[4])
        result.append(centre)

    # Rest: greedy farthest-from-any-selected.
    # At each step pick the unselected GCP whose minimum distance to any
    # already-selected GCP is the largest.  This maximises spatial coverage
    # across the survey area rather than simply re-visiting the perimeter,
    # which can cause clustering when the bounding box has elongated arms.
    while pool:
        idx = max(range(len(pool)),
                  key=lambda i: min(dist2d(pool[i][1], pool[i][2], r[1], r[2])
                                    for r in result))
        result.append(pool.pop(idx))

    return [item[0] for item in result], z_critical_labels


def _image_sort_score(px: float, py: float,
                      img_w: Optional[int], img_h: Optional[int],
                      gimbal_pitch: Optional[float],
                      z_critical: bool) -> float:
    """Return a sort score for one image (lower = higher priority).

    score = normalized_dist_from_center + nadir_weight * (0 if nadir else 1)

    nadir_weight = 1.0 for normal GCPs  → all nadir before any oblique
    nadir_weight = 0.3 for Z-critical   → well-centred obliques interleave with nadirs,
                                          naturally placing ~2-3 obliques in the top 8
    """
    if img_w and img_h:
        half_diag = math.sqrt(img_w**2 + img_h**2) / 2.0
        norm_dist = math.sqrt((px - img_w / 2.0)**2 + (py - img_h / 2.0)**2) / half_diag
    else:
        norm_dist = 0.5   # neutral fallback when dimensions unknown

    if gimbal_pitch is None:
        nadir_tier = 0    # assume nadir if pitch unavailable
    else:
        nadir_tier = 0 if abs(gimbal_pitch + 90.0) <= NADIR_TOL_DEG else 1

    nadir_weight = 0.3 if z_critical else 1.0
    return norm_dist + nadir_weight * nadir_tier


def _sort_images_for_gcp(img_map: dict, exif_map: dict, z_critical: bool) -> List[str]:
    """Return image filenames from img_map sorted by confidence score (best first)."""
    def score(fname):
        est  = img_map[fname]
        exif = exif_map.get(fname, {})
        return _image_sort_score(
            est['px'], est['py'],
            exif.get('img_w'), exif.get('img_h'),
            exif.get('gimbal_pitch'),
            z_critical,
        )
    return sorted(img_map.keys(), key=score)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_gcp_list(gcps: List[dict],
                    estimates: Dict[str, Dict[str, dict]],
                    sort_output: bool = True,
                    z_threshold: float = 0.05,
                    exif_map: Optional[Dict[str, dict]] = None,
                    dup_tolerance_m: float = 1.0,
                    omit_duplicates: bool = False) -> str:
    """
    Build gcp_list.txt content for GCPEditorPro / OpenDroneMap.

    Prefers projected coordinates (easting, northing, elevation) from the Emlid
    CSV over WGS-84 lat/lon, because GCPEditorPro expects the same projected
    coordinate system that the surveyor used.  The PROJ/EPSG header is derived
    from the Emlid CS name column via pyproj; falls back to the raw cs_name
    string, then to WGS-84 if projected coords are absent.

    Line 1: EPSG:xxxx or PROJ string
    Lines 2+: geo_x\tgeo_y\tgeo_z\tpx\tpy\timage_name\tgcp_label\tconfidence  (one per image)

    Tab-separated with trailing zeros stripped to match GCPEditorPro's download format,
    enabling plain diff comparison between pipeline output and confirmed GCP download.

    When sort_output=True, GCPs are ordered by structural priority (see _sort_gcps)
    and images within each GCP are ordered by confidence score (see _sort_images_for_gcp).
    When sort_output=False, GCPs appear in Emlid CSV order and images in match order.
    """
    def _fmt(v, decimals):
        """Format float to fixed decimals, stripping trailing zeros."""
        return f"{v:.{decimals}f}".rstrip('0').rstrip('.')

    gcp_by_label = {g['label']: g for g in gcps}

    # Prefer projected easting/northing/elevation when available
    use_projected = any(
        g.get('easting') is not None
        and g.get('northing') is not None
        and g.get('elevation') is not None
        for g in gcps
    )

    # Determine GCP order and which labels are Z-critical
    estimate_labels = set(estimates.keys())
    if sort_output:
        gcps_with_estimates = [g for g in gcps if g['label'] in estimate_labels]

        # Near-duplicate detection: demote co-located GCPs to end of list before sorting.
        if dup_tolerance_m > 0:
            gcps_main, demoted_triples = _separate_near_duplicates(
                gcps_with_estimates, dup_tolerance_m)
        else:
            gcps_main, demoted_triples = gcps_with_estimates, []

        sorted_gcps, z_critical_labels = _sort_gcps(gcps_main, z_threshold)

        if not omit_duplicates:
            sorted_gcps += [dup for dup, _, _ in demoted_triples]
    else:
        sorted_gcps = [g for g in gcps if g['label'] in estimate_labels]
        z_critical_labels = set()

    rows = []
    for gcp in sorted_gcps:
        gcp_label = gcp['label']
        img_map = estimates.get(gcp_label)
        if not img_map:
            continue
        if use_projected:
            x, y, z = gcp.get('easting'), gcp.get('northing'), gcp.get('elevation')
        else:
            x, y, z = gcp.get('lon'), gcp.get('lat'), gcp.get('ellip_alt_m')
        if None in (x, y, z):
            continue

        if sort_output and exif_map is not None:
            ordered_images = _sort_images_for_gcp(
                img_map, exif_map, gcp_label in z_critical_labels
            )
        else:
            ordered_images = list(img_map.keys())

        for img_name in ordered_images:
            est = img_map[img_name]
            cols = [
                _fmt(x, 3), _fmt(y, 3), _fmt(z, 3),
                _fmt(est['px'], 2), _fmt(est['py'], 2),
                img_name, gcp_label,
                est.get('confidence', 'projection'),
            ]
            if est.get('marker_bbox'):
                cols.append(est['marker_bbox'])
            rows.append('\t'.join(cols))

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


# ---------------------------------------------------------------------------
# GCP Classification (geo-12w)
# ---------------------------------------------------------------------------

def _classify_gcps(gcps: List[dict],
                   estimates: Dict[str, Dict[str, dict]],
                   sorted_labels: List[str],
                   n_control: int = 10) -> None:
    """
    Rename GCP labels in-place with GCP-* / CHK-* / DUP-* prefixes.

    sorted_labels: GCP labels in structural priority order (output of _sort_gcps,
                   with demoted near-duplicates appended at the end).

    Prefix assignment:
        Positions 1 .. n_control     → GCP-{base}   (control)
        Positions n_control+1 .. end → CHK-{base}   (check)

    Near-duplicates (GCPs not in sorted_labels) → DUP-{base}.

    Any existing GCP-*/CHK-*/DUP-* prefix is stripped before re-applying so
    that re-running the pipeline on already-prefixed labels is idempotent.
    """
    def _base(label: str) -> str:
        for pfx in ('GCP-', 'CHK-', 'DUP-'):
            if label.startswith(pfx):
                return label[len(pfx):]
        return label

    rename: Dict[str, str] = {}
    for rank, label in enumerate(sorted_labels, start=1):
        base = _base(label)
        if rank <= n_control:
            rename[label] = f'GCP-{base}'
        else:
            rename[label] = f'CHK-{base}'

    # Any GCP not in sorted_labels is a near-duplicate demoted outside the main
    # sort — give it DUP- prefix.
    for gcp in gcps:
        if gcp['label'] not in rename:
            rename[gcp['label']] = f'DUP-{_base(gcp["label"])}'

    # Apply to gcps list (in-place mutation of dicts)
    for gcp in gcps:
        gcp['label'] = rename.get(gcp['label'], gcp['label'])

    # Apply to estimates dict (rename keys)
    for old_label in list(estimates.keys()):
        new_label = rename.get(old_label)
        if new_label and new_label != old_label:
            estimates[new_label] = estimates.pop(old_label)

    n_ctrl = sum(1 for g in gcps if g['label'].startswith('GCP-'))
    n_chk  = sum(1 for g in gcps if g['label'].startswith('CHK-'))
    n_dup  = sum(1 for g in gcps if g['label'].startswith('DUP-'))
    print(f"  {n_ctrl} GCP-* (control),  {n_chk} CHK-* (check),  {n_dup} DUP-* (duplicate)")


# ---------------------------------------------------------------------------
# Stage 3 — Pixel Refinement (geo-56c)
# Extracted to coloredX.py; imported here for run_pipeline().
# ---------------------------------------------------------------------------

try:
    from .coloredX import refine_all_estimates as _refine_all_estimates
    from . import coloredX as _refine_module
except ImportError:
    from coloredX import refine_all_estimates as _refine_all_estimates
    import coloredX as _refine_module


def run_pipeline(images_dir: str,
                 emlid_csv_path: str,
                 reconstruction_path: Optional[str] = None,
                 fallback_radius_m: float = 50.0,
                 threads: int = 0,
                 nadir_only: bool = False,
                 sort_output: bool = True,
                 z_threshold: float = 0.05,
                 dup_tolerance_m: float = 1.0,
                 omit_duplicates: bool = False,
                 classify: bool = True,
                 n_control: int = 10,
                 refine_pixels: bool = True,
                 refine_limit: int = 0) -> Tuple[str, dict]:
    """
    Full pipeline: B1 → B2 → B3.

    Returns (gcp_txt_content, estimates_dict).

    If reconstruction_path is provided and valid, Mode B projection is used
    for images that have a matching shot in the reconstruction. All remaining
    images use Mode A (EXIF-based).

    nadir_only:    skip images whose gimbal pitch is not near -90°.
    sort_output:   order GCPs by structural priority and images by confidence score.
    z_threshold:   fraction of XY diagonal; Z extremes get priority slots only above
                   this ratio (default 0.05 = 5 %).
    classify:      rename GCP labels with GCP-*/CHK-*/DUP-* prefixes based on
                   structural sort order (geo-12w).  Requires sort_output=True.
    n_control:     number of top GCPs to label GCP-* (default 10); remainder → CHK-*.
    refine_pixels: run Stage-3 color-based pixel refinement after projection;
                   updates px/py to sub-pixel accuracy and adds marker_bbox column.
                   Requires opencv-python and numpy.
    refine_limit:  if > 0, refine only the first N (gcp, image) pairs in priority
                   order; useful for fast iteration during development.
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
    print("Estimating GCP locations via GCP,image projection geometry...")
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

    # Pre-sort: compute structural GCP order so that classification and
    # priority-aware refinement use the same ordering that _write_gcp_list
    # will use.  (_write_gcp_list will sort again — harmless redundancy.)
    sorted_gcps_order: List[dict] = []
    demoted_triples_presort: List[Tuple] = []
    if sort_output:
        print("Sorting GCPs by structural priority...")
        gcps_with_est = [g for g in gcps if g['label'] in estimates]
        if dup_tolerance_m > 0:
            gcps_main_pre, demoted_triples_presort = _separate_near_duplicates(
                gcps_with_est, dup_tolerance_m)
            if demoted_triples_presort:
                action = "omitted (--omit-duplicates)" if omit_duplicates else "labeled as duplicate and put at bottom of list"
                print(f"  WARNING: {len(demoted_triples_presort)} GCP(s) within "
                      f"{dup_tolerance_m:.1f} m of another — {action}")
        else:
            gcps_main_pre, demoted_triples_presort = gcps_with_est, []
        sorted_gcps_order, _ = _sort_gcps(gcps_main_pre, z_threshold)

    def _sorted_labels() -> List[str]:
        """Current labels of sorted_gcps_order + demoted, reflecting any renames."""
        return ([g['label'] for g in sorted_gcps_order] +
                [dup['label'] for dup, _, _ in demoted_triples_presort])

    # Classification (geo-12w): rename labels with GCP-*/CHK-*/DUP-* prefixes
    if classify:
        if not sorted_gcps_order:
            print("  WARNING: --classify requires sort_output; skipping classification.")
        else:
            # Pass only the non-demoted sorted labels; demoted triples are not in
            # this list so they fall through to the DUP-* path in _classify_gcps().
            _classify_gcps(gcps, estimates,
                           [g['label'] for g in sorted_gcps_order], n_control)
            # sorted_gcps_order dicts were mutated in-place; _sorted_labels() now
            # returns the renamed labels automatically.

    # B3 — Stage 3 pixel refinement (optional)
    if refine_pixels:
        estimates = _refine_all_estimates(
            estimates, exif_map, threads=threads,
            gcp_order=_sorted_labels() if sorted_gcps_order else None,
            refine_limit=refine_limit,
        )

    # B3 — Write outputs
    gcp_txt = _write_gcp_list(
        gcps, estimates,
        sort_output=sort_output,
        z_threshold=z_threshold,
        exif_map=exif_map,
        dup_tolerance_m=dup_tolerance_m,
        omit_duplicates=omit_duplicates,
    )

    return gcp_txt, estimates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='emlid2gcp: Emlid CSV + drone images → gcpeditpro.txt + pix4d.txt'
    )
    parser.add_argument('emlid_csv',  help='Emlid CSV file path')
    parser.add_argument('image_dir',  help='Directory of drone images')
    parser.add_argument('--reconstruction', default=None,
                        help='Path to opensfm/reconstruction.json (enables Mode B)')
    parser.add_argument('--out-dir',  default='.',
                        help='Output directory for gcpeditpro.txt and pix4d.txt (default: .)')
    parser.add_argument('--radius',   type=float, default=50.0,
                        help='Fallback footprint radius in metres (default 50)')
    parser.add_argument('--threads',  type=int,   default=0,
                        help='Worker threads (default: all CPUs)')
    parser.add_argument('--nadir-only', action='store_true',
                        help=f'Skip oblique images (gimbal pitch not within {NADIR_TOL_DEG}° of -90°)')
    parser.add_argument('--no-sort', action='store_true',
                        help='Output GCPs in Emlid CSV order and images in match order (disables structural priority sorting)')
    parser.add_argument('--z-threshold', type=float, default=0.05,
                        help='Fraction of XY diagonal; Z-extreme GCPs get priority slots only when Z range exceeds this ratio (default 0.05 = 5%%)')
    parser.add_argument('--dup-tolerance', type=float, default=1.0,
                        help='GCPs within this distance (metres) of a higher-priority GCP are '
                             'demoted to the end of the output list. Set to 0 to disable. (default: 1.0)')
    parser.add_argument('--omit-duplicates', action='store_true',
                        help='Omit near-duplicate GCPs from the output entirely instead of demoting them to the end')
    parser.add_argument('--no-classify', dest='classify', action='store_false',
                        help='Disable GCP-*/CHK-*/DUP-* label classification (classification runs by default)')
    parser.add_argument('--n-control', type=int, default=10,
                        help='Number of top-priority GCPs to label GCP-*; remainder → CHK-* (default 10)')
    parser.add_argument('--no-coloredX', dest='refine_pixels', action='store_false',
                        help='Disable color-based pixel refinement and marker bounding-box computation '
                             '(refinement runs by default; requires opencv-python and numpy)')
    parser.add_argument('--refine-limit', type=int, default=0,
                        help='Refine at most N (gcp,image) pairs in priority order; '
                             '0 = no limit (default). Useful for fast iteration during development.')
    parser.add_argument('--seed-dist-penalty', type=float, default=None,
                        help=f'Override _SEED_DIST_PENALTY (k in score /= 1+k*d). '
                             f'Default: {_refine_module._SEED_DIST_PENALTY}')
    parser.add_argument('--marker-size', type=float, default=None,
                        help=f'Expected physical marker size in metres for GSD-based '
                             f'size filtering. Default: {_refine_module._MARKER_SIZE_M}')
    parser.add_argument('--out-name', default='gcp_list.txt',
                        help='Filename for the gcp_list output (default: gcp_list.txt)')
    parser.set_defaults(classify=True, refine_pixels=True)
    parser.add_argument('--match-only', action='store_true',
                        help='Run image-to-GCP footprint matching only and print results without projecting pixels or writing files')
    args = parser.parse_args()

    if args.match_only:
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
        if args.seed_dist_penalty is not None:
            _refine_module._SEED_DIST_PENALTY = args.seed_dist_penalty
        if args.marker_size is not None:
            _refine_module._MARKER_SIZE_M = args.marker_size

        gcp_txt, estimates = run_pipeline(
            images_dir=args.image_dir,
            emlid_csv_path=args.emlid_csv,
            reconstruction_path=args.reconstruction,
            fallback_radius_m=args.radius,
            threads=args.threads,
            nadir_only=args.nadir_only,
            sort_output=not args.no_sort,
            z_threshold=args.z_threshold,
            dup_tolerance_m=args.dup_tolerance,
            omit_duplicates=args.omit_duplicates,
            classify=args.classify,
            n_control=args.n_control,
            refine_pixels=args.refine_pixels,
            refine_limit=args.refine_limit,
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gcp_out   = out_dir / args.out_name
        pix4d_out = out_dir / 'marks.csv'

        gcp_out.write_text(gcp_txt)
        pix4d_out.write_text(_write_pix4d(estimates))

        print(f'\nWrote {gcp_out}')
        print(f'Wrote {pix4d_out}')
