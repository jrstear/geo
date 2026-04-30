#!/usr/bin/env python3
"""
sight — Survey CSV + drone images → GCP pixel estimates.

Stages:
  B1  parse_survey_csv()           Parse a survey CSV (all solution statuses).
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
import re
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
FT_TO_M = 0.3048006096012192   # US survey foot
NADIR_TOL_DEG = 10.0   # pitch within 10° of -90° is treated as nadir

# ---------------------------------------------------------------------------
# B1 — Emlid CSV Parser
# ---------------------------------------------------------------------------

def parse_survey_csv(csv_path: str, fallback_crs: Optional[str] = None) -> List[dict]:
    """
    Parse a survey CSV (or any survey CSV) and return a list of GCP dicts.

    Column detection is header-based and case-insensitive substring matching, so
    column order and extra columns do not matter.

    Minimum required columns (one of):
      • Latitude + Longitude  (WGS84 degrees; ellipsoidal height used if present)
      • Easting + Northing    (projected; CRS taken from CS name column or fallback_crs)

    When only projected coordinates are present, lat/lon are derived via pyproj.

    All points are returned regardless of Solution status (FIX, FLOAT, SINGLE).
    The status is preserved in the 'solution_status' key so callers can filter
    or annotate as needed. Each dict has:
        label           : str   (Name column, or auto-generated gcp_N)
        lat             : float (WGS84 degrees, derived if not in CSV)
        lon             : float (WGS84 degrees, derived if not in CSV)
        ellip_alt_m     : float or None (WGS84 ellipsoidal height, metres)
        easting         : float or None (projected X in CRS units)
        northing        : float or None (projected Y in CRS units)
        elevation       : float or None (projected Z / geoid height in CRS units)
        cs_name         : str   (CRS description from CS name column, or '')
        solution_status : str   (e.g. 'FIX', 'FLOAT', 'SINGLE', or '')

    Raises ValueError if no valid rows are found or lat/lon cannot be determined.
    """
    import re as _re
    import io as _io

    # A lone " as a field value (e.g. from an Emlid Flow note entry) produces
    # ,"\n in the raw CSV, which breaks csv.DictReader by opening an unclosed
    # quoted field.  Strip any such trailing lone-quote before parsing.
    raw = open(csv_path, encoding='utf-8-sig').read()
    raw = _re.sub(r',"\n', ',\n', raw)

    gcps = []
    headers: List[str] = []
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

        for row in reader:
            try:
                gcp = {
                    'label':           (row.get(col_name) or '').strip(),
                    'lat':             float(row[col_lat]) if col_lat and row.get(col_lat) else None,
                    'lon':             float(row[col_lon]) if col_lon and row.get(col_lon) else None,
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

    # Derive lat/lon from easting/northing for any row that lacks them
    needs_latlon = [g for g in gcps if g['lat'] is None]
    if needs_latlon:
        try:
            from pyproj import Transformer as _Transformer
            _xfm_cache: dict = {}
            for g in needs_latlon:
                if g['easting'] is None or g['northing'] is None:
                    continue
                crs_src = g['cs_name'] or fallback_crs or ''
                # Try cs_name resolution first; fall back to --crs override if
                # the cs_name string is non-empty but unrecognised by
                # _cs_name_to_epsg (e.g. NAD27-era datums or vertical-suffixed
                # compound names that pyproj's name lookup misses).
                epsg = _cs_name_to_epsg(crs_src) if crs_src else None
                if not epsg:
                    epsg = fallback_crs or None
                if not epsg:
                    continue
                if epsg not in _xfm_cache:
                    _xfm_cache[epsg] = _Transformer.from_crs(epsg, 'EPSG:4326', always_xy=True)
                lon, lat = _xfm_cache[epsg].transform(g['easting'], g['northing'])
                g['lat'] = lat
                g['lon'] = lon
                # Backfill cs_name so _write_gcp_list can emit the correct CRS header
                if not g['cs_name']:
                    g['cs_name'] = epsg
        except ImportError:
            pass

    still_missing = [g['label'] for g in gcps if g['lat'] is None]
    if still_missing:
        hint = (
            "Provide Latitude/Longitude columns, or Easting+Northing with a "
            "CS name column or --crs EPSG:xxxx."
        )
        raise ValueError(
            f"{len(still_missing)} point(s) have no lat/lon and it could not be "
            f"derived: {still_missing}. {hint}"
        )

    # Derive ellipsoidal altitude from elevation for rows that have no explicit
    # Ellipsoidal Height column.  Two-step conversion:
    #   1. elevation (CRS native units) → metres via pyproj CRS linear unit factor
    #   2. orthometric metres → WGS84 ellipsoidal metres via NAVD88 geoid grid
    #      (us_noaa_g2018u0.tif, ships with pyproj-data).  Falls back to raw
    #      elevation in metres if the grid is unavailable (prints a warning).
    needs_ellip = [g for g in gcps
                   if g['ellip_alt_m'] is None and g['elevation'] is not None
                   and g['lat'] is not None]
    if needs_ellip:
        import sys as _sys
        try:
            from pyproj import CRS as _CRS, Transformer as _Transformer
            _unit_cache: dict = {}
            _vshift: Optional[object] = None
            _vshift_ok: Optional[bool] = None   # None = untried
            for g in needs_ellip:
                crs_src = g['cs_name'] or fallback_crs or ''
                epsg = _cs_name_to_epsg(crs_src) if crs_src else None
                if not epsg:
                    epsg = fallback_crs or None
                # Get linear unit factor (metres per CRS unit)
                if epsg not in _unit_cache:
                    try:
                        crs_obj = _CRS.from_user_input(epsg)
                        _unit_cache[epsg] = crs_obj.axis_info[0].unit_conversion_factor
                    except Exception:
                        _unit_cache[epsg] = 1.0
                elev_m = g['elevation'] * _unit_cache.get(epsg, 1.0)
                # Apply geoid correction once (lazy init)
                if _vshift_ok is None:
                    try:
                        _vshift = _Transformer.from_pipeline(
                            '+proj=vgridshift +grids=us_noaa_g2018u0.tif +multiplier=1')
                        _vshift_ok = True
                    except Exception:
                        _vshift_ok = False
                        print(
                            "  WARNING: NAVD88 geoid grid (us_noaa_g2018u0.tif) not found; "
                            "using orthometric elevation as approximate ellipsoidal height "
                            "(error ~10–30 m depending on location — pixel estimates will be "
                            "less accurate but pipeline will run).",
                            file=_sys.stderr,
                        )
                if _vshift_ok and _vshift is not None:
                    _, _, ellip_m = _vshift.transform(g['lon'], g['lat'], elev_m)
                    g['ellip_alt_m'] = ellip_m
                else:
                    g['ellip_alt_m'] = elev_m
        except ImportError:
            pass

    # Deduplicate labels: if two survey points share the same name (e.g. Emlid
    # auto-increment failure), rename 2nd occurrence to label-1, 3rd to label-2,
    # etc. — matching Pix4D behaviour so downstream tools see unique IDs.
    import sys as _sys
    _seen: dict = {}
    for g in gcps:
        orig = g['label']
        if orig not in _seen:
            _seen[orig] = 0
        else:
            _seen[orig] += 1
            new_label = f"{orig}-{_seen[orig]}"
            print(
                f"WARNING: same label {orig!r} on distinct coordinates — "
                f"renamed to {new_label!r} (occurrence {_seen[orig] + 1})",
                file=_sys.stderr,
            )
            g['label'] = new_label

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


def _load_cameras(path: str) -> dict:
    """
    Load a cameras.json file (ODM format) and return {(width, height): model_params}
    for Brown-model cameras only.  Key is the integer (w, h) tuple so callers can
    look up by image dimensions from EXIF.
    """
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    result = {}
    for params in raw.values():
        if params.get('projection_type') == 'brown':
            w, h = params.get('width'), params.get('height')
            if w and h:
                result[(int(w), int(h))] = params
    return result


def project_pixel_mode_a(exif: dict, gcp: dict,
                          camera_model: Optional[dict] = None) -> Optional[Tuple[float, float]]:
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
    (see docs/details/dji_m3e_camera_model.md).

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

    # Base requirements always needed
    if any(v is None for v in [cam_alt, gcp_alt, yaw_deg, roll_deg, img_w, img_h]):
        return None
    # EXIF focal length only needed when no calibrated model is available
    if camera_model is None and any(v is None for v in [focal_mm, focal35]):
        return None

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

    # --- Projection ---
    if camera_model is not None:
        # Calibrated Brown model: focal/principal point from cameras.json, with distortion
        w_max = max(img_w, img_h)
        fx = camera_model['focal_x'] * w_max
        fy = camera_model['focal_y'] * w_max
        cx = img_w / 2.0 + camera_model.get('c_x', 0.0) * w_max
        cy = img_h / 2.0 + camera_model.get('c_y', 0.0) * w_max
        xn = cam_x / cam_z
        yn = cam_y / cam_z
        r2 = xn*xn + yn*yn
        k1 = camera_model.get('k1', 0.0)
        k2 = camera_model.get('k2', 0.0)
        k3 = camera_model.get('k3', 0.0)
        p1 = camera_model.get('p1', 0.0)
        p2 = camera_model.get('p2', 0.0)
        radial = 1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        dx = 2.0*p1*xn*yn + p2*(r2 + 2.0*xn*xn)
        dy = p1*(r2 + 2.0*yn*yn) + 2.0*p2*xn*yn
        px = fx * (xn*radial + dx) + cx
        py = fy * (yn*radial + dy) + cy
    else:
        # EXIF-based pinhole (no distortion correction)
        scale = focal_mm / focal35
        sensor_diag = FULL_FRAME_DIAG_MM * scale
        aspect = img_w / img_h
        sensor_h = sensor_diag / math.sqrt(1 + aspect**2)
        sensor_w = sensor_h * aspect
        fx = focal_mm * img_w / sensor_w
        fy = focal_mm * img_h / sensor_h
        cx, cy = img_w / 2.0, img_h / 2.0
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

    # Intrinsics — OpenSfM uses two camera model conventions:
    #   perspective: 'focal' (one value, normalised by max(w,h))
    #   brown:       'focal_x'/'focal_y' + principal point 'c_x'/'c_y'
    #                + full Brown distortion k1,k2,k3,p1,p2
    w, h = camera['width'], camera['height']
    mwh = max(w, h)

    if 'focal' in camera:
        # Simple perspective model
        fx = fy = camera['focal'] * mwh
        cx_off = cy_off = 0.0
        k1 = camera.get('k1', 0.0)
        k2 = camera.get('k2', 0.0)
        k3 = p1 = p2 = 0.0
    else:
        # Brown model
        fx = camera['focal_x'] * mwh
        fy = camera['focal_y'] * mwh
        cx_off = camera.get('c_x', 0.0) * mwh
        cy_off = camera.get('c_y', 0.0) * mwh
        k1 = camera.get('k1', 0.0)
        k2 = camera.get('k2', 0.0)
        k3 = camera.get('k3', 0.0)
        p1 = camera.get('p1', 0.0)
        p2 = camera.get('p2', 0.0)

    # Normalized image coords + Brown distortion (radial + tangential)
    xn = p_cam[0] / p_cam[2]
    yn = p_cam[1] / p_cam[2]
    r2 = xn**2 + yn**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
    yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn

    px = fx * xd + cx_off + w / 2.0
    py = fy * yd + cy_off + h / 2.0

    if 0 <= px < w and 0 <= py < h:
        return (px, py)
    return None


# ---------------------------------------------------------------------------
# Iterative refinement: triangulate target 3D from color rays, re-project
# to projection-only images, optionally re-run color refinement using the
# improved seed. Produces four confidence labels: projection / tri_proj /
# color / tri_color. See docs/plans/tag-quality-consistency.md.
# ---------------------------------------------------------------------------

def _triangulate_target(target_gcp: dict,
                        color_obs: List[Tuple[str, dict, dict]],
                        cameras_by_wh: Optional[dict] = None
                        ) -> Optional[dict]:
    """
    Find the 3D point that, when forward-projected through each color
    observation's EXIF camera, best matches the color-refined pixel.

    color_obs: list of (fname, est_dict_with_px_py, exif_dict).
    Returns dict with refined lat/lon/ellip_alt_m + per-image residuals,
    or None if optimisation cannot run (fewer than 3 obs, missing scipy,
    no surveyed alt, or solver failure).

    Optimisation is in (dE, dN, dU) metres relative to the surveyed point;
    gives a well-scaled problem for least_squares.  Uses the existing
    forward projection (project_pixel_mode_a) so distortion handling
    matches the rest of the pipeline.
    """
    if len(color_obs) < 3:
        return None
    surv_lat = target_gcp.get('lat')
    surv_lon = target_gcp.get('lon')
    surv_alt = target_gcp.get('ellip_alt_m')
    if surv_lat is None or surv_lon is None or surv_alt is None:
        return None
    try:
        from scipy.optimize import least_squares  # type: ignore
    except ImportError:
        return None

    cos_lat = math.cos(math.radians(surv_lat))
    if abs(cos_lat) < 1e-9:
        return None

    def _residuals(delta_enu):
        dE, dN, dU = float(delta_enu[0]), float(delta_enu[1]), float(delta_enu[2])
        gcp_p = dict(target_gcp)
        gcp_p['lat'] = surv_lat + dN / METERS_PER_DEG_LAT
        gcp_p['lon'] = surv_lon + dE / (METERS_PER_DEG_LAT * cos_lat)
        gcp_p['ellip_alt_m'] = surv_alt + dU
        out: List[float] = []
        for _fname, est, exif in color_obs:
            cam_model = None
            if cameras_by_wh:
                cam_model = cameras_by_wh.get((exif.get('img_w'), exif.get('img_h')))
            proj = project_pixel_mode_a(exif, gcp_p, cam_model)
            if proj is None:
                # Out-of-frame penalty.  Large but finite so the solver can still progress.
                out.extend([1e4, 1e4])
            else:
                out.append(proj[0] - est['px'])
                out.append(proj[1] - est['py'])
        return out

    try:
        res = least_squares(_residuals, x0=[0.0, 0.0, 0.0],
                            method='lm', max_nfev=200)
    except Exception:
        return None
    if not getattr(res, 'success', False):
        return None

    # Per-image residual magnitudes at the solution
    final = res.fun
    per_img: List[Tuple[str, float]] = []
    for i, (fname, _est, _exif) in enumerate(color_obs):
        rx = final[2 * i]
        ry = final[2 * i + 1]
        per_img.append((fname, math.hypot(rx, ry)))
    if not per_img:
        return None

    return {
        'lat':           surv_lat + float(res.x[1]) / METERS_PER_DEG_LAT,
        'lon':           surv_lon + float(res.x[0]) / (METERS_PER_DEG_LAT * cos_lat),
        'ellip_alt_m':   surv_alt + float(res.x[2]),
        'per_img':       per_img,
        'max_resid_px':  max(d for _, d in per_img),
        'mean_resid_px': sum(d for _, d in per_img) / len(per_img),
        'surv_disagree_m': math.sqrt(float(res.x[0])**2 + float(res.x[1])**2 + float(res.x[2])**2),
        'n_used':        len(color_obs),
    }


def _triangulate_robust(target_gcp: dict,
                        color_obs: List[Tuple[str, dict, dict]],
                        cameras_by_wh: Optional[dict] = None,
                        eps_resid_px: float = 20.0,
                        max_drops: int = 2
                        ) -> Optional[dict]:
    """
    Triangulate, then drop the worst-fitting observation up to *max_drops* times
    if the max per-image residual still exceeds *eps_resid_px*.  Each drop
    requires that >=3 observations remain.  Returns the best result obtained,
    along with a 'dropped' list of fnames.
    """
    obs = list(color_obs)
    dropped: List[str] = []
    best = _triangulate_target(target_gcp, obs, cameras_by_wh=cameras_by_wh)
    if best is None:
        return None
    drops = 0
    while best['max_resid_px'] > eps_resid_px and drops < max_drops and len(obs) > 3:
        worst_fname = max(best['per_img'], key=lambda t: t[1])[0]
        obs = [(f, e, x) for (f, e, x) in obs if f != worst_fname]
        dropped.append(worst_fname)
        drops += 1
        new_res = _triangulate_target(target_gcp, obs, cameras_by_wh=cameras_by_wh)
        if new_res is None:
            break
        best = new_res
    best['dropped'] = dropped
    return best


def _iterative_refine(
        estimates: Dict[str, Dict[str, dict]],
        gcp_by_label: Dict[str, dict],
        exif_map: Dict[str, dict],
        cameras_by_wh: Optional[dict] = None,
        *,
        min_color_hits: int = 3,
        eps_resid_px: float = 120.0,
        eps_surv_m: float = 15.0,
        max_outlier_drops: int = 2,
        threads: int = 0,
        report_path: Optional[Path] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Pass 1: For each target with >= min_color_hits 'color' refinements, triangulate
    a refined 3D position from the color rays.  Sanity gate: max per-image residual
    < eps_resid_px AND |triangulated - surveyed| < eps_surv_m.  When the gate
    passes, re-project the refined 3D through EXIF for projection-only images of
    the same target and label them 'tri_proj' (px/py replaced).

    Pass 2: Re-run color refinement on tri_proj images using the new pixel as
    the seed.  On success, label 'tri_color' and store the refined pixel.

    Sanity-gate failures are reported as suspect targets but do not modify any
    estimates.  Mutates *estimates* in place; returns the same dict.

    Per-target metrics (n_color, triangulation residual, survey disagreement,
    n_promoted_tri_proj, n_promoted_tri_color, gate flag) are printed at end of
    run; if *report_path* is provided, also written there as TSV.
    """
    try:
        from . import coloredX as _refine_module  # type: ignore
    except (ImportError, ValueError):
        try:
            import coloredX as _refine_module  # type: ignore
        except ImportError:
            _refine_module = None  # type: ignore

    print("Iterative refinement (Pass 1: triangulate color rays + Pass 2: re-refine)...")

    per_target_report: List[dict] = []

    # Determine deterministic order — iterate over labels currently in estimates
    labels = list(estimates.keys())

    for label in labels:
        # Skip near-duplicate labels (same convention as coloredX)
        if label.startswith('DUP-') or re.search(r'-dup\d*$', label):
            continue
        gcp = gcp_by_label.get(label)
        if gcp is None:
            continue
        img_map = estimates.get(label) or {}

        # Collect color observations: those with confidence == 'color'
        color_obs: List[Tuple[str, dict, dict]] = []
        proj_only_fnames: List[str] = []
        for fname, est in img_map.items():
            exif = exif_map.get(fname)
            if exif is None:
                continue
            if est.get('confidence') == 'color':
                color_obs.append((fname, est, exif))
            else:
                proj_only_fnames.append(fname)

        record = {
            'label':            label,
            'n_color':          len(color_obs),
            'n_proj_only':      len(proj_only_fnames),
            'tri_resid_max_px': None,
            'surv_disagree_m':  None,
            'gate':             'skipped',
            'n_dropped':        0,
            'n_promoted_proj':  0,
            'n_promoted_color': 0,
        }

        if len(color_obs) < min_color_hits:
            record['gate'] = 'skipped'  # not enough color hits
            per_target_report.append(record)
            continue

        tri = _triangulate_robust(
            gcp, color_obs, cameras_by_wh=cameras_by_wh,
            eps_resid_px=eps_resid_px, max_drops=max_outlier_drops,
        )
        if tri is None:
            record['gate'] = 'failed'
            per_target_report.append(record)
            continue

        record['tri_resid_max_px'] = tri['max_resid_px']
        record['surv_disagree_m']  = tri['surv_disagree_m']
        record['n_dropped']        = len(tri.get('dropped', []))

        # Sanity gates
        if tri['max_resid_px'] > eps_resid_px:
            record['gate'] = 'COLOR_INCONSISTENT'
            per_target_report.append(record)
            continue
        if tri['surv_disagree_m'] > eps_surv_m:
            record['gate'] = 'SURVEY_DISAGREES'
            per_target_report.append(record)
            continue

        # Gate passed → Pass 1: re-project through EXIF for projection-only images
        refined_gcp = dict(gcp)
        refined_gcp['lat']         = tri['lat']
        refined_gcp['lon']         = tri['lon']
        refined_gcp['ellip_alt_m'] = tri['ellip_alt_m']

        promoted_to_tri_proj: List[str] = []
        for fname in proj_only_fnames:
            exif = exif_map.get(fname)
            if exif is None:
                continue
            cam_model = None
            if cameras_by_wh:
                cam_model = cameras_by_wh.get((exif.get('img_w'), exif.get('img_h')))
            proj = project_pixel_mode_a(exif, refined_gcp, cam_model)
            if proj is None:
                continue
            est = img_map[fname]
            est['_pre_iter_px'] = est['px']
            est['_pre_iter_py'] = est['py']
            est['px']         = float(proj[0])
            est['py']         = float(proj[1])
            est['confidence'] = 'tri_proj'
            promoted_to_tri_proj.append(fname)

        record['n_promoted_proj']  = len(promoted_to_tri_proj)

        # Pass 2: re-run color refinement on the new tri_proj seeds
        promoted_to_tri_color: List[str] = []
        if _refine_module is not None and promoted_to_tri_proj:
            for fname in promoted_to_tri_proj:
                exif = exif_map.get(fname) or {}
                path = exif.get('path')
                if not path:
                    continue
                est = img_map[fname]
                gsd = _refine_module._compute_gsd(exif)
                result = _refine_module._refine_single(path, est['px'], est['py'], gsd_m=gsd)
                if result is not None:
                    est['px']          = float(result['px'])
                    est['py']          = float(result['py'])
                    est['confidence']  = 'tri_color'
                    if result.get('marker_bbox') is not None:
                        est['marker_bbox'] = result['marker_bbox']
                    promoted_to_tri_color.append(fname)

        record['n_promoted_color'] = len(promoted_to_tri_color)
        record['gate'] = 'PASS'
        per_target_report.append(record)

    # Print summary
    n_total = len(per_target_report)
    n_pass = sum(1 for r in per_target_report if r['gate'] == 'PASS')
    n_susp = sum(1 for r in per_target_report if r['gate'] in ('COLOR_INCONSISTENT', 'SURVEY_DISAGREES'))
    n_skip = sum(1 for r in per_target_report if r['gate'] == 'skipped')
    n_fail = sum(1 for r in per_target_report if r['gate'] == 'failed')
    n_promoted_proj  = sum(r['n_promoted_proj']  for r in per_target_report)
    n_promoted_color = sum(r['n_promoted_color'] for r in per_target_report)
    print(f"  {n_total} target(s) examined: "
          f"{n_pass} pass, {n_susp} suspect, {n_skip} skipped (<{min_color_hits} color hits), {n_fail} failed.")
    print(f"  Promoted {n_promoted_proj} image(s) projection→tri_proj; "
          f"{n_promoted_color} of those further promoted tri_proj→tri_color.")

    suspects = [r for r in per_target_report
                if r['gate'] in ('COLOR_INCONSISTENT', 'SURVEY_DISAGREES')]
    if suspects:
        print(f"  Suspect targets (review carefully before tagging):")
        for r in suspects:
            tri_r = f"{r['tri_resid_max_px']:.1f} px" if r['tri_resid_max_px'] is not None else '—'
            sdis  = f"{r['surv_disagree_m']:.2f} m" if r['surv_disagree_m'] is not None else '—'
            print(f"    {r['label']:14s}  n_color={r['n_color']:2d}  "
                  f"max_resid={tri_r:>10s}  survey_disagree={sdis:>8s}  "
                  f"flag={r['gate']}")

    if report_path is not None:
        try:
            with open(report_path, 'w') as f:
                f.write("label\tn_color\tn_proj_only\ttri_resid_max_px\tsurv_disagree_m\t"
                        "n_dropped\tn_promoted_proj\tn_promoted_color\tgate\n")
                for r in per_target_report:
                    tri_r = f"{r['tri_resid_max_px']:.3f}" if r['tri_resid_max_px'] is not None else ''
                    sdis  = f"{r['surv_disagree_m']:.4f}" if r['surv_disagree_m'] is not None else ''
                    f.write(f"{r['label']}\t{r['n_color']}\t{r['n_proj_only']}\t"
                            f"{tri_r}\t{sdis}\t{r['n_dropped']}\t"
                            f"{r['n_promoted_proj']}\t{r['n_promoted_color']}\t{r['gate']}\n")
            print(f"  Wrote consistency report: {report_path}")
        except Exception as _e:
            print(f"  WARNING: could not write consistency report ({_e})")

    return estimates


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
            # Strip Emlid-specific numeric suffix e.g. "... (ftUS) (5)" → "... (ftUS)"
            import re as _re
            candidate = _re.sub(r'\s+\(\d+\)\s*$', '', candidate)
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
                      z_critical: bool,
                      nadir_weight: float = 0.2) -> float:
    """Return a sort score for one image (lower = higher priority).

    score = normalized_dist_from_center + nadir_weight * (0 if nadir else 1)

    nadir_weight controls how much obliques are penalised relative to nadirs:
      1.0 → all nadirs before any oblique
      0.4 → well-centred obliques interleave with edge-placed nadirs
      0.2 → obliques appear in top 7 even when most nadirs are centred (default)
    Z-critical GCPs use nadir_weight * 0.75 to slightly favour obliques for Z resolution.
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

    w = nadir_weight * 0.75 if z_critical else nadir_weight
    return norm_dist + w * nadir_tier


def _sort_images_for_gcp(img_map: dict, exif_map: dict, z_critical: bool,
                         nadir_weight: float = 0.2) -> List[str]:
    """Return image filenames from img_map sorted by confidence score (best first)."""
    def score(fname):
        est  = img_map[fname]
        exif = exif_map.get(fname, {})
        return _image_sort_score(
            est['px'], est['py'],
            exif.get('img_w'), exif.get('img_h'),
            exif.get('gimbal_pitch'),
            z_critical,
            nadir_weight=nadir_weight,
        )
    return sorted(img_map.keys(), key=score)


def _reproject_gcps_inplace(gcps: List[dict], src_crs: str, dst_crs: str) -> None:
    """
    Reproject easting/northing/elevation in each GCP dict from src_crs to dst_crs.

    Updates: easting, northing (reprojected), elevation (converted to dst_crs Z units),
             cs_name (set to dst_crs so _write_gcp_list emits the correct CRS header).

    Assumes src_crs elevation is in CRS-native units (feet for EPSG:6529, metres for
    EPSG:32613).  Converts Z via pyproj CRS linear unit factor.
    """
    from pyproj import Transformer, CRS
    if src_crs.upper() == dst_crs.upper():
        return
    xfm = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    # Determine Z scale: src units → dst units
    try:
        src_unit = CRS.from_user_input(src_crs).axis_info[0].unit_conversion_factor  # m per src unit
        dst_unit = CRS.from_user_input(dst_crs).axis_info[0].unit_conversion_factor  # m per dst unit
        z_scale = src_unit / dst_unit  # multiply src Z to get dst Z
    except Exception:
        z_scale = 1.0
    for g in gcps:
        if g.get('easting') is None or g.get('northing') is None:
            continue
        g['easting'], g['northing'] = xfm.transform(g['easting'], g['northing'])
        if g.get('elevation') is not None:
            g['elevation'] = g['elevation'] * z_scale
        g['cs_name'] = dst_crs


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_gcp_list(gcps: List[dict],
                    estimates: Dict[str, Dict[str, dict]],
                    sort_output: bool = True,
                    z_threshold: float = 0.05,
                    exif_map: Optional[Dict[str, dict]] = None,
                    dup_tolerance_m: float = 1.0,
                    omit_duplicates: bool = False,
                    nadir_weight: float = 0.2) -> str:
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

        # Near-duplicate detection: separate co-located GCPs from the structural
        # sort, then re-insert each one immediately after its closest primary so
        # duplicates are visually adjacent in the output.
        if dup_tolerance_m > 0:
            gcps_main, demoted_triples = _separate_near_duplicates(
                gcps_with_estimates, dup_tolerance_m)
        else:
            gcps_main, demoted_triples = gcps_with_estimates, []

        sorted_gcps, z_critical_labels = _sort_gcps(gcps_main, z_threshold)

        if not omit_duplicates and demoted_triples:
            # Interleave each duplicate immediately after its primary in
            # sorted_gcps.  When several duplicates share a primary, they form
            # a contiguous block: [primary, dup1, dup2, dup3, ...] in original
            # input order.  Spatial proximity (not labels) is used to find the
            # end of an existing dup block, so this works regardless of whether
            # _classify_gcps has already renamed the labels.
            for dup, primary, _dist in demoted_triples:
                try:
                    primary_idx = sorted_gcps.index(primary)
                except ValueError:
                    sorted_gcps.append(dup)
                    continue
                insert_at = primary_idx + 1
                while (insert_at < len(sorted_gcps)
                       and _gcp_distance_m(sorted_gcps[insert_at], primary) < dup_tolerance_m):
                    insert_at += 1
                sorted_gcps.insert(insert_at, dup)
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
                img_map, exif_map, gcp_label in z_critical_labels,
                nadir_weight=nadir_weight,
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
        nadir_only: bool = False,
        cameras_by_wh: Optional[dict] = None) -> Dict[str, Dict[str, dict]]:
    """
    Compute pixel estimates for all (image, GCP) pairs using Mode A (EXIF).

    nadir_only: if True, skip images whose gimbal pitch is not near -90°.
    cameras_by_wh: {(width, height): Brown model params} from _load_cameras(); when
                   a matching entry exists the calibrated Brown model is used instead
                   of the EXIF-derived pinhole.

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
        camera_model = None
        if cameras_by_wh:
            camera_model = cameras_by_wh.get((exif.get('img_w'), exif.get('img_h')))
        for label in gcp_labels:
            gcp = gcp_by_label.get(label)
            if gcp is None:
                continue
            result = project_pixel_mode_a(exif, gcp, camera_model)
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
                   n_control: int = 10,
                   dup_to_primary: Optional[Dict[str, str]] = None) -> None:
    """
    Rename GCP labels in-place with GCP-* / CHK-* prefixes (and -dup suffixes
    for near-duplicates).

    sorted_labels: GCP labels in structural priority order (output of _sort_gcps,
                   excluding near-duplicates).

    Prefix assignment for sorted_labels:
        Positions 1 .. n_control     → GCP-{base}   (control)
        Positions n_control+1 .. end → CHK-{base}   (check)

    Near-duplicates (GCPs not in sorted_labels):
        Each duplicate inherits its closest primary's role PREFIX (GCP- or
        CHK-) but keeps its own base name and gets a '-dup' suffix.  The
        1st, 2nd, 3rd, ... duplicate of the same primary gets '-dup',
        '-dup2', '-dup3', ... so they remain distinct.

        Example: primary '18' becomes 'CHK-18'; dup '18m' becomes
        'CHK-18m-dup' (NOT 'CHK-18-dup' — the dup's distinct base name
        '18m' is preserved).  A duplicate of a GCP-* primary becomes
        another GCP-* (the control count may exceed n_control); a
        duplicate of a CHK-* primary becomes another CHK-*.

        Requires dup_to_primary: {dup_label: primary_label} mapping using
        the ORIGINAL labels (before classification renames them).  Without
        this mapping, GCPs not in sorted_labels fall back to a legacy
        DUP-{base} prefix (kept for backward compatibility).

    Idempotency: any existing GCP-/CHK-/DUP- prefix is stripped before
    re-applying, and any trailing -dup\\d* suffix is also stripped, so
    re-running the pipeline on already-classified labels is a no-op.
    Surveyor-side suffixes like '-1', '-2' (e.g. 'GCP-127-1') are NOT
    stripped because the regex requires '-dup' specifically.
    """
    def _base(label: str) -> str:
        for pfx in ('GCP-', 'CHK-', 'DUP-'):
            if label.startswith(pfx):
                label = label[len(pfx):]
                break
        return re.sub(r'-dup\d*$', '', label)

    rename: Dict[str, str] = {}
    for rank, label in enumerate(sorted_labels, start=1):
        base = _base(label)
        if rank <= n_control:
            rename[label] = f'GCP-{base}'
        else:
            rename[label] = f'CHK-{base}'

    # Near-duplicates: inherit primary's role PREFIX (GCP-/CHK-), keep
    # the dup's own base name, and append -dup[N].  Group dups by primary
    # so multi-duplicate clusters get -dup, -dup2, -dup3.
    if dup_to_primary:
        dups_by_primary: Dict[str, List[str]] = {}
        for dup_label, primary_label in dup_to_primary.items():
            dups_by_primary.setdefault(primary_label, []).append(dup_label)
        for primary_label, dup_labels in dups_by_primary.items():
            primary_role = rename.get(primary_label)
            if primary_role is None:
                # Primary wasn't in sorted_labels — shouldn't happen, but
                # fall back gracefully.
                primary_role = f'GCP-{_base(primary_label)}'
            primary_prefix = 'CHK-' if primary_role.startswith('CHK-') else 'GCP-'
            for n, dup_label in enumerate(dup_labels):
                suffix = '-dup' if n == 0 else f'-dup{n + 1}'
                dup_base = _base(dup_label)
                rename[dup_label] = f'{primary_prefix}{dup_base}{suffix}'
    else:
        # Legacy fallback: anything not in sorted_labels → DUP-{base}.
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

    def _is_dup(lbl: str) -> bool:
        return bool(re.search(r'-dup\d*$', lbl))

    n_ctrl     = sum(1 for g in gcps if g['label'].startswith('GCP-'))
    n_chk      = sum(1 for g in gcps if g['label'].startswith('CHK-'))
    n_ctrl_dup = sum(1 for g in gcps if g['label'].startswith('GCP-') and _is_dup(g['label']))
    n_chk_dup  = sum(1 for g in gcps if g['label'].startswith('CHK-') and _is_dup(g['label']))
    n_legacy_dup = sum(1 for g in gcps if g['label'].startswith('DUP-'))
    if n_legacy_dup:
        print(f"  {n_ctrl} GCP-* (control),  {n_chk} CHK-* (check),  "
              f"{n_legacy_dup} DUP-* (legacy)")
    else:
        print(f"  {n_ctrl} GCP-* (control, incl. {n_ctrl_dup} -dup),  "
              f"{n_chk} CHK-* (check, incl. {n_chk_dup} -dup)")


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
                 survey_csv_path: str,
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
                 refine_limit: int = 0,
                 fallback_crs: Optional[str] = None,
                 nadir_weight: float = 0.2,
                 reproject_to: Optional[str] = None,
                 cameras_by_wh: Optional[dict] = None,
                 iterative: bool = False,
                 iter_eps_resid_px: float = 120.0,
                 iter_eps_surv_m: float = 15.0,
                 consistency_report_path: Optional[Path] = None) -> Tuple[str, dict]:
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
    gcps = parse_survey_csv(survey_csv_path, fallback_crs=fallback_crs)
    print(f"  {len(gcps)} targets")

    if reproject_to and gcps:
        _raw_cs = gcps[0].get('cs_name') or ''
        src = _cs_name_to_epsg(_raw_cs) or fallback_crs or _raw_cs
        if src and src.upper() != reproject_to.upper():
            try:
                _reproject_gcps_inplace(gcps, src, reproject_to)
                print(f"  reprojected coordinates: {src} → {reproject_to}")
            except Exception as _e:
                print(f"  WARNING: could not reproject {src} → {reproject_to}: {_e}")

    print(f"Reading EXIF from {images_dir}...")
    exif_map = read_image_exif_batch(images_dir)
    print(f"  {len(exif_map)} images with GPS data")

    print("Matching images to targets...")
    image_to_gcps = match_images_to_gcps(
        exif_map, gcps,
        fallback_radius_m=fallback_radius_m,
        threads=threads,
    )
    print(f"  {len(image_to_gcps)} images contain at least one target")

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
    print("Estimating target pixel locations via target+image projection geometry...")
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
                    cam_model = cameras_by_wh.get((exif.get('img_w'), exif.get('img_h'))) if cameras_by_wh else None
                    result = project_pixel_mode_a(exif, gcp, cam_model)
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
                                              nadir_only=nadir_only,
                                              cameras_by_wh=cameras_by_wh)

    # Pre-sort: compute structural GCP order so that classification and
    # priority-aware refinement use the same ordering that _write_gcp_list
    # will use.  (_write_gcp_list will sort again — harmless redundancy.)
    sorted_gcps_order: List[dict] = []
    demoted_triples_presort: List[Tuple] = []
    if sort_output:
        print("Sorting targets by structural priority...")
        gcps_with_est = [g for g in gcps if g['label'] in estimates]
        if dup_tolerance_m > 0:
            gcps_main_pre, demoted_triples_presort = _separate_near_duplicates(
                gcps_with_est, dup_tolerance_m)
            if demoted_triples_presort:
                action = "omitted (--omit-duplicates)" if omit_duplicates else "labeled with -dup suffix and interleaved after their primary"
                print(f"  WARNING: {len(demoted_triples_presort)} target(s) within "
                      f"{dup_tolerance_m:.1f} m of another — {action}")
        else:
            gcps_main_pre, demoted_triples_presort = gcps_with_est, []
        sorted_gcps_order, _ = _sort_gcps(gcps_main_pre, z_threshold)

    def _sorted_labels() -> List[str]:
        """Current labels of sorted_gcps_order + demoted, reflecting any renames."""
        return ([g['label'] for g in sorted_gcps_order] +
                [dup['label'] for dup, _, _ in demoted_triples_presort])

    # Classification (geo-12w): rename labels with GCP-*/CHK-* prefixes,
    # with -dup[N] suffixes for near-duplicates that inherit a primary's role.
    if classify:
        if not sorted_gcps_order:
            print("  WARNING: --classify requires sort_output; skipping classification.")
        else:
            # Build dup → primary mapping using ORIGINAL labels (before _classify_gcps
            # mutates them).  When omit_duplicates is set, pass None so duplicates
            # (which won't appear in output anyway) don't get -dup labels.
            dup_to_primary_pre = (
                {dup['label']: primary['label']
                 for dup, primary, _dist in demoted_triples_presort}
                if not omit_duplicates else None
            )
            _classify_gcps(gcps, estimates,
                           [g['label'] for g in sorted_gcps_order],
                           n_control,
                           dup_to_primary=dup_to_primary_pre)
            # sorted_gcps_order dicts were mutated in-place; _sorted_labels() now
            # returns the renamed labels automatically.

    # B3 — Stage 3 pixel refinement (optional)
    if refine_pixels:
        estimates = _refine_all_estimates(
            estimates, exif_map, threads=threads,
            gcp_order=_sorted_labels() if sorted_gcps_order else None,
            refine_limit=refine_limit,
        )

    # B3.5 — Iterative refinement (optional): triangulate target 3D from color
    # rays, re-project to projection-only images, optionally re-run color
    # refinement on the better seed. Produces 'tri_proj' and 'tri_color' labels.
    if iterative:
        # Rebuild gcp_by_label with post-classification labels (g['label'] was
        # renamed in-place by _classify_gcps so the original gcp_by_label built
        # at line ~1281 has stale keys).
        gcp_by_label_post = {g['label']: g for g in gcps}
        estimates = _iterative_refine(
            estimates, gcp_by_label_post, exif_map,
            cameras_by_wh=cameras_by_wh,
            eps_resid_px=iter_eps_resid_px,
            eps_surv_m=iter_eps_surv_m,
            threads=threads,
            report_path=consistency_report_path,
        )

    # B3 — Write outputs
    gcp_txt = _write_gcp_list(
        gcps, estimates,
        sort_output=sort_output,
        z_threshold=z_threshold,
        exif_map=exif_map,
        dup_tolerance_m=dup_tolerance_m,
        omit_duplicates=omit_duplicates,
        nadir_weight=nadir_weight,
    )

    return gcp_txt, estimates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='sight: Survey CSV + drone images → gcpeditpro.txt + pix4d.txt'
    )
    parser.add_argument('survey_csv',  help='GCP survey CSV file path')
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
    parser.add_argument('--crs', default=None,
                        help='Fallback CRS when CSV has no CS name column (e.g. EPSG:6529). '
                             'Used to derive lat/lon from easting/northing.')
    parser.add_argument('--nadir-weight', type=float, default=0.2,
                        help='Penalty applied to oblique images in the sort score '
                             '(0=treat obliques same as nadirs, 1=all nadirs first; default 0.2)')
    parser.add_argument('--out-name', default='targets.txt',
                        help='Filename for the tagging file output (default: targets.txt; '
                             'auto-set to {job}.txt from transform.yaml if present)')
    parser.set_defaults(classify=True, refine_pixels=True)
    parser.add_argument('--match-only', action='store_true',
                        help='Run image-to-GCP footprint matching only and print results without projecting pixels or writing files')
    parser.add_argument('--transform-yaml', default=None, metavar='FILE',
                        help='Path to transform.yaml written by transformer.py dc. '
                             'Auto-located in the survey CSV directory or cwd if omitted. '
                             'Provides survey_crs (fallback for --crs) and job name (fallback for --out-name).')
    parser.add_argument('--cameras', default=None, metavar='FILE',
                        help='Path to cameras.json (ODM Brown model). When provided, '
                             'replaces the EXIF-derived pinhole model with the calibrated '
                             'focal length, principal point, and radial/tangential distortion '
                             'coefficients for improved initial pixel projections.')
    parser.add_argument('--iterative', action='store_true',
                        help='After color refinement, triangulate target 3D from color '
                             'rays and re-project to projection-only images (tri_proj). '
                             'Then re-run color refinement on the new seeds (tri_color). '
                             'Pre-tagging suspect-target flags fall out as a side product.')
    parser.add_argument('--iter-eps-resid-px', type=float, default=120.0,
                        help='Sanity threshold: max per-image triangulation residual '
                             'in pixels for the gate to pass (default 120). Bounded by '
                             'EXIF camera-pose noise — per-image noise of ~50-100 px is '
                             'normal even when color refinement is correct.')
    parser.add_argument('--iter-eps-surv-m', type=float, default=15.0,
                        help='Sanity threshold: max distance (metres) between triangulated '
                             '3D and the surveyed coordinate for the gate to pass '
                             '(default 15). Bounded by EXIF GPS noise — drones without RTK '
                             'often have multi-metre GPS drift, so the optimizer-found 3D '
                             'will absorb that noise.')
    parser.add_argument('--consistency-report', default=None, metavar='FILE',
                        help='If set with --iterative, write per-target metrics '
                             '(triangulation residual, survey disagreement, gate flag) '
                             'to this TSV file alongside the printed summary.')
    args = parser.parse_args()

    # --- transform.yaml integration ---
    _yaml_path = None
    if args.transform_yaml:
        _yaml_path = Path(args.transform_yaml)
        if not _yaml_path.exists():
            import sys as _sys; _sys.exit(f'ERROR: transform.yaml not found: {_yaml_path}')
    else:
        for _candidate in [Path(args.survey_csv).parent / 'transform.yaml',
                            Path.cwd() / 'transform.yaml']:
            if _candidate.exists():
                _yaml_path = _candidate
                break

    if _yaml_path:
        try:
            # Minimal YAML reader — matches transformer.py's write_yaml format
            def _strip_inline_comment(s: str) -> str:
                # Strip trailing "# ..." comment, but only when '#' is preceded
                # by whitespace, so values containing a literal '#' aren't
                # corrupted.  Quoted values keep their quotes via .strip('"').
                _i = s.find(' #')
                return (s[:_i] if _i >= 0 else s).strip().strip('"')

            _transform: dict = {}
            _section = None
            for _raw in _yaml_path.read_text(encoding='utf-8').splitlines():
                _line = _raw.rstrip()
                if not _line or _line.lstrip().startswith('#'):
                    continue
                if _line.startswith('  '):
                    if _section:
                        _k, _, _v = _line.strip().partition(': ')
                        _transform.setdefault(_section, {})[_k] = _strip_inline_comment(_v)
                else:
                    _k, _, _v = _line.partition(': ')
                    _v = _strip_inline_comment(_v)
                    if not _v:
                        _section = _k.rstrip(':'); _transform[_section] = {}
                    else:
                        _section = None; _transform[_k] = _v
            print(f'Loaded transform.yaml: {_yaml_path}')
        except Exception as _e:
            print(f'WARNING: could not read transform.yaml ({_e}); ignoring')
            _transform = {}

        _survey_crs = _transform.get('survey_crs') or _transform.get('field_crs')
        if not _transform.get('survey_crs') and _transform.get('field_crs'):
            print("  DEPRECATION: transform.yaml uses 'field_crs'; rename to 'survey_crs'")
        _odm_crs     = _transform.get('odm_crs')
        _job_name    = _transform.get('job')
        _design_grid = _transform.get('design_grid') if isinstance(_transform.get('design_grid'), dict) else {}

        if _survey_crs and not args.crs:
            args.crs = _survey_crs
            print(f'  survey_crs → --crs {_survey_crs}')
        if _job_name and args.out_name == 'targets.txt':
            args.out_name = f'{_job_name}.txt'
            print(f'  job → --out-name {args.out_name}')
        if _odm_crs:
            print(f'  odm_crs → reproject output to {_odm_crs}')
    else:
        _odm_crs     = None
        _design_grid = {}
        _job_name    = None

    if args.match_only:
        print(f'Parsing {args.survey_csv}...')
        gcps = parse_survey_csv(args.survey_csv, fallback_crs=args.crs)
        print(f'  {len(gcps)} FIX targets: {[g["label"] for g in gcps]}')

        print(f'\nReading EXIF from {args.image_dir}...')
        exif_map = read_image_exif_batch(args.image_dir)
        print(f'  {len(exif_map)} images with GPS data')

        print(f'\nMatching images to targets (fallback radius={args.radius}m)...')
        image_to_gcps = match_images_to_gcps(
            exif_map, gcps,
            fallback_radius_m=args.radius,
            threads=args.threads,
        )

        print(f'\nResult: {len(image_to_gcps)} images contain at least one target')
        for fname, labels in sorted(image_to_gcps.items()):
            print(f'  {fname}: {labels}')
    else:
        if args.seed_dist_penalty is not None:
            _refine_module._SEED_DIST_PENALTY = args.seed_dist_penalty
        if args.marker_size is not None:
            _refine_module._MARKER_SIZE_M = args.marker_size

        # --- Load cameras.json if provided ---
        _cameras_by_wh = None
        if args.cameras:
            try:
                _cameras_by_wh = _load_cameras(args.cameras)
                sizes = [f"{w}×{h}" for w, h in _cameras_by_wh]
                print(f"Loaded cameras.json: {len(_cameras_by_wh)} Brown model(s) — {', '.join(sizes)}")
            except Exception as _ce:
                print(f"WARNING: could not load cameras.json ({_ce}); using EXIF pinhole")

        # Ensure output dir exists before writing any output files
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

        _consistency_path = (
            (Path(args.out_dir) / args.consistency_report)
            if args.consistency_report and not Path(args.consistency_report).is_absolute()
            else (Path(args.consistency_report) if args.consistency_report else None)
        )

        gcp_txt, estimates = run_pipeline(
                images_dir=args.image_dir,
                survey_csv_path=args.survey_csv,
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
                fallback_crs=args.crs,
                nadir_weight=args.nadir_weight,
                reproject_to=_odm_crs,
                cameras_by_wh=_cameras_by_wh,
                iterative=args.iterative,
                iter_eps_resid_px=args.iter_eps_resid_px,
                iter_eps_surv_m=args.iter_eps_surv_m,
                consistency_report_path=_consistency_path,
            )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gcp_out   = out_dir / args.out_name
        pix4d_out = out_dir / 'marks_design.csv'

        gcp_out.write_text(gcp_txt)
        pix4d_out.write_text(_write_pix4d(estimates))

        print(f'\nWrote {gcp_out}')
        print(f'Wrote {pix4d_out}')
