#!/usr/bin/env python3
"""
true_ortho.py -- True ortho post-processor for ODM orthophotos.

Replaces occlusion-affected pixels in an ODM DTM-orthophoto with correctly
projected camera imagery.  For each output pixel:

  1. Convert ortho pixel coords -> UTM via GeoTransform
  2. Convert UTM -> topocentric ENU via geodetic math (UTM -> LLA -> ECEF -> ENU)
  3. Look up ground Z from DTM (if provided) or use default Z
  4. Select best camera (most-nadir with valid projection)
     - If DSM provided: occlusion-aware ray-march filtering
     - If no DSM: simple most-nadir selection
  5. Project 3D point through camera model (brown distortion)
  6. Sample camera image via bilinear interpolation
  7. Write to output

Uses vectorized numpy operations for coordinate transforms (per-row).
Camera projection and image sampling are batched per-camera for locality.

Usage:
    conda run -n geo python accuracy_study/true_ortho.py \\
        reconstruction.topocentric.json \\
        orthophoto.tif \\
        image_dir/ \\
        -o true_ortho.tif \\
        [--dtm dtm.tif] \\
        [--dsm dsm.tif] \\
        [--crop "x_min,y_min,x_max,y_max"]  # UTM coords
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
except ImportError:
    raise RuntimeError("GDAL (osgeo) is required: conda install gdal")

try:
    from pyproj import CRS as ProjCRS, Transformer
except ImportError:
    raise RuntimeError("pyproj is required: conda install pyproj")

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    raise RuntimeError("scipy is required: conda install scipy")


# ---------------------------------------------------------------------------
# WGS84 ellipsoid (matching OpenSFM opensfm/geo.py)
# ---------------------------------------------------------------------------

WGS84_a = 6378137.0
WGS84_b = 6356752.314245


# ---------------------------------------------------------------------------
# Vectorised geodetic coordinate conversions
# ---------------------------------------------------------------------------

def ecef_from_lla_vec(lat_deg: np.ndarray, lon_deg: np.ndarray,
                      alt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 geodetic to ECEF, vectorised."""
    a2 = WGS84_a ** 2
    b2 = WGS84_b ** 2
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    L = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    return (
        (a2 * L + alt) * np.cos(lat) * np.cos(lon),
        (a2 * L + alt) * np.cos(lat) * np.sin(lon),
        (b2 * L + alt) * np.sin(lat),
    )


def lla_to_enu_vec(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt: np.ndarray,
    ref_lat: float, ref_lon: float, ref_alt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert arrays of geodetic lat/lon/alt to topocentric ENU.
    Matches OpenSFM geo.topocentric_from_lla, vectorised.
    """
    tx, ty, tz = ecef_from_lla_vec(lat_deg, lon_deg, alt)
    a2 = WGS84_a ** 2
    b2 = WGS84_b ** 2
    rlat = np.radians(ref_lat)
    rlon = np.radians(ref_lon)
    L = 1.0 / np.sqrt(a2 * np.cos(rlat) ** 2 + b2 * np.sin(rlat) ** 2)
    rx = (a2 * L + ref_alt) * np.cos(rlat) * np.cos(rlon)
    ry = (a2 * L + ref_alt) * np.cos(rlat) * np.sin(rlon)
    rz = (b2 * L + ref_alt) * np.sin(rlat)

    dx = tx - rx
    dy = ty - ry
    dz = tz - rz

    sa = np.sin(rlat)
    ca = np.cos(rlat)
    so = np.sin(rlon)
    co = np.cos(rlon)

    e = -so * dx + co * dy
    n = -sa * co * dx + (-sa * so) * dy + ca * dz
    u = ca * co * dx + ca * so * dy + sa * dz

    return e, n, u


def _enu_to_lla_scalar(
    x: float, y: float, z: float,
    ref_lat: float, ref_lon: float, ref_alt: float,
) -> Tuple[float, float, float]:
    """Scalar ENU to LLA (for camera centre conversion)."""
    a2 = WGS84_a ** 2
    b2 = WGS84_b ** 2
    rlat = np.radians(ref_lat)
    rlon = np.radians(ref_lon)
    L = 1.0 / np.sqrt(a2 * np.cos(rlat) ** 2 + b2 * np.sin(rlat) ** 2)
    rx = (a2 * L + ref_alt) * np.cos(rlat) * np.cos(rlon)
    ry = (a2 * L + ref_alt) * np.cos(rlat) * np.sin(rlon)
    rz = (b2 * L + ref_alt) * np.sin(rlat)

    sa = np.sin(rlat)
    ca = np.cos(rlat)
    so = np.sin(rlon)
    co = np.cos(rlon)

    ex = -so * x + (-sa * co) * y + (ca * co) * z + rx
    ey =  co * x + (-sa * so) * y + (ca * so) * z + ry
    ez =             ca       * y +  sa       * z + rz

    a = WGS84_a
    b = WGS84_b
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
    alt_out = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt_out)


# ---------------------------------------------------------------------------
# Reconstruction + camera model
# ---------------------------------------------------------------------------

def load_reconstruction(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty JSON array: {path}")
    return data[0]


def _get_cam_params(cam: dict):
    """Extract intrinsic parameters from OpenSFM camera dict."""
    w, h = cam["width"], cam["height"]
    mwh = max(w, h)
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
    cx = w / 2.0
    cy = h / 2.0
    return w, h, fx, fy, cx, cy, cx_off, cy_off, k1, k2, k3, p1, p2


def project_points_pinhole_vec(
    points_enu: np.ndarray,  # (N, 3)
    R: np.ndarray,           # (3, 3) world->camera
    t: np.ndarray,           # (3,) translation
    cam: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised pinhole projection (no distortion) for undistorted images.
    """
    w, h = cam["width"], cam["height"]
    mwh = max(w, h)
    fx = fy = cam.get("focal", cam.get("focal_x", 0.5)) * mwh
    if "focal_y" in cam:
        fy = cam["focal_y"] * mwh
    cx_off = cam.get("c_x", 0.0) * mwh
    cy_off = cam.get("c_y", 0.0) * mwh
    cx, cy = w / 2.0, h / 2.0

    p_cam = (R @ points_enu.T).T + t
    z_cam = p_cam[:, 2]
    valid = z_cam > 0.1
    z_safe = np.where(valid, z_cam, 1.0)

    px = fx * (p_cam[:, 0] / z_safe) + cx + cx_off
    py = fy * (p_cam[:, 1] / z_safe) + cy + cy_off

    margin = 2.0
    valid &= (px >= margin) & (px < w - margin) & (py >= margin) & (py < h - margin)
    return px, py, valid


def project_points_brown_vec(
    points_enu: np.ndarray,  # (N, 3)
    R: np.ndarray,           # (3, 3) world->camera
    t: np.ndarray,           # (3,) translation
    cam: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised brown projection of N 3D points through a camera.

    Returns:
        px, py: pixel coordinates (N,)
        valid:  boolean mask (N,) — True if point projects into frame
    """
    w, h, fx, fy, cx, cy, cx_off, cy_off, k1, k2, k3, p1, p2 = _get_cam_params(cam)

    # World to camera: p_cam = R @ p_world + t
    p_cam = (R @ points_enu.T).T + t  # (N, 3)

    z_cam = p_cam[:, 2]
    valid = z_cam > 0.1  # must be in front of camera

    # Avoid division by zero
    z_safe = np.where(valid, z_cam, 1.0)

    xn = p_cam[:, 0] / z_safe
    yn = p_cam[:, 1] / z_safe

    # Brown distortion
    r2 = xn ** 2 + yn ** 2
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    tang_x = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)
    tang_y = p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn

    xd = xn * radial + tang_x
    yd = yn * radial + tang_y

    px = fx * xd + cx + cx_off
    py = fy * yd + cy + cy_off

    # Bounds check
    margin = 2.0
    valid &= (px >= margin) & (px < w - margin) & (py >= margin) & (py < h - margin)

    return px, py, valid


def _detect_undistorted(image_dir: str, recon_path: str) -> Tuple[Optional[str], Optional[dict], bool]:
    """
    Auto-detect undistorted images alongside the reconstruction.

    If opensfm/undistorted/images/ exists relative to the reconstruction,
    prefer it over raw images. Also loads the undistorted reconstruction
    for its camera model (simple perspective, no distortion).

    Returns (image_dir, cameras_override, use_pinhole):
        - image_dir: path to use for image loading
        - cameras_override: camera dict from undistorted reconstruction, or None
        - use_pinhole: True if using undistorted images (pinhole projection)
    """
    recon_dir = os.path.dirname(recon_path)
    undist_img_dir = os.path.join(recon_dir, "undistorted", "images")
    undist_recon = os.path.join(recon_dir, "undistorted", "reconstruction.json")

    if os.path.isdir(undist_img_dir) and len(os.listdir(undist_img_dir)) > 0:
        cameras_override = None
        if os.path.exists(undist_recon):
            with open(undist_recon) as f:
                ur = json.load(f)
            if ur and "cameras" in ur[0]:
                cameras_override = ur[0]["cameras"]
        return undist_img_dir, cameras_override, True

    return image_dir, None, False


# ---------------------------------------------------------------------------
# DTM sampler
# ---------------------------------------------------------------------------

class DTMSampler:
    def __init__(self, path: str):
        self._ds = gdal.Open(path, gdal.GA_ReadOnly)
        if self._ds is None:
            raise FileNotFoundError(f"Cannot open: {path}")
        self._gt = self._ds.GetGeoTransform()
        self._band = self._ds.GetRasterBand(1)
        self._nodata = self._band.GetNoDataValue()
        self._w = self._ds.RasterXSize
        self._h = self._ds.RasterYSize

    def sample_row(self, x_utm: np.ndarray, y_utm: float) -> np.ndarray:
        """Sample DTM Z for a row of UTM x coords at fixed y. Returns NaN for no-data."""
        row = int((y_utm - self._gt[3]) / self._gt[5])
        cols = ((x_utm - self._gt[0]) / self._gt[1]).astype(int)

        result = np.full(len(x_utm), np.nan)
        in_bounds = (cols >= 0) & (cols < self._w) & (row >= 0) & (row < self._h)
        if not np.any(in_bounds):
            return result

        # Read full row from DTM
        if 0 <= row < self._h:
            row_data = self._band.ReadAsArray(0, row, self._w, 1)[0]
            valid_cols = cols[in_bounds]
            vals = row_data[valid_cols].astype(float)
            if self._nodata is not None:
                vals[vals == self._nodata] = np.nan
            result[in_bounds] = vals
        return result


    def sample_points(self, x_utm: np.ndarray, y_utm: np.ndarray) -> np.ndarray:
        """Sample DTM/DSM Z at arbitrary (x, y) UTM points. Returns NaN for no-data.

        Reads a single bounding-box block from the raster and indexes into it,
        rather than reading individual pixels (which has huge GDAL call overhead).
        """
        cols = ((x_utm - self._gt[0]) / self._gt[1]).astype(int)
        rows = ((y_utm - self._gt[3]) / self._gt[5]).astype(int)
        result = np.full(len(x_utm), np.nan)
        in_bounds = (cols >= 0) & (cols < self._w) & (rows >= 0) & (rows < self._h)
        if not np.any(in_bounds):
            return result

        bc = cols[in_bounds]
        br = rows[in_bounds]

        # Read bounding box that covers all points in one GDAL call
        c_min, c_max = int(bc.min()), int(bc.max())
        r_min, r_max = int(br.min()), int(br.max())
        block_w = c_max - c_min + 1
        block_h = r_max - r_min + 1

        block = self._band.ReadAsArray(c_min, r_min, block_w, block_h)
        if block is None:
            return result

        # Index into the block
        local_c = bc - c_min
        local_r = br - r_min
        vals = block[local_r, local_c].astype(float)
        if self._nodata is not None:
            vals[vals == self._nodata] = np.nan
        result[in_bounds] = vals
        return result


def check_occlusion_vec(
    cam_utm_x: float, cam_utm_y: float, cam_z: float,
    ground_x: np.ndarray, ground_y: np.ndarray, ground_z: np.ndarray,
    dsm_sampler: "DTMSampler",
    n_steps: int = 10,
) -> np.ndarray:
    """
    Ray-march from camera to each ground point, sampling DSM along the ray.

    Returns boolean mask (N,): True = unoccluded (camera can see ground point).

    For each ground point, samples the DSM at n_steps positions along the ray
    from the camera to the ground. If any DSM sample exceeds the ray height,
    the view is occluded.
    """
    n = len(ground_x)
    unoccluded = np.ones(n, dtype=bool)

    # Steps along ray: t=0 is camera, t=1 is ground. Sample interior only.
    t_vals = np.linspace(0.15, 0.85, n_steps)  # avoid endpoints

    for t in t_vals:
        # Interpolate positions along ray
        rx = cam_utm_x + t * (ground_x - cam_utm_x)
        ry = cam_utm_y + t * (ground_y - cam_utm_y)
        rz = cam_z + t * (ground_z - cam_z)  # ray height at this t

        # Only check points still considered unoccluded
        check_mask = unoccluded & np.isfinite(rx)
        if not np.any(check_mask):
            break

        check_idx = np.where(check_mask)[0]
        dsm_z = dsm_sampler.sample_points(rx[check_idx], ry[check_idx])

        # If DSM surface is above the ray, the view is blocked
        blocked = np.isfinite(dsm_z) & (dsm_z > rz[check_idx] + 0.5)  # 0.5m tolerance
        unoccluded[check_idx[blocked]] = False

    return unoccluded


# ---------------------------------------------------------------------------
# Image cache
# ---------------------------------------------------------------------------

class ImageCache:
    def __init__(self, image_dir: str, max_cached: int = 20):
        self._dir = image_dir
        self._max = max_cached
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str] = []

    def get(self, name: str) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]
        path = os.path.join(self._dir, name)
        if not os.path.exists(path):
            # Try with .tif suffix (undistorted images use name.JPG.tif)
            path = os.path.join(self._dir, name + ".tif")
            if not os.path.exists(path):
                return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if len(self._cache) >= self._max:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[name] = img
        self._order.append(name)
        return img


def sample_image_vec(img: np.ndarray, px: np.ndarray, py: np.ndarray,
                     valid: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling of image at (px, py) for valid pixels.
    Returns (N, 3) BGR array. Invalid pixels get [0, 0, 0].
    """
    h_img, w_img = img.shape[:2]
    n = len(px)
    result = np.zeros((n, 3), dtype=np.uint8)

    if not np.any(valid):
        return result

    # Work only on valid pixels
    vpx = px[valid]
    vpy = py[valid]

    x0 = vpx.astype(int)
    y0 = vpy.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp
    x0c = np.clip(x0, 0, w_img - 1)
    x1c = np.clip(x1, 0, w_img - 1)
    y0c = np.clip(y0, 0, h_img - 1)
    y1c = np.clip(y1, 0, h_img - 1)

    fx = (vpx - x0).reshape(-1, 1)
    fy = (vpy - y0).reshape(-1, 1)

    val = (
        (1 - fx) * (1 - fy) * img[y0c, x0c].astype(np.float32) +
        fx * (1 - fy) * img[y0c, x1c].astype(np.float32) +
        (1 - fx) * fy * img[y1c, x0c].astype(np.float32) +
        fx * fy * img[y1c, x1c].astype(np.float32)
    )
    result[valid] = np.clip(val, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_true_ortho(
    recon_path: str,
    ortho_path: str,
    image_dir: str,
    output_path: str,
    dtm_path: Optional[str] = None,
    dsm_path: Optional[str] = None,
    crop_utm: Optional[Tuple[float, float, float, float]] = None,
    max_cameras: int = 5,
    default_z: float = 0.0,
) -> None:
    """
    Generate a true orthophoto by reprojecting camera imagery.
    """

    # --- Load reconstruction ---
    print(f"Loading reconstruction: {recon_path}")
    recon = load_reconstruction(recon_path)
    cameras_dict = recon["cameras"]
    shots = recon["shots"]
    ref_lla = recon["reference_lla"]
    ref_lat = ref_lla["latitude"]
    ref_lon = ref_lla["longitude"]
    ref_alt = ref_lla["altitude"]
    print(f"  {len(shots)} shots, ref=({ref_lat:.6f}, {ref_lon:.6f}, {ref_alt:.1f})")

    # --- Auto-detect undistorted images ---
    undist_dir, undist_cams, use_pinhole = _detect_undistorted(image_dir, recon_path)
    if use_pinhole:
        image_dir = undist_dir
        if undist_cams:
            cameras_dict = undist_cams
        print(f"  Using undistorted images: {image_dir} (pinhole projection)")
    else:
        print(f"  Using original images: {image_dir} (brown distortion projection)")

    project_fn = project_points_pinhole_vec if use_pinhole else project_points_brown_vec

    # --- Pre-compute camera poses ---
    print("Pre-computing camera poses...")
    shot_names: List[str] = []
    cam_centres: List[np.ndarray] = []    # ENU
    cam_rotations: List[np.ndarray] = []  # R (3x3)
    cam_translations: List[np.ndarray] = []  # t (3)
    cam_models: List[dict] = []

    for shot_name, shot in shots.items():
        cam = cameras_dict[shot["camera"]]
        R = Rotation.from_rotvec(shot["rotation"]).as_matrix()
        t = np.array(shot["translation"], dtype=np.float64)
        C = -R.T @ t

        shot_names.append(shot_name)
        cam_centres.append(C)
        cam_rotations.append(R)
        cam_translations.append(t)
        cam_models.append(cam)

    n_cams = len(shot_names)
    cam_centres_arr = np.array(cam_centres)  # (N_cams, 3)
    print(f"  {n_cams} cameras loaded")

    # Compute camera UTM positions for spatial filtering
    cam_utm = np.zeros((n_cams, 2))
    wgs84_to_utm_tmp = None  # will set from ortho CRS

    # --- Open ortho for extent and CRS ---
    ortho_ds = gdal.Open(ortho_path, gdal.GA_ReadOnly)
    if ortho_ds is None:
        raise FileNotFoundError(f"Cannot open orthophoto: {ortho_path}")
    ortho_gt = ortho_ds.GetGeoTransform()
    ortho_srs_wkt = ortho_ds.GetProjection()
    ortho_w = ortho_ds.RasterXSize
    ortho_h = ortho_ds.RasterYSize
    n_bands = ortho_ds.RasterCount

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ortho_srs_wkt)
    epsg_code = srs.GetAuthorityCode(None)
    epsg_str = f"EPSG:{epsg_code}" if epsg_code else "EPSG:32613"
    print(f"Orthophoto: {ortho_w}x{ortho_h}, {n_bands} bands, CRS: {epsg_str}")

    # --- Coordinate transformers ---
    utm_to_wgs84 = Transformer.from_crs(epsg_str, "EPSG:4326", always_xy=True)
    wgs84_to_utm = Transformer.from_crs("EPSG:4326", epsg_str, always_xy=True)

    # Compute camera UTM positions
    for i in range(n_cams):
        C = cam_centres[i]
        lat, lon, alt = _enu_to_lla_scalar(C[0], C[1], C[2], ref_lat, ref_lon, ref_alt)
        x, y = wgs84_to_utm.transform(lon, lat)
        cam_utm[i] = [x, y]

    # --- Compute output region ---
    if crop_utm is not None:
        cx_min, cy_min, cx_max, cy_max = crop_utm
        col_min = int((cx_min - ortho_gt[0]) / ortho_gt[1])
        col_max = int(math.ceil((cx_max - ortho_gt[0]) / ortho_gt[1]))
        row_min = int((cy_max - ortho_gt[3]) / ortho_gt[5])
        row_max = int(math.ceil((cy_min - ortho_gt[3]) / ortho_gt[5]))
        col_min = max(0, col_min)
        col_max = min(ortho_w, col_max)
        row_min = max(0, row_min)
        row_max = min(ortho_h, row_max)
        out_w = col_max - col_min
        out_h = row_max - row_min
        out_gt = (
            ortho_gt[0] + col_min * ortho_gt[1],
            ortho_gt[1], 0.0,
            ortho_gt[3] + row_min * ortho_gt[5],
            0.0, ortho_gt[5],
        )
        print(f"Crop: cols [{col_min}, {col_max}), rows [{row_min}, {row_max})")
        print(f"  Output: {out_w}x{out_h} = {out_w * out_h:,} pixels")
    else:
        col_min, row_min = 0, 0
        out_w, out_h = ortho_w, ortho_h
        out_gt = ortho_gt
        print(f"Full ortho: {out_w}x{out_h} = {out_w * out_h:,} pixels")

    if out_w <= 0 or out_h <= 0:
        print("ERROR: empty crop region", file=sys.stderr)
        return

    # --- Load DTM / DSM ---
    dtm_sampler = None
    if dtm_path:
        print(f"Loading DTM: {dtm_path}")
        dtm_sampler = DTMSampler(dtm_path)

    dsm_sampler = None
    if dsm_path:
        print(f"Loading DSM: {dsm_path}")
        dsm_sampler = DTMSampler(dsm_path)

    # --- Read original ortho crop (for fallback + alpha) ---
    print("Reading original orthophoto crop...")
    ortho_crop = ortho_ds.ReadAsArray(col_min, row_min, out_w, out_h)
    if ortho_crop is None:
        print("ERROR: Failed to read orthophoto", file=sys.stderr)
        return

    has_alpha = (n_bands >= 4)
    if has_alpha:
        alpha = ortho_crop[3]
    else:
        alpha = np.full((out_h, out_w), 255, dtype=np.uint8)

    # --- Pre-compute UTM x coordinates for all columns ---
    col_indices = np.arange(out_w, dtype=np.float64)
    x_utm_all = out_gt[0] + col_indices * out_gt[1]

    # --- Create output buffer (start with copy of original RGB) ---
    out_buf = np.zeros((3, out_h, out_w), dtype=np.uint8)
    for b in range(min(3, n_bands)):
        out_buf[b] = ortho_crop[b]

    # --- Image cache ---
    img_cache = ImageCache(image_dir, max_cached=30)

    # --- Process: camera-centric approach ---
    # For each camera, determine which output pixels it can see,
    # project and sample, keeping the best (most-nadir) result per pixel.
    #
    # We store the best nadir angle per pixel; a camera only overwrites
    # if its nadir angle is smaller (more nadir).

    print(f"\nProcessing {out_w}x{out_h} pixels with {n_cams} cameras...")
    print("Strategy: camera-centric (iterate cameras, update best per pixel)")
    t0 = time.time()

    best_nadir = np.full((out_h, out_w), np.inf, dtype=np.float32)
    max_lateral_dist = 200.0  # metres

    # Pre-filter cameras that overlap the crop region
    crop_x_min = x_utm_all[0]
    crop_x_max = x_utm_all[-1]
    crop_y_min = out_gt[3] + (out_h - 1) * out_gt[5]
    crop_y_max = out_gt[3]

    crop_cx = (crop_x_min + crop_x_max) / 2
    crop_cy = (crop_y_min + crop_y_max) / 2
    crop_radius = math.sqrt((crop_x_max - crop_x_min) ** 2 +
                            (crop_y_max - crop_y_min) ** 2) / 2

    nearby_cams = []
    for i in range(n_cams):
        dx = cam_utm[i, 0] - crop_cx
        dy = cam_utm[i, 1] - crop_cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < crop_radius + max_lateral_dist:
            nearby_cams.append(i)

    print(f"  {len(nearby_cams)}/{n_cams} cameras overlap crop region")

    n_updated = 0
    n_projected = 0

    for ci, cam_idx in enumerate(nearby_cams):
        if ci % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Camera {ci}/{len(nearby_cams)} ({shot_names[cam_idx]})  "
                  f"elapsed={elapsed:.0f}s  updated={n_updated:,}", end="\r")

        sname = shot_names[cam_idx]
        C = cam_centres[cam_idx]
        R = cam_rotations[cam_idx]
        t_vec = cam_translations[cam_idx]
        cam = cam_models[cam_idx]
        cx_utm, cy_utm = cam_utm[cam_idx]

        # Load image for this camera
        img = img_cache.get(sname)
        if img is None:
            continue

        # Process row by row
        for local_row in range(out_h):
            y_utm = out_gt[3] + local_row * out_gt[5]

            # Quick row-level distance check
            dy = abs(cy_utm - y_utm)
            if dy > max_lateral_dist:
                continue

            # Column-level distance check
            dx = x_utm_all - cx_utm
            lateral_dist_sq = dx * dx + dy * dy
            col_mask = lateral_dist_sq < max_lateral_dist ** 2

            # Also skip transparent pixels
            col_mask &= alpha[local_row] > 0

            n_valid = np.sum(col_mask)
            if n_valid == 0:
                continue

            valid_cols = np.where(col_mask)[0]
            x_utm_valid = x_utm_all[valid_cols]

            # Get ground Z
            if dtm_sampler:
                z_ground = dtm_sampler.sample_row(x_utm_valid, y_utm)
                has_z = ~np.isnan(z_ground)
                if not np.any(has_z):
                    continue
                valid_cols = valid_cols[has_z]
                x_utm_valid = x_utm_valid[has_z]
                z_ground = z_ground[has_z]
            else:
                z_ground = np.full(len(valid_cols), default_z)

            # Convert UTM -> LLA
            lons, lats = utm_to_wgs84.transform(x_utm_valid, np.full_like(x_utm_valid, y_utm))

            # Convert LLA -> ENU
            e, n_coord, u = lla_to_enu_vec(lats, lons, z_ground, ref_lat, ref_lon, ref_alt)
            points_enu = np.column_stack([e, n_coord, u])  # (M, 3)

            # Compute nadir angle for this camera to each point
            vecs = points_enu - C  # (M, 3)
            dists = np.linalg.norm(vecs, axis=1)
            dists_safe = np.where(dists > 0, dists, 1.0)
            cos_nadir = -vecs[:, 2] / dists_safe
            nadir_angles = np.arccos(np.clip(cos_nadir, -1.0, 1.0))

            # Only process pixels where this camera is more nadir than current best
            better = nadir_angles < best_nadir[local_row, valid_cols]
            if not np.any(better):
                continue

            better_idx = np.where(better)[0]

            # DSM occlusion check: ray-march from camera to ground, reject if blocked
            if dsm_sampler is not None:
                b_x = x_utm_valid[better_idx] if len(x_utm_valid) > len(better_idx) else x_utm_all[valid_cols[better_idx]]
                b_y = np.full(len(better_idx), y_utm)
                b_z = z_ground[better_idx]
                unoccluded = check_occlusion_vec(
                    cx_utm, cy_utm, _enu_to_lla_scalar(C[0], C[1], C[2], ref_lat, ref_lon, ref_alt)[2],
                    b_x, b_y, b_z, dsm_sampler,
                )
                better_idx = better_idx[unoccluded]
                if len(better_idx) == 0:
                    continue

            pts_better = points_enu[better_idx]

            # Project through camera
            px, py, proj_valid = project_fn(pts_better, R, t_vec, cam)

            if not np.any(proj_valid):
                continue

            # Sample image at projected positions
            colors = sample_image_vec(img, px, py, proj_valid)

            # Update output for pixels where projection succeeded
            success_idx = better_idx[proj_valid]
            output_cols = valid_cols[success_idx]

            out_buf[0, local_row, output_cols] = colors[proj_valid, 2]  # R
            out_buf[1, local_row, output_cols] = colors[proj_valid, 1]  # G
            out_buf[2, local_row, output_cols] = colors[proj_valid, 0]  # B
            best_nadir[local_row, output_cols] = nadir_angles[success_idx].astype(np.float32)

            n_count = int(np.sum(proj_valid))
            n_projected += n_count
            n_updated += n_count

    n_filled = int(np.sum(np.isfinite(best_nadir) & (best_nadir < np.inf)))
    elapsed = time.time() - t0

    print(f"\n\nDone in {elapsed:.1f}s")
    print(f"  Total pixels:        {out_w * out_h:,}")
    print(f"  Filled from cameras: {n_filled:,} "
          f"({100.0 * n_filled / max(1, out_w * out_h):.1f}%)")
    print(f"  Total projections:   {n_projected:,}")
    non_alpha = np.sum(alpha > 0)
    print(f"  Non-transparent:     {non_alpha:,}")

    # --- Write output ---
    print(f"\nWriting output: {output_path}")
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path, out_w, out_h, 3, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
    )
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(ortho_srs_wkt)

    for b in range(3):
        out_ds.GetRasterBand(b + 1).WriteArray(out_buf[b])
    out_ds.FlushCache()
    out_ds = None
    ortho_ds = None
    print("Done.")


# ---------------------------------------------------------------------------
# Full-image tiled processing
# ---------------------------------------------------------------------------


def _process_tile(args_tuple):
    """Worker function for multiprocessing. Processes a single tile."""
    (tile_col, tile_row, tile_w, tile_h, col_off, row_off,
     ortho_path, image_dir, dtm_path, dsm_path, default_z,
     recon_data, ortho_gt, ortho_srs_wkt, epsg_str, use_pinhole) = args_tuple

    # Reconstruct camera data from shared dict
    shots = recon_data["shots"]
    cameras_dict = recon_data["cameras"]
    ref_lla = recon_data["reference_lla"]
    ref_lat, ref_lon, ref_alt = ref_lla["latitude"], ref_lla["longitude"], ref_lla["altitude"]

    # Read ortho tile alpha to check coverage
    ortho_ds = gdal.Open(ortho_path, gdal.GA_ReadOnly)
    n_bands = ortho_ds.RasterCount
    if n_bands >= 4:
        alpha = ortho_ds.GetRasterBand(4).ReadAsArray(col_off, row_off, tile_w, tile_h)
        if np.all(alpha == 0):
            ortho_ds = None
            return None  # empty tile
    else:
        # Read band 1 to check for all-zero
        b1 = ortho_ds.GetRasterBand(1).ReadAsArray(col_off, row_off, tile_w, tile_h)
        if np.all(b1 == 0):
            ortho_ds = None
            return None

    # Read full ortho tile
    ortho_crop = ortho_ds.ReadAsArray(col_off, row_off, tile_w, tile_h)
    has_alpha = n_bands >= 4
    alpha = ortho_crop[3] if has_alpha else np.full((tile_h, tile_w), 255, dtype=np.uint8)
    ortho_ds = None

    # Tile geotransform
    tile_gt = (
        ortho_gt[0] + col_off * ortho_gt[1], ortho_gt[1], 0.0,
        ortho_gt[3] + row_off * ortho_gt[5], 0.0, ortho_gt[5],
    )

    # UTM x coords for this tile
    x_utm_all = tile_gt[0] + np.arange(tile_w, dtype=np.float64) * tile_gt[1]

    # Coordinate transformers
    utm_to_wgs84 = Transformer.from_crs(epsg_str, "EPSG:4326", always_xy=True)
    wgs84_to_utm = Transformer.from_crs("EPSG:4326", epsg_str, always_xy=True)

    # Pre-compute camera poses + UTM positions
    shot_names = []
    cam_centres = []
    cam_rotations = []
    cam_translations = []
    cam_models = []
    cam_utm = []

    for sn, shot in shots.items():
        R = Rotation.from_rotvec(shot["rotation"]).as_matrix()
        t = np.array(shot["translation"], dtype=np.float64)
        C = -R.T @ t
        lat, lon, alt = _enu_to_lla_scalar(C[0], C[1], C[2], ref_lat, ref_lon, ref_alt)
        ux, uy = wgs84_to_utm.transform(lon, lat)
        shot_names.append(sn)
        cam_centres.append(C)
        cam_rotations.append(R)
        cam_translations.append(t)
        cam_models.append(cameras_dict[shot["camera"]])
        cam_utm.append((ux, uy))

    cam_utm = np.array(cam_utm)

    # DTM / DSM
    dtm_sampler = DTMSampler(dtm_path) if dtm_path else None
    dsm_sampler = DTMSampler(dsm_path) if dsm_path else None

    # Output buffer (start with ortho RGB)
    out_buf = np.zeros((3, tile_h, tile_w), dtype=np.uint8)
    for b in range(min(3, n_bands)):
        out_buf[b] = ortho_crop[b]

    # Image cache
    img_cache = ImageCache(image_dir, max_cached=30)

    best_nadir = np.full((tile_h, tile_w), np.inf, dtype=np.float32)
    max_lateral_dist = 200.0
    n_updated = 0

    # Filter cameras near tile
    crop_cx = (x_utm_all[0] + x_utm_all[-1]) / 2
    crop_cy = tile_gt[3] + (tile_h / 2) * tile_gt[5]
    crop_r = math.sqrt((x_utm_all[-1] - x_utm_all[0])**2 +
                        (tile_h * abs(tile_gt[5]))**2) / 2

    nearby = [i for i in range(len(shot_names))
              if math.sqrt((cam_utm[i, 0] - crop_cx)**2 +
                           (cam_utm[i, 1] - crop_cy)**2) < crop_r + max_lateral_dist]

    for cam_idx in nearby:
        sname = shot_names[cam_idx]
        C = cam_centres[cam_idx]
        R = cam_rotations[cam_idx]
        t_vec = cam_translations[cam_idx]
        cam = cam_models[cam_idx]
        cx_utm, cy_utm = cam_utm[cam_idx]

        img = img_cache.get(sname)
        if img is None:
            continue

        for local_row in range(tile_h):
            y_utm = tile_gt[3] + local_row * tile_gt[5]
            if abs(cy_utm - y_utm) > max_lateral_dist:
                continue

            dx = x_utm_all - cx_utm
            dy = cy_utm - y_utm
            col_mask = (dx * dx + dy * dy) < max_lateral_dist ** 2
            col_mask &= alpha[local_row] > 0
            if not np.any(col_mask):
                continue

            valid_cols = np.where(col_mask)[0]
            x_valid = x_utm_all[valid_cols]

            if dtm_sampler:
                z_ground = dtm_sampler.sample_row(x_valid, y_utm)
                has_z = ~np.isnan(z_ground)
                if not np.any(has_z):
                    continue
                valid_cols = valid_cols[has_z]
                x_valid = x_valid[has_z]
                z_ground = z_ground[has_z]
            else:
                z_ground = np.full(len(valid_cols), default_z)

            lons, lats = utm_to_wgs84.transform(x_valid, np.full_like(x_valid, y_utm))
            e, n_coord, u = lla_to_enu_vec(lats, lons, z_ground, ref_lat, ref_lon, ref_alt)
            points_enu = np.column_stack([e, n_coord, u])

            vecs = points_enu - C
            dists = np.linalg.norm(vecs, axis=1)
            dists_safe = np.where(dists > 0, dists, 1.0)
            cos_nadir = -vecs[:, 2] / dists_safe
            nadir_angles = np.arccos(np.clip(cos_nadir, -1.0, 1.0))

            better = nadir_angles < best_nadir[local_row, valid_cols]
            if not np.any(better):
                continue

            better_idx = np.where(better)[0]

            # DSM occlusion check
            if dsm_sampler is not None:
                b_x = x_utm_all[valid_cols[better_idx]]
                b_y = np.full(len(better_idx), y_utm)
                b_z = z_ground[better_idx]
                cam_alt = _enu_to_lla_scalar(C[0], C[1], C[2], ref_lat, ref_lon, ref_alt)[2]
                unoccluded = check_occlusion_vec(
                    cx_utm, cy_utm, cam_alt, b_x, b_y, b_z, dsm_sampler,
                )
                better_idx = better_idx[unoccluded]
                if len(better_idx) == 0:
                    continue

            pts_better = points_enu[better_idx]
            _proj_fn = project_points_pinhole_vec if use_pinhole else project_points_brown_vec
            px, py, proj_valid = _proj_fn(pts_better, R, t_vec, cam)
            if not np.any(proj_valid):
                continue

            colors = sample_image_vec(img, px, py, proj_valid)
            success_idx = better_idx[proj_valid]
            output_cols = valid_cols[success_idx]
            out_buf[0, local_row, output_cols] = colors[proj_valid, 2]
            out_buf[1, local_row, output_cols] = colors[proj_valid, 1]
            out_buf[2, local_row, output_cols] = colors[proj_valid, 0]
            best_nadir[local_row, output_cols] = nadir_angles[success_idx].astype(np.float32)
            n_updated += int(np.sum(proj_valid))

    n_filled = int(np.sum(best_nadir < np.inf))
    return (col_off, row_off, tile_w, tile_h, out_buf, n_filled, n_updated)


def process_true_ortho_full(
    recon_path: str,
    ortho_path: str,
    image_dir: str,
    output_path: str,
    dtm_path: Optional[str] = None,
    dsm_path: Optional[str] = None,
    default_z: float = 0.0,
    tile_size: int = 512,
    workers: int = 1,
) -> None:
    """Process the full orthophoto in tiles, optionally with multiprocessing."""
    from multiprocessing import Pool

    t0 = time.time()

    # Load reconstruction
    print(f"Loading reconstruction: {recon_path}")
    recon = load_reconstruction(recon_path)
    ref_lla = recon["reference_lla"]
    print(f"  {len(recon['shots'])} shots")

    # Auto-detect undistorted images
    undist_dir, undist_cams, use_pinhole = _detect_undistorted(image_dir, recon_path)
    if use_pinhole:
        image_dir = undist_dir
        if undist_cams:
            recon["cameras"] = undist_cams
        print(f"  Using undistorted images: {image_dir} (pinhole projection)")
    else:
        print(f"  Using original images: {image_dir} (brown distortion projection)")

    # Open ortho for metadata
    ortho_ds = gdal.Open(ortho_path, gdal.GA_ReadOnly)
    ortho_gt = ortho_ds.GetGeoTransform()
    ortho_srs_wkt = ortho_ds.GetProjection()
    ortho_w = ortho_ds.RasterXSize
    ortho_h = ortho_ds.RasterYSize
    n_bands = ortho_ds.RasterCount
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ortho_srs_wkt)
    epsg_str = f"EPSG:{srs.GetAuthorityCode(None) or '32613'}"
    ortho_ds = None
    print(f"Ortho: {ortho_w}x{ortho_h}, {n_bands} bands, CRS: {epsg_str}")

    # Build tile grid
    tiles = []
    tc = tr = 0
    for row_off in range(0, ortho_h, tile_size):
        th = min(tile_size, ortho_h - row_off)
        for col_off in range(0, ortho_w, tile_size):
            tw = min(tile_size, ortho_w - col_off)
            tiles.append((tc, tr, tw, th, col_off, row_off))
            tc += 1
        tr += 1

    n_tiles = len(tiles)
    print(f"Tile grid: {tile_size}px tiles, {n_tiles} total")

    # Prepare recon data for workers (serializable dict)
    recon_data = {
        "shots": recon["shots"],
        "cameras": recon["cameras"],
        "reference_lla": recon["reference_lla"],
    }

    # Build args for each tile
    tile_args = [
        (t[0], t[1], t[2], t[3], t[4], t[5],
         ortho_path, image_dir, dtm_path, dsm_path, default_z,
         recon_data, ortho_gt, ortho_srs_wkt, epsg_str, use_pinhole)
        for t in tiles
    ]

    # Create output file
    print(f"Creating output: {output_path}")
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path, ortho_w, ortho_h, 3, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
    )
    out_ds.SetGeoTransform(ortho_gt)
    out_ds.SetProjection(ortho_srs_wkt)
    out_ds.FlushCache()
    out_ds = None

    # Process tiles
    total_filled = 0
    total_updated = 0
    tiles_done = 0
    tiles_skipped = 0

    def write_tile(result):
        nonlocal total_filled, total_updated, tiles_done, tiles_skipped
        if result is None:
            tiles_skipped += 1
            tiles_done += 1
            return
        col_off, row_off, tw, th, buf, n_filled, n_updated = result
        ds = gdal.Open(output_path, gdal.GA_Update)
        for b in range(3):
            ds.GetRasterBand(b + 1).WriteArray(buf[b], col_off, row_off)
        ds.FlushCache()
        ds = None
        total_filled += n_filled
        total_updated += n_updated
        tiles_done += 1

    if workers <= 1:
        for i, args in enumerate(tile_args):
            elapsed = time.time() - t0
            rate = tiles_done / max(1, elapsed)
            eta = (n_tiles - tiles_done) / max(0.001, rate)
            print(f"\r  Tile {tiles_done+1}/{n_tiles}  "
                  f"filled={total_filled:,}  skip={tiles_skipped}  "
                  f"ETA={eta:.0f}s", end="", flush=True)
            result = _process_tile(args)
            write_tile(result)
    else:
        print(f"Using {workers} worker processes")
        with Pool(workers) as pool:
            for result in pool.imap_unordered(_process_tile, tile_args, chunksize=1):
                write_tile(result)
                if tiles_done % 10 == 0:
                    elapsed = time.time() - t0
                    rate = tiles_done / max(1, elapsed)
                    eta = (n_tiles - tiles_done) / max(0.001, rate)
                    print(f"\r  Tiles {tiles_done}/{n_tiles}  "
                          f"filled={total_filled:,}  skip={tiles_skipped}  "
                          f"ETA={eta:.0f}s", end="", flush=True)

    elapsed = time.time() - t0
    total_px = ortho_w * ortho_h
    print(f"\n\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total tiles:  {n_tiles} ({tiles_skipped} skipped as empty)")
    print(f"  Filled:       {total_filled:,} / {total_px:,} pixels")
    print(f"  Projections:  {total_updated:,}")

    # Convert to COG
    cog_path = output_path.replace(".tif", "_cog.tif")
    print(f"\nConverting to COG: {cog_path}")
    gdal.Translate(cog_path, output_path, format="COG",
                   creationOptions=["COMPRESS=LZW", "OVERVIEW_RESAMPLING=NEAREST"])
    print(f"  COG written: {cog_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="True ortho post-processor for ODM orthophotos.",
    )
    parser.add_argument("reconstruction",
                        help="Path to reconstruction.topocentric.json")
    parser.add_argument("orthophoto",
                        help="Path to ODM orthophoto (extent, GeoTransform, CRS)")
    parser.add_argument("image_dir",
                        help="Directory containing camera images")
    parser.add_argument("-o", "--output", default="true_ortho.tif",
                        help="Output path (default: true_ortho.tif)")
    parser.add_argument("--dtm", default=None,
                        help="DTM raster for ground Z values")
    parser.add_argument("--dsm", default=None,
                        help="DSM raster for occlusion detection")
    parser.add_argument("--crop", default=None,
                        help="UTM crop: x_min,y_min,x_max,y_max")
    parser.add_argument("--max-cameras", type=int, default=5,
                        help="Max candidate cameras per pixel (default: 5)")
    parser.add_argument("--default-z", type=float, default=0.0,
                        help="Default ground Z (ellipsoidal m) when no DTM")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes for full-image mode (default: 1)")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile size for full-image mode (default: 512)")
    args = parser.parse_args()

    crop = None
    if args.crop:
        parts = args.crop.split(",")
        if len(parts) != 4:
            parser.error("--crop must be x_min,y_min,x_max,y_max")
        crop = tuple(float(v) for v in parts)

    if crop:
        # Crop mode: single region
        process_true_ortho(
            recon_path=args.reconstruction,
            ortho_path=args.orthophoto,
            image_dir=args.image_dir,
            output_path=args.output,
            dtm_path=args.dtm,
            dsm_path=args.dsm,
            crop_utm=crop,
            max_cameras=args.max_cameras,
            default_z=args.default_z,
        )
    else:
        # Full-image tiled mode
        process_true_ortho_full(
            recon_path=args.reconstruction,
            ortho_path=args.orthophoto,
            image_dir=args.image_dir,
            output_path=args.output,
            dtm_path=args.dtm,
            dsm_path=args.dsm,
            default_z=args.default_z,
            tile_size=args.tile_size,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
