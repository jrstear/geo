#!/usr/bin/env python3
"""
ortho_uncertainty.py — Generate single-band GeoTIFF of estimated horizontal
positional uncertainty across an orthophoto.

Formula per pixel:
    sigma_horizontal = sigma_DTM(x,y) * tan(theta_camera(x,y)) + sigma_reconstruction

Where:
    sigma_DTM           = ground-surface elevation uncertainty (default 0.1 m)
    theta_camera(x,y)   = off-nadir angle of the most-nadir camera seeing that pixel
    sigma_reconstruction = base reconstruction accuracy (e.g. 0.035 m from rmse.py)

Usage:
    conda run -n geo python accuracy_study/ortho_uncertainty.py \\
        reconstruction.topocentric.json orthophoto.tif \\
        -o uncertainty.tif [--sigma-dtm 0.1] [--sigma-recon 0.035]

    # With DTM for elevation-aware processing:
    conda run -n geo python accuracy_study/ortho_uncertainty.py \\
        reconstruction.topocentric.json orthophoto.tif \\
        -o uncertainty.tif --dtm dtm.tif

The uncertainty field is spatially smooth (varies only with camera geometry),
so we compute on a coarse grid (~1 sample per metre) and bilinearly upsample
to full orthophoto resolution.  This makes processing feasible for large
orthophotos (e.g. 80k x 120k pixels at 5 cm GSD).

Requires: GDAL (osgeo.gdal), numpy, scipy, pyproj — all in conda env geo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from osgeo import gdal, osr

gdal.UseExceptions()

try:
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
except ImportError:
    raise RuntimeError("scipy is required: conda install scipy")

try:
    from pyproj import CRS as ProjCRS, Transformer
except ImportError:
    raise RuntimeError("pyproj is required: conda install pyproj")


# ---------------------------------------------------------------------------
# WGS84 ellipsoid (matching OpenSFM opensfm/geo.py)
# ---------------------------------------------------------------------------

_WGS84_a = 6378137.0
_WGS84_b = 6356752.314245


def _ecef_from_lla(lat_deg, lon_deg, alt):
    """WGS84 geodetic to ECEF. Accepts scalars or arrays."""
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


def _enu_to_projected_batch(
    enu_xyz: np.ndarray,
    ref_lat: float, ref_lon: float, ref_alt: float,
    transformer,
) -> np.ndarray:
    """Convert (N, 3) ENU points to projected CRS. Returns (N, 3)."""
    rx, ry, rz = _ecef_from_lla(ref_lat, ref_lon, ref_alt)
    sa = np.sin(np.radians(ref_lat))
    ca = np.cos(np.radians(ref_lat))
    so = np.sin(np.radians(ref_lon))
    co = np.cos(np.radians(ref_lon))

    x, y, z = enu_xyz[:, 0], enu_xyz[:, 1], enu_xyz[:, 2]
    ex = -so * x + (-sa * co) * y + (ca * co) * z + rx
    ey =  co * x + (-sa * so) * y + (ca * so) * z + ry
    ez =             ca       * y +  sa       * z + rz

    a, b = _WGS84_a, _WGS84_b
    ea2 = (a ** 2 - b ** 2) / a ** 2
    eb2 = (a ** 2 - b ** 2) / b ** 2
    p = np.sqrt(ex ** 2 + ey ** 2)
    theta = np.arctan2(ez * a, p * b)
    lon_r = np.arctan2(ey, ex)
    lat_r = np.arctan2(
        ez + eb2 * b * np.sin(theta) ** 3,
        p - ea2 * a * np.cos(theta) ** 3,
    )
    N = a / np.sqrt(1 - ea2 * np.sin(lat_r) ** 2)
    alt_out = p / np.cos(lat_r) - N

    lats = np.degrees(lat_r)
    lons = np.degrees(lon_r)
    px, py = transformer.transform(lons, lats)
    return np.column_stack([px, py, alt_out])


# ---------------------------------------------------------------------------
# Reconstruction loading
# ---------------------------------------------------------------------------


def load_cameras(recon_path: str) -> Tuple[dict, dict, dict]:
    """
    Load reconstruction.topocentric.json, extracting only needed fields.

    The full JSON can be 600+ MB due to the "points" field.  We discard
    points immediately to save memory.

    Returns (reference_lla, cameras, shots).
    """
    print("Loading reconstruction (this may take a minute for large files)...")
    t0 = time.time()

    with open(recon_path) as f:
        data = json.load(f)
    r = data[0]
    del data

    ref = r["reference_lla"]
    cameras = r["cameras"]

    raw_shots = r.pop("shots", {})
    r.pop("points", None)
    del r

    shots = {}
    for sname, sdata in raw_shots.items():
        shots[sname] = {
            "rotation": sdata["rotation"],
            "translation": sdata["translation"],
            "camera": sdata["camera"],
        }
    del raw_shots

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return ref, cameras, shots


def compute_camera_positions_utm(
    shots: dict,
    cameras: dict,
    ref: dict,
    target_epsg: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute camera positions in the target projected CRS.

    Returns:
        cam_pos_utm  : (N, 3) — camera positions in target CRS (x, y, z_m)
        shot_names   : list of shot filenames
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)

    shot_names = list(shots.keys())
    n = len(shot_names)
    enu_positions = np.empty((n, 3))

    for i, name in enumerate(shot_names):
        shot = shots[name]
        R = Rotation.from_rotvec(shot["rotation"]).as_matrix()
        t = np.array(shot["translation"], dtype=float)
        C = -R.T @ t
        enu_positions[i] = C

    cam_pos_utm = _enu_to_projected_batch(
        enu_positions,
        ref["latitude"], ref["longitude"], ref["altitude"],
        transformer,
    )

    return cam_pos_utm, shot_names


# ---------------------------------------------------------------------------
# Off-nadir angle on a coarse grid
# ---------------------------------------------------------------------------


def compute_offnadir_coarse(
    grid_x: np.ndarray,       # (gh, gw) — easting of coarse grid points
    grid_y: np.ndarray,       # (gh, gw) — northing
    grid_z: np.ndarray,       # (gh, gw) — ground elevation (m)
    cam_xy: np.ndarray,       # (N, 2) — camera x, y
    cam_z: np.ndarray,        # (N,) — camera altitude (m)
    k_nearest: int = 20,
) -> np.ndarray:
    """
    For each coarse grid point, find the most-nadir camera (smallest off-nadir
    angle) among the k nearest cameras.

    Off-nadir angle = arctan(horizontal_dist / vertical_dist) where
    vertical_dist = cam_z - ground_z.

    Returns off-nadir angle in radians, shape (gh, gw).
    """
    gh, gw = grid_x.shape

    # Build KDTree of camera XY positions
    tree = cKDTree(cam_xy)

    # Flatten grid
    gx_flat = grid_x.ravel()
    gy_flat = grid_y.ravel()
    gz_flat = grid_z.ravel()
    n_pts = gx_flat.size

    # Query k nearest cameras for each grid point
    query_pts = np.column_stack([gx_flat, gy_flat])
    dists_h, indices = tree.query(query_pts, k=min(k_nearest, len(cam_z)))

    # dists_h: (n_pts, k) — horizontal distances
    # indices: (n_pts, k) — camera indices

    if dists_h.ndim == 1:
        dists_h = dists_h[:, np.newaxis]
        indices = indices[:, np.newaxis]

    # Vertical distance from each camera to ground
    cam_alts = cam_z[indices]  # (n_pts, k)
    ground_alts = gz_flat[:, np.newaxis]  # (n_pts, 1)
    dz = cam_alts - ground_alts  # (n_pts, k)
    dz = np.maximum(dz, 1.0)  # avoid division by zero; cameras must be above ground

    # Off-nadir angle = arctan(horizontal_dist / vertical_dist)
    offnadir = np.arctan2(dists_h, dz)  # (n_pts, k)

    # Best (minimum) off-nadir angle per grid point
    best_offnadir = np.min(offnadir, axis=1)  # (n_pts,)

    return best_offnadir.reshape(gh, gw).astype(np.float32)


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------


def get_ortho_info(ortho_path: str) -> Tuple[list, int, int, str, int]:
    """Read orthophoto metadata. Returns (geotransform, width, height, proj_wkt, epsg)."""
    ds = gdal.Open(ortho_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open orthophoto: {ortho_path}")

    gt = list(ds.GetGeoTransform())
    w = ds.RasterXSize
    h = ds.RasterYSize
    proj_wkt = ds.GetProjection()

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj_wkt)
    epsg = int(srs.GetAuthorityCode(None))

    ds = None
    return gt, w, h, proj_wkt, epsg


def sample_dtm(
    dtm_path: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    default_elev: float,
) -> np.ndarray:
    """Sample DTM at projected coordinates. Returns elevations in metres."""
    ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open DTM: {dtm_path}")

    gt = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    nodata = ds.GetRasterBand(1).GetNoDataValue()

    # Convert to pixel coords
    col = ((grid_x - gt[0]) / gt[1]).astype(np.int32)
    row = ((grid_y - gt[3]) / gt[5]).astype(np.int32)

    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)

    c_min, c_max = int(col.min()), int(col.max())
    r_min, r_max = int(row.min()), int(row.max())

    data = ds.GetRasterBand(1).ReadAsArray(c_min, r_min, c_max - c_min + 1, r_max - r_min + 1)
    ds = None

    if data is None:
        return np.full_like(grid_x, default_elev, dtype=np.float32)

    elevations = data[row - r_min, col - c_min].astype(np.float32)
    if nodata is not None:
        elevations[elevations == nodata] = default_elev

    return elevations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_ortho(
    recon_path: str,
    ortho_path: str,
    output_path: str,
    dtm_path: Optional[str],
    sigma_dtm: float,
    sigma_recon: float,
    ground_elev: Optional[float],
    coarse_step_m: float,
    k_nearest: int,
):
    """
    Compute uncertainty on a coarse grid, then upsample to full ortho resolution.
    """
    t0 = time.time()

    # --- Load orthophoto metadata ---
    gt, ortho_w, ortho_h, proj_wkt, epsg = get_ortho_info(ortho_path)
    px_size = gt[1]
    print(f"Orthophoto: {ortho_w} x {ortho_h}, EPSG:{epsg}, pixel size {px_size:.4f} m")

    # --- Load reconstruction ---
    ref, cameras, shots = load_cameras(recon_path)
    print(f"Loaded {len(shots)} camera shots")

    # --- Compute camera positions in UTM ---
    cam_pos, shot_names = compute_camera_positions_utm(shots, cameras, ref, epsg)
    print(f"Camera altitude range: {cam_pos[:, 2].min():.1f} - {cam_pos[:, 2].max():.1f} m")

    median_cam_z = np.median(cam_pos[:, 2])

    # Default ground elevation
    if ground_elev is not None:
        default_elev = ground_elev
    else:
        default_elev = median_cam_z - 75.0
        print(f"Estimated ground elevation: {default_elev:.1f} m (camera median - 75 m)")

    # --- Build coarse grid ---
    # Coarse step in pixels
    step_px = max(1, int(round(coarse_step_m / px_size)))
    coarse_w = (ortho_w + step_px - 1) // step_px + 1
    coarse_h = (ortho_h + step_px - 1) // step_px + 1

    print(f"Coarse grid: {coarse_w} x {coarse_h} (step {step_px} px = {step_px * px_size:.2f} m)")

    # Coarse grid coordinates (pixel centres)
    cols_c = np.arange(coarse_w, dtype=np.float64) * step_px + 0.5
    rows_c = np.arange(coarse_h, dtype=np.float64) * step_px + 0.5
    cols_c = np.minimum(cols_c, ortho_w - 0.5)
    rows_c = np.minimum(rows_c, ortho_h - 0.5)

    col_grid, row_grid = np.meshgrid(cols_c, rows_c)
    grid_x = gt[0] + col_grid * gt[1]
    grid_y = gt[3] + row_grid * gt[5]

    # --- Ground elevation ---
    if dtm_path is not None:
        print("Sampling DTM elevations...")
        grid_z = sample_dtm(dtm_path, grid_x, grid_y, default_elev)
        print(f"  DTM range: {grid_z.min():.1f} - {grid_z.max():.1f} m")
    else:
        grid_z = np.full((coarse_h, coarse_w), default_elev, dtype=np.float32)

    # --- Compute off-nadir angles on coarse grid ---
    print(f"Computing off-nadir angles (k={k_nearest} nearest cameras)...")
    t1 = time.time()
    offnadir = compute_offnadir_coarse(
        grid_x, grid_y, grid_z,
        cam_pos[:, :2], cam_pos[:, 2],
        k_nearest=k_nearest,
    )
    print(f"  Done in {time.time() - t1:.1f}s")
    print(f"  Off-nadir range: {np.degrees(offnadir.min()):.1f} - {np.degrees(offnadir.max()):.1f} deg")

    # --- Compute uncertainty on coarse grid ---
    uncertainty_coarse = sigma_dtm * np.tan(offnadir) + sigma_recon
    print(f"  Uncertainty range: {uncertainty_coarse.min():.4f} - {uncertainty_coarse.max():.4f} m")

    # --- Write coarse grid to temporary GeoTIFF ---
    coarse_gt = [
        gt[0],
        gt[1] * step_px,
        0.0,
        gt[3],
        0.0,
        gt[5] * step_px,
    ]

    coarse_path = output_path + ".coarse.tif"
    driver = gdal.GetDriverByName("GTiff")
    coarse_ds = driver.Create(
        coarse_path, coarse_w, coarse_h, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=3"],
    )
    coarse_ds.SetGeoTransform(coarse_gt)
    coarse_ds.SetProjection(proj_wkt)
    coarse_band = coarse_ds.GetRasterBand(1)
    coarse_band.SetNoDataValue(-9999.0)
    coarse_band.WriteArray(uncertainty_coarse)
    coarse_ds.FlushCache()
    coarse_ds = None

    # --- Upsample to full resolution using GDAL Warp (bilinear) ---
    print(f"Upsampling to full resolution ({ortho_w} x {ortho_h})...")
    t2 = time.time()

    import multiprocessing
    n_threads = multiprocessing.cpu_count()
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        width=ortho_w,
        height=ortho_h,
        outputBounds=[
            gt[0],                          # minX
            gt[3] + gt[5] * ortho_h,        # minY
            gt[0] + gt[1] * ortho_w,        # maxX
            gt[3],                           # maxY
        ],
        resampleAlg=gdal.GRA_Bilinear,
        multithread=True,
        warpOptions=[f"NUM_THREADS={n_threads}"],
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            f"NUM_THREADS={n_threads}",
        ],
    )
    result = gdal.Warp(output_path, coarse_path, options=warp_opts)
    if result is None:
        raise RuntimeError("GDAL Warp failed")

    # Set band description and nodata
    result.GetRasterBand(1).SetDescription("Horizontal positional uncertainty (m)")
    result.GetRasterBand(1).SetNoDataValue(-1.0)
    result.FlushCache()
    result = None

    print(f"  Upsampled in {time.time() - t2:.1f}s")

    # --- Mask to orthophoto coverage ---
    # Set uncertainty to nodata where the orthophoto has no valid pixels
    # (transparent alpha or all-zero RGB).
    print("Masking to orthophoto coverage...")
    ortho_ds2 = gdal.Open(ortho_path, gdal.GA_ReadOnly)
    out_ds = gdal.Open(output_path, gdal.GA_Update)
    unc_band = out_ds.GetRasterBand(1)
    block_h = 512
    for row_off in range(0, ortho_h, block_h):
        bh = min(block_h, ortho_h - row_off)
        # Check alpha channel if present, else check RGB for all-zero
        if ortho_ds2.RasterCount >= 4:
            alpha_block = ortho_ds2.GetRasterBand(4).ReadAsArray(0, row_off, ortho_w, bh)
            no_data_mask = alpha_block == 0
        else:
            r = ortho_ds2.GetRasterBand(1).ReadAsArray(0, row_off, ortho_w, bh)
            g = ortho_ds2.GetRasterBand(2).ReadAsArray(0, row_off, ortho_w, bh)
            b = ortho_ds2.GetRasterBand(3).ReadAsArray(0, row_off, ortho_w, bh)
            no_data_mask = (r == 0) & (g == 0) & (b == 0)
        unc_block = unc_band.ReadAsArray(0, row_off, ortho_w, bh)
        unc_block[no_data_mask] = -1.0
        unc_band.WriteArray(unc_block, 0, row_off)
    out_ds.FlushCache()
    ortho_ds2 = None
    out_ds = None
    print("  Masked.")

    # Clean up coarse temp file
    import os
    os.remove(coarse_path)

    # --- Summary stats from output (excluding nodata) ---
    out_ds = gdal.Open(output_path, gdal.GA_ReadOnly)
    band = out_ds.GetRasterBand(1)
    stats = band.ComputeStatistics(False)
    out_ds = None
    val_min, val_max, val_mean, val_std = stats
    print(f"\nOutput statistics:")
    print(f"  Min:  {val_min:.4f} m")
    print(f"  Max:  {val_max:.4f} m")
    print(f"  Mean: {val_mean:.4f} m")
    print(f"  StdDev: {val_std:.4f} m")

    # --- Generate RGBA COG with red-to-green colormap ---
    # Green = low uncertainty (good), Yellow = moderate, Red = high uncertainty (bad)
    # Color range is distributed over the actual data range for this job.
    print("\nGenerating colored RGBA COG overlay...")
    t3 = time.time()
    rgba_path = output_path.replace(".tif", "_overlay.tif")
    raw_tmp = output_path  # the float32 file we just created

    out_ds = gdal.Open(raw_tmp, gdal.GA_ReadOnly)
    unc_w = out_ds.RasterXSize
    unc_h = out_ds.RasterYSize
    unc_gt = out_ds.GetGeoTransform()
    unc_srs = out_ds.GetProjection()

    # Create RGBA temp file (tiled for COG conversion)
    rgba_tmp = rgba_path + ".tmp.tif"
    drv = gdal.GetDriverByName("GTiff")
    rgba_ds = drv.Create(rgba_tmp, unc_w, unc_h, 4, gdal.GDT_Byte,
                         options=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256",
                                  "COMPRESS=DEFLATE"])
    rgba_ds.SetGeoTransform(unc_gt)
    rgba_ds.SetProjection(unc_srs)

    # Process in blocks
    block_h = 512
    nodata_val = -1.0
    for row_off in range(0, unc_h, block_h):
        bh = min(block_h, unc_h - row_off)
        unc_block = out_ds.GetRasterBand(1).ReadAsArray(0, row_off, unc_w, bh)

        r = np.zeros((bh, unc_w), dtype=np.uint8)
        g = np.zeros((bh, unc_w), dtype=np.uint8)
        b = np.zeros((bh, unc_w), dtype=np.uint8)
        a = np.zeros((bh, unc_w), dtype=np.uint8)

        valid = (unc_block != nodata_val) & np.isfinite(unc_block)
        if np.any(valid):
            # Normalize to 0-1 over the data range
            t_val = np.clip((unc_block[valid] - val_min) / max(val_max - val_min, 1e-6), 0.0, 1.0)

            # Green (0) → Yellow (0.5) → Red (1.0)
            r[valid] = np.clip(t_val * 2.0 * 255, 0, 255).astype(np.uint8)          # 0→255 over first half, 255 for second half
            g[valid] = np.clip((1.0 - t_val) * 2.0 * 255, 0, 255).astype(np.uint8)  # 255 for first half, 255→0 over second half
            b[valid] = 0
            a[valid] = 180  # semi-transparent overlay

        rgba_ds.GetRasterBand(1).WriteArray(r, 0, row_off)
        rgba_ds.GetRasterBand(2).WriteArray(g, 0, row_off)
        rgba_ds.GetRasterBand(3).WriteArray(b, 0, row_off)
        rgba_ds.GetRasterBand(4).WriteArray(a, 0, row_off)

    # Burn in legend at top-right corner (outside corridor, won't occlude data)
    try:
        import cv2
        legend_w, legend_h = 280, 100
        margin = 20
        lx = unc_w - legend_w - margin
        ly = margin
        if lx > 0 and ly + legend_h < unc_h:
            # Read the legend region
            legend_r = rgba_ds.GetRasterBand(1).ReadAsArray(lx, ly, legend_w, legend_h)
            legend_g = rgba_ds.GetRasterBand(2).ReadAsArray(lx, ly, legend_w, legend_h)
            legend_b = rgba_ds.GetRasterBand(3).ReadAsArray(lx, ly, legend_w, legend_h)
            legend_a = rgba_ds.GetRasterBand(4).ReadAsArray(lx, ly, legend_w, legend_h)

            # Build legend as BGR image for cv2 drawing
            legend_img = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
            legend_img[:, :] = (40, 40, 40)  # dark background

            font = cv2.FONT_HERSHEY_SIMPLEX
            M = 3.28084  # M_TO_FT

            # Title
            cv2.putText(legend_img, "Positional uncertainty", (8, 18),
                        font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # Color ramp bar
            bar_x, bar_y, bar_w, bar_h = 8, 28, legend_w - 16, 20
            for px_i in range(bar_w):
                t = px_i / max(bar_w - 1, 1)
                r_val = int(min(t * 2.0 * 255, 255))
                g_val = int(min((1.0 - t) * 2.0 * 255, 255))
                cv2.line(legend_img, (bar_x + px_i, bar_y),
                         (bar_x + px_i, bar_y + bar_h), (0, g_val, r_val), 1)

            # Min/max labels
            cv2.putText(legend_img, f"{val_min * M:.2f} ft", (bar_x, bar_y + bar_h + 14),
                        font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
            max_label = f"{val_max * M:.2f} ft"
            (tw, _), _ = cv2.getTextSize(max_label, font, 0.35, 1)
            cv2.putText(legend_img, max_label, (bar_x + bar_w - tw, bar_y + bar_h + 14),
                        font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

            # Mean label centered
            mean_label = f"mean: {val_mean * M:.2f} ft"
            (tw2, _), _ = cv2.getTextSize(mean_label, font, 0.35, 1)
            cv2.putText(legend_img, mean_label, ((legend_w - tw2) // 2, bar_y + bar_h + 30),
                        font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

            # Green/red labels
            cv2.putText(legend_img, "low", (bar_x, bar_y - 4),
                        font, 0.3, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(legend_img, "high", (bar_x + bar_w - 24, bar_y - 4),
                        font, 0.3, (0, 0, 200), 1, cv2.LINE_AA)

            # Write legend pixels to RGBA bands
            rgba_ds.GetRasterBand(1).WriteArray(legend_img[:, :, 2], lx, ly)  # R
            rgba_ds.GetRasterBand(2).WriteArray(legend_img[:, :, 1], lx, ly)  # G
            rgba_ds.GetRasterBand(3).WriteArray(legend_img[:, :, 0], lx, ly)  # B
            legend_alpha = np.full((legend_h, legend_w), 230, dtype=np.uint8)
            rgba_ds.GetRasterBand(4).WriteArray(legend_alpha, lx, ly)
            print(f"  Legend burned in at ({lx}, {ly})")
    except ImportError:
        print("  cv2 not available — skipping legend burn-in")

    rgba_ds.FlushCache()
    rgba_ds = None
    out_ds = None

    # Convert to COG
    gdal.Translate(rgba_path, rgba_tmp, format="COG",
                   creationOptions=["COMPRESS=DEFLATE", "OVERVIEW_RESAMPLING=NEAREST"])

    os.remove(rgba_tmp)
    print(f"  Colored overlay: {rgba_path}")
    print(f"  Color ramp: green ({val_min:.4f}m) → yellow → red ({val_max:.4f}m)")
    print(f"  Generated in {time.time() - t3:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Raw uncertainty: {output_path}")
    print(f"  RGBA COG overlay: {rgba_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate horizontal positional uncertainty overlay GeoTIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "reconstruction", help="Path to reconstruction.topocentric.json",
    )
    parser.add_argument(
        "orthophoto", help="Path to orthophoto GeoTIFF (for extent/CRS/resolution)",
    )
    parser.add_argument(
        "-o", "--output", default="uncertainty.tif",
        help="Output GeoTIFF path (default: uncertainty.tif)",
    )
    parser.add_argument(
        "--dtm", default=None,
        help="Path to DTM GeoTIFF for ground elevation (optional; flat model if omitted)",
    )
    parser.add_argument(
        "--sigma-dtm", type=float, default=0.1,
        help="DTM surface uncertainty in metres (default: 0.1)",
    )
    parser.add_argument(
        "--sigma-recon", type=float, default=0.035,
        help="Base reconstruction uncertainty in metres (default: 0.035)",
    )
    parser.add_argument(
        "--ground-elev", type=float, default=None,
        help="Ground elevation in metres for flat model (default: estimated from cameras)",
    )
    parser.add_argument(
        "--coarse-step", type=float, default=1.0,
        help="Coarse grid spacing in metres (default: 1.0). Uncertainty is smooth, "
             "so 1-5 m spacing is usually sufficient.",
    )
    parser.add_argument(
        "--k-nearest", type=int, default=20,
        help="Number of nearest cameras to consider per grid point (default: 20)",
    )

    args = parser.parse_args()

    process_ortho(
        recon_path=args.reconstruction,
        ortho_path=args.orthophoto,
        output_path=args.output,
        dtm_path=args.dtm,
        sigma_dtm=args.sigma_dtm,
        sigma_recon=args.sigma_recon,
        ground_elev=args.ground_elev,
        coarse_step_m=args.coarse_step,
        k_nearest=args.k_nearest,
    )


if __name__ == "__main__":
    main()
