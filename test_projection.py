#!/usr/bin/env python3
"""
Test Mode A (EXIF-based) pixel projection against ground truth from gcpeditpro.txt.
Validates the pinhole camera math documented in docs/dji_m3e_camera_model.md.
"""
import math

# ---------------------------------------------------------------------------
# Ground truth from ~/stratus/ghostrider/gcpeditpro.txt
# Format: easting northing elev px py filename gcpname
# ---------------------------------------------------------------------------
GROUND_TRUTH = [
    # (filename, true_px, true_py)
    ("DJI_20260212133306_0137_V.JPG", 2672.1155855500006, 2100.1229708323303),
    ("DJI_20260212135629_0028_V.JPG", 3072.5489462212327, 2285.5928463778696),
    ("DJI_20260212135601_0011_V.JPG", 3260.2319576800937, 2103.2569991707064),
    ("DJI_20260212133304_0136_V.JPG", 2669.461563147476,  1422.7047883545247),
    ("DJI_20260212135627_0026_V.JPG", 3059.913143591404,  1285.7198045883051),
    ("DJI_20260212135630_0029_V.JPG", 3063.431395586837,  2809.1185250918966),
    ("DJI_20260212135600_0010_V.JPG", 3266.8301786947086, 1280.0213471724887),
    ("DJI_20260212135602_0012_V.JPG", 3262.869258854054,  2762.219684978168),
]

# EXIF data pulled from exiftool -n for each image
# (lat, lon, abs_alt_m, gimbal_yaw_deg, gimbal_roll_deg, img_w, img_h, focal_mm, focal_35mm)
EXIF = {
    "DJI_20260212133306_0137_V.JPG": dict(lat=35.3912053055556, lon=-106.161318583333, abs_alt=1968.821, yaw=21.40,  roll=180.0, w=5280, h=3956, focal=12.29, focal35=24),
    "DJI_20260212135629_0028_V.JPG": dict(lat=35.3912025,       lon=-106.161399861111, abs_alt=1943.004, yaw=-72.60, roll=0.0,   w=5280, h=3956, focal=12.29, focal35=24),
    "DJI_20260212135601_0011_V.JPG": dict(lat=35.3913291111111, lon=-106.161260416667, abs_alt=1941.049, yaw=108.10, roll=0.0,   w=5280, h=3956, focal=12.29, focal35=24),
    "DJI_20260212133304_0136_V.JPG": dict(lat=35.3913566388889, lon=-106.1612495,      abs_alt=1967.178, yaw=-158.30,roll=0.0,   w=5280, h=3956, focal=12.29, focal35=24),
    "DJI_20260212135627_0026_V.JPG": dict(lat=35.3911472222222, lon=-106.161189777778, abs_alt=1945.075, yaw=107.60, roll=180.0, w=5280, h=3956, focal=12.29, focal35=24),
    "DJI_20260212135630_0029_V.JPG": dict(lat=35.3912315833333, lon=-106.161506888889, abs_alt=1942.452, yaw=107.30, roll=180.0, w=5280, h=3956, focal=12.29, focal35=24),
    "DJI_20260212135600_0010_V.JPG": dict(lat=35.3913718611111, lon=-106.161422916667, abs_alt=1939.368, yaw=-72.10, roll=180.0, w=5280, h=3956, focal=12.29, focal35=24),
    "DJI_20260212135602_0012_V.JPG": dict(lat=35.3912930555556, lon=-106.161124444444, abs_alt=1942.134, yaw=-72.20, roll=180.0, w=5280, h=3956, focal=12.29, focal35=24),
}

# GCP 1 from emlid.csv
GCP = dict(
    lat=35.39123866,
    lon=-106.1613153,
    elev_ellip_ft=6134.472,   # WGS84 ellipsoidal height, feet
)
GCP_elev_m = GCP['elev_ellip_ft'] * 0.3048   # → meters

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
METERS_PER_DEG_LAT = 111319.9  # metres per degree latitude

def enu_from_gps(cam_lat, cam_lon, gcp_lat, gcp_lon, cam_alt_m, gcp_alt_m):
    """Return (dE, dN, dU) in metres from camera to GCP."""
    mid_lat = math.radians((cam_lat + gcp_lat) / 2)
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(mid_lat)
    dE = (gcp_lon - cam_lon) * meters_per_deg_lon
    dN = (gcp_lat - cam_lat) * METERS_PER_DEG_LAT
    dU = gcp_alt_m - cam_alt_m
    return dE, dN, dU

def sensor_from_35mm(focal_mm, focal35_mm, img_w, img_h):
    """Derive sensor dims (mm) and focal length in pixels from 35mm-equiv focal."""
    scale = focal_mm / focal35_mm
    # Full-frame diagonal = sqrt(36^2 + 24^2) mm
    full_diag = math.sqrt(36**2 + 24**2)
    sensor_diag = full_diag * scale
    aspect = img_w / img_h
    sensor_h = sensor_diag / math.sqrt(1 + aspect**2)
    sensor_w = sensor_h * aspect
    fx = focal_mm * img_w / sensor_w
    fy = focal_mm * img_h / sensor_h
    cx = img_w / 2.0
    cy = img_h / 2.0
    return fx, fy, cx, cy

def project_nadir(dE, dN, dU, yaw_deg, roll_deg, fx, fy, cx, cy):
    """
    Project a point (dE, dN, dU) relative to nadir camera into pixel (px, py).

    Camera model (ENU frame, Gimbal pitch=-90 assumed):
      yaw_deg : GimbalYawDegree (CW from North)
      roll_deg: GimbalRollDegree (0 or 180 for DJI M3E)

    Convention (roll=0):
      Image top → drone forward (yaw direction)
      Image right → 90° CW from forward
      Optical axis → ENU down (0, 0, -1)

    roll=180: camera rotated 180° around optical axis → X_cam and Y_cam both negate.
    """
    if dU >= 0:
        return None  # GCP above or at camera height — can't project

    psi = math.radians(yaw_deg)   # CW from North

    # Camera axes in ENU frame (roll=0):
    # X_cam (image right): 90° CW from forward in horizontal plane
    #   forward = (sin psi, cos psi, 0)  [East=sin, North=cos]
    #   right   = (cos psi, -sin psi, 0)
    # Y_cam (image down): opposite of forward = (-sin psi, -cos psi, 0)
    # Z_cam (optical):    (0, 0, -1)  [into scene = down]

    Xx, Xy = math.cos(psi), -math.sin(psi)   # X_cam in ENU
    Yx, Yy = -math.sin(psi), -math.cos(psi)  # Y_cam in ENU

    if abs(roll_deg - 180.0) < 1.0:
        # Roll=180: negate both X and Y axes
        Xx, Xy = -Xx, -Xy
        Yx, Yy = -Yx, -Yy

    # Project into camera frame (Z component: -dU since ENU Z_cam = (0,0,-1))
    cam_x = Xx * dE + Xy * dN          # dU term is 0 for X_cam
    cam_y = Yx * dE + Yy * dN          # dU term is 0 for Y_cam
    cam_z = -dU                          # positive when GCP below camera

    # Pinhole projection
    px = fx * cam_x / cam_z + cx
    py = fy * cam_y / cam_z + cy
    return px, py

# ---------------------------------------------------------------------------
# Run test
# ---------------------------------------------------------------------------
print("=" * 80)
print("Mode A (EXIF) Pixel Projection Validation")
print("GCP 1 ellipsoidal height: {:.2f} m".format(GCP_elev_m))
print("=" * 80)

errors = []
for fname, true_px, true_py in GROUND_TRUTH:
    e = EXIF[fname]
    fx, fy, cx, cy = sensor_from_35mm(e['focal'], e['focal35'], e['w'], e['h'])

    dE, dN, dU = enu_from_gps(e['lat'], e['lon'], GCP['lat'], GCP['lon'],
                               e['abs_alt'], GCP_elev_m)

    result = project_nadir(dE, dN, dU, e['yaw'], e['roll'], fx, fy, cx, cy)

    if result is None:
        print(f"{fname}: PROJECTION FAILED (dU={dU:.1f}m)")
        continue

    pred_px, pred_py = result
    err_x = pred_px - true_px
    err_y = pred_py - true_py
    err_mag = math.sqrt(err_x**2 + err_y**2)
    errors.append(err_mag)

    roll_lbl = f"roll={e['roll']:.0f}"
    print(f"\n{fname} ({roll_lbl}, yaw={e['yaw']:+.1f}°, dH={-dU:.1f}m)")
    print(f"  True:  px={true_px:.1f}, py={true_py:.1f}")
    print(f"  Pred:  px={pred_px:.1f}, py={pred_py:.1f}")
    print(f"  Error: dx={err_x:+.1f}, dy={err_y:+.1f}, |e|={err_mag:.1f} px")
    print(f"  Camera: dE={dE:.2f}m, dN={dN:.2f}m, fx={fx:.1f}px")

if errors:
    print("\n" + "=" * 80)
    print(f"Mean error:   {sum(errors)/len(errors):.1f} px")
    print(f"Max error:    {max(errors):.1f} px")
    print(f"Min error:    {min(errors):.1f} px")
    print("=" * 80)
