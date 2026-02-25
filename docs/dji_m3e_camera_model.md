# DJI M3E Camera Model and Pixel Projection

## Confirmed EXIF Tags (from ~/stratus/ghostrider/raw/)

```
Camera Model Name    : M3E
Focal Length         : 12.29 mm
Focal Length 35mm    : 24 mm
Image Width          : 5280 px
Image Height         : 3956 px
Gimbal Pitch Degree  : -89.9 to -90.0  (nadir — straight down)
Gimbal Yaw Degree    : varies           (heading, degrees CW from true North)
Gimbal Roll Degree   : 0.0 or 180.0    (see roll convention below)
Flight Pitch Degree  : varies           (body attitude, NOT used for projection)
Flight Yaw Degree    : varies           (body attitude, NOT used for projection)
Flight Roll Degree   : varies           (body attitude, NOT used for projection)
GPS Latitude         : varies
GPS Longitude        : varies
Absolute Altitude    : varies           (WGS84 ellipsoidal height, meters — USE THIS)
Relative Altitude    : varies           (AGL above takeoff, meters — NOT reliable for projection)
Field Of View        : NOT PRESENT in M3E EXIF — derive from FocalLength + FocalLengthIn35mmFormat
EXIF Orientation     : 1 (always — image pixels are stored as-is, roll correction is manual)
```

**Important**: All camera parameters MUST be read from EXIF per image.
Do not hardcode values based on drone model name. Different projects may use
different cameras. The pipeline must work generically from EXIF alone.

## Critical Height System Finding (validated 2026-02-25)

**Use AbsoluteAltitude (WGS84 ellipsoidal, meters) for the camera height.**
**Use the Emlid CSV `Ellipsoidal height` column (feet → meters) for the GCP height.**

Do NOT use:
- `Relative Altitude` (AGL above takeoff — gives 3900+ px error)
- Emlid `Elevation` column (NAVD88 geoid height — gives 2× worse projection error)

The AbsoluteAltitude EXIF field on DJI M3E is the GPS ellipsoidal height (WGS84),
not barometric. Confirmed by consistency with the Emlid ellipsoidal height readings
and validated against 8 ground-truth pixel observations.

```python
# Correct height calculation:
cam_alt_m = exif['AbsoluteAltitude']            # WGS84 ellipsoidal, meters
gcp_alt_m = emlid_row['Ellipsoidal height'] * 0.3048  # ft to meters
dU = gcp_alt_m - cam_alt_m                      # negative when GCP below camera
```

## GimbalRoll Convention

GimbalRoll ∈ {0°, 180°}. Both values appear in the same dataset.
EXIF Orientation is always 1 — the JPEG pixels are NOT corrected for roll.
Roll MUST be applied in the projection math.

- **Roll = 0°**: Image top faces drone forward (GimbalYaw direction). Standard.
- **Roll = 180°**: Camera is rotated 180° around the optical axis. Image is
  flipped: top faces backward from drone, left/right swapped. Negate both
  X and Y camera axes in the projection (see Mode A math below).

## Deriving Sensor Dimensions from EXIF

`FieldOfView` is NOT present in M3E EXIF. Use `FocalLength` + `FocalLengthIn35mmFormat`:

```python
import math

# Inputs from EXIF (always available on DJI M3E)
focal_mm   = exif['FocalLength']              # e.g. 12.29
focal35_mm = exif['FocalLengthIn35mmFormat']  # e.g. 24
img_w      = exif['ImageWidth']               # e.g. 5280
img_h      = exif['ImageHeight']              # e.g. 3956

# Derive sensor size from 35mm equivalence
# Full-frame diagonal = sqrt(36^2 + 24^2) = 43.27 mm
FULL_FRAME_DIAG = math.sqrt(36**2 + 24**2)
scale = focal_mm / focal35_mm
sensor_diag_mm = FULL_FRAME_DIAG * scale

aspect = img_w / img_h
sensor_h_mm = sensor_diag_mm / math.sqrt(1 + aspect**2)
sensor_w_mm = sensor_h_mm * aspect

# Focal length in pixels (fx ≈ fy ≈ 3660 px for M3E)
fx = focal_mm * img_w / sensor_w_mm
fy = focal_mm * img_h / sensor_h_mm
cx = img_w / 2.0                       # principal point (assume center)
cy = img_h / 2.0
# Verified: fx=fy=3659.7 px for M3E at 12.29mm, 5280x3956
```

## Mode A: EXIF-Based Pixel Projection

Validated against 8 ground-truth observations (gcpeditpro.txt vs computed):
**mean error 73.6 px, range 12–126 px** at altitudes 7–99 m above GCP.
Dominant error source: camera GPS positioning (3–5 m RMS → 50–100 px at 70 m AGL).

### Validated Python implementation

```python
import math

MPD_LAT = 111319.9  # metres per degree latitude (approximate)

def project_mode_a(cam_lat, cam_lon, cam_abs_alt_m,
                   gcp_lat, gcp_lon, gcp_ellip_alt_m,
                   gimbal_yaw_deg, gimbal_roll_deg,
                   focal_mm, focal35_mm, img_w, img_h):
    """
    Project GCP world coords to pixel (px, py) using EXIF-only camera model.
    Returns (px, py) or None if GCP is behind or out of frame.

    cam_abs_alt_m   : EXIF AbsoluteAltitude (WGS84 ellipsoidal, meters)
    gcp_ellip_alt_m : Emlid 'Ellipsoidal height' * 0.3048 (WGS84 ellipsoidal, meters)
    gimbal_yaw_deg  : GimbalYawDegree (CW from North)
    gimbal_roll_deg : GimbalRollDegree (0 or 180)
    """
    # --- Sensor geometry ---
    FULL_FRAME_DIAG = math.sqrt(36**2 + 24**2)
    scale = focal_mm / focal35_mm
    sensor_diag = FULL_FRAME_DIAG * scale
    aspect = img_w / img_h
    sensor_h = sensor_diag / math.sqrt(1 + aspect**2)
    sensor_w = sensor_h * aspect
    fx = focal_mm * img_w / sensor_w
    fy = focal_mm * img_h / sensor_h
    cx, cy = img_w / 2.0, img_h / 2.0

    # --- ENU displacement from camera to GCP ---
    mid_lat = math.radians((cam_lat + gcp_lat) / 2)
    dE = (gcp_lon - cam_lon) * MPD_LAT * math.cos(mid_lat)
    dN = (gcp_lat - cam_lat) * MPD_LAT
    dU = gcp_ellip_alt_m - cam_abs_alt_m   # negative when GCP below camera

    if dU >= 0:
        return None  # GCP at or above camera

    # --- Camera axes in ENU (nadir camera, pitch = -90°) ---
    # Convention (roll=0):
    #   Image top  → drone forward (GimbalYaw direction)
    #   Image right → 90° CW from forward in horizontal plane
    #   Optical axis → ENU-Down
    psi = math.radians(gimbal_yaw_deg)   # CW from North
    Xx, Xy = math.cos(psi), -math.sin(psi)    # X_cam (image right) in ENU XY
    Yx, Yy = -math.sin(psi), -math.cos(psi)   # Y_cam (image down)  in ENU XY

    if abs(gimbal_roll_deg - 180.0) < 1.0:
        # Roll=180: both image axes flip (camera rotated 180° around optical axis)
        Xx, Xy, Yx, Yy = -Xx, -Xy, -Yx, -Yy

    # --- Project to camera frame ---
    cam_x = Xx * dE + Xy * dN   # dU term = 0 (Z_cam is vertical)
    cam_y = Yx * dE + Yy * dN
    cam_z = -dU                  # positive into scene

    # --- Pinhole projection ---
    px = fx * cam_x / cam_z + cx
    py = fy * cam_y / cam_z + cy

    if 0 <= px < img_w and 0 <= py < img_h:
        return (px, py)
    return None
```

### For non-nadir GimbalPitch
Apply the full ZYX rotation matrix using all three gimbal angles (pitch, yaw, roll)
via scipy.spatial.transform.Rotation. The nadir simplification above holds when
GimbalPitch ∈ [-91°, -89°], which covers all M3E mapping images.

## Mode B: Reconstruction-Based Pixel Projection

### reconstruction.json Format (researched 2026-02-25 from OpenSfM source)

File location: `<odm_task_output>/opensfm/reconstruction.json`
Format: JSON array (list of reconstructions; use index [0] for the main one).

```json
[{
  "reference_lla": {
    "latitude": 35.391,      // WGS84, origin of topocentric ENU frame
    "longitude": -106.161,
    "altitude": 1870.0       // ellipsoidal, meters
  },
  "cameras": {
    "v2 dji m3e 5280 3956 perspective 0.6916": {
      "focal":  0.6921,      // normalized: focal_px / max(width, height)
      "width":  5280,
      "height": 3956,
      "k1":     0.012,       // radial distortion coefficient
      "k2":    -0.008,       // radial distortion coefficient
      "projection_type": "perspective"
    }
  },
  "shots": {
    "DJI_20260212133306_0137_V.JPG": {
      "rotation":    [rx, ry, rz],   // angle-axis (Rodrigues) vector, 3 elements
      "translation": [tx, ty, tz],   // world-to-camera translation
      "camera":      "v2 dji m3e 5280 3956 perspective 0.6916",
      "gps_position": [dE, dN, dU],  // camera GPS in ENU topocentric (metres)
      "gps_dop":    2.5
    }
  }
}]
```

### Coordinate System (confirmed via OpenSfM TopocentricConverter)

**ENU topocentric** relative to `reference_lla`:
- x = East (metres)
- y = North (metres)
- z = Up (metres)

Verified: `TopocentricConverter(lat, lon, alt).to_topocentric(lat+ε, lon, alt)` returns
positive y (North) and `to_topocentric(lat, lon+ε, alt)` returns positive x (East).

### Rotation Convention

`rotation` is an **angle-axis (Rodrigues) vector**: `[rx, ry, rz]`.
The rotation matrix R = `Rotation.from_rotvec([rx,ry,rz]).as_matrix()` (scipy).

Transform convention: **world → camera**
```
p_cam = R @ p_world + t
```
where `p_world` is in ENU topocentric metres relative to `reference_lla`.

Camera position in world: `origin = -R.T @ t` (confirmed via `pose.get_origin()`).

### Mode B Projection Implementation

```python
import numpy as np
from scipy.spatial.transform import Rotation

def project_mode_b(gcp_lat, gcp_lon, gcp_ellip_alt_m,
                   shot, camera, reference_lla):
    """
    Project GCP to pixel using SfM-refined camera pose.

    shot          : dict from reconstruction.json shots[filename]
    camera        : dict from reconstruction.json cameras[shot['camera']]
    reference_lla : dict with 'latitude', 'longitude', 'altitude'
    """
    from opensfm.geo import TopocentricConverter  # or reimplement without OpenSfM dep
    tc = TopocentricConverter(reference_lla['latitude'],
                              reference_lla['longitude'],
                              reference_lla['altitude'])

    # GCP in ENU topocentric
    p_world = np.array(tc.to_topocentric(gcp_lat, gcp_lon, gcp_ellip_alt_m))

    # Camera pose
    R = Rotation.from_rotvec(shot['rotation']).as_matrix()
    t = np.array(shot['translation'])
    p_cam = R @ p_world + t   # 3D point in camera frame

    if p_cam[2] <= 0:
        return None  # behind camera

    # Camera intrinsics (OpenSfM normalized focal)
    w, h = camera['width'], camera['height']
    max_dim = max(w, h)
    focal_px = camera['focal'] * max_dim
    k1 = camera.get('k1', 0.0)
    k2 = camera.get('k2', 0.0)

    # Normalized image coordinates (before distortion)
    xn = p_cam[0] / p_cam[2]
    yn = p_cam[1] / p_cam[2]

    # Radial distortion: r^2 = xn^2 + yn^2
    r2 = xn**2 + yn**2
    distort = 1 + k1 * r2 + k2 * r2**2
    xd, yd = xn * distort, yn * distort

    # Pixel coordinates (principal point = center)
    px = focal_px * xd + w / 2.0
    py = focal_px * yd + h / 2.0

    if 0 <= px < w and 0 <= py < h:
        return (px, py)
    return None
```

### TopocentricConverter Without OpenSfM Dependency

If importing opensfm is not available in the plugin environment:

```python
import math

def topocentric_from_lla(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """ENU coordinates (x=E, y=N, z=U) relative to reference point."""
    MPD_LAT = 111319.9
    mid_lat = math.radians((lat + ref_lat) / 2)
    x = (lon - ref_lon) * MPD_LAT * math.cos(mid_lat)
    y = (lat - ref_lat) * MPD_LAT
    z = alt - ref_alt
    return x, y, z
```

### Validation Status
No ghostrider ODM reconstruction.json currently available to validate Mode B accuracy.
The R3 issue should be re-opened when a reconstruction becomes available.
Expected accuracy: ±5–20 px (SfM residuals dominate over GPS noise).

## Accuracy Notes (Measured)

| Mode | Mean error | Max error | Dominant source |
|------|-----------|-----------|-----------------|
| EXIF (Mode A) | **73.6 px** | 126 px | Camera GPS (3–5 m RMS → ~50–100 px at 70 m AGL) |
| Reconstruction (Mode B) | ~5–20 px (est.) | TBD | SfM residuals + GCP coordinate accuracy |

Measured on 8 ground-truth images from ~/stratus/ghostrider/ at 7–99 m above GCP.
At 256×256 px sub-image crop, Mode A is sufficient: worst-case GCP is within ~1
sub-image width of center, almost always visible within the crop.

## Validation Ground Truth

- Images: `~/stratus/ghostrider/raw/` (1738 images, DJI M3E)
- Emlid CSV: `~/stratus/ghostrider/emlid.csv`
- Manually-tagged reference: `~/stratus/ghostrider/gcpeditpro.txt`
  (8 entries for GCP "1" with hand-tagged pixel coordinates)
- Test script: `~/git/geo/test_projection.py`

## WebODM Plugin Development Note

The auto-gcp plugin lives in `~/git/webodm/coreplugins/auto-gcp/` (git repo).
Test iterations may use the live WebODM container, but all changes must be
committed to the repo and verified to work in a freshly-built container.
Do not rely on modifications made directly in the running container.
