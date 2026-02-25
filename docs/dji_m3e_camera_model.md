# DJI M3E Camera Model and Pixel Projection

## Confirmed EXIF Tags (from ~/stratus/ghostrider/raw/)

```
Camera Model Name    : M3E
Focal Length         : 12.29 mm
Focal Length 35mm    : 24 mm
Image Width          : 5280 px
Image Height         : 3956 px
Gimbal Pitch Degree  : -90.00   (nadir — straight down)
Gimbal Yaw Degree    : varies   (heading, degrees from true North)
Gimbal Roll Degree   : +180.00  (DJI convention: camera mounted inverted on gimbal)
Flight Pitch Degree  : varies   (body attitude)
Flight Yaw Degree    : varies
Flight Roll Degree   : varies
GPS Latitude         : varies
GPS Longitude        : varies
Relative Altitude    : varies   (AGL, meters)
Field Of View        : varies   (diagonal FOV, degrees — read per-image, do NOT hardcode)
```

**Important**: All camera parameters MUST be read from EXIF per image.
Do not hardcode values (e.g., do not hardcode focal length or sensor dimensions
based on the M3E model name). Different projects may use different cameras.

## Deriving Sensor Dimensions from EXIF

Physical sensor dimensions are not always in EXIF but can be derived:

```python
import math

# Inputs from EXIF
focal_mm = 12.29        # FocalLength
fov_diag_deg = ...      # FieldOfView (diagonal)
img_w = 5280            # ImageWidth
img_h = 3956            # ImageHeight

# Compute sensor diagonal from diagonal FOV + focal length
fov_diag_rad = math.radians(fov_diag_deg)
sensor_diag_mm = 2 * focal_mm * math.tan(fov_diag_rad / 2)

# Distribute to width/height by image aspect ratio
aspect = img_w / img_h
sensor_h_mm = sensor_diag_mm / math.sqrt(1 + aspect**2)
sensor_w_mm = sensor_h_mm * aspect

# Focal length in pixels
fx = focal_mm * img_w / sensor_w_mm   # pixels
fy = focal_mm * img_h / sensor_h_mm   # pixels
cx = img_w / 2.0                       # principal point (assume center)
cy = img_h / 2.0
```

## Mode A: EXIF-Based Pixel Projection

For nadir images (GimbalPitch ≈ -90°), the projection simplifies:

```
Camera at (lat_c, lon_c, alt_c_m)
GCP at (lat_g, lon_g, elev_g_m)

Step 1: Convert to local ENU (East-North-Up) centered at camera
  dE = haversine_east(lat_c, lon_c, lat_c, lon_g)  [m]
  dN = haversine_north(lat_c, lon_c, lat_g, lon_c) [m]
  dU = elev_g_m - (camera_elev_m)                   [m, negative = below camera]

  (Use pyproj or haversine for lat/lon → meters)

Step 2: Rotate ENU by gimbal yaw (yaw = degrees from North, CW)
  yaw_rad = math.radians(gimbal_yaw_deg)
  # Camera X axis points East rotated by yaw (image right → heading direction)
  # Camera Y axis points North rotated by yaw
  # Camera Z axis points DOWN (-U)

  cam_x =  dE * cos(yaw_rad) + dN * sin(yaw_rad)
  cam_y = -dE * sin(yaw_rad) + dN * cos(yaw_rad)
  cam_z = -dU   # positive z = into scene (down)

Step 3: Apply GimbalRoll=180 correction
  # DJI mounts the camera inverted; GimbalRoll=180 means image is upright
  # Effect: flip cam_x axis (image columns run right = East for yaw=0, but
  #         image rows run DOWN = South for yaw=0 after roll flip)
  cam_x = -cam_x   # flip horizontal due to roll=180

Step 4: Pinhole projection (if cam_z > 0, GCP is in front of camera)
  if cam_z <= 0:
      return None  # GCP behind camera

  px = fx * cam_x / cam_z + cx
  py = fy * cam_y / cam_z + cy

  if 0 <= px < img_w and 0 <= py < img_h:
      return (px, py)
  return None
```

**Note**: For non-nadir angles (GimbalPitch ≠ -90°), apply full rotation matrix
using all three gimbal angles (pitch, yaw, roll). The nadir simplification
above collapses pitch to -90° which makes cam_z = distance below camera.

## Mode B: Reconstruction-Based Pixel Projection (TODO — research in R3)

When `opensfm/reconstruction.json` is available from a prior ODM run,
use the SfM-refined camera poses instead of EXIF gimbal angles.

Key questions for R3 to answer:
1. What coordinate system are positions/rotations in? (OpenSfM uses local
   coordinates; need the world→local transform to convert GCP coordinates)
2. What rotation convention? (angle-axis? rotation matrix?)
3. Does reconstruction.json include distortion coefficients (k1, k2)?
4. How to get world→local transform? (may be in `reference_lla.json` or
   `topocentric_proj` field)
5. Validate by projecting GCPs from `~/stratus/ghostrider/emlid.csv` using
   the reconstruction from a ghostrider ODM run, checking against manually
   tagged positions in `~/stratus/ghostrider/gcpeditpro.txt`.

## Accuracy Notes

| Mode | Expected error | Dominant source |
|------|---------------|-----------------|
| EXIF | ±30–150 px | Gimbal yaw uncertainty (1-2° → ~50-100 px at 300ft) |
| Reconstruction | ±5–20 px | GCP coordinate accuracy + SfM residuals |

For shift-click confirmation, ±150 px is sufficient for a 256×256 sub-image
centered at the estimate, as long as the GCP is at least partially visible.

## Validation Ground Truth

- Images: `~/stratus/ghostrider/raw/`
- Emlid CSV: `~/stratus/ghostrider/emlid.csv`
- Manually-tagged reference: `~/stratus/ghostrider/gcpeditpro.txt`
  Use this to check projected pixel accuracy for Mode A and Mode B.
