# GCP Accuracy Experiment Framework — Technical Specification

_An agent-ready specification. Each component section is self-contained and can be
handed to a separate Claude Code agent with no additional context._

---

## Context

### End-to-end pipeline

`csv2gcp.py` reads an Emlid rover CSV plus drone images and produces `gcp_list.txt`
(one row per GCP×image pair). The GCPEditorPro Angular app loads that file, lets the
user confirm or correct pixel tags, then exports two files:

- `gcp_confirmed.txt` — confirmed GCP-* rows only (used as ODM control input)
- `chk_confirmed.txt` — confirmed CHK-* rows only (held out; used for accuracy validation)

WebODM runs ODM bundle adjustment using `gcp_confirmed.txt` and produces
`reconstruction.json` (OpenSfM output) at:

```
{task_dir}/assets/odm_opensfm/reconstruction.json
```

### gcp_list.txt format

Tab-separated text. Line 1 is a CRS header (EPSG code or PROJ string). Remaining
lines are data rows:

```
EPSG:6529
geo_x\tgeo_y\tgeo_z\tpx\tpy\timage_name\tgcp_label\tconfidence[\tmarker_bbox]
```

- `geo_x`, `geo_y`, `geo_z` — surveyed projected coordinates in the CRS declared on
  line 1 (easting, northing, elevation). Trailing zeros stripped.
- `px`, `py` — pixel coordinates (float, trailing zeros stripped).
- `image_name` — bare filename, no path.
- `gcp_label` — e.g. `GCP-1`, `CHK-3`, `DUP-A`.
- `confidence` — `projection` (EXIF-based Mode A) or `reconstruction` (SfM Mode B).
- `marker_bbox` — optional 9th column; ignore when reading for RMSE purposes.

`gcp_confirmed.txt` and `chk_confirmed.txt` use the same format, containing only the
rows the user confirmed in GCPEditorPro.

### reconstruction.json structure

```json
[{
  "cameras": {
    "v2 unknown unknown 4000 3000 perspective 0": {
      "width": 4000,
      "height": 3000,
      "focal": 0.85,
      "k1": -0.05,
      "k2": 0.01
    }
  },
  "shots": {
    "DJI_20260212134359_0475_V.JPG": {
      "camera": "v2 unknown unknown 4000 3000 perspective 0",
      "rotation": [rx, ry, rz],
      "translation": [tx, ty, tz],
      "gps_position": [lat, lon, alt],
      "gps_dop": 1.0
    }
  },
  "reference_lla": {
    "latitude": 34.123,
    "longitude": -106.456,
    "altitude": 1850.0
  },
  "points": {}
}]
```

Key conventions:

- `rotation` — Rodrigues vector (axis-angle compact form). Convert to 3×3 matrix via
  `scipy.spatial.transform.Rotation.from_rotvec(r).as_matrix()` or the manual
  Rodrigues formula. This is the world-to-camera rotation R such that
  `p_cam = R @ p_world + t`.
- `translation` — world-to-camera translation vector t (not camera position).
  Camera center in world coords: `C = -R^T @ t`.
- `focal` — normalised focal length: `focal_px = focal * max(width, height)`.
- `k1`, `k2` — Brown radial distortion coefficients (applied in normalised
  coordinates before multiplying by focal_px).
- `reference_lla` — WGS-84 geodetic origin of the local ENU reconstruction frame.

### Coordinate transform: reconstruction ENU → projected CRS

The reconstruction frame is a local ENU (East-North-Up) coordinate system centred at
`reference_lla`. A 3D point `p_enu = [E, N, U]` in this frame can be converted to
projected coordinates (e.g. EPSG:6529) as follows:

```python
from pyproj import Transformer, CRS
import math

def enu_to_projected(p_enu, ref_lat_deg, ref_lon_deg, ref_alt_m, epsg):
    """
    Convert a point in local ENU (metres) to projected CRS coordinates.

    Uses flat-earth ENU approximation (valid to <1 mm at typical survey scales
    of a few km). If higher accuracy is needed, use full ECEF intermediary.
    """
    METERS_PER_DEG_LAT = 111319.9
    mid_lat_rad = math.radians(ref_lat_deg)

    # ENU → WGS-84 offset
    dlon = p_enu[0] / (METERS_PER_DEG_LAT * math.cos(mid_lat_rad))
    dlat = p_enu[1] / METERS_PER_DEG_LAT
    lon = ref_lon_deg + dlon
    lat = ref_lat_deg + dlat
    alt = ref_alt_m + p_enu[2]

    xfm = Transformer.from_crs('EPSG:4326', epsg, always_xy=True)
    x, y = xfm.transform(lon, lat)
    return x, y, alt
```

The `epsg` string must match the CRS declared in the gcp_list.txt header (e.g.
`'EPSG:6529'`). Pass `always_xy=True` so pyproj treats the first argument as
longitude.

Note: the reconstruction Z axis is Up (metres above the WGS-84 ellipsoid at
`reference_lla.altitude`). The projected Z from this transform is therefore
ellipsoidal altitude in metres. The survey CSV stores orthometric elevation in the
local CRS units (feet for EPSG:6529). This unit/datum mismatch must be handled in
rmse_calc.py — see the Surveyed Z section below.

---

## Pipeline Optimization for RMSE Experiments

### Key insight: only sparse SfM is needed

RMSE measures where bundle adjustment *thinks* GCP/CHK points are in 3D space vs.
surveyed positions. This answer comes entirely from sparse SfM + bundle adjustment.
The following stages are **irrelevant for RMSE** and must be skipped in the
experiment driver:

| Stage | RMSE relevance | Skip? |
|---|---|---|
| Feature extraction | Builds keypoint graph | Required |
| Feature matching | Builds image connectivity | Required |
| Sparse SfM + bundle adjustment | **This IS the RMSE** | Required |
| Dense reconstruction (MVS) | Point density for DSM | **Skip** |
| Filter points | Clean point cloud | **Skip** |
| Meshing | Surface for ortho | **Skip** |
| Texturing | Color on mesh | **Skip** |
| DEM / orthophoto | Final deliverables | **Skip** |

Skipping dense reconstruction saves **10–20× per run**. Cost drops from ~$0.60–0.80
(full ODM at medium quality) to ~$0.03–0.10 (sparse only).

### Do NOT attempt image subsets

SfM requires a fully-connected image graph. Selecting only images near GCPs removes
the bridging images that chain GCP clusters together, causing the reconstruction to
fail or split into disconnected components. The speedup from skipping dense
reconstruction already dominates any image-subset benefit. Blackening non-GCP image
regions breaks feature matching for the same reason.

### Implementation: two options

#### Option A — OpenSfM direct (preferred, ~20–45 min/run for ~1400 images)

ODM's `dataset` stage creates `opensfm/config.yaml` with the GCP file path, camera
priors, and coordinate reference. Run that first, then invoke OpenSfM stages directly,
stopping before `compute_depthmaps`:

```bash
# Step 1: ODM dataset stage only (populates opensfm/config.yaml with GCP path etc.)
docker run --rm -v /data/project:/datasets/project opendronemap/odm:3.3.0 \
  --project-path /datasets project \
  --rerun-from dataset --end-with dataset

# Step 2: OpenSfM sparse reconstruction only (no depth maps / dense)
docker run --rm --entrypoint bash \
  -v /data/project:/datasets/project opendronemap/odm:3.3.0 \
  -c 'cd /datasets/project && \
      opensfm detect_features opensfm && \
      opensfm match_features opensfm && \
      opensfm create_tracks opensfm && \
      opensfm reconstruct opensfm'
```

`reconstruction.json` path: `/datasets/project/opensfm/reconstruction.json`

**Verify before implementing**: confirm `--end-with dataset` is a valid flag in
`opendronemap/odm:3.3.0` (`docker run opendronemap/odm:3.3.0 --help | grep end-with`).
If unavailable, fall back to Option B.

#### Option B — ODM minimal flags (fallback, ~1.5–2 hr/run)

Still runs lowest-quality dense reconstruction but skips all downstream stages:

```bash
docker run --rm -v /data/project:/datasets/project opendronemap/odm:3.3.0 \
  --project-path /datasets project \
  --pc-quality lowest --skip-3dmodel --skip-report --orthophoto-resolution 100
```

`reconstruction.json` path: `/datasets/project/odm_opensfm/reconstruction.json`

**Note the path difference** — `rmse_calc.py` and `run.sh` must use the correct path
for whichever option is implemented.

### Revised cost estimates

| Approach | Time / run | Cost / run | 50-run ablation |
|---|---|---|---|
| Full ODM medium (original plan) | 6–10 hr | $0.60–0.80 | ~$40 |
| Option B: ODM + skip-3dmodel + lowest | 1.5–2 hr | $0.15–0.20 | ~$10 |
| Option A: OpenSfM sparse only | 20–45 min | $0.03–0.08 | ~$3 |

---

## Component 1: rmse_calc.py

**Location:** `/Users/jrstear/git/geo/TargetSighter/rmse_calc.py`

**Purpose:** Given a completed ODM reconstruction and a set of user-confirmed check
points, triangulate each check point from its confirmed pixel observations, convert to
the survey CRS, and report RMSE statistics.

**Dependencies:** `numpy`, `scipy`, `pyproj` (all present in the `geo` conda env).

### CLI

```
conda run -n geo python TargetSighter/rmse_calc.py \
    reconstruction.json \
    chk_confirmed.txt \
    emlid.csv \
    [--crs EPSG:6529]
```

Arguments:

| Argument | Description |
|---|---|
| `reconstruction.json` | Path to ODM OpenSfM output |
| `chk_confirmed.txt` | GCPEditorPro confirmed CHK-* export (same format as gcp_list.txt) |
| `emlid.csv` | Emlid rover CSV with surveyed ground-truth coordinates |
| `--crs EPSG:xxxx` | Override CRS if not detectable from chk_confirmed.txt header |

### Algorithm

#### Step 1 — Parse reconstruction.json

```python
import json
import numpy as np
from scipy.spatial.transform import Rotation

with open(recon_path) as f:
    recon = json.load(f)[0]   # always a list; take element 0

cameras = recon['cameras']
shots   = recon['shots']
ref_lla = recon['reference_lla']
```

For each shot name that appears in `chk_confirmed.txt`, extract:

```python
shot = shots[image_name]
cam  = cameras[shot['camera']]

R = Rotation.from_rotvec(shot['rotation']).as_matrix()   # 3×3 world→camera
t = np.array(shot['translation'])                         # world→camera translation
C = -R.T @ t                                              # camera centre in world (ENU)

w, h       = cam['width'], cam['height']
focal_px   = cam['focal'] * max(w, h)
k1         = cam.get('k1', 0.0)
k2         = cam.get('k2', 0.0)
cx, cy     = w / 2.0, h / 2.0
```

If a shot name from `chk_confirmed.txt` is absent from `shots`, skip that observation
and emit a warning to stderr.

#### Step 2 — Parse chk_confirmed.txt

Read line 1 as the CRS string. Parse remaining lines as tab-separated fields. Group
observations by `gcp_label`:

```python
obs_by_label = {}   # label → [(image_name, px, py), ...]
for line in lines[1:]:
    fields = line.rstrip('\n').split('\t')
    geo_x, geo_y, geo_z, px, py, image_name, label, confidence = fields[:8]
    obs_by_label.setdefault(label, []).append((image_name, float(px), float(py)))
```

#### Step 3 — Parse emlid.csv (surveyed ground truth)

Reuse `parse_survey_csv()` from `csv2gcp.py`. Import it directly:

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # TargetSighter/
from csv2gcp import parse_survey_csv, _cs_name_to_epsg
```

Build a lookup from label to surveyed position in the projected CRS:

```python
survey_gcps = parse_survey_csv(emlid_csv, fallback_crs=crs_override)
survey_by_label = {g['label']: g for g in survey_gcps}
```

The surveyed Z is `g['elevation']` (orthometric, in CRS units). If `g['elevation']`
is None, fall back to `g['ellip_alt_m'] / FT_TO_M` if the CRS uses feet — but log a
warning. Do not attempt geoid separation: use whatever elevation the survey CSV
provides, matching what GCPEditorPro sees in gcp_list.txt.

#### Step 4 — Triangulate each CHK label

For each label, collect all observations that have a matching shot in the
reconstruction:

```python
rays = []   # list of (origin_C, direction_d) in ENU (world) coords
for (image_name, px, py) in obs_by_label[label]:
    if image_name not in shots:
        continue
    # Unproject pixel to normalised camera coordinates
    xn = (px - cx) / focal_px
    yn = (py - cy) / focal_px
    # Apply inverse radial distortion (iterative, 3 iterations is sufficient)
    xnd, ynd = xn, yn
    for _ in range(3):
        r2 = xnd**2 + ynd**2
        factor = 1 + k1 * r2 + k2 * r2**2
        xnd = xn / factor
        ynd = yn / factor
    # Ray in camera frame: [xnd, ynd, 1]
    d_cam = np.array([xnd, ynd, 1.0])
    # Ray in world (ENU) frame: R^T @ d_cam  (R is world→camera, so R^T is camera→world)
    d_world = R.T @ d_cam
    d_world /= np.linalg.norm(d_world)
    rays.append((C, d_world))
```

If fewer than 2 rays are available for a label, skip it and emit a warning.

**Linear DLT triangulation** (N ≥ 2 rays):

For each ray i with origin `Ci` and unit direction `di`, the constraint is that the
unknown point X lies on the ray: `(X - Ci) × di = 0`. This expands to 2 independent
linear equations per ray. Construct the system `A x = 0` (shape `2N × 4` in
homogeneous coordinates, or `2N × 3` in Cartesian):

```python
# Ax = b formulation (Cartesian, more numerically stable than homogeneous for
# well-conditioned problems)
# From (I - d d^T)(X - C) = 0 → (I - d d^T) X = (I - d d^T) C
A = []
b = []
for C_i, d_i in rays:
    M = np.eye(3) - np.outer(d_i, d_i)   # projects out the ray direction
    A.append(M)
    b.append(M @ C_i)
A = np.vstack(A)   # shape (3N, 3)
b = np.concatenate(b)
# Least-squares solution
X_enu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
```

#### Step 5 — Convert to projected CRS and compare

Convert `X_enu` (ENU, metres) to the survey CRS using `enu_to_projected()` (see
Context section above). Use the CRS declared on line 1 of `chk_confirmed.txt`, or the
`--crs` override.

```python
x_proj, y_proj, z_ellip_m = enu_to_projected(
    X_enu, ref_lla['latitude'], ref_lla['longitude'], ref_lla['altitude'], crs
)
```

The reconstructed Z is ellipsoidal altitude in metres. The surveyed Z is orthometric
elevation in the CRS units (for EPSG:6529: US survey feet). Convert the surveyed Z
to metres if needed by checking the CRS axis units via pyproj:

```python
from pyproj import CRS as ProjCRS
crs_obj = ProjCRS.from_user_input(crs)
axis_units = crs_obj.axis_info[0].unit_name   # e.g. 'US survey foot'
FT_TO_M = 0.3048006096012192
if 'foot' in axis_units.lower():
    x_proj_m = x_proj * FT_TO_M
    y_proj_m = y_proj * FT_TO_M
    z_survey_m = survey_z * FT_TO_M
else:
    x_proj_m = x_proj
    y_proj_m = y_proj
    z_survey_m = survey_z
```

Compute residuals (all in metres):

```python
dX = x_proj_m - survey_x_m
dY = y_proj_m - survey_y_m
dZ = z_ellip_m - z_survey_m   # note: geoid separation not corrected here
d3D = math.sqrt(dX**2 + dY**2 + dZ**2)
```

#### Step 6 — Output

**stdout** (JSON, for piping to experiment driver):

```json
{
  "rms_x": 0.042,
  "rms_y": 0.038,
  "rms_z": 0.121,
  "rms_3d": 0.134,
  "mean_dz": -0.015,
  "std_dz": 0.120,
  "n": 5,
  "points": [
    {"label": "CHK-3", "dX": 0.031, "dY": -0.055, "dZ": 0.098, "d3D": 0.115},
    ...
  ]
}
```

**stderr** (human-readable summary):

```
CHK RMSE report — N=5
  RMS_X  =  0.042 m
  RMS_Y  =  0.038 m
  RMS_Z  =  0.121 m
  RMS_3D =  0.134 m
  mean_Z =  -0.015 m  std_Z = 0.120 m

Per-point:
  CHK-3   dX=+0.031  dY=-0.055  dZ=+0.098  d3D=0.115
  ...
```

### Acceptance criteria

1. **Ghostrider gulch smoke test** (run once chk_confirmed.txt and reconstruction.json
   are available):
   `RMS_Z` must be in the range 0.05–0.30 ft (converted to metres: 0.015–0.091 m) —
   consistent with Pix4D check point Z errors for this site.

2. **Synthetic unit test** (can be run now):
   - Generate a fake reconstruction with 3 known camera poses and a known 3D point.
   - Create `chk_confirmed.txt` rows by forward-projecting the point through each
     camera (using the same pinhole model).
   - Run `rmse_calc.py` and assert `RMS_3D < 0.001 m`.
   - Test command:
     ```
     conda run -n geo python TargetSighter/rmse_calc.py \
         /tmp/synthetic_recon.json \
         /tmp/synthetic_chk.txt \
         /tmp/synthetic_emlid.csv \
         --crs EPSG:6529
     ```

3. **Error handling**:
   - Missing shot name → warning on stderr, observation skipped, not fatal.
   - Fewer than 2 rays for a label → warning, label skipped, still reports other labels.
   - Non-convergent lstsq (rank-deficient A) → warning + label skipped.

---

## Component 2: experiment_gen.py

**Location:** `/Users/jrstear/git/geo/TargetSighter/experiment_gen.py`

**Purpose:** Given a master tag file (all confirmed GCP-* and CHK-* rows in pipeline
priority order) and an experiment config, produce a trimmed `gcp_experiment.txt`
suitable for submission to ODM as the GCP control file.

### CLI

```
conda run -n geo python TargetSighter/experiment_gen.py \
    master_tags.txt \
    --config config.json \
    --out gcp_experiment.txt
```

Arguments:

| Argument | Description |
|---|---|
| `master_tags.txt` | Merged confirmed tags in priority order (see Input Format below) |
| `--config config.json` | Experiment config; see schema below |
| `--out gcp_experiment.txt` | Output path (ODM-compatible GCP file) |

### Config schema

```json
{
  "control_labels": ["GCP-1", "GCP-2", "GCP-3"],
  "images_per_label": 7,
  "description": "top3-7imgs"
}
```

- `control_labels` — list of label strings to include as ODM control points. Must be
  a subset of labels present in `master_tags.txt`. Labels not in the master file are
  silently ignored (emit a warning).
- `images_per_label` — integer or `{"GCP-1": 5, "GCP-2": 10, ...}` dict. When an
  integer, apply uniformly to all labels. When a dict, fall back to the integer value
  (or 7 if the key is absent). Rows are taken in the order they appear in
  `master_tags.txt` (pipeline priority order: best images first).
- `description` — freeform string written as a comment in the output file header.

### Input format: master_tags.txt

Produced by merging `gcp_confirmed.txt` + `chk_confirmed.txt`. The merge is done by
the experiment driver (not by this script). `master_tags.txt` has the same format as
`gcp_list.txt`: CRS header on line 1, then tab-separated data rows. Rows appear in
pipeline structural priority order (GCP-1 rows first, then GCP-2, …, then CHK-*).

The merge order must preserve intra-label row order. Across labels, GCP-* labels
come first in structural priority order, CHK-* labels come after.

### Algorithm

```python
# 1. Parse header
with open(master_tags_path) as f:
    lines = f.readlines()
crs_header = lines[0].rstrip('\n')

# 2. Parse data rows; group by label preserving order
from collections import defaultdict
rows_by_label = defaultdict(list)
for line in lines[1:]:
    line = line.rstrip('\n')
    if not line:
        continue
    fields = line.split('\t')
    label = fields[6]   # 7th column (0-indexed: 6)
    rows_by_label[label].append(line)

# 3. Load config
import json
with open(config_path) as f:
    config = json.load(f)
control_labels  = config['control_labels']
images_per_label = config['images_per_label']
description = config.get('description', '')

def n_images(label):
    if isinstance(images_per_label, int):
        return images_per_label
    return images_per_label.get(label, images_per_label.get('default', 7))

# 4. Select rows
output_rows = []
for label in control_labels:
    if label not in rows_by_label:
        print(f"WARNING: label {label!r} not in master_tags.txt", file=sys.stderr)
        continue
    output_rows.extend(rows_by_label[label][:n_images(label)])

# 5. Write output
with open(out_path, 'w') as f:
    if description:
        f.write(f'# {description}\n')
    f.write(crs_header + '\n')
    for row in output_rows:
        f.write(row + '\n')
```

Notes:

- The output file must NOT include the `# description` comment line if ODM is the
  consumer; ODM's GCP parser treats line 1 as the CRS string and will reject a
  comment. Add the comment only as a second line after the CRS — or omit it entirely
  and store the description only in the filename. Prefer omitting from the file body;
  record description in the results_table.csv instead.
- The CRS header is passed through verbatim from `master_tags.txt` — do not
  transform coordinates.
- Row order in the output file mirrors the order rows appear in `master_tags.txt`
  (preserves priority ordering within each label).

### Acceptance criteria

1. Output file parses without error when passed to ODM as `--gcp gcp_experiment.txt`.
2. Row count equals `sum(min(n_images(l), len(rows_by_label[l])) for l in control_labels)`.
3. No rows from labels not in `control_labels` appear in the output.
4. CRS header is identical to the one in `master_tags.txt` (byte-for-byte).

---

## Component 3: experiment_driver.py

**Location:** `/Users/jrstear/git/geo/TargetSighter/experiment_driver.py`

**Purpose:** Orchestrate a matrix of ODM runs that vary GCP count, image count per
GCP, and pipeline parameters. For each run: generate the GCP file, submit to cloud,
collect RMSE, write a results table.

**Note:** Cloud infrastructure (AWS Batch job definitions, IAM roles, S3 buckets) is
specified separately in `docs/cloud-infra-spec.md`. This component assumes those
resources exist.

### Responsibilities

1. **Define experiment matrix** — parametrise over:
   - `control_labels`: subsets of GCP-* labels (e.g. top-3, top-5, top-7, top-10)
   - `images_per_label`: integer values (e.g. 3, 5, 7, 10)
   - `nadir_weight`: float (controls pipeline image ordering; re-run `csv2gcp.py`
     with different weights to produce different `master_tags.txt` variants)
   - Any other pipeline parameter that affects the GCP file layout.

2. **Generate GCP files** — for each run config, call `experiment_gen.py` to produce
   `gcp_experiment_{run_id}.txt` in a local staging directory.

3. **Submit Batch jobs** — upload the GCP file to S3, submit an ODM Batch job (job
   definition: `odm-run`) and, when that completes, an RMSE Batch job (job
   definition: `rmse-calc`). See `docs/cloud-infra-spec.md` for API details.

4. **Poll for completion** — use `boto3` to poll Batch job status. Implement
   exponential back-off (start: 30 s, max: 5 min).

5. **Collect results** — download `rmse_report.json` from S3 for each completed run.

6. **Write results_table.csv** — one row per run:

   ```
   run_id, description, control_labels, n_labels, images_per_label, nadir_weight,
   rms_x_m, rms_y_m, rms_z_m, rms_3d_m, mean_dz_m, std_dz_m, n_points,
   batch_job_id, completed_at
   ```

### CLI

```
conda run -n geo python TargetSighter/experiment_driver.py \
    --master-tags master_tags.txt \
    --chk-confirmed chk_confirmed.txt \
    --emlid emlid.csv \
    --matrix matrix.json \
    --staging /tmp/experiment_staging/ \
    --s3-bucket my-experiment-bucket \
    --results results_table.csv
```

### matrix.json schema

```json
{
  "base_control_labels": ["GCP-1","GCP-2","GCP-3","GCP-4","GCP-5","GCP-6","GCP-7"],
  "label_subsets": [
    ["GCP-1","GCP-2","GCP-3"],
    ["GCP-1","GCP-2","GCP-3","GCP-4","GCP-5"],
    ["GCP-1","GCP-2","GCP-3","GCP-4","GCP-5","GCP-6","GCP-7"]
  ],
  "images_per_label_values": [3, 5, 7],
  "nadir_weight_values": [0.2, 1.0],
  "chk_confirmed": "chk_confirmed.txt",
  "emlid_csv": "emlid.csv"
}
```

The driver generates the Cartesian product of `label_subsets × images_per_label_values`
(and optionally `nadir_weight_values` if re-running the pipeline is feasible). For
`nadir_weight` variations, a separate `master_tags_{nw}.txt` must be pre-generated by
running `csv2gcp.py` with `--nadir-weight {nw}` before invoking the driver.

### Concurrency model

Use `concurrent.futures.ThreadPoolExecutor` (not multiprocessing) for polling
multiple Batch jobs simultaneously. Maximum 8 concurrent polls to avoid AWS API
rate limits.

---

## File locations (for implementing agents)

| File | Path |
|---|---|
| Existing pipeline (reference) | `/Users/jrstear/git/geo/TargetSighter/csv2gcp.py` |
| `parse_survey_csv()` | In `csv2gcp.py` — import directly |
| `_cs_name_to_epsg()` | In `csv2gcp.py` — import directly |
| `project_pixel_mode_b()` | In `csv2gcp.py` — reference for reconstruction.json camera math |
| Test data root | `/Users/jrstear/stratus/ghostrider gulch/` |
| Emlid CSV | `/Users/jrstear/stratus/ghostrider gulch/emlid.csv` (confirm exact name on disk) |
| gcp_list.txt (full pipeline output) | `/Users/jrstear/stratus/ghostrider gulch/gcp_list.txt` |
| gcp_confirmed.txt | Same directory (available after tagging bead geo-chk-data) |
| chk_confirmed.txt | Same directory (available after tagging bead geo-chk-data) |
| reconstruction.json | `{task_dir}/assets/odm_opensfm/reconstruction.json` (path TBD after first WebODM run) |

---

## Concurrency and agent assignment

These components have no code dependencies on each other and can be implemented in
parallel by separate agents:

- **Agent A**: `rmse_calc.py` — no dependencies on experiment_gen or driver.
- **Agent B**: `experiment_gen.py` — no dependencies on rmse_calc or driver.
- **Agent C**: Cloud infrastructure (Terraform) — defined in `docs/cloud-infra-spec.md`;
  no Python code dependencies.

Agent workflow for each component:

1. Read this spec and the reference file `csv2gcp.py`.
2. Implement the component.
3. Run the synthetic acceptance test.
4. Report result: `bd update geo-XXX --status=in_progress` during work,
   `bd close geo-XXX` on completion.
5. Do NOT commit until the user confirms testing passed.
6. After confirmation, in order: `bd sync` → `git commit` → `git push`.

---

## Open questions / deferred decisions

- **Geoid separation for Z comparison**: `rmse_calc.py` compares triangulated
  ellipsoidal altitude against surveyed orthometric elevation without geoid correction.
  This will introduce a systematic Z offset. For RMSE trending across experiments the
  offset is constant, so relative comparisons are valid. Absolute accuracy requires a
  geoid model (e.g. GEOID18 via `geoid18.py`). Address in a follow-up bead.

- **chk_confirmed.txt availability**: Until the user tags CHK-* rows in GCPEditorPro,
  the real-data smoke test for `rmse_calc.py` cannot be run. The synthetic test is
  always runnable. Record this dependency in bead notes.

- **reconstruction.json path**: The exact `task_dir` is determined by WebODM at task
  creation time. Agent A should parameterise the path via CLI argument and document
  how to find it (WebODM task UUID in the URL → `/webodm/app/media/project/{pid}/task/{tid}/`).
