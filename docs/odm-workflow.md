# Survey-Quality ODM Workflow

End-to-end process for producing survey-quality orthophotos with OpenDroneMap,
using Emlid GNSS survey data and GCPEditorPro pixel tagging.

---

## Overview

```
BSN design-grid control (.dc)
    └─ extract_dc_points.py ─────────────────────────────┐
                                                          ▼
Emlid rover CSV ──► csv2gcp.py ──► {job}_tagged.txt ──► GCPEditorPro
                    (+ images)      (pixel estimates)    (manual confirm)
                                                          │
                                            {job}_confirmed.txt
                                            (GCP- + CHK-, EPSG:3618, feet)
                                                          │
                                           prepare_odm.py │
                                          ┌───────────────┴───────────────┐
                                          ▼                               ▼
                               {job}_control.txt                {job}_check.txt
                             (GCP- only, EPSG:32613)         (CHK- only, EPSG:32613)
                                          │                               │
                                     ODM run                        rmse_calc.py
                                    (opensfm)                      (accuracy QC)
                                          │
                                    orthomosaic
                                    (EPSG:32613 out)
                                          │
                                  reproject + shift
                                   ─► delivery CRS
```

---

## File naming convention

| Stage | File | Contents |
|-------|------|----------|
| BSN input | `F{job}.dc` | Trimble design-grid control |
| BSN control | `F{job}_points.csv` | Monuments in EPSG:3618 (from `extract_dc_points.py`) |
| Emlid survey | `{job}_{date}.csv` | All points, Emlid Flow export |
| Pipeline input | `{job}_{filter}.csv` | Subset for tagging (e.g. `aztec_3_9.csv`) |
| Tagging file | `{job}_tagged.txt` | csv2gcp.py output for GCPEditorPro |
| GCPEditorPro export | `{job}_confirmed.txt` | All confirmed observations (GCP- + CHK-) |
| ODM control | `{job}_control.txt` | GCP- only, EPSG:32613, metres |
| RMSE check | `{job}_check.txt` | CHK- only, EPSG:32613, metres |

**Do not use `gcp_list.txt`** — ambiguous name (contains both GCP and CHK observations).

---

## Critical CRS rules

ODM receives XYZ ground coordinates and must map them to the 3D world. Avoid any
CRS ambiguity that could cause a feet/metres mismatch:

| CRS | Use | Notes |
|-----|-----|-------|
| **EPSG:32613** (WGS 84 / UTM 13N, metres) | ODM control + RMSE check files | **Always use this for ODM** |
| **EPSG:3618** (NAD83 NM Central, feet) | Field survey, internal analysis | CSV/QGIS only |
| **EPSG:6529** (NAD83(2011) NM Central, feet) | Emlid native output | Same zone as 3618; convert before ODM |

**Why EPSG:32613 for ODM?**  EPSG:3618 and 6529 are 2D CRS — they define XY units (US
survey feet) but not vertical units.  ODM assumes Z is in metres for any 2D CRS, which
causes a ~3.28× Z scale error when Z is supplied in feet.  EPSG:32613 is a well-defined
2D horizontal CRS in metres; ODM treats Z as metres and the XY transformation is
unambiguous.

`prepare_odm.py` converts automatically:
- XY: `pyproj` transform EPSG:6529 → EPSG:32613
- Z: multiply by `FT_TO_M = 0.3048006096012192` (US survey foot)

---

## Step-by-step

### 1. Prepare control monuments (one-time per job)

```bash
cd ~/stratus/{job}
python python/extract_dc_points.py   # → F{job}_points.csv  (EPSG:3618)
```

### 2. Build tagging file

```bash
conda run -n geo python GCPSighter/csv2gcp.py \
    "{job}_{filter}.csv" \
    images/ \
    --out-name "{job}"
# → {job}.txt    (tagging file)
# → marks.csv   (image observations for import into GCPEditorPro)
```

### 3. Tag and confirm in GCPEditorPro

1. Open GCPEditorPro
2. Load tagging file + images directory
3. Import marks.csv (pre-populated pixel estimates)
4. Review each GCP- and CHK- point; confirm observations
5. Export → save as **`{job}_confirmed.txt`**

GCP- labels = ground control (used by ODM to georeference)
CHK- labels = independent check points (NOT given to ODM; used for accuracy QC only)

### 4. Split into control + check files

```bash
conda run -n geo python GCPSighter/prepare_odm.py \
    {job}_confirmed.txt \
    --out-dir ~/stratus/{project}/ \
    --stem {job}
# → {job}_control.txt   (GCP- only, EPSG:32613)
# → {job}_check.txt     (CHK- only, EPSG:32613)
```

### 5. Run ODM on EC2

Copy the control file to the project and launch via `odm-bootstrap.sh`:

```bash
# On your Mac
scp ~/stratus/{project}/{job}_control.txt  ec2:/home/ec2-user/project/

# On EC2 — bootstrap handles S3 sync of images + runs ODM + syncs output + shuts down
bash infra/ec2/scripts/odm-bootstrap.sh
```

Recommended ODM flags (medium quality, 1385-image corridor):
```
--gcp /path/to/{job}_control.txt
--pc-quality medium
--feature-quality high
--orthophoto-resolution 5
--optimize-disk-space
--max-concurrency 16
```

Expected runtime: ~20 hours on m5.4xlarge (16 vCPU). See `docs/cloud-infra-spec.md`.

### 6. Verify accuracy with rmse_calc.py

After the ODM run completes and `opensfm/reconstruction.json` is available:

```bash
conda run -n geo python GCPSighter/rmse_calc.py \
    {odm_project}/opensfm/reconstruction.json \
    ~/stratus/{project}/{job}_check.txt \
    "{job}_{filter}.csv"
```

Expected accuracy (250 ft AGL, drone RTK active, 5 BSN monument GCPs):

| Component | Expected |
|-----------|----------|
| Horizontal RMS | 0.08–0.12 ft (0.024–0.037 m) |
| Vertical RMS | 0.12–0.18 ft (0.037–0.055 m) |
| GCP residuals (ODM report) | < 0.05 m |

RMS_Z should be < 0.10 m after geoid correction. If it is near 4,000 m, the Z unit
conversion was not applied (feet-as-metres error).

### 7. Deliver

```bash
# Reproject to state plane (optional — QGIS handles on-the-fly)
gdalwarp -s_srs EPSG:32613 -t_srs EPSG:3618 odm_orthophoto.tif ortho_3618.tif

# Apply BSN design-grid shift for delivery (Aztec job)
python package.py \
    --no-tile \
    --shift-x 1546702.929 \
    --shift-y -3567.471 \
    ortho_3618.tif
```

---

## Aztec Highway F100340 — specific notes

- Survey CSV: `~/stratus/aztec/Aztec Highway 3_9.csv` (filtered 3/9 points)
- Confirmed file: `~/stratus/aztec/gcp_list.txt` (historical name; contains GCP- + CHK-)
- aztec3 control: `~/stratus/aztec3/aztec_control.txt`
- aztec3 check: `~/stratus/aztec3/aztec_check.txt`
- Design-grid shift: state_E + 1,546,702.929 ft; state_N − 3,567.471 ft
- CRS detail: see `~/stratus/aztec/jrs/Control_Info_F100340_AZTEC.md`
- Prior run issues: see `~/stratus/aztec/jrs/ortho-analysis.md`

### GCP distribution (corridor)

10 GCPs for a 1385-image corridor are marginal. Recommended distribution:
- Alternating sides of road every ~500 ft
- GCPs at both ends + 2 mid-corridor
- Z-critical points at high/low elevation extremes
- CHK points distributed throughout (not clustered near GCPs)

See `docs/aztec-gcp-analysis.md` for full placement analysis.
