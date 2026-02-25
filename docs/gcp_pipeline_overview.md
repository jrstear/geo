# GCP Auto-Identification Pipeline — Design Overview

## Goal

Automate the GCP workflow: raw drone images + Emlid rover CSV →
fully-tagged `gcpeditpro.txt`, ready for OpenDroneMap. Minimal manual effort.

## End-to-End Workflow

```
1. Create WebODM project, add images + Emlid CSV
2. (Optional) Run initial ODM task with no GCPs
   → produces opensfm/reconstruction.json for enhanced accuracy
3. Click "Generate GCP Estimates" (WebODM plugin, or CLI)
   a. Parse Emlid CSV → GCP label, lat, lon, elev          [B1]
   b. Batch EXIF read per image (exiftool, parallel)        [B1]
   c. Footprint match → which images contain which GCPs     [B1]
   d. Pixel projection per (image, GCP) pair                [B2]
      - Mode A (default): EXIF gimbal angles + GPS
      - Mode B (enhanced): reconstruction.json camera poses
   e. Write gcpeditpro.txt + .estimates.json sidecar        [B3]
4. Open GCPEditorPro with gcpeditpro.txt
   - Images per GCP pre-selected, estimated pixel shown     [G1]
   - Zoom column: cropped sub-images centered at estimate   [G2]
   - Shift-click to confirm: yellow → green markers         [G3]
   - c/t confirmed/total badge on GCP list page             [G4]
5. Save gcpeditpro.txt; upload to WebODM
6. Run final ODM task with GCPs → high-accuracy model
7. (Optional) Train auto-detector from confirmed GCPs       [A1–A4]
```

## Repositories & Branches

| Repo | Path | Branch |
|------|------|--------|
| WebODM | `~/git/webodm` | `feature/auto-gcp-pipeline` |
| GCPEditorPro | `~/git/GCPEditorPro` | `feature/auto-gcp-pipeline` |
| geo (tools) | `~/git/geo` | `main` |

## Camera Pose Modes

### Mode A — EXIF (default, no prior ODM run needed)

Reads per-image from EXIF via exiftool:
- `GPSLatitude`, `GPSLongitude`, `RelativeAltitude` — camera position
- `GimbalPitchDegree`, `GimbalYawDegree`, `GimbalRollDegree` — orientation
- `FocalLength` (mm), `FieldOfView` (diagonal, degrees), `ImageWidth`, `ImageHeight`
- All sensor dims derived from EXIF — nothing hardcoded per drone model

For nadir images (GimbalPitch ≈ -90°), expected accuracy: ±30–150 px.
Good enough for the shift-click workflow (GCP appears in ~256×256 sub-image crop).

### Mode B — Reconstruction (optional, requires prior ODM task)

Reads `opensfm/reconstruction.json` from a prior WebODM task. SfM-refined
camera positions and orientations give ±5–20 px accuracy.

Auto-detected: if a `reconstruction.json` is present/specified, use Mode B;
otherwise fall back to Mode A.

## Outputs from B3

**`gcpeditpro.txt`** — standard ODM GCP format:
```
<proj4 string>
<x> <y> <z> <px> <py> <filename> <gcpname>
...
```
One line per (image, GCP) pair. x/y/z from Emlid CSV (projected coords).

**`gcpeditpro.estimates.json`** — sidecar for GCPEditorPro pre-population:
```json
{
  "gcpName": {
    "image.JPG": { "px": 1234, "py": 567, "mode": "exif" }
  }
}
```

## Phases

### Phase 1 — Research (all parallel)

| Issue | Topic | Docs target |
|-------|-------|-------------|
| R1 | WebODM plugin API + task result file access | `docs/webodm_plugin_spec.md` |
| R2 | GCPEditorPro storage model + .txt import path | `docs/gcpeditorpro_ui_spec.md` |
| R3 | EXIF pinhole math validation + reconstruction.json format | `docs/dji_m3e_camera_model.md` |

### Phase 2 — Backend Python Pipeline

- **B1** Emlid CSV parser + parallel footprint matcher (extends gcp.py)
- **B2** Pixel projection: Mode A (EXIF) and Mode B (reconstruction.json)
- **B3** Pipeline runner combining B1+B2 + file writers

### Phase 3 — WebODM Plugin

- **W1** "Generate GCP Estimates" plugin button in task UI

### Phase 4 — GCPEditorPro Enhancements

- **G1** Pre-populate image/pixel from .estimates.json
- **G2** Zoomed sub-image column view
- **G3** Shift-click confirmation + sidecar .confirmed.json
- **G4** c/t count badge on GCP list
- **G5** +/- zoom buttons in full image view

### Phase 5 — Auto-Detector

- **A1** (geo-bwb) Extract sub-image training data from confirmed GCPs
- **A2** (geo-d83) Haar cascade training (orange clay pigeon + shape)
- **A3** (geo-kit) CNN/YOLO training pipeline
- **A4** GCPEditorPro: classifier selection, crop bounds, Detect GCPs button

## Existing Issues Fate

| Issue | Action | New role |
|-------|--------|----------|
| geo-9z5 | Update | Split across B1 (footprint) + B2 (projection) |
| geo-bwb | Update | A1, depends on G3 |
| geo-d83 | Update | A2, depends on A1 |
| geo-kit | Update | A3, depends on A1 |
| geo-7dj | Close | Superseded by W1 |
| geo-xkt | Close | Superseded by G1/G2/G3 |
| geo-dbi | Close | Orange clay pigeon chosen; alt tools deprioritized |
