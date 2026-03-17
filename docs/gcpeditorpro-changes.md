# GCPEditorPro — Pipeline Integration Changes

This document describes a set of changes made to GCPEditorPro on the
`feature/auto-gcp-pipeline` branch.  The changes are designed to support an
end-to-end workflow for drone surveys where GCP targets are surveyed with an Emlid Reach
rover, optionally consist of colored X's (surveyor paint or clay pigeons), and then tagged in GCPEditorPro before photogrammetric reconstruction in
OpenDroneMap / WebODM.

The companion pipeline tool `emlid2gcp.py` (in the `geo` repository) converts
the Emlid rover CSV into a `gcp_list.txt` that already contains per-image pixel
estimates — where each GCP target is expected to appear in each drone image —
computed from camera EXIF (GPS, gimbal angles, sensor dimensions) and,
optionally, from SfM-refined camera poses.  For colored-X targets it also runs
a color-detection pass that narrows the estimate to ±5–30 px.  GCPEditorPro's
role in this workflow is confirmation: the user verifies or corrects each pixel
estimate rather than placing every tag from scratch.

The full tagging workflow is documented in `docs/tagging-guide.md`.

---

## Changes

### 1. Confidence column (8th column) support

The pipeline writes an 8th tab-separated column containing a confidence value
for each row: `projection` (EXIF-derived estimate), `color` (color-detection
refined), or `reconstruction` (SfM-refined).  GCPEditorPro now reads and
preserves this column on load and writes it back on export.

On load, any file containing a `projection` or `reconstruction` value is
flagged as a pipeline-generated file (`hasPipelineEstimates = true`), which
gates the pipeline-specific UI described below.  Files without the column
load and behave exactly as before.

### 2. Zoom view

A new **zoom view** was added as the primary tagging interface for pipeline
files.  It presents a two-panel layout: a scrollable column of cropped
sub-image thumbnails on the left, and a full-resolution image panel on the
right.  Hovering over a thumbnail selects it in the right panel.  This allows
the user to work through images rapidly without switching screens.

Images are shown in the order they appear in the file — which is the
confidence-ranked order written by `sight.py` — so the user works through
the best-quality images first.

### 3. Adaptive crop from marker bounding box (9th column)

When the color-detection pass locates the target marker, it records its
bounding box as an optional 9th column (`x1,y1,x2,y2` in full-image pixel
coordinates).  Zoom view reads this column and scales the right-panel crop to
frame the bounding box at a fixed display size, so the target is immediately
recognisable regardless of GSD.

### 4. Auto-enable zoom view for pipeline files

When a pipeline-generated file is loaded (`hasPipelineEstimates = true`), zoom
view is enabled automatically.  This saves the user from having to discover and
toggle the view mode on every session.  The toggle remains available so the user
can switch back to grid view if needed.

### 5. Spacebar confirmation shortcut

In zoom view, pressing **Space** confirms the current pixel estimate in place —
equivalent to clicking on the crosshair.  This is particularly fast when the
color-detection estimate is accurate: the user works through images pressing
Space for good estimates and clicking to correct poor ones, without moving the
mouse between images.  After confirming, the view advances automatically to the
next unconfirmed image.

### 6. GCP ordering preserved on save

**Bug fix.**  Previously, when the user clicked OK in the tagger, the current
GCP's rows were removed from `storage.imageGcps` and pushed to the end of the
array.  After tagging several GCPs, the exported `gcp_list.txt` had those GCPs
relocated to the tail in click order, destroying the structural ordering
written by `emlid2gcp.py`.  The fix records the first-row index before
filtering and splices the updated rows back at that position, so the global GCP
order is preserved across save/reload cycles.

### 7. Per-GCP progress badges and summary line

Each GCP row in the list now shows a **confirmed / total** badge coloured
red (0–2), amber (3–6), or green (≥ 7) to indicate tagging coverage.  At the
top of the list, a summary line counts how many GCP-\* control points have
reached green, coloured by the same thresholds.  This gives the user a clear
target: the session is complete when the summary line is green.

These thresholds reflect ODM's practical weighting behaviour and the USGS /
ASPRS minimum of 3 confirmed images per point for a usable constraint.

### 8. GCP-\* / CHK-\* / DUP-\* map pins

The pipeline prefixes every GCP label to indicate its photogrammetric role:
`GCP-*` (top 10 structural control points), `CHK-*` (check points for
independent accuracy validation), `DUP-*` (near-duplicates excluded from
normal ordering).  The map now renders these with distinct styles: `GCP-*`
pins are large, labelled, and ratio-coloured by the red/amber/green progress
thresholds; `CHK-*` pins are smaller and unlabelled; `DUP-*` pins use a
separate symbol.  This gives a spatial read of tagging progress — the user
can see control coverage at a glance and watch pins turn green.

### 9. Scroll-to-zoom centered on cursor (geo-b49)

Previously, zooming in the full-image panel required holding **Shift** while
scrolling.  The Shift guard has been removed: plain scroll now zooms in/out,
always centered on the current cursor position.  This makes navigation
significantly faster — pan to the area of interest, then scroll to zoom in.

### 10. Shift-click to un-tag / revert to estimated (geo-cvm)

A confirmed (green) tag can be reverted to the pipeline's original estimated
(yellow) state by **shift-clicking** anywhere in the image — in the full-image
panel, in a thumbnail in the left column, or in a grid-view thumbnail.  This
restores the pipeline's original `px`/`py` estimate and `confidence` value.

Previously, the only way to undo an accidental confirmation was to re-click
the correct pixel manually.

### 11. Per-image north compass and tilt indicator (geo-rrn)

Each image in zoom view — both the thumbnail strip and the large panel — now
shows a small orientation overlay in the top-left corner:

- **Compass arrow** — red tip points north.  Derived from `GimbalYawDegree`
  (falling back to `FlightYawDegree`) in DJI XMP metadata.  Rotation is
  `−GimbalYawDegree` so the red tip stays fixed on north as the drone heading
  varies.
- **Camera icon + tilt angle** — a white-outline camera silhouette followed by
  the degrees-from-nadir angle: `(90 + GimbalPitchDegree)°`, so 0° = straight
  down, 45° = typical oblique pass, 90° = horizontal.

Both elements are hidden when the relevant EXIF fields are absent.  The overlay
is read from `storage.service.ts` via a single `exifr.parse(file, {xmp: true})`
call that caches both yaw and pitch in `getOrientation()`.

### 12. Split confirmed export

The export screen now offers three downloads when a pipeline file is loaded:

- **`gcp_list.txt`** — all rows (confirmed + unconfirmed estimates), in original
  file order.  Use this to resume tagging in a later session.
- **`gcp_confirmed.txt`** — confirmed `GCP-*` control point rows only.  Supply
  this to ODM/WebODM for the reconstruction.
- **`chk_confirmed.txt`** — confirmed `CHK-*` check point rows only.  Use this
  after reconstruction to compute independent RMSE figures (ODM has no native
  check-point concept, so these are held out entirely).

For non-pipeline files the export screen shows the original single confirmed
download, unchanged.

---

## File changes (summary)

| File | Change |
|------|--------|
| `src/app/gcps-utils.service.ts` | Read/write 8th confidence column; read 9th marker_bbox column; `hasPipelineEstimates` flag; `generateExtrasNames()` returns `'Marker Bbox'` for col 8 |
| `src/app/images-tagger/images-tagger.component.ts` | Zoom view; adaptive crop; auto-enable; spacebar confirm; file-order image sequencing; ordering bug fix; write `'confirmed'` confidence; shift-click untag; compass/tilt per image |
| `src/app/images-tagger/images-tagger.component.html` | Zoom view layout (full image + crop panels); crosshair; bbox crop; compass/tilt overlay on large panel |
| `src/app/images-tagger/images-tagger.component.scss` | Pill-shaped compass overlay; large-image-compass positioning |
| `src/app/sub-image-crop/sub-image-crop.component.ts` | Shift-click unpin output; compass/tilt overlay on thumbnail canvas |
| `src/app/smartimage/smartimage.component.ts` | Scroll-to-zoom on cursor; shift-click unpin output; removed shift-to-zoom guard |
| `src/app/gcps-map/gcps-map.component.ts` | Progress badges; summary line; GCP-\*/CHK-\*/DUP-\* pin styles |
| `src/app/gcps-map/gcps-map.component.html` | Summary line; pin rendering |
| `src/app/export-config/export-config.component.ts` | Split export (control / check / full); pipeline vs non-pipeline branching |
| `src/app/export-config/export-config.component.html` | Three download buttons; confidence + marker_bbox columns in preview table |
| `src/app/load-config-txt/load-config-txt.component.html` | Confidence + marker_bbox columns in input preview table |
