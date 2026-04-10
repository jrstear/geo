# GCPEditorPro Changes

This document describes changes made to GCPEditorPro on the
`feature/auto-gcp-pipeline` branch.

---
## 1. Progress indicators

Each row in the target list now shows a **confirmed / total** badge colored
red (0–2), amber (3–6), or green (≥ 7) to indicate tagging coverage.  Pin colors match their badges, and a summary line above the list shows how many GCP- points have reached green (same color thresholds).  The coloring is consistent with ASPRS guidelines of 3 minimum and 7+ recommended.  This gives users clear progress indicators: tagging is sufficient when the badge/pin/summary is green.

## 2. Scroll-to-zoom on cursor

Previously, zooming in the full-image panel required holding **Shift** while
scrolling.  Plain (unshifted) scroll now zooms, and zoom is centered on the current cursor position.  This behavior is consistent with QGIS — pan to the area of interest, cursor on target, scroll to zoom.

## 3. Zoom view

A **zoom view** was added, with a two-panel layout: a scrollable column of cropped and smoothed thumbnails on the left, and a larger non-smoothed image panel on the
right.  Hovering over a thumbnail selects it in the right panel.

## 4. Compass and tilt

In zoom view, if image files include sufficient metadata, small overlays in the top-left corners of the images provide additional context info:

- **Compass arrow** — red tip points north.  Derived from `GimbalYawDegree`
  (falling back to `FlightYawDegree`) in DJI XMP metadata.  Rotation is
  `−GimbalYawDegree` so the red tip stays fixed on north as the drone heading
  varies.
- **Camera icon + tilt angle** — a white-outline camera silhouette followed by
  the degrees-from-nadir angle: `(90 + GimbalPitchDegree)°`, so 0° = straight
  down, 45° = typical oblique pass, 90° = horizontal.

## 5. Tag status

If the input file contains an (optional) 8th column containing one of the below values, it is interpreted as pixel coordinate confidence, and zoom is set as the default tagging view.  Valid values are: `projection` (EXIF-derived estimate), `color` (color-detection refined), `reconstruction` (SfM-refined),  and `tagged` (manual selection).  This value is read on input, updated during tagging, and written upon download, and its values are shown on the download screen.   Tagged thumbnails have a green border, estimated are yellow, and no-status is grey.

## 6. Space to tag

In zoom view, pressing **Space** confirms the current image's estimated tag position as `tagged` (equivalent to clicking in the image).  After confirming, the view advances automatically to the next unconfirmed image, enabling easy acceptance of excellent estimates.

## 7. Shift-click to un-tag

A `tagged` location can be reverted to its (estimated) state by a **shift-click** anywhere in the image.

**Bug fix** - Previously, the only way to undo an accidental confirmation was to click
the correct pixel position.  Untagging is better, eg if the target can not be found or is badly distorted.

## 8. Thumbnail bounding box

If the input file contains an (optional) 9th column, it is interpreted as a target bounding box, and determines the source crop region for the thumbnail.  This enables upstream tools to set this (eg via automated pattern detection) for efficient review (without zooming).

## 9. Target ordering

Targets and images are ordered according to the input file, enabling upstream tools to order them as desired (eg, rank targets by dispersion, and images by a mix of nadir and oblique).

**Bug fix** - Previously, tagging shifted rows to the end of downloaded lists, destroying the ordering
of the original input file.  Now the original ordering is preserved, enabling multi-session tagging with consistent ordering.

---

## File changes (summary)

| File | Change |
|------|--------|
| `src/app/gcps-utils.service.ts` | Read/write 8th confidence column; read 9th marker_bbox column; normalize legacy `'confirmed'`/`'mouse_click'` → `'tagged'` on import; `hasPipelineEstimates` flag (true when any row has `projection`/`reconstruction`/`tagged`); `generateExtrasNames()` names `extras[0]` as `'Marker Bbox'` (col 9) |
| `src/app/images-tagger/images-tagger.component.ts` | Zoom view; adaptive crop (uses marker_bbox when present); auto-enable when `hasPipelineEstimates`; spacebar confirm + advance to next unconfirmed; file-order image sequencing; ordering bug fix; writes `'tagged'` confidence on pin (and reverts to estimate confidence on unpin); shift-click untag; compass/tilt overlay on large panel |
| `src/app/images-tagger/images-tagger.component.html` | Zoom view layout (full image + crop panels); crosshair; compass/tilt overlay on large panel |
| `src/app/images-tagger/images-tagger.component.scss` | Pill-shaped compass overlay; large-image-compass positioning |
| `src/app/sub-image-crop/sub-image-crop.component.ts` | **NEW** — canvas-based thumbnail with adaptive marker_bbox crop, crosshair pin, click/shift-click handlers, compass/tilt overlay |
| `src/app/smartimage/smartimage.component.ts` | Scroll-to-zoom on cursor (removed shift-to-zoom guard); shift-click unpin; new `autoFit` / `cropFocus` modes; `cropBox` overlay synced to mirror sub-image-crop selection on the large panel |
| `src/app/smartimage/smartimage.component.html` | Adds `.fill` class binding (driven by `autoFit`); adds `cropBoxDiv` overlay element; removes "press SHIFT to zoom" hint message |
| `src/app/smartimage/smartimage.component.scss` | `.smart-image.fill` styles (fill mode for fixed-size panels) |
| `src/app/gcps-map/gcps-map.component.ts` | Progress badges; summary line; GCP-\*/CHK-\*/DUP-\* pin styles |
| `src/app/gcps-map/gcps-map.component.html` | Summary line; pin rendering |
| `src/app/gcps-map/gcps-map.component.scss` | Viewport-fill layout (scrollable target list, fixed map) |
| `src/app/export-config/export-config.component.ts` | Split export (control / check / full); pipeline vs non-pipeline branching; export uses `inputFileName` to derive `{job}_tagged.txt` |
| `src/app/export-config/export-config.component.html` | Three download buttons; confidence + marker_bbox columns in preview table |
| `src/app/load-config-txt/load-config-txt.component.ts` | Propagates `hasPipelineEstimates` from parse result; sets `inputFileName` for export naming |
| `src/app/load-config-txt/load-config-txt.component.html` | Confidence + marker_bbox columns in input preview table |
| `src/app/storage.service.ts` | Adds `hasPipelineEstimates` flag and `inputFileName`; adds `ImageInfo.getOrientation()` (cached XMP yaw/pitch read for compass+tilt overlays) |
| `src/app/app.module.ts` | Declares the new `SubImageCropComponent` |
## Suggestions
- change "Ground Control Points" on target list page to "Targets", since there are both GCP and CHK points in the list
- rename GCPEditorPro as TargetTaggerPro