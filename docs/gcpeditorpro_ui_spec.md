# GCPEditorPro UI Enhancements — Specification

Repo: `~/git/GCPEditorPro`
Branch: `feature/auto-gcp-pipeline`
Stack: Angular, TypeScript, Bootstrap

## G1 — Pre-Populate Image List from .estimates.json

### Trigger
When the user loads a `.txt` GCP file in GCPEditorPro, check if a sidecar
`<name>.estimates.json` file exists alongside it. If so, pre-populate image
associations and estimated pixel positions.

### .estimates.json Format
```json
{
  "gcpLabel": {
    "image_filename.JPG": { "px": 1234, "py": 567, "mode": "exif" }
  }
}
```
- `mode`: `"exif"` or `"reconstruction"` (informational only)
- `px`, `py`: estimated pixel coordinates (fractional okay)

### Implementation Notes
- In `load-config-txt.component.ts`, after parsing the `.txt` file, check for
  the `.estimates.json` sidecar and load it.
- Populate `storage.imageGcps` with the estimated associations.
- Also populate `storage.images` with the image filenames from the estimates,
  so the images-tagger doesn't require the user to drag-drop images that are
  already listed.
- In images-tagger, the filter-by-distance logic should still work (images
  without GPS coords remain unfiltered).
- Research needed (R2): confirm the exact data path in storage.service.ts.

---

## G2 — Zoomed Sub-Image Column View

### Goal
Replace (or augment) the current images-tagger layout with a view that shows
cropped sub-images centered at the estimated GCP pixel, so the user can quickly
confirm/refine without hunting through full-resolution images.

### Layout

```
┌─────────────────────────────────────────────────────┐
│ GCP: "3"    Confirmed: 2/5   [Standard View] [Zoom] │
├──────────────────┬──────────────────────────────────┤
│  Sub-image col   │  Full image of selected sub-image │
│  (scroll)        │  (pan + zoom + shift-click)       │
│  ┌────────────┐  │                                   │
│  │ DJI_001.JPG│◄─┤── selected (green outline)        │
│  │  [crop]    │  │                                   │
│  └────────────┘  │  ← GCP marker shown               │
│  ┌────────────┐  │  ← Pan/zoom + shift-click to      │
│  │ DJI_002.JPG│  │    refine pixel location          │
│  │  [crop]    │  │                                   │
│  └────────────┘  │  [ − ]  [ + ]  (zoom buttons)    │
│  ...             │                                   │
└──────────────────┴──────────────────────────────────┘
```

### Sub-Image Behavior
- **Crop size**: 256×256 px (configurable constant) extracted from full image,
  centered at estimated (px, py).
- **Rendering**: Use Canvas API to draw the cropped region. Do not load the
  full image into DOM multiple times — load once, draw crops.
- **Outline colors**:
  - Gray: image loaded, not yet viewed
  - Yellow: estimated pixel shown, not yet confirmed by user
  - Green: user has shift-clicked to confirm pixel location
- **Selection**: Click a sub-image to select it → shown in full-image panel.
  Default selection: first sub-image in the list.
- **Ordering**: Sort by confirmation status (unconfirmed first), then by
  distance to GCP.

### Full Image Panel Behavior
- Shows the full image of the currently selected sub-image.
- GCP marker rendered at current (px, py) estimate.
- Standard pan/zoom via mouse drag and scroll wheel.
- **+/- buttons** (bottom-right, see G5): coarse zoom steps like Google Maps.
- **Shift-click**: sets the precise GCP pixel for this image. Updates:
  - The sub-image crop (re-centered on confirmed pixel)
  - Sub-image outline: yellow → green
  - confirmation state in .confirmed.json sidecar

### Mode Toggle
Add a toggle button "Zoom View / Standard View" near the top of images-tagger.
Standard view = current images-tagger behavior (unchanged).
Zoom view = new sub-image column layout (G2).

Persist the user's preference in localStorage (`gcpEditorZoomView = true/false`).

---

## G3 — Confirmation State Sidecar

### Sidecar File: `<name>.confirmed.json`
Written alongside the `.txt` file. Updated on every shift-click.
```json
{
  "gcpName": {
    "image_filename.JPG": true
  }
}
```
A GCP pixel is "confirmed" when the user shift-clicks it (yellow → green).
A simple "I accept the estimate" confirmation is also valid (e.g., shift-clicking
exactly at the shown marker location).

### Storage Model Addition
Add `confirmed: boolean` to the `ImageGcp` interface in `gcps-utils.service.ts`.
Persist to `.confirmed.json` whenever an image is confirmed/unconfirmed.

### Marker Colors in images-tagger (existing SmartImage component)
- **Yellow pin**: pixel position is from estimate (not yet confirmed)
- **Green pin**: user has shift-click confirmed

If the pin color is not currently configurable in SmartimageComponent, add a
`pinColor: string` input to the component.

---

## G4 — c/t Count Badge on GCP List

On the `gcps-map` page, each GCP entry should show a badge like `2/5` where:
- `2` = number of images with a confirmed pixel for this GCP
- `5` = total number of images in which this GCP appears (per estimates)

Color logic:
- Red: 0 confirmed
- Yellow: some confirmed, not all
- Green: all confirmed

Reads from `storage.imageGcps` + `confirmed` field.

---

## G5 — Zoom +/- Buttons in Full Image View

In the full-image panel of the Zoom View (G2), add two buttons bottom-right:
- `[+]` — zoom in by 1.5× centered on current view
- `[-]` — zoom out by 1.5×

Style: round buttons with transparent background, similar to Google Maps zoom
controls. CSS positioned `absolute; bottom: 12px; right: 12px`.

These supplement the existing scroll-wheel zoom which is "too touchy" on
high-resolution images.

---

## R2 Research Findings (confirmed from GCPEditorPro source)

### Data Model (storage.service.ts)

- `storage.gcps: GCP[]` — `{ name, northing, easting, elevation }`
- `storage.imageGcps: ImageGcp[]` — `{ geoX, geoY, geoZ, imX, imY, imgName, gcpName }`
- `imX === 0 && imY === 0` is confirmed "not tagged" sentinel (existing code uses this)

### Injection Point (G1)

`load-config-txt.component.ts` **line 167** — after `txtParseResult` is assigned.
Insert `.estimates.json` sidecar check and `storage.imageGcps` population here.

### SmartimageComponent — Pin Color (G3)

Pin color is NOT currently configurable. Need to add:
```typescript
@Input() pinColor: string = 'yellow';   // 'yellow' | 'green'
```
to `smartimage.component.ts` and use it in the pin rendering logic.

### File System Access for Sidecar (G3)

- **Electron mode**: IPC via `main.js` — use `ipcRenderer.invoke('write-file', path, content)`
- **Web mode**: FileReader API only (can read, but cannot write sidecar files without user interaction / download)
  - Workaround for web: store confirmed state in `localStorage` keyed by filename hash

### GCP List Badge (G4)

Update `GcpInfo` class (in `gcps-map.component.ts` or shared types) to add:
```typescript
confirmCount: number;
totalCount: number;
```
Then update the `gcps-map` template to show `{{ gcp.confirmCount }}/{{ gcp.totalCount }}` badge.

### Implementation Order

G3 (sidecar + confirmed field) → G1 (pre-populate) → G2 (zoom view) → G4 (badge) → G5 (zoom buttons)
