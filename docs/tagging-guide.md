# GCP Tagging Guide

This guide walks through the full workflow from raw Emlid rover data and drone
images to a confirmed `gcp_list.txt` ready for OpenDroneMap or WebODM.

---

## What you need

- Emlid Reach rover export CSV (one row per GCP survey point)
- Raw drone images (the full flight folder)
- Python environment with `emlid2gcp.py` dependencies installed (`setup.sh`)
- [GCPEditorPro](https://uav4geo.com/software/gcpeditorpro)

---

## Step 1 — Generate `gcp_list.txt`

```bash
python GCPSighter/emlid2gcp.py  emlid.csv  /path/to/images/  --out-dir ./output
```

This reads the Emlid CSV and every image in the folder, then:

1. Parses GCP survey positions (projected easting / northing / elevation, or
   WGS-84 lat/lon/alt when projected coordinates are absent).
2. Reads EXIF from every image (GPS position, gimbal angles, sensor dimensions).
3. Matches each image to the GCPs it is likely to contain, based on the camera
   footprint.
4. Projects the estimated pixel location of each GCP into its matching images.
5. Classifies each GCP as **GCP-\*** (control), **CHK-\*** (check), or **DUP-\*** (near-duplicate).
6. Writes `gcp_list.txt` and `marks.csv` to `--out-dir`.

### Output order (important — do not re-sort)

The pipeline orders GCPs and images deliberately:

**GCPs are sorted by structural priority** — the sequence that locks down your
model's geometry as quickly as possible with the fewest tags:

| Order | GCP selected as | Why first |
|-------|-----------------|-----------|
| 1st | Most distal from centroid | Sets one anchor of the bounding box |
| 2nd | Most distal from #1 | Defines global scale and orientation |
| 3rd | Highest elevation *(hilly sites only)* | Prevents vertical drift upward |
| 4th | Lowest elevation *(hilly sites only)* | Prevents vertical drift downward |
| 5th | Closest to centroid | Anti-doming center pin |
| 6th–10th | Remaining, perimeter-first | Redundancy — strongest structural value |
| 11th+ | Remaining, interior-first | Check points — hold-out accuracy validation |

Z-priority slots (3rd and 4th) activate only when the site's elevation range
exceeds 5 % of the horizontal span.  Flat sites skip straight from 2nd to the
center pin.

**Images within each GCP are sorted by confidence** — most reliable first:
well-centred shots (less lens distortion) before edge shots, nadir before
oblique for most GCPs.  For Z-critical GCPs (elevation extremes), well-centred
obliques are interleaved with nadirs early in the list because oblique angles
provide the parallax that nails vertical accuracy.

The practical implication: **working through GCPs and images in the order shown
gives you the best photogrammetric result for the least effort.**

### GCP classification (GCP-\* / CHK-\* / DUP-\*)

The pipeline renames every GCP label with a prefix that reflects its role:

| Prefix | Positions | Role |
|--------|-----------|------|
| **GCP-\*** | Top 10 by structural priority | Control points — used in bundle adjustment |
| **CHK-\*** | Positions 11+ | Check points — for independent accuracy validation |
| **DUP-\*** | Near-duplicates (within 1 m of a higher-priority GCP) | Excluded from normal ordering |

> **Note:** All points are passed to ODM as control — ODM has no native check-point
> concept.  The GCP-\*/CHK-\* distinction is organisational: it guides your tagging
> priority and will inform future hold-out RMSE reporting.

The prefix is stripped and reapplied on each run, so re-running the pipeline on
already-labelled data is safe.

### Useful flags

| Flag | Effect |
|------|--------|
| `--no-sort` | Output GCPs in Emlid CSV order, images in match order |
| `--z-threshold 0.02` | Lower threshold to promote Z slots on modest terrain (default 0.05) |
| `--n-control 7` | Adjust how many top GCPs become GCP-\* (default 10) |
| `--reconstruction path/to/reconstruction.json` | Use SfM-refined camera poses for ±5–20 px accuracy instead of EXIF-only ±30–150 px |

---

## Step 2 — Load into GCPEditorPro

1. Open GCPEditorPro.
2. **Import GCP file** → select `gcp_list.txt`.
3. **Import images** → point to the same raw images folder.
4. GCPEditorPro will show each GCP with its estimated pixel position already
   placed (shown as a yellow marker — unconfirmed estimate).

> The GCP list is displayed in file order.  Do not re-sort — the order is
> meaningful (see above).

---

## Step 3 — Confirm tags in the zoom view

Switch to **zoom view** (toggle in the top-right of the tagger).  The zoom view
shows a cropped sub-image centred on the pipeline's pixel estimate, with a
crosshair overlay.  When the pipeline detected a marker bounding box, the right
panel automatically zooms to match the crop scale so you can see the target clearly
without scrolling.

**Workflow for each GCP:**

1. Select the GCP from the list — work top to bottom (structural priority order).
2. Images are shown in confidence order — best images first.
3. For each image, check whether the crosshair lands on the GCP target.
   - If correct (or close enough to click precisely): **click the target** to
     confirm.  The marker turns green.
   - If the estimate is wrong: click the correct location.
   - If the target is not visible: leave it unconfirmed and move to the next image.
4. Aim to confirm at least 7 images per GCP-\* control point, working from the
   top of the list.  Because images are pre-sorted, confirming the first 7 gives
   you the 7 highest-quality matches — there is no need to hunt for better ones
   further down.
5. After completing all GCP-\* control points, work through CHK-\* check points
   the same way.  Check points are not required for reconstruction, but confirmed
   check-point tags enable independent accuracy validation.

### Progress indicators

Each GCP row in the list shows a **confirmed / total** badge coloured by how
many images have been confirmed for that GCP:

| Confirmed | Badge colour | What it means |
|-----------|--------------|---------------|
| 0–2 | Red | Insufficient — ODM may poorly weight this GCP |
| 3–6 | Amber | Usable, but worth adding more if time allows |
| ≥ 7 | Green | Good coverage — move on |

This colour scheme applies equally to GCP-\* and CHK-\* points.

### Map view

The GCP map uses two pin styles:

- **GCP-\* control points** — distinctive labelled pins showing the GCP name.
  The pin colour follows the same red / amber / green thresholds as the badge.
  As you confirm tags and return to the map, the pins update colour in real time.
- **CHK-\* check points** — smaller unlabelled pins.
- **DUP-\* near-duplicates** — shown as a distinct symbol; generally low priority.

The map gives you an intuitive spatial read of your progress: you can see the
two distal anchors at opposite corners of the site, the center pin, the
elevation extremes on high and low ground, and the additional perimeter
points — and you can watch them turn green one by one as you work through the list.
**Your goal is to get all GCP-\* labelled pins to green.**

At the top of the GCP list, a **summary line** shows your overall progress:

> `4 / 10 GCP-* control points sufficiently tagged`

- The **denominator** is the total number of GCP-\* control points in the file.
- The **numerator** counts how many have reached green (≥ 7 confirmed images).
- The summary line itself is coloured by the same rules (red / amber / green)
  applied to the numerator.
- CHK-\* check points have their own badge progress but do not count toward the
  GCP-\* summary.

**You're done with control tagging when the summary line turns green** — all
GCP-\* control points have ≥ 7 confirmed images.  CHK-\* tagging is bonus
coverage that improves accuracy reporting.

The same **confirmed / total** badge is shown in the zoom view header for the
currently selected GCP, so you always know where you stand without switching back
to the list.

---

## Step 4 — Export and upload

From the export screen:

- **Download `gcp_list.txt`** — all rows (confirmed + unconfirmed estimates).
  Use this to resume work in a later session.
- **Download `gcp_confirmed.txt`** — confirmed GCP-\* control point tags only,
  in ODM/WebODM format.  Upload this file to WebODM when running your final
  reconstruction.
- **Download `chk_confirmed.txt`** — confirmed CHK-\* check point tags only.
  Use this file post-reconstruction to compute independent RMSE accuracy figures.

In WebODM: add `gcp_confirmed.txt` to the task, run with GCPs enabled.  The
confirmed GCP-\* tags will anchor the point cloud to real-world coordinates.

> Because ODM has no native check-point concept, `chk_confirmed.txt` is not
> supplied to ODM.  Check-point accuracy validation (comparing surveyed vs
> photogrammetric coordinates for the held-out points) must be done manually
> using `marks.csv` after reconstruction.

---

## Minimum requirements (USGS / ASPRS)

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| GCP-\* control points | **3** | Hard floor for constraining a block; 5+ strongly recommended for redundancy and outlier detection |
| CHK-\* check points | **3** | USGS NSSDA minimum for a publishable accuracy assessment; aim for 20+ on larger surveys |
| Confirmed images per GCP | **3** | Below 3 ODM may poorly weight the point (amber badge) |
| Confirmed images for strong constraint | **7** | Green badge target — provides solid redundancy |

> Reference: *USGS National Geospatial Program — Lidar Base Specification* and
> *ASPRS Positional Accuracy Standards for Digital Geospatial Data (2015)* both
> specify a minimum of 3 independent check points for accuracy reporting.  The
> ASPRS standard further recommends ≥ 20 check points for surveys larger than
> 500 km².

---

## Quick-reference targets

| What | Target |
|------|--------|
| GCP-\* control points (minimum) | 3 (USGS/ASPRS floor); pipeline default is top 10 |
| CHK-\* check points (minimum) | 3 (USGS accuracy-reporting requirement) |
| Confirmed images per GCP-\* control point | ≥ 7 (green badge) |
| Confirmed images per CHK-\* check point | ≥ 7 (green badge) — bonus coverage |
| GCP-\* control points that must reach green | All of them (summary line) |
| Summary line colour when done | Green |
| Minimum confirmed images for any value | 3 (amber) |
