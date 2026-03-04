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
5. Writes `gcp_list.txt` and `marks.csv` to `--out-dir`.

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
| 6th+ | Remaining, perimeter-first | Redundancy |

Z-priority slots (3rd and 4th) activate only when the site's elevation range
exceeds 5 % of the horizontal span.  Flat sites skip straight from 2nd to the
center pin.

**Images within each GCP are sorted by confidence** — most reliable first:
well-centerd shots (less lens distortion) before edge shots, nadir before
oblique for most GCPs.  For Z-critical GCPs (elevation extremes), well-centerd
obliques are interleaved with nadirs early in the list because oblique angles
provide the parallax that nails vertical accuracy.

The practical implication: **working through GCPs and images in the order shown
gives you the best photogrammetric result for the least effort.**

### Useful flags

| Flag | Effect |
|------|--------|
| `--no-sort` | Output GCPs in Emlid CSV order, images in match order |
| `--z-threshold 0.02` | Lower threshold to promote Z slots on modest terrain (default 0.05) |
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
shows a cropped sub-image centerd on the pipeline's pixel estimate, with a
crosshair overlay.

**Workflow for each GCP:**

1. Select the GCP from the list — work top to bottom (structural priority order).
2. Images are shown in confidence order — best images first.
3. For each image, check whether the crosshair lands on the GCP target.
   - If correct (or close enough to click precisely): **click the target** to
     confirm.  The marker turns green.
   - If the estimate is wrong: click the correct location.
   - If the target is not visible: leave it unconfirmed and move to the next image.
4. Aim to confirm at least 7 images per GCP, working from the top of the list.
   Because images are pre-sorted, confirming the first 7 gives you the 7
   highest-quality matches — there is no need to hunt for better ones further down.

### Progress indicators

Each GCP row in the list shows a **confirmed / total** badge coloured by how
many images have been confirmed for that GCP:

| Confirmed | Badge colour | What it means |
|-----------|--------------|---------------|
| 0–2 | Red | Insufficient — ODM may ignore or poorly weight this GCP |
| 3–6 | Amber | Usable, but worth adding more if time allows |
| ≥ 7 | Green | Good coverage — move on |

### Map view

The GCP map uses two pin styles:

- **Top-7 GCPs** — distinctive labelled pins showing the GCP name.  The pin
  colour follows the same red / amber / green thresholds as the badge.  As you
  confirm tags and return to the map, the pins update colour in real time.
- **Other GCPs** — standard pins, no label, no ratio colouring.

The map gives you an intuitive spatial read of your progress: you can see the
two distal anchors at opposite corners of the site, the center pin, the
elevation extremes on high and low ground, and the additional perimeter points —
and you can watch them turn green one by one as you work through the list.
**Your goal is to get all 7 labelled pins to green.**

At the top of the GCP list, a **summary line** shows your overall progress:

> `4 / 7 top GCPs sufficiently tagged`

- The **denominator is always 7** — the first 7 GCPs in the file, which are the
  structurally most important.
- The **numerator** counts how many of those 7 have reached green (≥ 7 confirmed).
- The summary line itself is coloured by the same rules applied to the numerator
  (red / amber / green).
- You can tag other GCPs beyond the top 7 — and you'll see their individual
  badges update — but only the top 7 count toward the summary line.

**You're done when the summary line turns green** — all 7 priority GCPs have ≥ 7
confirmed images.  Everything else is bonus coverage.

The same **confirmed / total** badge is shown in the zoom view header for the
currently selected GCP, so you always know where you stand without switching back
to the list.

---

## Step 4 — Export and upload

From the export screen:

- **Download `gcp_list.txt`** — all rows (confirmed + unconfirmed estimates).
  Use this to resume work in a later session.
- **Download `gcp_confirmed.txt`** — confirmed tags only, in ODM/WebODM format.
  Upload this file to WebODM when running your final reconstruction.

In WebODM: add `gcp_confirmed.txt` to the task, run with GCPs enabled.  The
confirmed tags you provided will anchor the point cloud to real-world coordinates.

---

## Quick-reference targets

| What | Target |
|------|--------|
| Confirmed images per GCP | ≥ 7 (green badge) |
| GCPs that must reach green | Top 7 in file order |
| Summary line colour when done | Green |
| Minimum to get any value from a GCP | 3 (amber) |
