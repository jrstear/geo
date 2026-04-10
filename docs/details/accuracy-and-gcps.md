# GCP Accuracy: How It Works and What to Do Next

This document answers three questions Isaiah raised after the Red Rocks accuracy
report:

1. Would having the correct coordinate transform *before* Pix4D processing have
   produced a better model?
2. What is the relationship between number of GCPs and RMSE — and what would it
   have taken to hit the 0.1 ft target?
3. Is it worth regenerating the model?

---

## How GCPs actually affect model generation

SfM photogrammetric processing (Pix4D, OpenDroneMap, Metashape) works in two stages.

**Stage 1 — Bundle adjustment (BA).** The software finds matching features across
overlapping images (tie points) and simultaneously optimizes camera positions,
orientations, and lens distortion parameters.  Without GCPs this produces a
*relative* model — correct internal shape but floating in an arbitrary coordinate
system, with no connection to real-world coordinates and potentially significant
systematic distortion.

**Stage 2 — GCP integration.** GCPs are introduced as hard constraints *inside* the
bundle adjustment, not applied afterward.  The BA re-runs and adjusts camera poses
and distortion parameters so that every GCP's known real-world position matches its
observed position in each image.  This is a full re-optimization of the model — it
actively reshapes the geometry, not just the coordinate frame.

### The doming problem

The most important thing GCPs correct — and the thing a post-hoc coordinate transform
*cannot* correct — is **doming** (also called the bowl effect).

Without well-distributed GCPs, SfM models develop a systematic warp: the center of
the model bulges up or depresses relative to the edges.  This is caused by a
systematic bias in estimating radial lens distortion during the BA.  Once baked into
the camera model, it cannot be removed by shifting or rotating the output.  A rigid
transform applied after processing moves the dome to a new location; it does not
flatten it.

The only fix is a **center-pin GCP** included in the BA — a point near the center of
the survey block that forces the BA to find lens distortion parameters that are
consistent with the ground truth at the center of the model, not just the edges.

---

## Would the correct transform before processing have helped?

Isaiah applied an easting shift of 2,222,906.89 ft to the GCP coordinates before
running Pix4D.  Analysis of the March 2026 survey (260303.csv) against the December
2025 survey (2512.csv) found that the complete transform is:

| Axis | Correction needed |
|------|-------------------|
| Easting | subtract 2,222,911.07 ft (Isaiah's shift was 4.18 ft short) |
| Northing | subtract 58.73 ft (not applied at all) |
| Elevation | subtract 4.17 ft (not applied at all) |

**However:** all three of these corrections are *uniform* across every GCP.  We
verified this directly — 49 of 50 aerial targets in the two surveys match to
sub-millimeter precision once the full transform is applied, with essentially zero
point-to-point variation in the residuals.

A uniform offset on all GCPs does not change their *relative* geometry.  The bundle
adjustment uses relative geometry to constrain model shape.  Applying the correct
full 3D transform post-hoc gives an identical result to having used the correct
coordinates a priori.

**Conclusion: reprocessing to fix the transform is not justified.  The accuracy
numbers would be the same.**

---

## Why 0.33 ft RMSE, and what would have hit 0.1 ft?

The 0.1 ft (~3 cm) target is achievable but requires getting several things right
simultaneously.  The factors in order of impact:

### 1. Flight altitude and GSD — the hard ceiling

Achievable RMSE is bounded by ground sample distance (GSD), the real-world size of
one image pixel.  The practical floor is 1–2× GSD.

| Altitude AGL | Typical GSD | Achievable RMSE (1–2× GSD) |
|---|---|---|
| 400 ft (120 m) | ~2 cm / 0.065 ft | 0.07–0.13 ft |
| 200 ft (60 m) | ~1 cm / 0.033 ft | 0.03–0.07 ft |
| 130 ft (40 m) | ~0.7 cm / 0.023 ft | 0.02–0.05 ft |

At 400 ft AGL, 0.1 ft sits right at the theoretical limit — achievable only with
near-perfect GCP placement and tagging.  At higher altitudes it is not achievable
regardless of how many GCPs are used.  **This is the single biggest lever.**

### 2. GCP distribution — more important than count

For a long narrow corridor (road survey), the dominant failure mode is longitudinal
bowing — the ends lift or sag relative to the center, giving the block a "banana"
shape.  The critical placements are:

- **End anchors** at both termini — corrects bowing along the long axis
- **Center pin** — corrects doming; this is the most commonly omitted point and the
  most impactful single addition
- **Perimeter fill** every 0.5–1 mile along the corridor

Published research on GCP count vs. RMSE for typical survey geometry (at constant
GSD and good tagging quality):

| GCP count | Typical RMSE |
|-----------|--------------|
| 3 | 3–5× GSD |
| 5–6 (perimeter only, no center) | 2–3× GSD |
| 7–10 (perimeter + center pin) | 1–2× GSD |
| 10–15 | 1–1.5× GSD |
| 15+ | ~1× GSD (plateau — diminishing returns) |

For a 6-mile corridor at 400 ft AGL aiming for < 0.1 ft: **9–11 GCPs with good
geometry** (both ends + center + quarter-mile spacing, alternating sides of road) is
the right prescription.  Fewer than 7, or 7 poorly distributed (e.g. clustered on
one side or all along one edge), will not hit the target regardless of tagging
quality.

### 3. GCP tagging accuracy

Every misplaced click in the GCP tagger adds noise to the BA constraints.  A 5-pixel
tagging error at 400 ft AGL corresponds to roughly 10 cm of real-world position
error at that GCP.  With 7 images per GCP the errors partially average out, but
systematic bias — always clicking a few pixels off-center — propagates directly into
model accuracy.

The practical targets are:

| Parameter | Minimum | Target |
|-----------|---------|--------|
| Images confirmed per GCP | 3 | 7 |
| GCP-* control points | 3 | 7 (of 10 candidates) |
| Check points (CHK-*) | 3 | 7 |

Colored-X targets (spray-painted X marks or orange clay pigeons) help here: their
high-contrast pattern lets the image pipeline auto-detect the centroid to ±5–30 px,
which the operator then refines to ±1–3 px in the zoom view.  Generic targets
(printed paper squares, sandbags) offer no automatic detection and rely entirely on
manual clicking.

---

## Verdict

### Do not reprocess the Red Rocks model

The $300–500 reprocess would produce the same RMSE because:

- The transform error is a uniform offset → same BA result regardless of when it is
  applied
- The 0.33 ft RMSE at aerial targets reflects the actual geometric quality of the
  bundle adjustment given the flight altitude, GCP count/distribution, and tagging
  precision of that job — not the coordinate frame

The 0.33 ft result is the model.  Shifting its coordinate frame does not change it.

### To hit < 0.1 ft on the next job

In order of impact:

1. **Reduce flight altitude.**  If flying at 400 ft, drop to 200 ft.  This halves
   GSD and halves the achievable RMSE floor.  It is the single largest lever and
   costs nothing except longer flight time and more images.

2. **Place GCPs using structural geometry.**  End anchors at both ends of the
   corridor, center pin at mid-corridor, then perimeter fill every ~0.5 mile
   alternating sides.  9–11 points for a 6-mile section.  A GCP placement advisor
   tool is in development (see `docs/plans/gcp-placement-advisor.md`) that will generate
   these locations automatically from a corridor polygon and export a KML for field
   navigation.

3. **Tag 7+ images per GCP.**  The colored-X detection pipeline (emlid2gcp.py →
   GCPEditorPro zoom view) directly addresses this.  The auto-detection narrows the
   pixel estimate to ±5–30 px; the spacebar-confirm workflow lets the operator
   accept or correct estimates image by image without hunting.

4. **Process in OpenDroneMap / WebODM instead of Pix4D.**  ODM produces equivalent
   photogrammetric output, supports the same GCP format (gcp_list.txt), and costs
   nothing per run.  The per-job Pix4D licensing and upload costs are the expense
   worth eliminating — not the reprocess.

5. **Use the flight planning tool** (in development, geo-8sk) to derive altitude
   from the RMSE target rather than guessing.  Given a target of 0.1 ft and a camera
   model, the tool computes required GSD → altitude, then estimates image count,
   flight time, battery swaps, and processing cost — so the trade-offs are visible
   before the flight, not after.

---

## Summary table

| Factor | Current job | Target for next job |
|--------|-------------|---------------------|
| Flight altitude | Unknown | ≤ 200 ft AGL |
| GSD | Unknown | ≤ 1 cm |
| GCP count | Unknown | 9–11 |
| GCP distribution | Unknown | End anchors + center pin + perimeter fill |
| Images per GCP | Unknown | ≥ 7 confirmed |
| GCP target type | Unknown | Colored X (spray paint or clay pigeon) |
| Processing software | Pix4D ($300–500/job) | WebODM (free) |
| Expected RMSE | 0.33 ft achieved | < 0.1 ft |
