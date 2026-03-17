# RMSE Explained: Photogrammetric Accuracy for the Working Surveyor

This document is written for someone who already knows what GCPs are and has
seen a Pix4D accuracy report — but wants to understand what the numbers
actually mean, why check points matter more than control points for judging
accuracy, and how the new open-source pipeline produces the same report.

---

## 1. What RMSE Is and How to Read It

**Root Mean Square Error (RMSE)** is the standard way to express how far a set
of measured positions are from their true positions, on average.  It is not
the same as the average error — it penalizes large misses more heavily than
small ones, which makes it a better summary of real-world performance.

The formula for a single axis (say, Z / elevation) with N check points:

```
dZ_i  = reconstructed_Z_i  −  surveyed_Z_i   (residual for point i)

RMSE_Z  =  sqrt( (dZ_1² + dZ_2² + ... + dZ_N²) / N )
```

Photogrammetry software reports this separately for each axis and then as a
combined 3D total:

```
RMSE_X  =  sqrt( mean(dX_i²) )
RMSE_Y  =  sqrt( mean(dY_i²) )
RMSE_Z  =  sqrt( mean(dZ_i²) )

RMSE_3D =  sqrt( RMSE_X² + RMSE_Y² + RMSE_Z² )
```

**What "0.2 ft RMSE Z" means in practice:**

Imagine you placed 10 check points across your survey area and ran them through
the model.  If RMSE Z = 0.2 ft, the typical elevation error is about 0.2 ft.
Most points will be within 0.2–0.3 ft; an occasional outlier might reach 0.4 ft.
It does not mean every point is exactly 0.2 ft off — some will be closer,
some farther.  It is a statistical summary of the whole population.

For reference, the USGS NSSDA Accuracy Standard at 95% confidence is
approximately 1.96 × RMSE, so a 0.2 ft RMSE Z corresponds to roughly
0.39 ft accuracy at the 95% confidence level.

---

## 2. GCPs (Control Points) vs. CHKs (Check Points)

### What they are

- **GCPs (control points)** are ground-surveyed positions whose coordinates are
  fed into the bundle adjustment.  The software uses them as constraints while
  it solves for camera positions and lens distortion.  In the new pipeline,
  these are the points labeled **GCP-\*** in the `gcp_confirmed.txt` file.

- **CHKs (check points)** are ground-surveyed positions that are withheld from
  the bundle adjustment entirely.  After processing is complete, the software
  reprojects the surveyed coordinates into the finished model and measures
  where they land.  These are labeled **CHK-\*** and held back in
  `chk_confirmed.txt`.

### Why GCP residuals are not a reliable accuracy measure

GCPs were used as constraints during bundle adjustment.  The optimization
actively moved camera poses to minimize the distance between each GCP's
known world coordinates and its observed pixel position.  The residuals you
see for a GCP in the quality report are the leftovers after the solver has
already done its best to honor that point.  A small GCP residual simply
confirms the solver converged — it does not tell you how accurate the model
is at locations the solver never saw.

Think of it like this: if you fit a curve through five data points, those
five points will lie exactly on the curve by definition.  The curve's accuracy
is only tested by points it was not fitted to.

### Why CHK residuals are the real accuracy number

Check points are independent witnesses.  They were never shown to the bundle
adjustment.  Their residuals measure actual model error — the gap between what
the model predicts and what the ground truth says — at locations spread across
the survey area.  A CHK RMSE of 0.1 ft is a meaningful, defensible accuracy
claim.  A GCP RMSE of 0.1 ft is not.

The USGS NSSDA standard requires a minimum of 3 independent check points for a
publishable accuracy assessment.  The pipeline targets 7 confirmed CHK-\* points
for the same reason it targets 7 confirmed GCP-\* points — statistical
confidence and spatial distribution both improve with more.

---

## 3. What Pix4D Computes and Reports

Pix4D's quality report includes a "Check Points" table.  For each check point,
it computes the signed residual on each axis and an overall RMS per axis:

```
dX_i  = X_reconstructed_i  −  X_surveyed_i
dY_i  = Y_reconstructed_i  −  Y_surveyed_i
dZ_i  = Z_reconstructed_i  −  Z_surveyed_i

RMS_X  =  sqrt( (dX_1² + dX_2² + ... + dX_N²) / N )
RMS_Y  =  sqrt( (dY_1² + dY_2² + ... + dY_N²) / N )
RMS_Z  =  sqrt( (dZ_1² + dZ_2² + ... + dZ_N²) / N )

RMS_3D =  sqrt( RMS_X² + RMS_Y² + RMS_Z² )
```

For a check point, "reconstructed" means: take the camera pose and lens model
that the bundle adjustment produced, and use those to triangulate where the
surveyed ground position projects to in 3D space.  The difference between
that triangulated 3D position and the surveyed position is the residual.

Pix4D also reports the mean error (bias) and standard deviation per axis.  A
large mean error with small standard deviation indicates a systematic offset
(e.g., a coordinate reference system mismatch).  A small mean with large
standard deviation indicates random noise (pointing error, target distortion).
Both patterns look similar in the headline RMSE — the mean/std breakdown is
where you diagnose which problem you have.

---

## 4. Why Z Error Is Typically 2–3× Horizontal Error

In a purely nadir (straight-down) drone survey, all cameras are pointed
directly at the ground.  Two overlapping nadir images of the same ground
feature differ primarily in their horizontal viewing angle — one is slightly
to the left, one slightly to the right.  That horizontal parallax is what lets
the software triangulate X and Y positions accurately.

Elevation (Z) is determined by how much a feature shifts horizontally between
two images as a function of its height above the ground plane — in other words,
by the vertical parallax.  In nadir imagery, features at different elevations
shift very little relative to each other because all cameras are looking almost
straight down.  The vertical parallax signal is weak, and Z accuracy suffers.

The practical result: at a nadir-only survey, Z RMSE is typically 2–3× the
horizontal RMSE.  This is a geometric property of the imaging geometry, not a
software limitation.

**Oblique imagery breaks this pattern.**  When some cameras are tilted 30–45°
from nadir, they view the scene from a much steeper oblique angle.  Features
at different elevations now produce strong horizontal shifts between the nadir
and oblique views — a much larger vertical parallax signal.  Adding even a
small proportion of oblique images (one oblique pass alongside standard nadir
passes) substantially improves Z accuracy, often bringing it within 1.5× the
horizontal error.

This is why the pipeline deliberately interleaves nadir and oblique images in
the first 7 image slots for each GCP.  You need both view types in the bundle
adjustment to get accurate 3D positions — pure nadir tagging produces good X/Y
and mediocre Z.

---

## 5. What "Sufficiently Tagged" Means

When you tag a GCP pixel in an image, you are providing the bundle adjustment
with one ray: "from this camera, at this pixel, this 3D point was visible."
The bundle adjustment triangulates the 3D position of the GCP by finding where
multiple rays from multiple cameras intersect.

With only 1 or 2 images tagged, triangulation is either impossible or highly
unstable — small pointing errors cause large position errors.  With 3 images
from reasonably different viewpoints, a position can be determined, but outlier
rejection is weak (if one tag is off, you cannot tell which one).  With 7
images, the intersection is overdetermined: the solver can detect and
downweight any single bad tag, and the statistical averaging significantly
reduces the effect of per-image pointing errors.

The practical threshold of **7 confirmed images per GCP** is where the
accuracy benefit per additional image starts to plateau.  Below 3 is
insufficient; 3–6 is usable; 7+ is the target for high-accuracy work.

The pipeline's image sort guarantees that if you stop at 7, you have tagged the
7 best images — best-centered on the sensor (less lens distortion) with a mix
of nadir and oblique shots for full 3D constraint.  There is no value in
hunting further down the list unless the first 7 images have problems (poor
lighting, occluded marker).

The same threshold applies to CHK-\* check points: 7 confirmed images each
gives the triangulation enough overdetermination to produce a reliable
independent position for the accuracy check.

---

## 6. How the Open-Source Pipeline Produces the Same Report

Pix4D's check-point accuracy report is not magic — it is a straightforward
application of the triangulation equations above.  The same calculation can be
reproduced with open-source tools once you have the camera poses.

### The tools

| Tool | Role |
|------|------|
| `sight.py` | Parses Emlid rover CSV + drone image EXIF; outputs `gcp_confirmed.txt` and `chk_confirmed.txt` |
| WebODM / OpenDroneMap | Runs the bundle adjustment using `gcp_confirmed.txt`; writes `reconstruction.json` |
| `rmse_calc.py` (planned) | Reads `reconstruction.json` + `chk_confirmed.txt`; computes per-axis RMSE |

### What reconstruction.json contains

After WebODM processes your images with GCPs, it writes a
`reconstruction.json` file that contains the fully optimized camera pose for
every image: position (X, Y, Z in world coordinates), rotation (3×3 matrix or
quaternion), and focal length / distortion parameters.  This is the same
information Pix4D uses internally to compute check-point residuals.

### How rmse_calc.py will work

For each confirmed CHK-\* tag (image + pixel coordinates from `chk_confirmed.txt`):

1. Look up the camera pose for that image in `reconstruction.json`.
2. Back-project the pixel through the camera model to produce a ray in world
   space — a line from the camera center in the direction of the tagged pixel.
3. Collect all rays for this CHK point (one per tagged image).
4. Triangulate the 3D intersection of those rays using a least-squares solver.
5. Compare the triangulated position to the ground-surveyed coordinates from
   the Emlid CSV.
6. Record `dX`, `dY`, `dZ` for this point.

After processing all CHK points:

```
RMSE_X  =  sqrt( mean(dX_i²) )
RMSE_Y  =  sqrt( mean(dY_i²) )
RMSE_Z  =  sqrt( mean(dZ_i²) )
RMSE_3D =  sqrt( RMSE_X² + RMSE_Y² + RMSE_Z² )
```

The output will be a table formatted identically to the Pix4D check-point
report: per-point residuals, per-axis RMSE, mean error, and standard
deviation.  The numbers are directly comparable — same math, same
interpretation.

The `--reconstruction` flag in `sight.py` already supports reading
`reconstruction.json` to improve pixel projection accuracy from ±30–150 px
(EXIF-only) to ±5–20 px (SfM-refined camera poses).  The same file path will
be the input to `rmse_calc.py`.

---

## 7. What the Experiments Will Tell Us

With a reproducible open-source pipeline and ground truth from the Emlid rover,
we can now run controlled experiments that answer questions Pix4D's proprietary
workflow makes difficult or expensive to test.

### Configurations to vary

- **GCP count:** 3, 5, 7, 10, 15 control points — holding geometry and imagery constant
- **GCP geometry:** end-anchors only vs. end-anchors + center pin vs. full
  perimeter — the doming test
- **Images per GCP:** confirm 3, 5, 7, or all available — the tagging effort trade-off
- **Nadir/oblique mix:** 100% nadir vs. 80/20 nadir/oblique vs. 60/40 — the Z accuracy test
- **Flight altitude:** 130 ft, 200 ft, 400 ft AGL — verifying the 1–2× GSD rule

### What we measure

For each configuration, `rmse_calc.py` produces:

- RMSE X, Y, Z, 3D for the withheld CHK-\* set
- Per-point residuals to identify spatial patterns (edge error, doming signature)
- Mean error (systematic bias) vs. standard deviation (random noise)

### What we learn

These experiments directly answer:

1. **How many GCPs are enough?**  Where does the RMSE curve flatten?  Is 7
   meaningfully better than 5, or is it already on the plateau?

2. **Does the center-pin GCP actually matter?**  Comparing a run with vs.
   without the centroid GCP isolates the doming effect empirically rather than
   relying on theoretical arguments.

3. **How many images per GCP matter?**  If 5 confirmed images gives the same
   RMSE as 7, the tagging target can be lowered without sacrificing accuracy.

4. **Does adding oblique images improve Z accuracy on this camera / flight
   altitude?**  The theory predicts yes, but the magnitude is site- and
   geometry-dependent.

5. **Does the open-source pipeline match Pix4D?**  Running the same imagery
   through both and comparing CHK RMSE numbers gives a direct calibration of
   how WebODM compares to the proprietary benchmark on real survey data.

The Ghostrider Gulch dataset — 944 images, ground-surveyed GCPs with confirmed
pixel tags, known Emlid rover coordinates — is the test bed.  Every
configuration can be evaluated against the same ground truth without flying
again.
