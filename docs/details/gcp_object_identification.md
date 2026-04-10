# GCP Object Identification Strategy (Orange Clay Pigeons)

This document outlines the recommended object identification approach for Ground Control Points (GCPs) structured as an "X" of orange clay pigeons in drone imagery, and the downstream pixel refinement and crop-communication pipeline.

## The Hybrid Pipeline Strategy

Given the tiny size of GCPs in high-resolution drone imagery and the distinct color signature of orange clay pigeons, the following staged pipeline is recommended.  Stages 1–3 are implemented first; Stage 4 (CNN/DNN) is deferred until validated training data exists.

### Stage 1: Color-based Candidate Detection (Pre-filter)
*   **Method**: Convert imagery to HSV color space and use `cv2.inRange()` to isolate the specific orange pigment.
*   **Goal**: Eliminate 99% of the pixels (grass, dirt, water) in milliseconds to identify high-probability "clusters".

### Stage 2: Geometric Refinement (Shape Check)
*   **Method**: Use `cv2.findContours` on the orange masks. Look for clusters of 4-5 blobs arranged in a cross/X geometry.
*   **Goal**: Distinguish intentional patterns from incidental orange noise (e.g., orange leaves or trash).

### Stage 3: Pixel Refinement and Marker Delineation (geo-56c)

Once a candidate region is identified (or seeded by an existing pipeline projection estimate), this stage refines the pixel estimate to sub-pixel accuracy and computes a tight bounding box around the full marker extent.  Results are written back to `gcp_list.txt` and consumed by GCPEditorPro for adaptive zoom cropping.

#### 3a — Anomaly-relative color detection

Rather than hardcoding per-color HSV ranges, detect marker pixels by their deviation from the local background:
- Sample a surrounding ring/annulus to characterise the local terrain colour distribution.
- Flag pixels that deviate strongly from that distribution as candidate marker pixels.
- Robust to lighting, soil type, and vegetation variation.

Fallback: try per-color HSV masks (orange / red / white) in sequence; first to produce a coherent blob wins.

#### 3b — Search: expanding annuli (nearest-first)

Seed from the pipeline's rough pixel estimate `(px, py)`.  Search outward in expanding annuli, stopping at the first annulus that contains candidate marker pixels.  This ensures that in images containing multiple GCPs the nearest marker (not an arbitrary one) is always found.

Hard cap at ~300 px radius; fall back to unrefined estimate if exceeded.

#### 3c — Capture: flood-fill to full marker extent

Once candidate pixels are found, flood-fill connected non-natural pixels outward, with no hard boundary (the connected component may extend beyond the initial search radius).  Compute the tight axis-aligned bounding box of the connected component.

Sanity checks: discard if the component is too small (noise), too large (merged with another GCP marker), or has an aspect ratio inconsistent with an X shape.

#### 3d — Localisation: arm-cluster line intersection

The true intersection of the X arms is *not* in general the centroid of the colored pixel set (centroid drifts toward whichever arm has more pixels, e.g. if one arm is partially occluded or the mark is asymmetric):

1. Run PCA on the colored pixel set to obtain two principal-axis directions.
2. Cluster pixels by which eigenvector they project onto most strongly → arm 1 pixels, arm 2 pixels.
3. Fit an independent line through each cluster (e.g. `cv2.fitLine`).
4. Compute the geometric intersection of the two fitted lines.

This is robust to unequal arm lengths, partial occlusion, and perspective distortion.  If the skeleton-based approach is available, use it as a cross-check or fallback.

Confidence metric: fraction of expected marker pixels found, residual of the line fits, and the search radius at which the marker was found (smaller = higher confidence).

#### 3e — Output: updated columns in gcp_list.txt

When refinement succeeds, the row for that (GCP, image) pair is updated:

| Column | Index | Name | Notes |
|--------|-------|------|-------|
| 4 | 3 | `px` | **Replaced** with refined sub-pixel X coordinate |
| 5 | 4 | `py` | **Replaced** with refined sub-pixel Y coordinate |
| 8 | 7 | `confidence` | Changed from `projection`/`reconstruction` to `color_refined` |
| 9 | 8 | `marker_bbox` | `"x1,y1,x2,y2"` — tight bounding box of the full marker, comma-separated integer pixel coordinates |

Column 9 is absent for rows where refinement did not find a marker (backward compatible: downstream tools fall back to their default crop when this column is missing).

The bounding box is the tight fit around the connected non-natural-color component with no added padding.  GCPEditorPro and other consumers (training data extractors, etc.) add their own configurable margin at use time.

#### GCPEditorPro adaptive crop integration (geo-s6p)

`SubImageCropComponent` currently draws a fixed 256×256 px crop centered on `(imX, imY)`.  When `marker_bbox` is present, it instead:

1. Parses `"x1,y1,x2,y2"` from the `ImageGcp` object.
2. Computes a padded source rectangle, e.g. expand each edge by 10% of the larger bbox dimension, centered on the refined `(px, py)`.
3. Scales the padded rectangle to fill the 256×256 canvas (aspect ratio preserved; letterbox if needed).
4. Draws the crosshair at the refined `(px, py)` within the scaled view.

This means the canvas always shows the entire GCP marker at consistent apparent size regardless of altitude, GSD, or marker physical dimensions — substantially improving the tagging experience in zoom view.

#### Threading

Each (GCP, image) pair is independent.  Stage 3 is embarrassingly parallel and should run in a `ThreadPoolExecutor` consistent with the rest of the pipeline.  Apply after GCP and image ordering so that the highest-priority images are processed first (allowing early termination if a time budget applies).

---

### Stage 4: Lightweight CNN/DNN Verification (Deferred)

*   **Status**: Deferred until sufficient labeled training data exists (see geo-bwb, geo-kit, geo-d83).
*   **Method**: Train a custom CNN or small YOLO (v8-nano) model on tightly cropped sub-images (e.g. 128×128 px).
*   **Goal**: Confidently classify Stage 2 candidates as true GCP targets, reducing false positives from incidental orange objects.
*   **Note**: The `marker_bbox` column produced by Stage 3 provides natural positive-sample crops for training this model.

---

## Comparison of OpenCV Approaches

| Approach | Accuracy | Compute | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- |
| **Haar Cascades** | Medium-Low | Very Low | Fast, standard. | Grayscale only; high false-positive rate. |
| **Template Matching**| High | Low | Simple. | Fails on rotation and scale changes. |
| **Custom CNN/DNN** | **Very High** | High | Robust; learns color + shape. | Requires labeled training data. |

## Training Data Requirements (geo-bwb)

*   **Positive Samples**: Tightly cropped (5-10% padding) square images of the "X" targets.  The `marker_bbox` column from Stage 3 provides these automatically once Stage 3 is running.
*   **Negative Samples**: "Hard negatives" containing similar-colored non-target objects or typical terrain features.
*   **Resolution**: Must preserve Ground Sample Distance (GSD) to ensure pigeons are discernible.
