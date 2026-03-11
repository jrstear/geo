# GCP and Image Ordering for Optimal Photogrammetric Accuracy

## Motivation

The first 3–5 GCPs tagged in GCPEditorPro have an outsized effect on photogrammetric
accuracy.  They establish the initial rigid transformation that moves the floating
point cloud into world space (scale, rotation, translation).  If those first points
are spatially poor choices, the software's initial registration is weak, and all
subsequent tags build on that shaky foundation.

Similarly, within a given GCP, the order of images tagged matters.  Images where the
target falls near the sensor center have less radial lens distortion and are more
accurate.  Images at oblique angles provide Z (elevation) parallax that purely nadir
images cannot.

`emlid2gcp.py` estimates pixel coordinates for every (GCP, image) pair before
writing output.  Sorting is applied at that write step — after all pixel projections
are complete — so the ordering has access to the full result set.  A summary line
is printed: `Sorted output for optimal accuracy with minimal tagging`.

---

## Algorithm 1 — GCP Structural Priority Sort

**Goal:** establish the spatial "box" of the project as quickly as possible.

```
Input:  list of GCPs with (easting, northing, elevation)
Output: same list, reordered by structural importance
```

### Step-by-step

1. **Centroid** — compute `(mean_easting, mean_northing)` over all GCPs.

2. **XY diagonal** — `sqrt((max_east − min_east)² + (max_north − min_north)²)`.
   Used as the horizontal scale reference.

3. **Z significance test** — `z_range = max_elevation − min_elevation`.
   Z extremes are only promoted to priority slots when:
   ```
   z_range > z_threshold × xy_diagonal
   ```
   Default `z_threshold = 0.05` (5 %).

   **What this means in practice:**

   | Site | XY diagonal | Z range | Ratio | Z promoted? |
   |------|-------------|---------|-------|-------------|
   | Flat parking lot | 500 ft | 3 ft | 0.6 % | No |
   | Subdivision pad | 2,500 ft | 20 ft | 0.8 % | No |
   | Rolling terrain | 2,500 ft | 200 ft | 8 % | **Yes** |
   | Mountain survey | 3,000 ft | 800 ft | 27 % | **Yes** |

   The default of 5 % catches meaningfully hilly terrain while ignoring the minor
   elevation noise present in all flat sites.  The surveyor can lower this (e.g.
   `--z-threshold 0.02`) if modest topography on a smaller site should still
   activate Z-priority slots.

4. **Slot assignment** (greedy — once placed, a GCP is removed from the pool):

   | Slot | Selected as | Why |
   |------|-------------|-----|
   | 1 | `argmax(dist_from_centroid)` | Most distal — sets one anchor of the bounding box |
   | 2 | `argmax(dist_from_slot_1)` | Furthest from slot 1 — defines global scale and orientation |
   | 3 | `argmax(elevation)` *(Z significant only)* | Top of the vertical lever; prevents upward drift |
   | 4 | `argmin(elevation)` *(Z significant only)* | Bottom of the lever; prevents downward drift |
   | next | `argmin(dist_from_centroid)` | Center point — pins the model, prevents doming |
   | rest | sorted by `dist_from_centroid` descending | Perimeter-first redundancy |

   If Z is not significant, slots 3 and 4 are skipped and the centroid point
   moves up to slot 3.

### Fallback

If `easting`/`northing`/`elevation` are absent (projected coordinates not
available), the algorithm falls back to `lat`/`lon`/`ellip_alt_m`.  Degree-based
distance is used for centroid/distal calculations (haversine or simple Euclidean —
close enough at survey scale).

### `--no-sort` flag

When `--no-sort` is passed, GCPs are output in the same order they appear in the
input Emlid CSV, allowing upstream tools or the surveyor to control sequencing.
The existing lexical/numeric fallback sort is also disabled.

---

## Algorithm 2 — Image Confidence Sort (per GCP)

**Goal:** for each GCP, show the highest-confidence images first.
A good rule of thumb is ~8 images per GCP for solid accuracy; the sort ensures
that if the user stops at 8, they have tagged the 8 most valuable images.

```
Input:  for one GCP — dict of {filename → {px, py, mode}}
        global exif_map with img_w, img_h, gimbal_pitch per filename
        z_critical: bool — whether this GCP is a Z-extreme point
Output: filenames sorted best-first
```

### Distance from image center

```
normalized_dist = sqrt((px − w/2)² + (py − h/2)²) / (sqrt(w² + h²) / 2)
```

Values range 0 (dead center) to ~1 (corner of frame).  Lower is better — less
radial lens distortion.

### Nadir vs. oblique

Reuses the existing `is_nadir(exif)` predicate (`gimbal_pitch` within
`NADIR_TOL_DEG = 10°` of −90°).

Both nadir and oblique images have a place in the first 7 tags:

- **Nadir** (pitch near −90°) gives the best horizontal (X, Y) accuracy.
- **Oblique** shots provide the parallax needed for accurate Z (elevation).

Pushing all obliques to the back means the user may never tag any if they stop
at 7.  The `nadir_weight` parameter controls how aggressively obliques are
promoted.

### Sort score formula

```
score = normalized_dist + nadir_weight × (0 if nadir else 1)
```

Images are sorted by score ascending (lower = higher priority).

| GCP type | `nadir_weight` | Effect |
|----------|----------------|--------|
| Normal | `--nadir-weight` (default **0.2**) | Well-centred obliques appear in the top 7, interleaved with centred nadirs |
| Z-critical (slots 3 or 4) | `nadir_weight × 0.75` (default **0.15**) | Slightly stronger oblique promotion for the elevation-extreme GCPs that most need parallax |

Crossover point: an oblique beats a nadir when `nadir_weight < nadir_norm_dist`.
At the default 0.2, an oblique with `dist = 0.11` (score 0.31) beats a nadir
with `dist > 0.20`.  Raise `--nadir-weight` toward 1.0 to push obliques later;
lower it toward 0 to treat both tiers equally.

**Example** (nadir_weight = 0.2):

| Image | dist | nadir? | score |
|-------|------|--------|-------|
| A | 0.10 | Yes | 0.10 |
| B | 0.11 | No  | 0.11 + 0.20 = **0.31** |
| C | 0.25 | Yes | 0.25 |
| D | 0.15 | No  | 0.15 + 0.20 = **0.35** |
| E | 0.40 | Yes | 0.40 |
| F | 0.55 | Yes | 0.55 |

→ Order: A, C, B, D, E, F  (two obliques appear in the top 4 naturally)

### What counts as Z-critical?

A GCP is Z-critical when Z is significant (per the threshold test) AND the GCP's
elevation places it in the top or bottom 20 % of the site's elevation range:

```
z_deviation = abs(elevation − mean_elevation) / z_range
z_critical   = z_significant AND z_deviation > 0.80
```

### Fallback when EXIF fields are missing

- `img_w` / `img_h` absent → `normalized_dist = 0.5` (neutral; no advantage or penalty)
- `gimbal_pitch` absent → treated as nadir (tier 0) to avoid penalising the image

---

## CLI Interface

New arguments added to `emlid2gcp.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-sort` | off | Disable ordering; output GCPs in Emlid CSV order, images in match order |
| `--z-threshold` | `0.05` | Fraction of XY diagonal; Z extremes promoted to priority slots only above this ratio |
| `--nadir-weight` | `0.2` | Oblique penalty in image sort (0 = treat equally, 1 = all nadir first) |

`run_pipeline()` gains matching keyword arguments:
`sort_output: bool = True`, `z_threshold: float = 0.05`, `nadir_weight: float = 0.2`.

---

## Implementation Notes

### New pure functions (add before `_write_gcp_list`)

```python
def _sort_gcps(gcps: List[dict], z_threshold: float) -> List[dict]:
    """Return gcps reordered by structural priority."""
    ...

def _image_sort_score(px: float, py: float, img_w: int, img_h: int,
                      gimbal_pitch: Optional[float], z_critical: bool) -> float:
    """Return sort score for one image (lower = higher priority)."""
    ...

def _sort_images_for_gcp(img_map: dict, exif_map: dict,
                          z_critical: bool) -> List[str]:
    """Return image filenames sorted by confidence score."""
    ...
```

### Changes to `_write_gcp_list`

Signature gains: `sort_output: bool`, `z_threshold: float`, `exif_map: dict`.

When `sort_output=True`:
- Replace the existing lexical/numeric label sort (lines 535–539) with
  `sorted_gcps = _sort_gcps(gcps, z_threshold)` and iterate over that list.
- Determine which GCP labels are Z-critical (output of `_sort_gcps` can tag them).
- Replace `for img_name, est in img_map.items()` with the ordered list from
  `_sort_images_for_gcp(img_map, exif_map, z_critical=...)`.

### Changes to `run_pipeline`

- Add `sort_output=True` and `z_threshold=0.05` parameters.
- Pass `exif_map` (already computed) and the new parameters to `_write_gcp_list`.
- After projection, before writing: print
  `"Sorting output for optimal accuracy with minimal tagging..."` when `sort_output=True`.

### CLI changes

- `--no-sort` sets `sort_output=False`.
- `--z-threshold` sets `z_threshold`.
- Both passed through to `run_pipeline()`.

---

## GCPEditorPro Ordering (see geo-1i7)

For the sorted output to benefit the user, GCPEditorPro must present GCPs and
images in the order they appear in the input `.txt` file — not re-sort them.

**What needs investigation (GCPEditorPro/src/app):**

1. **GCP list order** — a previous session concluded GCPEditorPro sorts GCPs
   numerically.  Find the Angular code responsible, verify, and change it to
   preserve file-insertion order.

2. **Image list per GCP** — when a GCP is selected in the tag view, are images
   presented in file order?  This has not been investigated.  If the component
   re-sorts or shuffles images, the ordering from `emlid2gcp.py` has no effect.

The fix is in `GCPEditorPro/src/app` — likely the component that builds the GCP
list and the image carousel/grid for each selected GCP.
