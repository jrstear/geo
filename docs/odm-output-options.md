# ODM Output Options: Split-Merge, COG, and Compression

Reference for large-dataset processing and orthophoto optimization decisions.
Covers lessons from Aztec (1393 imgs) and planning for Red Rocks (6428 imgs).

---

## Split-Merge (`--split` / `--split-overlap`)

### What it does

ODM clusters images into geographic sub-models of ~N images each, runs the
**full pipeline independently** on each (SfM → dense → mesh → texture → DEM →
ortho), then merges outputs into a single deliverable. Memory scales with
sub-model size, not total dataset size.

### Memory impact (all heavy stages benefit)

| Stage | Aztec full (1393 imgs) | Aztec split N=300 (~5 sub-models) |
|---|---|---|
| Dense MVS | ~32 GB | ~7 GB |
| Meshing | >64 GB (OOM on m5.4xlarge) | ~13 GB |
| Texturing | ~100+ GB (OOM on r5.4xlarge) | ~20 GB |
| Orthophoto | ~212 GB (OOM) | ~42 GB |

For Red Rocks (6428 imgs, N=300 → ~21 sub-models), even an **m5.4xlarge
(64 GB) becomes viable** — changes the instance sizing story entirely.

### Choosing N (images per sub-model)

`--split N` sets the **target images per sub-model** (not number of
sub-models). ODM clusters geographically so each sub-model is a coherent
area, not a random split.

**Total sub-models ≈ total images / N** (roughly — overlap images appear in
two adjacent sub-models).

| N | Sub-models (Aztec) | Sub-models (Red Rocks) | Notes |
|---|---|---|---|
| 100 | ~14 | ~64 | Very low memory, many seams, fragile SfM |
| 200 | ~7 | ~32 | Reasonable minimum |
| **300** | **~5** | **~21** | **Recommended starting point** |
| 500 | ~3 | ~13 | More GCPs per model, fewer seams |

**Practical floor**: each sub-model needs at least 2–3 GCPs and enough image
connectivity for stable SfM. Below ~100–150 images, sparse areas or few GCPs
per sub-model can cause failures.

Guidelines:
- Smaller N → less memory, more sub-models, more boundary seams, more total
  compute (overlap images processed twice)
- Larger N → approaches single-model memory, fewer seams, faster total runtime
- For Aztec (9 GCPs): N=300 → ~2 GCPs/sub-model (workable)
- For Red Rocks (50 GCPs): N=300 → ~2–3 GCPs/sub-model (workable);
  N=500 → ~4 GCPs/sub-model (better coverage)

### Choosing M — overlap in metres (`--split-overlap M`)

M is the boundary buffer: images within M metres of a sub-model boundary are
included in **both** adjacent sub-models, giving the merge step shared
geometry to align and blend seams.

M should be **at least 2–3× the image footprint radius** at your AGL:

- At 250 ft AGL (76 m), DJI M3E footprint ≈ 110 m × 80 m → radius ~55 m
- Minimum: ~80 m (1–2 images of overlap at each boundary)
- **Recommended: 100–150 m** for comfortable seam alignment
- Too small: merge fails or produces visible seams in ortho/DEM
- Too large: boundary images processed many times, negates memory savings

### Recommended values

| Dataset | `--split` | `--split-overlap` | Instance |
|---|---|---|---|
| Aztec (1393 imgs) | 300 | 100 | r5.4xlarge (128 GB) |
| Red Rocks (6428 imgs) | 300 | 100 | m5.4xlarge (64 GB) or better |

### Downsides

**Seams.** Visible boundary lines can appear in the orthophoto, especially
with color/exposure variation across flight lines or terrain elevation jumps
at boundaries. DEM merge is usually cleaner than ortho merge.

**Reduced global accuracy.** Each sub-model bundle-adjusts independently;
tie points don't cross boundaries, so global geometry constraints are weaker.
Well-distributed GCPs largely mitigate this — GCPs act as the global anchor.

**More total compute.** Overlap images run through the full pipeline twice.
Expect ~1.3–1.5× the single-model compute time, but at much lower peak
memory (so cheaper instance possible).

**No mid-pipeline adoption.** Split restructures the project directory into
`submodels/submodel_0000/`, `submodel_0001/`, etc. at the dataset stage.
Cannot be added to an in-progress run. Must be set from the start.

**One sub-model failure stops merge.** If one sub-model's SfM fails, the
full merge fails. Individual sub-models can be rerun with `--rerun-from`
targeting that sub-model's directory.

### Example command

```bash
docker run --rm -v /data/project:/datasets/project opendronemap/odm:3.3.0 \
  --project-path /datasets project \
  --split 300 --split-overlap 100 \
  --pc-quality medium \
  --dsm --dtm \
  --orthophoto-resolution 5
```

---

## Orthophoto Resolution (`--orthophoto-resolution`)

Units: cm/pixel. Lower number = higher resolution = more memory.

| Value | vs. 2 cm GSD | Memory (relative) | Notes |
|---|---|---|---|
| 2 | 1× (native) | 1× (~212 GB for Aztec) | OOM on anything < 256 GB |
| 5 | 2.5× | ~1/6 (~34 GB) | Still excellent; recommended default |
| 10 | 5× | ~1/25 (~8 GB) | Good for inspection, large-area overview |

**For Aztec/Red Rocks surveys at 250 ft AGL:** 5 cm/px is 2.5× native GSD
and is sharp enough for all practical deliverable and accuracy-check uses.

Rerun orthophoto only (after a failed full run):

```bash
docker run --rm -v /data/project:/datasets/project opendronemap/odm:3.3.0 \
  --project-path /datasets project \
  --rerun-from odm_orthophoto \
  --orthophoto-resolution 5
```

---

## Cloud-Optimized GeoTIFF (COG)

### What it is

A regular GeoTIFF stores data as scanlines (row 1, row 2...). Reading any
small geographic area requires scanning past all preceding rows — expensive
for large files.

A COG has two structural additions:

**Internal tiling**: The image is stored as small rectangular blocks
(typically 256×256 or 512×512 px). Reading any area fetches only the
intersecting tiles — no scanning.

**Internal overviews (pyramids)**: Pre-computed downsampled versions of the
full image at multiple zoom levels (1:2, 1:4, 1:8, 1:16, 1:32...) stored
inside the file. A viewer zoomed out reads the appropriate overview level
rather than the full-resolution data.

The "cloud" part: a specific byte layout plus ghost metadata block lets HTTP
clients fetch individual tiles via range requests — enabling efficient S3
serving without downloading the whole file.

### Why QGIS is slow (and COG fixes it)

Without overviews, zooming out in QGIS on a large orthophoto forces QGIS to
read the **entire full-resolution raster** and downsample it on the fly — for
a 2 cm/px corridor orthophoto that can mean reading 10–50 GB per render.

Google Maps is fast because it pre-renders a tile pyramid at every zoom level.
QGIS with a COG does the same: reads the appropriate overview level, fetches
only visible tiles.

**A VRT does not help if the underlying files have no overviews.** The VRT
just points to the underlying GeoTIFFs; if those lack overviews, QGIS still
reads full resolution.

### Building overviews

In QGIS: Raster → Build Pyramids (on the underlying file, not the VRT).

Via GDAL in-place:
```bash
gdaladdo -r average odm_orthophoto.tif 2 4 8 16 32 64
```

Via `gdal_translate` (creates a new file with tiling + overviews + compression
in one pass — preferred):
```bash
gdal_translate -of COG \
  -co COMPRESS=LZW \
  -co PREDICTOR=2 \
  -co OVERVIEW_LEVEL=AUTO \
  -co OVERVIEW_RESAMPLING=AVERAGE \
  -co BIGTIFF=IF_SAFER \
  odm_orthophoto.tif deliverable_orthophoto_cog.tif
```

### ODM's `--orthophoto-tiled` flag

Outputs a tiled GeoTIFF (not scanline), which is the first prerequisite for
COG. Does **not** add overviews — still slow for zoomed-out rendering. Only
half the fix; post-processing with `gdal_translate -of COG` is still needed.

---

## Compression

### LZW (lossless)

Best for final deliverables. Universal support, no quality loss.

Always pair with `PREDICTOR=2` (horizontal differencing) — significantly
improves compression ratio on continuous raster data:

```bash
-co COMPRESS=LZW -co PREDICTOR=2
```

**Downsides:**
- **Write speed**: Compressing a 50 GB orthophoto can add 20–40 min. This is
  why ODM does not enable it by default.
- **Modest ratio**: For photographic aerial imagery (already visually complex),
  LZW typically achieves only 30–50% size reduction.
- **Read CPU overhead**: Decompression on every tile read. Negligible on NVMe;
  still usually a net win on network storage (less I/O).

### JPEG-in-TIFF (lossy, for web/storage)

For internal use or web serving where storage is constrained, JPEG compression
within the GeoTIFF gives 5–10× better size reduction at visually
indistinguishable quality (quality 85):

```bash
-co COMPRESS=JPEG -co JPEG_QUALITY=85 -co PHOTOMETRIC=YCBCR
```

`PHOTOMETRIC=YCBCR` converts RGB to YCbCr before JPEG compression — improves
ratio by ~30% over direct RGB JPEG. Not lossless; not appropriate for primary
deliverable archiving.

### ODM flags

```bash
--orthophoto-compression LZW    # equivalent to post-processing LZW
--orthophoto-tiled              # tiled output (not full COG — no overviews)
```

Doing compression at ODM time vs post-processing produces identical output,
but post-processing is slightly better because:
1. ODM's intermediate pipeline doesn't benefit from the compressed orthophoto
2. You can inspect raw output before committing to a packaging decision
3. Overviews can be added in the same `gdal_translate` pass

---

## Recommended Workflow

### After any ODM run

```bash
# Create deliverable COG (lossless, with overviews — fast in QGIS):
gdal_translate -of COG \
  -co COMPRESS=LZW \
  -co PREDICTOR=2 \
  -co OVERVIEW_LEVEL=AUTO \
  -co OVERVIEW_RESAMPLING=AVERAGE \
  -co BIGTIFF=IF_SAFER \
  odm_orthophoto/odm_orthophoto.tif \
  deliverable/orthophoto_cog.tif

# Optional: smaller web version (lossy, ~10× smaller):
gdal_translate -of COG \
  -co COMPRESS=JPEG \
  -co JPEG_QUALITY=85 \
  -co PHOTOMETRIC=YCBCR \
  -co OVERVIEW_LEVEL=AUTO \
  -co OVERVIEW_RESAMPLING=AVERAGE \
  odm_orthophoto/odm_orthophoto.tif \
  deliverable/orthophoto_web.tif
```

### Red Rocks full run (recommended)

```bash
docker run --rm -v /data/project:/datasets/project opendronemap/odm:3.3.0 \
  --project-path /datasets project \
  --split 300 --split-overlap 100 \
  --pc-quality medium \
  --dsm --dtm \
  --orthophoto-resolution 5
```

Then post-process with `gdal_translate -of COG` as above.

---

## Open Questions / Future Work

- Does `geo/package.py` currently add overviews, or just LZW? If not, add
  `gdal_translate -of COG` step to the packaging workflow.
- Evaluate whether `--orthophoto-resolution 5` is sufficient for BSN
  deliverable spec, or if 2 cm is required (needs larger instance or split).
- For production pipeline (geo-8fg automated stage switching): dense MVS is
  still GPU-bound regardless of split; split reduces memory at meshing/texturing
  but not the GPU benefit window for dense.
