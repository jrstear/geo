# DTM artifacts in road corridors — investigation and mitigation plan

**Bead:** geo-3xhz

## Problem statement

Photogrammetric DTMs produced by ODM (SMRF + OpenPointClass classification on
SfM/MVS point clouds) show systematic artifacts along road corridors. On the
aztec7 highway survey:

- **False multi-foot elevation steps** crossing the road, visible as 1-foot
  contour lines that don't correspond to real terrain
- **Building classification (class 6) bleeding** from actual roadside structures
  out onto the road surface, in the orange-V pattern radiating from utility
  poles and their shadows
- **Noisy ground classification** along road edges where the road meets
  shoulders, ditches, or curbs

These artifacts make the DTM unreliable as a deliverable surface for highway
corridor surveys — exactly the kind of survey we most want to support.

## Root causes (validated by inspection in CloudCompare)

1. **Featureless asphalt.** SfM/MVS dense matching relies on texture
   correlation. Asphalt is low-texture and semi-specular, so dense matching
   produces sparse, noisy points compared to surrounding terrain.

2. **Shadow contamination.** Pole and wire shadows are among the few high-
   contrast features on a road surface. MVS latches onto them, but shadow
   positions shift between images (different sun angles within a flight),
   so dense matching produces a "smeared band" of points at incorrect
   elevations. (Haala & Rothermel 2012; Stathopoulou & Remondino 2019.)

3. **SMRF's blindness to context.** SMRF uses a sliding morphological window
   comparing each point to a local minimum surface. It has no concept of
   "road" vs "building". When buildings sit close to the road and noisy road
   points are elevated, the building classification expands outward because
   SMRF can't distinguish "noisy road point near a building" from "building
   edge". The aztec7 ODM run used `smrf_window=18.0` — an 18m radius window
   that easily lets nearby buildings dominate the classification near road
   edges.

4. **Poles as classification anchors.** A reconstructed pole is a vertical
   cluster of points. SMRF correctly classifies it as non-ground, but the
   classification boundary radiates outward and captures nearby shadow-
   induced noise points in the same class — producing the V/wing pattern.

## Why your intuition (semantic priors) is well-supported

The user's intuition — "tell the classifier this is a highway" — matches the
state of the art in the literature. Two strong precedents:

- **Boulch et al. (2018), "SnapNet"** — project 2D semantic labels from
  images onto 3D points via known camera poses. ODM's OpenSfM output already
  contains the camera poses; this is the missing glue.
- **Stathopoulou & Remondino (2019)** — directly showed that semantic
  labelling (including shadow masking) before dense matching significantly
  improved reconstruction quality.

The pieces all exist in open source — what's missing is the integration into
the ODM pipeline.

## Mitigation options, ranked by effort

### Tier 1 — PDAL pipeline tweaks (no ML, days of work)

Lowest effort, most likely to give immediate improvement. All can be applied
post-ODM as a re-classification + re-rasterization step against the existing
LAZ output.

| Change | Why |
|---|---|
| Add `filters.outlier` (statistical, mean_k=8, multiplier=1.5) **before** SMRF | Removes the noisy elevated points on asphalt that trigger misclassification |
| Add `filters.elm` before SMRF | Removes Extended Local Minimum (low-noise) points |
| Try `filters.csf` (Cloth Simulation Filter, Zhang 2016) with `rigidness=3` (flat terrain) and `threshold=0.2` instead of SMRF | CSF drapes a virtual cloth over the inverted cloud — handles flat elevated surfaces like roads better than morphological opening |
| Lower `smrf_threshold` from 0.5 → 0.3, increase `smrf_window` from 18 → 25–33 | Tighter elevation tolerance, larger context window |

**Expected outcome:** smoother road surface ground classification, fewer
"building bleed" artifacts at road/structure interfaces. Expect partial
improvement, not a fix.

### Tier 2 — GIS priors via PDAL crop (days to a week)

Use external road geometry as a constraint. The simplest version:

1. Obtain a road polygon — from OpenStreetMap, by digitizing the centerline
   from the orthophoto, or from a survey CAD file
2. Buffer it appropriately (lane widths + shoulders)
3. Use PDAL `filters.crop` to isolate points within the road polygon
4. Within the road polygon, force ground classification + apply more
   aggressive outlier removal
5. Merge the corrected road points back with the rest of the cloud
6. Re-run `gdal_rasterize` / ODM's `dem.py` to regenerate the DTM

**Precedent:** Kaiser et al. (2017) trained aerial image segmentation using
OSM labels as ground truth — this is the same idea, applied at the cloud
classification stage rather than the labelling stage.

**Expected outcome:** clean road surface in the DTM regardless of what
SMRF thought. Trade-off: depends on quality of road polygon; misses road
geometry that isn't in the prior.

### Tier 3 — Image semantics → point cloud labels (weeks of work)

The state-of-the-art approach. Pipeline:

1. Run **SegFormer** (Xie et al. 2021) pretrained on **Cityscapes** on each
   input image. Cityscapes classes include road, sidewalk, building, pole,
   vegetation, vehicle — exactly what we need. Available pretrained on
   HuggingFace; inference is a few lines of `transformers` code.
2. For each 3D point in the dense cloud, project back to the images that see
   it (using camera poses from `reconstruction.topocentric.json`). Look up
   the semantic label of the corresponding image pixels.
3. Majority-vote or weighted-vote across views to assign a semantic class to
   each 3D point.
4. Override SMRF's ground/non-ground decision: if image semantics say "road"
   and the point is within ~1 m of a local ground plane, force to ground.
   If image semantics say "pole" or "building", force to non-ground
   regardless of SMRF.
5. Re-rasterize the DTM from the corrected ground points.

**Open-source building blocks:**
- SegFormer: HuggingFace `transformers` library
- Camera projection: copy from `sight.py` Mode B, or use OpenSfM's API
- Point cloud handling: PDAL or `laspy`

**Expected outcome:** the highest-quality DTM, semantically consistent with
the imagery. Closes most of the gap with commercial tools (Pix4Dmatic's
"surface type" classification works on similar principles).

**Risk:** complexity, latency (inference + back-projection at full res is
slow), and sensitivity to Cityscapes domain shift (Cityscapes is street-
level, not aerial — fine-tuning may be needed for best results).

### Tier 4 — Contribute to ODM upstream (months)

The right long-term home for this is ODM itself, not a post-processing
script. ODM already has the image inputs, camera poses, and dense cloud in
one place. A semantic-classification stage between `odm_filterpoints` and
`odm_dem` would benefit every ODM user. Coordinate with the ODM team
(Piero Toffanin / OpenDroneMap) before investing here — they may have
opinions about scope and architecture.

## Suggested first experiment

Start with **Tier 1, on aztec7's existing LAZ**:

1. Sync `s3://stratus-jrstear/bsn/aztec7/odm_georeferencing/odm_georeferenced_model.laz`
   locally
2. Run a PDAL pipeline that does outlier removal → CSF (rigidness=3) → ground filter
3. Re-rasterize the DTM with `gdal_grid` or PDAL's `writers.gdal`
4. Diff against the original `dtm.tif` in QGIS — look at the same highway
   stretch where we saw the V-shaped artifact
5. Measure: how much did the false elevation steps shrink?

If Tier 1 closes most of the gap, we may not need Tiers 2/3 for production
use. If Tier 1 helps but doesn't fix the building-bleed near poles, escalate
to Tier 2 (road polygon prior). Tier 3 is the long-term direction but
shouldn't precede a ground-truth measurement of how much Tier 1+2 already
gain.

## Reference papers

| Citation | Relevance |
|---|---|
| Pingel et al. (2013), *ISPRS J. Photogramm. Remote Sens.* 77, 21–30 | Original SMRF paper. 85% Kappa default, 90% optimized. Good at avoiding Type I errors, only "acceptable" at avoiding Type II — exactly our highway artifact. |
| Zhang et al. (2016), *Remote Sensing* 8(6) 501 | Cloth Simulation Filter (CSF). Less parameter-sensitive than SMRF; better for flat urban terrain. |
| Serifoglu Yilmaz et al. (2020), *Int. J. Digital Earth* | Benchmarks SMRF/CSF/PMF/PTIN on LiDAR vs photogrammetric clouds. SMRF best on LiDAR; PTIN often better on photogrammetric clouds. |
| Haala & Rothermel (2012), *ISPRS Annals* I-3 | Shadow boundaries as systematic error source in dense matching. |
| Stathopoulou & Remondino (2019), *ISPRS Archives* XLII-2/W9 | Semantic photogrammetry — masking shadow regions before MVS reduces noise. |
| Boulch et al. (2018), "SnapNet", *Computers & Graphics* 71, 189–198 | Project 2D semantic labels onto 3D points via known camera poses. Directly applicable. |
| Kaiser et al. (2017), *IEEE TGRS* 55(11), 6054–6068 | OSM as training labels for aerial semantic segmentation. |
| Xie et al. (2021), "SegFormer", *NeurIPS 2021* | Transformer-based semantic segmentation. Pretrained on Cityscapes (with road/building/pole/vegetation classes). |
| Ferrer-González et al. (2020), *Remote Sensing* 12(15) 2447 | UAV corridor mapping accuracy — confirms DTM errors concentrate at road surfaces and shadow zones. |

(Citations are from training data through early 2025 — verify exact volumes
against Google Scholar before formal use.)
