# True Ortho Comparison: aztec7 Results

**Date:** 2026-04-01
**Related beads:** geo-ksd (productionize true ortho), geo-51u (proof of concept)

## Summary

The true ortho post-processor (true_ortho.py) produced **significantly worse** results
than ODM's standard orthophoto on aztec7.  The approach is shelved pending fundamental
improvements.

## Comparison Data

42 targets evaluated.  "Offset" = estimated distance from survey coordinate to visible
target center in the orthophoto (smaller = better).

| Metric | Original ODM ortho | True ortho (no DSM) | True ortho (with DSM) |
|---|---|---|---|
| Typical offset | 0.2–1.0 ft | 1.5–2.2 ft | 1.5–2.2 ft |
| Worst offset | ~3.3 ft (GCP-104) | ~2.8 ft | ~2.8 ft |
| Duplicated targets | None | CHK-109 | CHK-109 |
| False power line breaks | None | CHK-109 | CHK-109 |
| DSM vs no-DSM difference | N/A | None observed | None observed |

**The original ODM ortho is consistently better**, often by 2x.
**No measurable difference** between DSM and no-DSM true ortho variants.

## Runtimes (r5.4xlarge on-demand, 16 vCPU, 124 GB RAM)

| Variant | Workers | Duration | Cost (@$1.008/hr) |
|---|---|---|---|
| ODM standard ortho | N/A (ODM internal) | ~67 min | ~$1.13 |
| True ortho (no DSM) | 8 | ~7.3 hr | ~$7.36 |
| True ortho (with DSM) | 8 | ~8.2 hr | ~$8.27 |

True ortho is ~7x slower AND produces worse results.

## Root Cause Analysis

1. **DTM vs mesh projection:** ODM projects through a 3D mesh that captures actual
   surface geometry at sub-pixel resolution.  Our true ortho projects through the DTM,
   a gridded approximation.  Every pixel has lateral error from DTM interpolation.

2. **Naive camera selection:** ODM uses texrecon for camera selection, considering view
   angle, image quality, color consistency, and seam optimization.  Our "most nadir wins"
   selection picks cameras without considering projection quality at each pixel.

3. **Camera model mismatch:** The full run used undistorted images with pinhole projection.
   The relationship between undistorted pixel coordinates and the reconstruction's camera
   poses may have subtle inconsistencies.  The proof-of-concept crop tests used original
   (distorted) images with the brown model and showed improvement — suggesting the
   undistorted path introduced errors.

4. **Mosaic seam artifacts:** The duplicate target at CHK-109 shows adjacent tiles picking
   different cameras that project the same target to different ortho positions.  ODM's
   texrecon avoids this with global seam optimization.

## What the Proof of Concept Got Right (and Wrong)

The small crop tests (400×400 pixels) showed genuine improvement because:
- Used original images with the correct brown distortion model
- Small crops have minimal camera switching (fewer seam artifacts)
- Compared against a specific bad case (GCP-102 near power lines)

Scaling to full ortho (37K tiles, 14 workers) exposed the real problems.

## Recommendation

Shelve the true ortho approach (geo-ksd → backlog).  If revisited:
- Use ODM's mesh instead of DTM for projection
- Use ODM's texturing algorithm for camera selection, or integrate into ODM's pipeline
- The visibility-aware occlusion idea (geo-51u) is still valid in principle but needs
  much better projection accuracy before the occlusion check adds value
- Consider contributing to ODM's texturing stage upstream rather than reimplementing

## Artifacts

| File | Location | Size |
|---|---|---|
| Original ODM ortho | `s3://stratus-jrstear/bsn/aztec7/odm_orthophoto/odm_orthophoto.original.tif` | 1.2 GB |
| True ortho (no DSM) | `s3://stratus-jrstear/bsn/aztec7/odm_orthophoto/true_orthophoto_nodm_cog.tif` | 2.2 GB |
| True ortho (with DSM) | `s3://stratus-jrstear/bsn/aztec7/odm_orthophoto/true_orthophoto_dsm_cog.tif` | 2.2 GB |
| Comparison spreadsheet | User's local analysis | N/A |
| rmse reports (3 variants) | `~/stratus/aztec7/rmse_report_*.html` | ~31 MB each |
| Session transcript | `~/.claude/projects/-Users-jrstear-git-geo/55893a1a-*.jsonl` | N/A |
