# experimental/

Code that has been validated for correctness but is not currently part of the
production pipeline. Kept here for reference and possible revival.

## true_ortho.py

A true-orthophoto post-processor for ODM orthophotos. For each output pixel it
re-projects the ground point through the most-nadir camera (with optional DSM
occlusion ray-march), instead of using ODM's standard DTM-orthophoto path.

**Status:** Shelved on 2026-04-01 (bead **geo-ksd**, P4/backlog). Full
side-by-side analysis with annotated examples lives in
[`true-ortho-comparison.md`](true-ortho-comparison.md).

**Why shelved.** A small-crop proof of concept (geo-51u) showed clear visible
improvement at survey targets, so the post-processor was integrated into the
EC2 pipeline as a final stage and validated end-to-end on aztec7. Full-site
results were disappointing:

- ~2 ft horizontal offset at targets vs ~1 ft for ODM's standard ortho
- ~7× slower than ODM's ortho stage
- Mosaic artifacts: duplicate targets, false breaks in linear features
  (power lines), seam visibility
- No measurable improvement from adding DSM occlusion ray-march vs no-DSM

**Suspected root causes:**

1. **DTM projection vs mesh projection.** This implementation projects each
   ortho pixel onto the bare-earth DTM, while ODM's pipeline projects onto the
   reconstructed mesh. The mesh captures local surface geometry (curbs, target
   markers, vegetation tops) that the DTM smooths away.
2. **Naive camera selection vs texrecon.** This implementation picks the
   most-nadir camera per pixel. ODM uses texrecon, which does view-consistent
   selection with seam optimization across neighbouring pixels.
3. **Possible camera-model mismatch with ODM's undistorted images.** ODM emits
   `undistorted/` images that have already had Brown distortion removed, but
   this code re-applies a Brown model. The combined effect may be non-trivial.

**If revisited:** the right architecture is probably to use ODM's reconstructed
mesh as the projection surface and adopt a view-consistent camera-selection
strategy. At that point it may be more productive to contribute to ODM upstream
rather than maintain a separate post-processor (see bead **geo-005**).

The 4-stage `__main__` and `--compare` flag (geo-x5e) are still useful as a
quick A/B harness if anyone wants to revisit. EC2 integration was removed in
commit `194f91a`; the standalone script is self-contained.
