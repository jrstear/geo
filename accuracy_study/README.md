# accuracy_study/

Experimental scripts and tooling supporting the accuracy investigation around
the survey-quality ODM workflow. These are research tools, not production
pipeline stages — they generate variants, run ablations, and produce overlays
that feed back into design decisions in `sight.py`, `rmse.py`, and the EC2
pipeline.

For pipeline context see [`docs/odm-workflow.md`](../docs/odm-workflow.md).
For the broader experiment framework see
[`docs/plans/experiment-framework-spec.md`](../docs/plans/experiment-framework-spec.md).

## Scripts

### `compare_refinement.py`
Compare one or more refinement algorithm outputs against a human-confirmed
ground truth `gcp_confirmed.txt`. Reports per-row pixel deltas and summary
statistics. Used during pixel-refinement tuning (bead **geo-mfs**).

```bash
python compare_refinement.py gcp_confirmed.txt gcp_list-baseline.txt gcp_list-variant.txt
```

### `chk_image_study.py`
Mini-study: vary image count per CHK label (and optionally exclude outliers)
to characterize how CHK RMSE responds to image count and lens-edge distortion.
Generates variants of `chk_list.txt` and runs `rmse.py` on each, printing a
comparison table. Result from aztec7 confirmed top-7 images per CHK beats
all-images for RMS_H (lens-edge distortion is a real factor).

### `experiment_gen.py`
GCP file variant generator for ablation experiments. Reads a master tag file
plus an experiment config JSON (or CLI flags) and writes a trimmed
`gcp_experiment.txt` containing only the selected control labels and the
requested number of images per label. Feeds the experiment driver
(bead **geo-dk5**) and the experiment framework epic (**geo-4fu**).

### `ortho_uncertainty.py`
Generate a single-band GeoTIFF showing per-pixel estimated horizontal
positional uncertainty across an orthophoto. Formula:

```
sigma_h(x,y) = sigma_DTM * tan(theta_camera(x,y)) + sigma_reconstruction
```

where `theta_camera(x,y)` is the off-nadir angle of the most-nadir camera
viewing each pixel. The output TIF can be embedded in the rmse.py HTML report
via `--uncertainty` (see odm-workflow §6). Supports an optional DTM for
elevation-aware processing.

### `test_projection.py`
Validates `sight.py`'s `project_pixel_mode_a()` (EXIF-based pinhole) and
`project_pixel_mode_b()` (full 3D rotation, oblique-capable) against
GCPEditorPro-confirmed pixel observations. Used to characterize per-pixel
projection accuracy for the validation note in `sight.py`. Bead reference:
**geo-2j8.4** (Validate Mode B projection accuracy).

### `analyze_color.py`
LAB color-space analysis of detected vs ground-truth markers. Investigates
whether color consensus across multiple confirmed observations of the same
target can be used as a post-pass filter on coloredX refinements (the **R4**
filter referenced in [`docs/details/gcp_object_identification.md`](../docs/details/gcp_object_identification.md)).

## Open questions (Claude Opus, while drafting this README)

These are the parts of the study plan that weren't obvious to me from the
scripts alone. They're not blockers — just things whose answers would help a
new reader (or future-me) reconstruct the intended plan:

1. **Which study is currently the priority?** Several beads are P0/P1
   (geo-7x1 ortho accuracy, geo-h32x ortho-level rmse mode, geo-4fu experiment
   framework epic). Is there a defined order, or are they parallel tracks?

2. **What is "master_tags.txt"?** `experiment_gen.py` reads a master tag file
   in pipeline priority order. How is that file produced — is it just a
   manually-curated `gcp_list.txt` after a clean tagging session, or is there
   tooling that builds it?

3. **R4 filter status.** `analyze_color.py` produces stdout-only analysis.
   Has the R4 filter been implemented in `coloredX.py` based on those results,
   or is the analysis still informing the decision?

4. **uncertainty overlay validation.** The formula in `ortho_uncertainty.py`
   uses a fixed `sigma_DTM` (default 0.1 m) and `sigma_reconstruction`
   (default 0.035 m). Have those defaults been calibrated against measured
   ortho residuals on a test site, or are they nominal estimates?

5. **`true_ortho.py` (now in `experimental/`).** The full-site comparison on
   aztec7 was disappointing (see `experimental/README.md`). Does the accuracy
   study still want a "true ortho" track, or has that question been answered
   for the foreseeable future?
