# Tag-Quality Consistency: Iterative Sight + Trail + Bad-Tag Detector

**Epic bead:** geo-wst0

This document describes a coordinated set of changes to improve tag-quality QA
*before* paying for an ODM run on EC2. It introduces (1) iterative refinement
inside `sight.py` that uses color-found pixels to triangulate a better 3D and
re-project to projection-only images, (2) an internal-consistency report that
flags suspect targets pre-tagging, (3) an optional multi-line trail propagated
through the pipeline so the deliverable can confirm the iterative idea actually
helps, and (4) a pre-ODM bad-tag detector that compares user tags to anchor
estimates.

Two validation gates are built into the sequence so that effort can stop early
if the underlying iterative idea doesn't pan out empirically.

---

## Problem

Today's pipeline catches mistagging *after* ODM, via `rmse.py` against the
GCP-constrained reconstruction. That worked for aztec7 — `rmse.py` flagged
CHK-14 (mis-tagged base station, dH 490× median) and CHK-18 — but only after
the full EC2 ODM run had completed. The cost: time, money, and a re-run after
fixing the tags.

Two complementary opportunities to catch problems earlier:

1. **Inside sight.py** — `sight.py`'s color refinement gives sub-pixel pixel
   estimates for some (target, image) pairs but falls back to raw EXIF
   projection (~30–150 px error) for others. When color works on multiple
   images of the same target, those refined pixels back-project as 3D rays
   that converge on the target's true 3D position. That convergence (or
   failure to converge) carries information sight.py is currently throwing
   away.

2. **After tagging, before ODM** — comparing user-tagged pixel positions
   against sight.py's estimates per target reveals catastrophic mistags
   (entire-target wrong feature) in cases where the user did not use sight.py's
   estimate as a starting point.

Both signals are independent of `reconstruction.topocentric.json` and run
locally — no EC2 spend required to use them.

---

## Empirical findings on aztec7

A direct numeric study using `aztec7/aztec_tagged-with14and18.txt` (the
pre-correction tagging where CHK-14 and CHK-18 were known wrong) and
`tmp/F100340_AZTEC.txt` (the matching sight.py estimates) showed:

### CHK-14 (10 bad tags, surveyed at 238452.081, 4085679.434, 1767.173)

Tags split bimodally against estimates:

- 4 tags at offset ≈ 0–2 px (user accepted sight's color-refined estimate
  exactly — but sight's color refinement had locked onto the wrong feature
  on those images, so user and sight were both wrong, agreeing)
- 6 tags at 200–530 px offset

### CHK-18 (16 bad tags)

Same pattern: 5 tags at offset ≈ 0, 11 tags at ~190 px offset in a consistent
direction. The user systematically tagged the base station instead of the
target.

### Color vs. projection estimate quality

For the 31 mixed-source targets (≥3 color hits + ≥1 projection-only) compared
against the *corrected* `aztec_tagged.txt` as truth:

| source | n | resid median | resid p90 | resid max |
|---|---|---|---|---|
| color (when it locks correctly) | 451 | 1.4 px | 180 px | 397 px |
| projection (raw EXIF) | 171 | 91 px | 179 px | 239 px |

Color estimates are ~65× better than projection estimates by median error —
when color works. Color failures are bimodal: typically ≈0 px or 100+ px, no
middle ground. This is the room for improvement: when color worked on most
images of a target, projection-only images on that same target are 50–200 px
off purely from EXIF pose noise, and iterative reprojection can recover that.

### Single-statistic discrimination

A naive per-target threshold on median residual or median offset does **not**
cleanly separate known-bad from known-good targets — several known-good
targets have similar individual stats due to sight.py estimate noise on lone
images. The discrimination is strong *within* a target (anchor cluster vs.
outliers) but weak *between* targets on a single statistic. This is why the
detector ranks a review queue rather than auto-rejecting.

---

## Empirical validation (2026-04-25, geo-nu8f)

After implementing the iterative refinement (geo-z7xw) and consistency
report (geo-9fzr), an end-to-end validation was run on aztec12 images
(1385 images, 49 targets) using `aztec11_tagged.txt` as ground truth.
Result: **mixed**. The implementation works as designed but the lift on
this dataset is smaller than the planning hypothesis assumed.

### Per-source residual to tagged truth (lenient gates 120 px / 15 m)

| source     | n   | median | use case |
|------------|-----|--------|----------|
| `color`      | 490 | **1.6 px** | gold standard when refinement succeeds |
| `tri_color`  | 42  | 114 px     | mixed — sub-pixel for ~4 targets, 100–300 px for others |
| `tri_proj`   | 4   | 172 px     | essentially same as `projection` |
| `projection` | 139 | 177 px     | baseline — raw EXIF projection of surveyed_xyz |

### Per-target outcomes

- **Clean wins** (CHK-101, CHK-103, CHK-119, GCP-126): tri_color lands
  sub-pixel against tagged truth — exactly the design goal.
- **Modest wins** (CHK-114): tri_proj 121 px vs projection 167 px — real
  ~30 % reduction in the seed-to-truth gap, but small in absolute terms.
- **Failures** (CHK-130): tri_proj 216 px vs projection 142 px — *worse*
  than the baseline because one outlier color hit poisoned the
  triangulation. The consistency gate did not catch it (max residual was
  72 px after one outlier drop, comfortably below the 120 px threshold).

### CHK-18 result

Did **not** fire `SURVEY_DISAGREES`. Color refinement on aztec12 found the
actual marker; survey disagreement was 8 m (normal EXIF noise).
The original CHK-18 problem was a **user tagging error**, not a sight.py
color-refinement error — the user clicked the base station instead of the
target. That class of error is what the post-tagging bad-tag detector
(geo-v4tu) is designed for, not the sight-time iterative refinement.

### Why the lift is smaller than predicted

Per-camera EXIF GPS+IMU noise is the dominant contributor to projection
error in this dataset (~50–100 px per camera, independent across images).
Triangulation absorbs surveyed_xyz error (which is small for an RTK
survey) but cannot reduce per-camera EXIF noise. Re-projecting a
triangulated 3D through camera B still inherits camera B's pose error.
The planning hypothesis assumed surveyed_xyz error and pose error were
comparable; on aztec12 (RTK-surveyed, consumer drone EXIF), pose error
dominates by an order of magnitude.

### Why strict gates also fail

Setting `--iter-eps-resid-px 30` rejected **all 46 targets** as suspect.
The EXIF noise floor (~50–100 px per camera) makes <30 px residuals
unattainable on real data. Loose gates admit too many bad targets;
strict gates reject every target. Single-statistic gating cannot
distinguish "honest EXIF noise" from "outlier-driven misfit" because
both produce residuals in the same range.

### Decision

- Iterative refinement (geo-z7xw) and the consistency report (geo-9fzr)
  ship as `--iterative` / `--consistency-report` opt-in flags.
  **Default behavior is unchanged** (no flag = no iterative pass).
- The pre-ODM bad-tag detector (geo-v4tu) is the higher-leverage path
  for catching user-tagging errors that motivate the epic. Promote it
  to next priority.
- The trail-sidecar work (geo-l941, geo-m8h9, geo-gr1i) is contingent
  on iterative becoming reliably useful — defer until the algorithm
  has a stronger outlier-robustness scheme. Possible improvements
  worth a future bead:
    - RANSAC-style aggressive outlier dropping until residual converges
    - Cross-image color-signature consistency check before accepting a
      color hit as a triangulation ray
    - Use of SfM-refined poses (when a reconstruction is available) for
      sharper triangulation
- The triangulation-math consolidation with rmse.py (geo-i733) is also
  deferred — there's no urgency until iterative is broadly useful.

---

## Confidence labels (four)

Sight.py output column 8 today contains `projection` or `color`. After
iterative refinement, four labels:

| label | how derived | accuracy | use as anchor? |
|---|---|---|---|
| `color`     | surveyed seed → color refinement succeeded | sub-pixel | yes |
| `tri_color` | triangulated seed → color refinement succeeded | sub-pixel | yes |
| `tri_proj`  | triangulated 3D re-projected through EXIF, no color refinement | ~5–20 px | no, but better baseline than projection |
| `projection`| surveyed seed, no color refinement, no triangulation possible | 30–150+ px | no |

Quality ordering: `{color, tri_color}` > `tri_proj` > `projection`.

Anchors (used by the bad-tag detector to define consensus) are `color` +
`tri_color` only. Sub-pixel accuracy is the membership criterion, not the
specific path that got there.

---

## Pass structure inside sight.py

```
Pass 0 (existing):
  for each (target, image):
    project surveyed_xyz through EXIF camera
    optional color refinement around projection
    label: color (refinement succeeded) or projection (it didn't)

Pass 1 (new): triangulate from color rays
  for each target with >= 3 'color' hits:
    rays_i = back-project (px_color_i, py_color_i) through EXIF camera_i
    target_3D, residual = least-squares ray intersection
    if residual high: drop worst-fitting ray, retriangulate (max 1-2 drops)
    if residual < eps_tri AND |target_3D - surveyed_xyz| < eps_surv:
      for each projection-only image j:
        re-project target_3D through EXIF camera_j → new pixel
        label: tri_proj
    else:
      flag target as suspect (color and survey disagree)

Pass 2 (new): iterative re-refinement
  for each tri_proj pixel:
    re-run color refinement with this pixel as seed (instead of surveyed_xyz projection)
    if color marker found: relabel tri_color
    else: stays tri_proj

Pass 3 (optional): re-triangulate including tri_color rays
  if new residual >> Pass 1 residual: demote bad tri_color rays back to tri_proj
```

### Sanity gates (Pass 1)

- `eps_tri` (triangulation residual): how far apart the rays are in 3D. Color
  refinement is sub-pixel so honest convergence should be a few feet at most.
  Suggested starting value: 2 ft. To tune empirically.
- `eps_surv` (survey disagreement): how far the triangulated 3D is from the
  surveyed coordinate. RTK survey is typically <1 ft; if rays converge to a
  point >5 ft from the survey, either color found a wrong feature or the
  survey coordinate is wrong. Suggested starting value: 5 ft.

When the sanity gate fails, do not auto-reproject — emit a flag in the
consistency report. Most cases of CHK-18 / CHK-127 / CHK-105-style failures
should land here.

---

## Estimate trail sidecar (`--emit-trail`)

Orthogonal flag from `--iterative`. When set, sight.py writes a separate
`{job}_trail.txt` sidecar file recording the full chain for tuples where
iterative actually fired. The main `{job}.txt` is unchanged — it still
contains exactly one row per (image, label) tuple, just with new column-8
label values (`tri_proj`, `tri_color`) in addition to the existing
`projection`/`color`.

The sidecar uses the **same row format as `{job}.txt`** (tab-separated, EPSG
header, eight columns). Any tool that parses `{job}.txt` parses the sidecar.
One row per (image, label, source) for trail-worthy tuples; tuples where
Pass 0 `color` succeeded need not appear (the main file already has the
best — and only — estimate). Within a tuple's rows in the sidecar, ordering
is best-first: `tri_color` > `tri_proj` > `projection`.

| Pass 0 result | Sidecar rows for this tuple |
|---|---|
| color found                              | (none — main file has the only useful estimate) |
| Pass 1 + Pass 2 promoted to tri_color    | `tri_color`, `tri_proj`, `projection` |
| Pass 1 succeeded, Pass 2 didn't promote  | `tri_proj`, `projection` |
| Pass 1 unable (no triangulation)         | (none — main file has the only estimate, label `projection`) |

The main `{job}.txt` always contains the best row per tuple. Diagnostic
tools that want the trail read both files and join by (image, label).

### Why a sidecar instead of multi-line in the main file

The original plan was to allow multiple rows per (image, label) in the main
`{job}.txt` and `{job}_tagged.txt` files, with a "first line wins" rule for
tools that didn't understand the new format. That model carried real
diagnostic value (per-tuple trail visible end-to-end) but pushed format
changes into every tool that touched the file:

- `transform.py split` would need dedupe-by-tuple logic.
- `rmse.py` would need to consume multi-line input.
- **GCPEditorPro** would need a data-model migration from
  `Map<(image,label), pixel>` to `Map<(image,label), List<pixel>>`,
  redefined save/load semantics, round-trip tests, and UI changes.

That last item was the most expensive and highest-risk change in the epic.

The sidecar approach captures the same diagnostic value at a fraction of
the cost:

- `transform.py split` reads `{job}_tagged.txt` exactly as today.
- `rmse.py` optionally reads `{job}_trail.txt` for the per-target delta
  table; everything else is unchanged.
- GCPEditorPro never sees the sidecar. No data-model migration.

What's lost: untag-reversibility in GCPEditorPro (a hypothetical property
of the multi-line model that doesn't exist today anyway, so we just don't
gain it), and single-canonical-file semantics (but `transform.yaml`
already establishes a sidecar pattern in this codebase).

Migrating sidecar contents into the main file later is straightforward
if it ever becomes desirable — the format is already identical. Starting
sidecar-only is a cheap reversible decision; starting multi-line-in-main
is an expensive irreversible one.

---

## Internal consistency report (sight.py)

Side-output of Pass 1 and (optionally) Pass 3. Per-target metrics, printed at
end of run and written to a small report file:

```
Target          n_color  tri_resid  surv_disagree  promoted  flag
-------         -------  ---------  -------------  --------  ----
GCP-104              12       0.3 ft       0.4 ft         5  ok
CHK-105               5     188.0 px       2.1 ft         0  COLOR_INCONSISTENT
CHK-18               11       0.2 ft       89.0 ft        4  SURVEY_DISAGREES
CHK-127               5     183.6 px       3.0 ft         0  COLOR_INCONSISTENT
```

CHK-18 should fire `SURVEY_DISAGREES` from this analysis pre-tagging — color
rays converge tightly (residual low), but to a 3D point ~90 ft from the
recorded survey coordinate. That is the same signal `rmse.py` discovered
post-ODM, computed from sight.py's own data without needing ODM at all.

---

## Pre-ODM bad-tag detector

Runs after tagging, before ODM. Reads `{job}_tagged.txt` plus the original
sight.py output. Per target:

1. **Anchors** = tags whose pixel matches the `color` or `tri_color` estimate
   within ~10 px. These are images where the user accepted sight's high-
   confidence pixel.
2. **Anchor consensus** = mean offset of anchor tags from estimates.
3. **Per-tag residual** = distance from each tag's offset to anchor consensus.
4. **Suspect tag** = residual > 50 px.
5. **Suspect target** = composite of low anchor fraction, high suspect-tag
   count, high max residual, and bimodality of offset distribution.

Output: ranked HTML report. Per suspect target, annotated crops showing the
estimated position vs the tagged position for each suspect tag. User clicks
through to GCPEditorPro to fix.

**Why depend on iterative refinement first**: with `tri_color` available as a
second class of anchor, more targets have ≥3 anchors and the consensus is
stronger. Targets that are pure-projection (no color, no tri_color) get a
weaker version of the analysis but it still runs.

This is distinct from `rmse.py` (post-ODM, against reconstruction) and
`geo-aw0` (also post-ODM, triangulation spread on reconstruction). All three
catch some overlapping problems; each is cheaper-and-earlier than the next.

---

## Three-stage QA chain

| stage | runs when | anchored on | EC2 cost to find issue | catches |
|---|---|---|---|---|
| sight.py internal consistency | pre-tagging | color rays + surveyed_xyz | $0 | bad surveys, color refinement misfires |
| pre-ODM bad-tag detector | post-tagging, pre-ODM | user tags vs anchor estimates | $0 | individual mistags, wrong-feature mistakes |
| `rmse.py` | post-ODM | bundle-adjusted reconstruction | full ODM run | residual ground-truth errors, ortho-level errors |

Each stage's job is to reduce the work for the next. None replace each other.

---

## Implementation sequence

Beads in original execution order, with current status reflecting the
2026-04-25 validation findings:

1. **geo-z7xw** — sight.py iterative refinement (Pass 1 + Pass 2) — **DONE**, opt-in via `--iterative`
2. **geo-9fzr** — sight.py internal consistency report — **DONE** (bundled with #1), opt-in via `--consistency-report`
3. **geo-nu8f** — MVP gate — **DONE**, mixed result documented above; recommendation is to pivot to v4tu
4. **geo-l941** — sight.py `--emit-trail` sidecar output — **DEFERRED**, contingent on iterative becoming broadly useful
5. **geo-m8h9** — `rmse.py` reads sidecar for per-target delta table — **DEFERRED**, same contingency
6. **geo-gr1i** — Stage-2 gate: validate sidecar value — **DEFERRED**, same contingency
7. **geo-v4tu** — Pre-ODM bad-tag detector — **NEXT**, promoted to active priority

Two beads from the original plan were closed earlier as superseded by the
sidecar approach: `geo-ef0i` (`transform.py split` multi-line awareness — no
longer needed since the main file format is unchanged) and `geo-mlev`
(GCPEditorPro multi-line data model — no longer needed since GCPEditorPro
never sees the sidecar).

One follow-on bead exists for triangulation-math consolidation with
rmse.py: **geo-i733** (P3, deferred until iterative is broadly useful).

Updated dependency graph (deferred beads dimmed):

```
geo-z7xw ─┐                                                   (deferred)
          ├─→ geo-nu8f ─→ ⟦geo-l941 ─→ geo-m8h9 ─→ geo-gr1i⟧
geo-9fzr ─┘                  │
                             └────→ geo-v4tu  ← NEXT
```

### MVP gate (geo-nu8f) acceptance criteria

- Projection-residual median for projection-only images on aztec7 / redrocks
  drops materially after iterative (target: from ~90 px today to <20 px)
- `tri_color` promotion rate is non-trivial (target: ≥30% of would-be-
  projection images get promoted)
- CHK-18 fires `SURVEY_DISAGREES` flag pre-tagging on aztec7
- False-positive rate on known-good targets is acceptable (manual review)

If these do not hold, stop and pivot. Multi-line trail and GCPEditorPro work
are gated on this passing.

### Stage-2 gate (geo-gr1i) acceptance criteria

- For known-suspect targets in aztec7, the sidecar-driven trail in `rmse.py`
  output distinguishes "user clicked near `tri_color`" (sight's fault) from
  "user clicked far from all estimates" (user error)
- Across many tagged points, `tagged - tri_color` delta is consistently
  smaller than `tagged - projection` delta (i.e., iterative refinement is
  measurably helpful at deliverable level)
- Sidecar file size and storage cost are acceptable in practice

If these do not hold, ship what's built; the sidecar simply isn't enabled by
default. The pre-ODM bad-tag detector (`geo-v4tu`) and main-pipeline behavior
are unaffected since they don't depend on the sidecar's diagnostic value.

---

## Open questions

- **Threshold tuning** — `eps_tri`, `eps_surv`, anchor-residual cutoff, and
  bad-tag detector composite weights all need empirical tuning across
  multiple jobs (aztec7, redrocks, future). Each gate is a chance to adjust.
- **No-color-at-all targets** — a target where Pass 0 found zero color hits
  cannot triangulate. Pass 1 is skipped; everything stays `projection`. The
  bad-tag detector falls back to per-target median consensus on these. Lower
  discrimination power, but unchanged from today.
- **Iterative convergence** — Pass 3 (re-triangulate including tri_color
  rays) is opt-in. Whether to default it on depends on whether the residual
  almost always drops or sometimes oscillates. Validate empirically.
- **Sidecar contract** — `{job}_trail.txt` is generated by sight.py and not
  edited downstream. Tools join with the main file by (image, label) keys.
  If a user manually edits the main file in a way that breaks key alignment,
  the join silently produces incomplete trails. Mitigation: documented
  contract; sidecar regeneration is a single sight.py rerun away.
- **Backwards compatibility** — runs without `--emit-trail` produce no
  sidecar, and `rmse.py` / the bad-tag detector simply skip trail
  diagnostics in that case. Pre-trail jobs continue to work unchanged.
- **Future merge of sidecar into main file** — if the sidecar approach
  proves valuable enough to be worth the integration cost, the same trail
  data could later be merged into a multi-line `{job}.txt` and the sidecar
  retired. This document keeps that door open but does not commit to it.
