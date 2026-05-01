# Pix4D Process Support: geo tool ports + odium two-path

**Epic bead:** geo-qx4b
**Spawned from:** 2026-04-29 prep for Jon ↔ Isaiah meeting on `~/stratus/redrocks/elevation/TIN_GCP_EVALUATION_RESPONSE.md`
**Status:** Living planning doc — revised post-meeting.

**Update 2026-05-01:** The Thread 1 P2 tools-layer set is shipped end-to-end.
`geo-6gni` (Pix4D `quality_report.pdf` parser + rmse.py integration),
`geo-hbxf` (rmse.py LandXML TIN axis), and `geo-y3j3` (check_tags
toolchain-agnostic input + simplified gate) all closed. `geo-fcio` reframed
and shipped via `transform.py split` emitting `{job}_6529_tagged_marks.csv`.
The unified rmse.html report now renders three independent accuracy axes
side-by-side under an `ODM | Pix4D × Reconstruction/Orthophoto/TIN`
hierarchy — cross-validation on aztec13 already surfaced new findings
worth investigating (`geo-0h61`, `geo-xzds`). The Thread 2 odium agent
layer (geo-frmv / geo-seoj / geo-wsu1 / geo-vmdn) remains open. See bead
status updates below.

This document captures the threads worth supporting in geo and odium so that
Isaiah's Pix4D process benefits from the same catch-errors-early discipline
that ODM jobs have today, without forcing him to switch toolchains. Most items
are already itemised in the Tier 0–4 sections of `TIN_GCP_EVALUATION_RESPONSE.md`;
this doc translates them into bead-tracked work and groups them around the
three meeting threads.

---

## Threads to discuss with Isaiah

### Thread 1 — Which geo tools port to Pix4D today

The honest separation between "ports cleanly" and "needs work":

| Tool | Pix4D fit | Bead | Status |
|---|---|---|---|
| `transform.py dc` → `transform.yaml` | **Ports today, toolchain-agnostic.** Run on every job's `.dc` regardless of Pix4D vs ODM. Mar-6 email chaos becomes structured data. | (already shipped) | ✅ shipped |
| Pix4D `quality_report.pdf` parser → post-bundle accuracy + outlier checks | Pix4Dmatic v2.0.x does not emit a JSON QR — pdfplumber over the PDF tables is the practical path. Post-bundle checks: `verified_marks_count` vs `marks_count` mismatch (solver rejected operator marks), per-GCP XYZ residuals, per-CHK RMS_H/RMS_Z, severity-classified quality checks (matches/dataset/camera_opt/gcps/checkpoints/atps). Standalone `pix4d_qr.py` + `rmse.py --pix4d-qr` flag wires it into the unified report. | geo-6gni | ✅ **closed 2026-05-01** (b828af0, 89cb0e7) |
| Pre-run marks-count hard-stop (Pix4D marks file OR `_tagged.txt`) | Catches bigMem (50 GCPs, 0 marked) **before** the bundle runs, at $0 cost. Bundled with check_tags' toolchain-agnostic input refactor — same parser, both toolchains. Strictly cheaper than QR for this class of failure. | geo-y3j3 | ✅ **closed 2026-04-30** (282cabd) |
| `rmse.py` extended for TIN-sampled Z | New `tin.py` standalone module (LandXML 1.2 parsing, barycentric sampling, `<Application>`-based tool detection). `rmse.py --tin PATH [--tin-source pix4d|odm]` adds `tin_dZ` column under producing tool's TIN sub-section. | geo-hbxf | ✅ **closed 2026-05-01** (7b35c91) |
| `check_tags` toolchain-agnostic input + simplify gate | Three input modes (sight tagged.txt, Pix4D `history.p4mpl`, Pix4D marks CSV). 3-tier rule (red/amber/green at <3, 3-6, ≥7) replaces the heuristic 0.7 gate. Fairness clamp on visible-image count when sight estimates available. Auto-locates sibling sight estimates from filename pattern. Plus `--vs OTHER` for tagger-vs-tagger comparison. | geo-y3j3 | ✅ **closed** (see above; same bead) |
| Surveyor-handoff CSV validator | **Deprioritised.** Stratus plans to require `.dc` files going forward — the `.dc` carries CRS/datum/geoid metadata explicitly, so the CSV validator becomes a fallback rather than a primary gate. | geo-vtc0 | open, P3 |
| LandXML metadata-vs-data validator | **Deprioritised.** If `.dc` is required and CRS is locked at survey-handoff time, the declared-vs-actual mismatch becomes much harder to produce. Residual risk is Pix4D project-CRS misconfig (the original `Red Rocks TIN.xml` mode), but it's no longer the highest-leverage gate. | geo-75sw | open, P3 |
| `sight.py` → Pix4D marks CSV exporter | **Reframed.** Isaiah: Pix4D doesn't distinguish external-process estimates from human-tagged marks — exactly the gap the geo *confidence* attribute (`color`/`tri_color`/`tri_proj`/`projection`/`tagged`) was created to fill. Reframe: tag in GCPEditorPro (where confidence is meaningful), then export human-confirmed marks to Pix4D's CSV format. `transform.py split` now emits `{job}_6529_tagged_marks.csv` from a tagged file, paired with sight's `{job}_6529_color_marks.csv` (color-only pre-marks) and `{job}_6529_targets.csv` (GCP coords with monument dedup). See `docs/plans/sight-to-pix4d-import.md` for the joint workflow doc. | geo-fcio | ✅ **closed 2026-04-30** (fab751e); **geo-qkip** ✅ closed (62bf575) for the targets-CSV sibling |

**Question for Isaiah after this discussion:** of the P2 set (QR parser,
rmse-vs-TIN, agnostic check_tags), which would actually save the most pain
on the next job? That's your "ship first" candidate.

> **Resolved 2026-05-01:** ship-first question is moot — all three P2 items
> shipped. The unified `rmse.html` (with `--pix4d-qr` and `--tin`) renders
> three independent accuracy axes per target side-by-side, and on the first
> dataset run (aztec13) already surfaced a 3× CHK RMS_Z disagreement between
> ODM and Pix4D worth investigating (`geo-0h61`) plus a NAD83 datum-vintage
> offset visible in `mean_dZ` (`geo-xzds`). Next-job benefit is now baked in
> rather than a choice.

### Thread 2 — odium with two state-paths

Most novel angle. The TIN response doc focuses on tools; odium is an *agent*
that can drive the ones that aren't GUI-bound.

#### Concrete validation: Isaiah's Taos workflow (2026-04-29)

Captured during the meeting. Surveyor delivered a `.dc` with **no embedded
CRS** and on a **localized frame** (off by thousands of feet from any obvious
state-plane fit). What Isaiah did manually, mapped to odium tools:

| Step | What he did | Odium tool |
|---|---|---|
| 1. Spotted missing CRS | Manual inspection | handoff validator detects missing CRS field (extends geo-vtc0) |
| 2. Guessed NM 2258 from job address | Domain knowledge | `epsg_candidates_from_latlon` |
| 3. Visualised in QGIS, saw kft-scale offset | Manual | `dc_to_geo_preview` — project under each candidate CRS, emit GPKG |
| 4. Recognised "local frame" not "wrong CRS" | Domain reasoning | odium pattern: systematic offset/rotation = localised, not wrong CRS |
| 5. Field-shot 5-7 flagged stakes with EMLID | Irreducible field work | (no automation — but odium tells him *what* to shoot) |
| 6. **"Asked my AI to pattern-match" → rough transform** | Ad-hoc AI assist | **`recover_transform_from_correspondences`** — already prototyped as `compare_and_helmert.py` in the dec-4 eval |
| 7. Used rough transform to locate actual control points | Manual lookup | `apply_transform_and_match` |
| 8. Shot actuals, built EMLID localisation, ran job | Standard flow | hands off to ODM or Pix4D fork |

Steps 1, 2, 3, 4, 6, 7 are all first-class odium work. Bead **geo-frmv** lifts
the Taos workflow from ad-hoc into the agent. The pattern-match transform
recovery (step 6) re-uses `compare_and_helmert.py` from the dec-4 evaluation
as the algorithmic core. **First concrete demonstration that the odium
"common upstream" stage delivers value neither Pix4D nor ODM offers natively.**

#### Concrete validation 2: substitute-control-point recommender

Isaiah, follow-up: the *actually interesting* agent capability isn't the CRS
recovery — that's tractable from common sense + spatial inspection — it's the
**field-time substitute-point recommendation**. Scenario:

> CRS is fine. Localization is solved. Two control points have been shot in.
> Operator goes to where the third should be — *it's not there.* Disturbed,
> never set, gone. Operator wants the agent to look at the `.dc`, find another
> previously-measured point that is (a) likely to be findable on the ground
> and (b) good geometry for improving the localization fit, and surface
> "go find THIS point instead."

Tracked as **geo-seoj** (P2, sibling of geo-frmv under the epic). Algorithm:

- **Findability prior** — regex/keyword on `.dc` per-point descriptions.
  Durable markers (`rebar`, `iron pin`, `brass cap`, `monument`) score high;
  ephemeral (`paint`, `lath`, `flag`) score low. Penalise neighbourhoods
  where multiple points have already failed to be found.
- **Localization leverage** — D-optimal experimental design. For each
  candidate, simulate Helmert/affine re-fit with the candidate added,
  measure trace-of-covariance reduction across the AOI. Cheap proxy:
  maximise triangle area with existing baseline + distance from centroid of
  already-tied points (avoids colinearity, rewards spread).
- **Combine** — rank = findability_prior × leverage_gain.

This is the kind of capability **no field tool offers today**. Pix4D doesn't
suggest field action. EMLID's app shows a point list, not a leverage score.
Surveyors do this with judgment + experience. odium can do it with a `.dc`
parser, a Helmert fit, and an objective function — all small.

Two field-side use cases captured (Taos CRS recovery + substitute-point
recommender) is enough to anchor "the odium common-upstream stage is real
and demonstrably valuable."


```
       customer accuracy spec  (Tier 0.2 → flight-planning advisor)
                   ↓
       surveyor .dc / CSV  (Tier 0.1 → handoff validator)
                   ↓
       transform.py dc → transform.yaml  (Tier 1.1, today)
                   ↓
       single-frame + CRS check  (Tier 1.5, handoff validator continues)
                   ↓
       flight planning advice (AGL ↔ GSD ↔ accuracy)
                   ↓
             ┌─────┴─────┐
             ↓           ↓
           ODM         Pix4D
       (sight, tag,    (Pix4D run,
       check_tags,      QR parser
       odm,             marks_count
       rmse)            hard-stop,
                        rmse-vs-TIN)
             ↓           ↓
             └─────┬─────┘
                   ↓
        packager + lineage.json  (Tier 4, toolchain-agnostic)
```

Beads:
- **geo-wsu1** — odium two-path state machine (common upstream → fork → common downstream).
- **geo-vmdn** — flight-planning advisor (AGL ↔ GSD ↔ accuracy, ASPRS-anchored).
  Genuinely new value over either toolchain. Surfaces customer-impossible
  specs at SOW time, not at delivery: e.g. "you want CHK RMS_H ≤ 0.1 ft → that's
  <0.6 GSD on a 5 cm ortho, ASPRS-impossible at 250 ft AGL with a Mavic 3E;
  drop AGL or accept 0.2 ft."
- **geo-z9u0** — `lineage.json` writer in `packager/`. Highest-impact item from
  Tier 4. Toolchain-agnostic. Would have made the dec-4 forensic chain
  unnecessary. Different scope from per-tag trail sidecar (geo-l941) — this
  is full deliverable provenance.

**Questions for Isaiah:**
- Would you actually run odium against a Pix4D job, or is your Pix4D process
  too GUI-bound for an agent to drive meaningfully?
- Is the value mostly in the **gates** (handoff validator, marks_count
  hard-stop, rmse) or in the **advice** (flight planning, accuracy spec at
  SOW time)?

### Thread 3 — `check_tags`: rescoped toolchain-agnostic + pre-run marks gate

Updated with Isaiah's input plus a Jon insight on layering. Original framing
was "the 0.7 gate is mediocre, maybe retire it." Isaiah added the
toolchain-agnostic angle: `{job}_tagged.txt` and a Pix4D marks file carry
similar information (per-image pixel observations of GCP/CHK targets), so
**check_tags can serve both toolchains** if its input parser is generalised.

Layering insight: the bigMem-class failure (zero marks on a GCP) is detectable
**pre-run** by inspecting the marks file directly — no need to wait for the
post-bundle quality report. So the `marks_count >= 3` hard-stop belongs here,
not in geo-6gni. QR-level checks stay there but their scope shrinks to
genuinely post-bundle questions (verified-marks-vs-placed-marks, residuals,
RMS).

Combined scope under **geo-y3j3** (raised to P2):

1. **Toolchain-agnostic input.** Accept either `{job}_tagged.txt` (geo native)
   or a Pix4D project history log (`history.p4mpl` — ASCII, regex-parseable
   sequence of `MarkTiePoint { ... }` blocks). Internal representation uniform
   after parsing.
2. **Pre-run 3-tier guideline matching GCPEditorPro.** Definitive thresholds
   from `GCPEditorPro/src/app/gcps-map/gcps-map.component.ts`:

   | Confirmed marks | Tier | Policy |
   |---|---|---|
   | `< 3` | red | **FAIL** (hard-stop, exit 1) — bigMem class |
   | `3 ≤ n < 7` | amber | **WARN** (pass, flagged in report) |
   | `n ≥ 7` | green | clean pass |

   Both thresholds clamped to `min(threshold, total_visible_images_for_GCP)` —
   a GCP visible in only 4 images can't reach 7 marks, so green becomes 4 for
   it. Applies fairness across small-image-count targets.

3. **Drop the magic 0.7 composite gate.** Surface the per-target consistency
   report as a reviewer aid only.

**Real Pix4D fixture available**:
`/Volumes/Stratus Files/survey/BSN/aztec highway/aztec highway matic/history.p4mpl`
— 352 `MarkTiePoint` operations across 42 GCPs, 8-15 marks each. All would
land green. Use as the canonical clean-pass test. Failure-case fixture
(zero/under-marked GCPs) needs either synthetic mutation or Isaiah's bigMem
`history.p4mpl`.

The two checks (geo-y3j3 pre-run, geo-6gni post-run) answer different
questions and aren't redundant:

- **Pre-run**: did the operator actually place enough marks?
- **Post-run**: did the bundle accept and use them, with what accuracy?

Honest carry-over from the original doubt:

> tri_proj/tri_color remain non-default because results were mediocre — we
> may be bumping the noise floor set by EXIF pose accuracy that no pre-bundle
> refinement can beat. What check_tags reliably catches is catastrophic
> per-target mis-tags (aztec7 CHK-14: 490× median dH); those score far above
> any threshold. The simpler `≥ 3 anchor-quality marks` rule captures the
> same catch with a cleaner mental model.

---

## Beads created from this prep

(Status as of 2026-05-01. Priority reflects post-Isaiah-feedback
adjustments and subsequent revisions.)

### Tools layer (Thread 1) — shipped

| ID | Title | Status |
|---|---|---|
| geo-6gni | Pix4Dmatic `quality_report.pdf` parser + `rmse.py --pix4d-qr` integration | ✅ closed |
| geo-hbxf | `rmse.py --tin`: LandXML TIN-sampled Z as third accuracy axis | ✅ closed |
| geo-y3j3 | `check_tags` toolchain-agnostic input + 3-tier gate + `--vs` comparison | ✅ closed |
| geo-fcio | `transform.py split` emits `{job}_6529_tagged_marks.csv` for Pix4D import | ✅ closed |
| geo-qkip | sight emits `{job}_6529_targets.csv` (Pix4D-import GCP coords, monument dedup) | ✅ closed (added during impl, not in original plan) |

### Tools layer — open

| ID | Pri | Title | Notes |
|---|---|---|---|
| geo-3ui | P2 | `rmse.py` independent triangulation on Pix4D camera params + marks | Demoted from P1; gated by root.p4m CBOR schema discovery |
| geo-z9u0 | P2 | packager: per-deliverable `lineage.json` writer | Tier 4, toolchain-agnostic |
| geo-4j6p | P2 | `docs/odm-workflow.md` diagram: add rmse.py's Pix4D consumption | Filed 2026-05-01 |
| geo-vtc0 | P3 | Surveyor-handoff CSV validator | Deprioritised — `.dc` required |
| geo-75sw | P3 | LandXML metadata-vs-data validator | Deprioritised — `.dc` required |

### Odium agent layer (Thread 2) — open

| ID | Pri | Title |
|---|---|---|
| geo-frmv | P2 | odium: CRS / transform recovery from `.dc` (Taos use case) |
| geo-seoj | P2 | odium: substitute control-point recommender (field-side, novel) |
| geo-wsu1 | P3 | odium two-path state machine: common upstream forks to ODM/Pix4D |
| geo-vmdn | P3 | odium: flight-planning advisor (AGL ↔ GSD ↔ accuracy) |

### Investigations / follow-ups surfaced by the unified report

| ID | Pri | Title |
|---|---|---|
| geo-0h61 | P1 | Investigate ODM vs Pix4D CHK accuracy gap on aztec (3× dZ disagreement) |
| geo-xzds | P1 | NAD83 realization-aware Helmert recovery from monument deltas |
| geo-1tz1 | P1 | GCPEditorPro: RETURN on tag screen saves and advances to next target |

### Epic

| ID | Pri | Title |
|---|---|---|
| geo-qx4b | P2 | Pix4D process support: geo tool ports + odium two-path (this epic) |

Dependencies wired:
- geo-qx4b (epic) depends on the original 9 children.
- geo-wsu1 (odium two-path) depends on geo-vmdn (flight advisor) and geo-6gni
  (QR parser, now closed) — wsu1's QR-parser prerequisite is satisfied.

---

## Meeting notes (revise live)

### Isaiah's reactions — captured 2026-04-29

**Handoff validator (geo-vtc0):** deprioritise. Stratus plans to require `.dc`
files from surveyors going forward; CSV validation is a fallback for the
no-`.dc` case.

**LandXML validator (geo-75sw):** deprioritise. If `.dc` is required upstream
and CRS is locked, the declared-vs-actual mismatch is much harder to produce.
Residual risk is Pix4D project-CRS misconfig (the original `Red Rocks TIN.xml`
mode), but no longer the highest-leverage gate.

**QR parser (geo-6gni) and rmse-vs-TIN (geo-hbxf):** confirmed direct interest.
Both stay P2.

**sight → Pix4D marks (geo-fcio):** Pix4D doesn't distinguish external-process
estimates from human-tagged marks at the input layer. That's exactly the gap
the geo `confidence` attribute (in `{job}.txt`, `{job}_tagged.txt`, surfaced
through GCPEditorPro) was created to fill. **Reframed**: rather than feeding
sight estimates into Pix4D (where the confidence weighting would be lost),
use sight + GCPEditorPro as Pix4D's *tagging frontend*, then export the
human-confirmed marks to Pix4D's CSV. Pending Isaiah's decision on whether
he'd actually tag in GCPEditorPro instead of Pix4D's GUI.

**check_tags (geo-y3j3):** raised to P2. Isaiah's insight: `{job}_tagged.txt`
and a Pix4D marks file carry similar information, so check_tags can serve
both toolchains if its input parser is generalised. Bundle the gate
simplification with the agnostic input refactor.

### Open questions for follow-up

- ~~Would Isaiah tag in GCPEditorPro instead of Pix4D's GUI? (gates fcio scope)~~
  **Resolved.** fcio shipped with both directions: `{job}_6529_color_marks.csv`
  (sight estimates as Pix4D pre-marks, color-confidence only) and
  `{job}_6529_tagged_marks.csv` (human-confirmed GCPEditorPro tags). Isaiah
  picks per job. See `docs/plans/sight-to-pix4d-import.md`.
- ~~Has anyone tested whether Pix4DMatic accepts pre-marked image-coordinate
  CSV input? Mapper does.~~ **Pending Isaiah's first import test on a real
  job; the Matic API path is documented in the joint-review plan.**
- ~~For check_tags toolchain-agnostic input: what does a Pix4D marks export
  actually look like?~~ **Resolved.** `extract_pix4d_marks.py` parses
  `history.p4mpl` directly (replaying `MarkTiePoint` / `RemoveTiePoints`
  blocks); check_tags consumes either that, the extracted CSV, or a sight
  `_tagged.txt` interchangeably.
- **New:** what's the root cause of the 3× CHK RMS_Z gap between ODM and
  Pix4D on aztec13? Tracked as **geo-0h61** with hypothesis list.
- **New:** does pyproj's 6529↔2258 transform leave a NAD83 datum-realization
  residual large enough to confound `rmse --tin` comparisons? The aztec13
  TIN `mean_dZ = -0.16 ft` suggests yes, but small. Tracked as **geo-xzds**.

### Action items

- ~~Ship-first candidate~~ — **moot; all three Thread 1 P2 items shipped 2026-04-30 / 2026-05-01.**
- ~~Decision pending from Isaiah on fcio~~ — **resolved; both pre-mark and
  tagged-mark exporters ship.**
- **Open threads:** odium agent layer (frmv, seoj, wsu1, vmdn) is the next
  major front; no Thread 2 work has begun. The 3× ODM-vs-Pix4D CHK Z gap
  (geo-0h61) and the realization-vintage Helmert (geo-xzds) are the two
  highest-value follow-ups uncovered by the cross-validation in the unified
  rmse.html report.

---

## Cross-references

- `~/stratus/redrocks/elevation/TIN_GCP_EVALUATION_RESPONSE.md` — full diagnosis
  + Tier 0–4 recommendations (the source material for this plan).
- `docs/plans/tag-quality-consistency.md` — overlapping epic geo-wst0 (sight
  iterative refinement, internal-consistency report, pre-ODM bad-tag
  detector, trail sidecar). The check_tags simplification (geo-y3j3) is a
  course-correction within that epic.
- `docs/odm-workflow.md` — the canonical ODM pipeline diagram; the two-path
  state machine extends this with a Pix4D fork.
- `~/git/geo-samples/odm-ortho-error/README.md` — "Accuracy claims relative
  to image resolution" section is the prior art the flight-planning advisor
  (geo-vmdn) builds on.
