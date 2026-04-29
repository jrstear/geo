# Pix4D Process Support: geo tool ports + odium two-path

**Epic bead:** geo-qx4b
**Spawned from:** 2026-04-29 prep for Jon ↔ Isaiah meeting on `~/stratus/redrocks/elevation/TIN_GCP_EVALUATION_RESPONSE.md`
**Status:** Living planning doc — revise during/after the meeting.

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

| Tool | Pix4D fit | Bead | Pri |
|---|---|---|---|
| `transform.py dc` → `transform.yaml` | **Ports today, toolchain-agnostic.** Run on every job's `.dc` regardless of Pix4D vs ODM. Mar-6 email chaos becomes structured data. | (already shipped) | – |
| Pix4D `quality_report.json` parser → post-bundle accuracy + outlier checks | **~30 lines.** Post-run gate: `verified_marks_count` vs `marks_count` mismatch (catches mis-tagged marks the bundle rejected); per-GCP XYZ residuals; per-CHK RMS_H/RMS_Z. Different question than the pre-run marks-count check (which moves to geo-y3j3). | geo-6gni | **P2** |
| Pre-run marks-count hard-stop (Pix4D marks file OR `_tagged.txt`) | Catches bigMem (50 GCPs, 0 marked) **before** the bundle runs, at $0 cost. Bundled with check_tags' toolchain-agnostic input refactor — same parser, both toolchains. Strictly cheaper than QR for this class of failure. | geo-y3j3 | **P2** |
| `rmse.py` extended for TIN-sampled Z | **Needs work.** Replaces the Flask app in `~/stratus/redrocks/elevation/` and the duplicated `examine_tins*.py` analysis. Toolchain-agnostic. **Confirmed direct interest.** | geo-hbxf | **P2** |
| `check_tags` toolchain-agnostic input + simplify gate | **Newly elevated.** Isaiah's observation: `{job}_tagged.txt` and a Pix4D marks file carry similar information, so check_tags can serve both toolchains if it accepts either input. Bundle with the gate-simplification (drop the magic 0.7 score, replace with the `≥ 3 anchor-quality marks` rule that mirrors `marks_count >= 3` for Pix4D). | geo-y3j3 | **P2** |
| Surveyor-handoff CSV validator | **Deprioritised.** Stratus plans to require `.dc` files going forward — the `.dc` carries CRS/datum/geoid metadata explicitly, so the CSV validator becomes a fallback rather than a primary gate. | geo-vtc0 | P3 |
| LandXML metadata-vs-data validator | **Deprioritised.** If `.dc` is required and CRS is locked at survey-handoff time, the declared-vs-actual mismatch becomes much harder to produce. Residual risk is Pix4D project-CRS misconfig (the original `Red Rocks TIN.xml` mode), but it's no longer the highest-leverage gate. | geo-75sw | P3 |
| `sight.py` → Pix4D marks CSV exporter | **Reframed.** Isaiah: Pix4D doesn't distinguish external-process estimates from human-tagged marks — exactly the gap the geo *confidence* attribute (`color`/`tri_color`/`tri_proj`/`projection`/`tagged`) was created to fill. Feeding raw sight estimates to Pix4D would erase that distinction Pix4D doesn't make in the first place. **Reframe**: use sight + GCPEditorPro as the *tagging frontend* (where confidence is meaningful), then export the human-confirmed result to Pix4D's marks CSV format. Pix4D treats every input mark as ground truth — correct, because they are. Isaiah skips Pix4D's GUI tagging step. Pending his decision on whether he'd actually tag in GCPEditorPro vs Pix4D's GUI. | geo-fcio | P3 |

**Question for Isaiah after this discussion:** of the P2 set (QR parser,
rmse-vs-TIN, agnostic check_tags), which would actually save the most pain
on the next job? That's your "ship first" candidate.

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

(Priority reflects post-Isaiah-feedback adjustments.)

| ID | Pri | Title |
|---|---|---|
| geo-qx4b | P2 | Pix4D process support: geo tool ports + odium two-path (this epic) |
| geo-6gni | **P2** | Pix4D `quality_report.json` parser with marks_count hard-stop |
| geo-hbxf | **P2** | `rmse.py` extension: TIN-sampled Z as third accuracy axis |
| geo-y3j3 | **P2** | `check_tags`: toolchain-agnostic input + simplify gate (raised) |
| geo-z9u0 | P2 | packager: per-deliverable `lineage.json` writer |
| geo-frmv | **P2** | odium: CRS / transform recovery from `.dc` with missing or wrong-frame CRS (Taos use case) |
| geo-seoj | **P2** | odium: substitute control-point recommender when monument is disturbed/missing (field-side, novel) |
| geo-vtc0 | P3 | Surveyor-handoff CSV validator (deprioritised — `.dc` required) |
| geo-75sw | P3 | LandXML metadata-vs-data validator (deprioritised — `.dc` required) |
| geo-fcio | P3 | `sight.py` → Pix4D marks CSV exporter (reframed; pending Isaiah's tagging-tool decision) |
| geo-wsu1 | P3 | odium two-path state machine: common upstream forks to ODM or Pix4D |
| geo-vmdn | P3 | odium: flight-planning advisor (AGL ↔ GSD ↔ accuracy) |

Dependencies wired:
- geo-qx4b (epic) depends on all 9 children.
- geo-wsu1 (odium two-path) depends on geo-vmdn (flight advisor) and geo-6gni
  (QR parser) — those are concrete prerequisites for the Pix4D fork.

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

- Would Isaiah tag in GCPEditorPro instead of Pix4D's GUI? (gates fcio scope)
- Has anyone tested whether Pix4DMatic accepts pre-marked image-coordinate
  CSV input? Mapper does. (still relevant if fcio proceeds)
- For check_tags toolchain-agnostic input: what does a Pix4D marks export
  actually look like? Format spec needed before geo-y3j3 implementation.

### Action items

- _Ship-first candidate: _
- _Decision pending from Isaiah on fcio: _
- _Other open threads: _

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
