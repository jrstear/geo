# sight.py → Pix4D Matic Import Workflow

**Bead:** geo-qkip
**Status:** ready for review with Isaiah after first sight-output import test

This document captures the proposed workflow for using sight.py output as
the input to Pix4D Matic, instead of (or in addition to) raw Emlid CSV
exports. Written for joint review with Isaiah after he tries an import.

---

## Goal

Run sight.py after the survey, then import its output into Pix4D Matic and
finish tagging there. Isaiah keeps Pix4D as his processing engine; sight
gives him pre-vetted GCP coordinates, pre-classified GCP/CHK roles, and a
starter set of color-confidence pixel marks for the easy images.

---

## File bundle

After `sight.py` runs, two files form the Pix4D Matic import bundle:

```
{job}_{epsg}_targets.csv          ← GCP coordinates (survey CRS)
{job}_{epsg}_color_marks.csv      ← marks for the import (image-pixel space)
```

Concrete example (aztec):

```
aztec_6529_targets.csv            (59 rows: 41 paint targets + 18 monuments)
aztec_6529_color_marks.csv        (493 rows: 1 row per (image, target) where
                                    sight color refinement found the marker
                                    sub-pixel)
```

### Targets file schema

```
label,X,Y,Z,type
CHK-101,349825.0585,653586.7032,1767.7902,target
GCP-104,349373.8999,653016.1558,1764.3627,target
1,1111697.1230,2121313.6030,5668.3700,monument
18m,1156886.3920,2160373.7220,5829.4850,monument
```

- **label** carries sight's classification: `CHK-NNN` (check), `GCP-NNN`
  (control), `CHK-NNN-dup` (near-duplicate suffix). Monument labels stay
  bare (Emlid `Name`): `1`, `18m`, `12m`, etc.
- **X, Y, Z** in survey CRS (the Emlid native CRS, e.g. EPSG:6529 ftUS for
  NM Central) — *not* the design grid, *not* the ODM metric CRS.
- **type** = `target` (paint marker, RTK-shot) or `monument` (control point,
  imported reference or RTK shot of an imported point).

### Marks file schema

```
Filename,Label,PixelX,PixelY
DJI_20260309171745_0197_V.JPG,CHK-123,2652.30,1485.51
```

- Standard Pix4D image-points format.
- Pixel coordinates are CRS-independent (image-pixel space).
- Includes only color-confidence sight estimates (the sub-pixel-accurate
  ones). Absence of a mark on a given (image, target) pair means "needs
  Isaiah's manual click in the Pix4D UI."

---

## Pix4D Matic terminology

Confirmed from Isaiah's `history.p4mpl`:

- **TiePoint** — umbrella term in Pix4D Matic for any GCP or checkpoint.
- Default role on import = used as **GCP** (constrains bundle adjustment).
- `UseTiePointAsCheckpoint { isCheckpoint: true }` flips a tie point to
  **checkpoint** role (withheld from bundle, used for accuracy QC only).
- Isaiah can override the role per-target in the UI after import.

So sight's `CHK-` / `GCP-` labels are *suggestions*. If Isaiah wants to
keep his existing convention of which targets are control vs check, he
can re-classify each target in Pix4D after import — sight's labeling
doesn't lock him in.

---

## Pros

1. **Consistent label naming across tools.** sight's `CHK-101` / `GCP-104`
   propagates into Pix4D, matching the labels Isaiah sees in ODM,
   `rmse.py`, and `check_tags`. Today his Pix4D projects use bare
   numerics with spaces (`131 2`, `128 1`); switching to sight's labels
   removes a real cognitive-translation cost.
2. **Filtered, vetted output.** Emlid raw includes setup shots, prior-
   session shots, FLOAT-quality rows, possibly old job data — sight
   already filters Origin=Local-without-match and applies dedup. Cleaner
   input than raw export.
3. **Quality-aware exclusion.** sight's pre-tagging quality checks
   (geo-40vs) flag FLOAT shots, low sample averaging, high RMS. Future
   work could exclude flagged rows from the import bundle.
4. **Pairs symmetrically with the marks file.** The bundle is one
   coordinated handoff for the Pix4D side.
5. **Decouples the import from Emlid-export discipline.** "Did Isaiah
   remember to do a full export this time?" becomes irrelevant — sight
   produces what's needed.
6. **Monument dedup.** Each physical monument appears once in the
   targets file. Where both an Origin=Local import row and an
   Origin=Global RTK shot exist for the same monument, sight prefers the
   RTK shot (since that's the verified position). For aztec: 5 RTK +
   13 Local-only = 18 unique physical monuments.

## Cons

1. **Convention shift for Isaiah.** Today his Pix4D projects use
   `131 2` (space-separated, bare ids); sight outputs `CHK-131-2`
   (hyphenated, prefixed). If he adopts sight's labels, his project
   history changes naming convention. Mitigation: a `--pix4d-naming bare`
   flag could strip prefixes and replace `-` with ` ` for sub-points if
   he pushes back. **Deferred until needed.**
2. **Yet another file to manage.** Naming is busy:
   `{job}_6529.csv` (.dc-derived monuments),
   `{job}_emlid_6529.csv` (Emlid raw),
   `{job}_targets.csv` (ODM CRS, all surveyed),
   `{job}_targets_design.csv` (design grid, all surveyed),
   `{job}_6529_color_marks.csv` (Pix4D marks),
   `{job}_6529_targets.csv` (Pix4D GCP coords) ← new.
   Mitigation: documentation. The naming pattern is consistent with
   existing files.
3. **Loss of Emlid per-row metadata.** RMS, samples, solution status —
   all gone in the slim sight output. Pix4D doesn't use these, so likely
   OK, but Isaiah loses the option to QA from this file alone. He can
   always check the original Emlid CSV.
4. **Redundant data flow vs Emlid → Pix4D direct.** If full Emlid
   exports work for Pix4D, the new sight output is duplication. The
   benefits above (label consistency, filtering, dedup) make the
   duplication worthwhile, but only if Isaiah agrees they're worth
   adopting.

## Risks

### CRS-vintage / datum-realization mismatch

**This is the biggest risk and warrants its own discussion.**

NAD83 isn't one fixed datum — it's a family of *realizations* refined as
the geodetic network grows and tectonic motion accumulates:

| name           | year   | typical sources                                         |
|----------------|--------|---------------------------------------------------------|
| NAD83 (1986)   | 1986   | classical triangulation + early GPS                     |
| NAD83 (HARN)   | ~1990s | high-accuracy regional readjustments                    |
| NAD83 (NSRS2007) | 2007 | nationwide CORS-based readjustment                      |
| **NAD83 (2011)** | 2011 | tied to ITRF2008 epoch 2010.0; current standard for RTK |
| NAD83 (2022)   | TBD    | upcoming — will replace NAD83                           |

**For NM, the regional shift between NAD83(86) and NAD83(2011) is ~12-16
ft horizontally** — exactly what `geo-40vs` Check 2b detected on the
aztec dataset.

The same projection (NM Central state plane, ftUS) appears in multiple
EPSG codes, one per realization:

| EPSG     | realization     | typical use                                     |
|----------|-----------------|-------------------------------------------------|
| 32108    | NAD83 (1986)    | very old datasheets                             |
| **2258** | NAD83 (1986/HARN era) | **Isaiah's Pix4D project CRS**            |
| 3618     | NAD83 (NSRS2007) | mid-2000s deliverables                         |
| **6529** | NAD83 (2011)    | **Jon's Emlid output, modern RTK**              |

Same projection in all of these — the only difference is the realization.
Coords differ by ~12-16 ft between 2258 and 6529 in NM.

#### What's actually happening in aztec

Three data sources are all tagged "EPSG:6529" by the pipeline but in
different realizations:

1. **`.dc` file** (Trimble field book): coords are in whatever realization
   the customer's project was set up in — *probably* NAD83(86) or
   NSRS2007. `transform.py dc` tagged them as EPSG:6529 without verifying
   the realization.
2. **Emlid CSV `Origin=Local` rows**: imported customer coords. May or
   may not match the .dc — depends on what reference Isaiah typed in.
3. **Emlid CSV `Origin=Global` rows**: fresh RTK shots in NAD83(2011) via
   the surveyor's NTRIP base. Genuinely 6529.

Check 2b's 12-16 ft delta between the .dc-derived `aztec_6529.csv` and
the Emlid Local rows shows the .dc is in an older realization, not 2011.

#### Implications for the sight-to-Pix4D workflow

- **Pixel marks are unaffected.** Image-pixel coords have no datum.
- **GCP coordinate file IS affected.** sight emits `aztec_6529_targets.csv`
  with values in NAD83(2011). Isaiah's Pix4D project is set to
  **EPSG:2258** (older NAD83). If Isaiah loads the 2011 coords into a
  2258 project without realizing it, every GCP will be 12-16 ft
  off horizontally; bundle adjustment will either reject the GCPs as
  outliers or absorb the bias as systematic camera-pose error.
- **Isaiah's existing workflow has been getting away with this** because
  Emlid Flow's localization step uses customer-frame monuments to compute
  a Helmert transform, absorbing the realization shift into the
  localization. After localization, his RTK shots are *numerically* in
  the customer's frame (2258), even though they were collected in 2011.

#### Options for handling the mismatch

| option | what changes | trade-off |
|---|---|---|
| **A. Re-pin Pix4D project CRS to 6529** | Isaiah changes the project CRS to NAD83(2011). | Customer's published reference may be in an older realization → deliverables may not match. |
| **B. sight emits in customer's CRS** | sight reprojects 6529 → 2258 (or 3618) before writing. | Requires knowing customer's realization (currently unknown / hand-typed). pyproj's 6529→2258 is approximate (the per-region tectonic adjustment is empirical, not exact). |
| **C. Emlid-side localization absorbs the shift** | Isaiah continues localizing in the field as today. sight's output goes through that localization. | Status quo. Works but doesn't fix the underlying mismatch. |
| **D. Realization-aware Helmert recovery** (future bead) | sight uses observed Local↔Global pairs (and Check 2b's monument deltas) to recover the realization transform empirically. | Most accurate; needs implementation. See **geo-xzds**. |

For the **first import test with Isaiah**, recommend **option C
(unchanged workflow on his side)** — he treats sight's GCP file the same
way he'd treat an Emlid CSV: load it, apply localization, proceed.
Validate that the bundle adjustment converges and check residuals match
his historical numbers.

After that, if the import is successful, decide between A/B/D for the
durable fix.

### Other risks

- **Pix4D may not auto-handle the `type` column.** The schema is
  `label,X,Y,Z,type`. Pix4D Matic's import dialog generally lets you
  pick which column is which, so it should be tolerant — but worth a
  one-shot test before committing to it. If Pix4D Matic objects, sight
  could emit a slim 4-column variant.
- **Origin=Local rows for monuments that were never RTK-shot are
  unverified.** sight's monument dedup keeps the Local for these (no
  Global to prefer). If the Local was hand-typed from a customer PDF and
  has typos, Pix4D doesn't know. Today Isaiah catches these in localization;
  he'll keep catching them.

---

## Open questions for Isaiah

After his first import test:

1. Do the labels (`CHK-101`, `GCP-104`, `CHK-131-2`) work in his Pix4D
   workflow, or does he prefer bare ids with spaces? If the latter, we
   add `--pix4d-naming bare`.
2. Does Pix4D Matic accept the `type` column, or do we need a slim
   4-column file?
3. Does importing the `aztec_6529_color_marks.csv` (493 rows) save him
   meaningful time vs hand-clicking, or are the projection-quality images
   that he still needs to click manually most of the work?
4. Which option (A/B/C/D) for the CRS-vintage mismatch matches his
   project setup constraints? Specifically: is he locked into 2258 by
   the customer's reference data, or can he move his project CRS to 6529?

---

## Open questions for further geo design

1. Should sight emit a paint-targets-only variant (no monuments) for
   workflows where monument inclusion is undesired? Currently the file
   has both. (Probably no — Isaiah wants both.)
2. Should sight provide a `--pix4d-strict-fix` flag that excludes RTK-
   FLOAT shots from the targets file? geo-40vs already flags them; this
   would extend it to filtering. (Defer until a real case shows up.)
3. Should the marks file's confidence threshold default change once we
   have data on Isaiah's experience? E.g., if `tri_color` proves reliable
   on real datasets, default to `tri_color` (after `--iterative` becomes
   default-on).

---

## Related work

- **geo-s074** — Origin-based monument filter (foundation for this work).
- **geo-40vs** — pre-tagging quality checks (Local↔Global localization
  residuals, .dc vs Emlid import discrepancies, RTK quality).
- **geo-frmv** — odium CRS recovery from .dc files with missing or
  wrong-frame CRS.
- **geo-xzds** — NAD83 realization-aware Helmert recovery from monument
  deltas. Option D above; sibling of geo-frmv.
- **geo-9d66** (closed) — GCPEditorPro compass overlay fix; unrelated
  but landed during the same retrospective.
