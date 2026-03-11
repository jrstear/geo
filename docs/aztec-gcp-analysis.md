# Aztec NM — GCP Analysis and Flight Recommendations

**Job:** F100340 AZTEC
**Today's flight:** US 550 corridor only (~6 mi, KMZ provided)
**NM 516 / southern monuments:** from a previous job — not in scope today
**Customer requirement:** 55 control points + check points
**Accuracy targets:** 0.1 ft horizontal, 0.3 ft vertical

---

## What is actually in the provided data

### Point file (F100340 AZTEC.dc → F100340_AZTEC_points.csv)

| Category | Count | Notes |
|---|---|---|
| **True survey monuments** | **18** | Brass/alum caps, NGS VCM — usable as GCPs |
| Structure as-built detail | 28 | TOF, CWB wall, pipe inverts, manholes — all in a 300×700 ft area at one bridge/retaining wall structure; not aerial targets |
| Road alignment geometry | 36 | PC, PT, PI, B.O.P., EOP, CL crossover — horizontal only, no elevation |
| Base station setups | 12 | GNSS antenna heights (2–9 ft), not ground elevation |
| Observed points | 12 | Repeated antenna setups — not GCPs |

The 18 true survey monuments are the usable GCPs.  The original count of "48 3D control
points" included 28 construction detail points that are not GCP targets.

### Today's flight: US 550 only

The customer-provided KMZ covers **US 550** (northeast of Aztec, ~6 miles).
The 13 southern/western monuments (IDs 1–13) are from a **previous job** — not in scope.

| Corridor | Monuments in scope | Notes |
|---|---|---|
| US 550 (today) | **5** (IDs 14–18) | The only relevant monuments for today's flight |
| NM 516 / previous job | 13 (IDs 1–13) | Out of scope — different job |

**With only 5 monuments on US 550, the customer's "55 control points" almost certainly
requires setting new targets in the field.**  The 5 existing caps provide the anchor points;
new pigeon targets fill the gaps.

### Side-of-road distribution

US 550 runs **northeast** in this area.  "East/right" side = the southeast face of the road.

- US 550 monuments: **3 LT (northwest/left) / 2 RT (southeast/right)** — reasonably balanced
- NM 516 monuments: all appear as "right" of US 550 because they are on a different road
  southwest of the US 550 corridor — side-of-road relative to NM 516 is unknown without
  that route's centerline

The all-red appearance in Google Earth (all GCPs on one side) is because the KMZ only
shows US 550, and the NM 516 monuments are all to the southwest of it.  Loaded together
they look one-sided, but they're actually on a completely separate road.

**Ideal setup:** 2–3 GCPs on each side per corridor, alternating along the length.

### Structure detail cluster (2000-series points)

28 points (IDs 2000–2035: TOF, CWB WALL, inv pipe, mh inlet, etc.) are packed into a
~300 × 700 ft area.  These are bridge/retaining wall construction as-built measurements
at the US 550 / NM 516 interchange.  They were included in the DC file as 69KI control
records (formal survey control) but they are:
- Impossible to identify individually from altitude
- All within one cluster — no structural GCP value
- Labeled "inv pipe", "TOF STEP G-H", etc. — structure elements, not targets

These are **excluded from all GCP selection.**  The original "center pin" (ID 2000,
"inv pipe") in the preliminary analysis was one of these — it has been replaced by
a true survey monument.

---

## The two things that actually drive RMSE

### 1. Flight altitude — the dominant factor

Camera: **DJI Mavic 3 Enterprise (M3E)** or **Matrice 4E** — both use a 4/3" CMOS 20 MP
wide camera (12.3 mm FL, 5280 px), identical to the standard Mavic 3.
**GSD is unchanged:** altitude × 2.664e-4 ft/px (e.g. 200 ft → 0.053 ft/px, 300 ft → 0.080 ft/px).

*Note on the Matrice 4E's medium-tele camera (70 mm equiv, 1/1.3" 48 MP sensor): GSD is
roughly 3× lower at the same altitude — excellent for close-range detail but rarely used for
wide-area corridor mapping.  All tables below assume the wide camera.*

The table below shows **practical expected RMSE** — real-world achievable with good GCP
distribution and accurate target tagging.  Values use empirical GSD multipliers (not
theoretical minimum): 3 GCPs ≈ 5× GSD, 5 GCPs ≈ 3× GSD, 7 GCPs ≈ 2× GSD,
10 GCPs ≈ 1.25× GSD.  The theoretical floor (1× GSD, perfect conditions) is always lower.

| GCPs | 100 ft AGL | 150 ft AGL | 200 ft AGL | 300 ft AGL | 400 ft AGL |
|---|---|---|---|---|---|
| 3 | 0.13 / 0.26 | 0.20 / 0.39 | 0.26 / 0.53 | 0.39 / 0.78 | 0.52 / 1.05 |
| 5 | **0.08 / 0.17 ✓** | 0.12 / 0.24 | 0.17 / 0.33 | 0.25 / 0.49 | 0.33 / 0.66 |
| 7 | **0.05 / 0.10 ✓** | **0.07 / 0.15 ✓** | **0.10 / 0.20 ✓** | 0.16 / 0.32 | 0.21 / 0.43 |
| 10 | **0.03 / 0.07 ✓** | **0.05 / 0.10 ✓** | **0.07 / 0.13 ✓** | 0.10 / 0.20 ⚠ | 0.13 / 0.27 |
| 15–18 | **✓** | **✓** | **✓** | 0.10 / 0.20 ⚠ | 0.13 / 0.27 |

*Values are horizontal / vertical RMSE in feet.  ✓ = meets 0.1 ft H / 0.3 ft V with margin.
⚠ = right at the limit with best-case tagging — no margin for error (see note below).*

**The customer specified 300 ft AGL.** At that altitude with the M3E wide camera (GSD = 0.080 ft/px):

| Scenario at 300 ft | Expected H RMSE |
|---|---|
| 10 GCPs, excellent pigeon targets, auto-detect | ~0.10–0.11 ft — borderline |
| 10 GCPs, raw brass caps (2.5 px — very hard to tag) | ~0.14–0.17 ft — likely fails |
| 7 GCPs (realistic US 550 monument count) | ~0.16 ft — fails |
| Theoretical absolute floor (unachievable in practice) | 0.080 ft |

**The 300 ft / 0.1 ft H combination is contradictory.**  The table ⚠ value (0.10 ft) assumes
perfect pigeon targets, perfect GCP distribution, and zero tagging error.  Any one of those
degrading — and 300 ft gives you ~0.12–0.17 ft horizontal.  The customer may not know
these two requirements are in conflict.

**200 ft AGL** gives GSD = 0.053 ft/px.  With 7+ good GCPs: ~0.10 ft H (meeting target
with margin).  With 10 GCPs and pigeon targets: ~0.07 ft H (comfortable margin).

Beyond 10–15 GCPs the values are identical — adding more monuments provides no further
RMSE improvement.  This is a hard limit set by physics (GSD), not by effort.

### 2. GCP distribution — more important than count

**For a linear corridor project, two properties matter:**

**A. Side-of-road alternation (cross-track constraint)**
All GCPs on one side allows the block to tilt perpendicular to the road regardless
of how many GCPs are used.  Minimum: 2 GCPs on each side per corridor.

**B. Along-track spacing (accumulation of systematic error)**
Systematic errors — camera model imperfections, lens distortion, atmospheric
refraction — accumulate between GCP constraints.  The practical rule:

| Corridor length | GCPs needed (non-RTK) | Spacing |
|---|---|---|
| 6 mi (today's US 550) | 6–12 | ~1 per mile |
| 20 mi | 10–20 | 1 per mile |
| 50 mi | 25–50 | 1 per mile |
| Any length, with RTK | 3–5 | RTK constrains position; GCPs calibrate boresight only |

The "10 GCPs is sufficient" finding from compact-block studies does **not** apply to
long corridors.  For a 50-mile AOI, 10 GCPs means 5-mile gaps where systematic errors
accumulate unchecked.  GCP count must scale with corridor length.

For today's 6-mile US 550 flight: 6–10 GCPs well-spaced along the route is the target.
The 5 existing caps provide anchors; new pigeon targets fill the gaps in between.
Every GCP here matters — there is no redundancy to spare.

### RTK — the M3E / Matrice 4E difference that actually matters

Both the M3E and Matrice 4E are RTK-capable.  RTK fundamentally changes GCP requirements:

| Mode | GCPs needed | Why |
|---|---|---|
| **Non-RTK** (standard GPS) | 1 per mile along corridor | Systematic errors accumulate between constraints |
| **RTK active** | 3–5 per corridor (any length) | RTK constrains position to 1–2 cm; GCPs only calibrate boresight tilt |
| **PPK (post-processed)** | 3–5, same as RTK | Equivalent accuracy, resolved in office |

**RTK does NOT change GSD or altitude requirements.**  The 300 ft / 0.1 ft H conflict
is a camera physics problem (GSD = 0.080 ft/px), not a positioning problem.  RTK cannot
make a 300 ft image resolve sub-pixel.  The altitude recommendation (≤ 200 ft for a
realistic path to 0.1 ft H) is unchanged.

**If RTK is active today:**
- The 1-per-mile spacing rule is irrelevant.  3 well-distributed GCPs (one near each
  end + one mid-corridor) is enough for boresight calibration.
- The 5 existing caps may be more than sufficient without any new pigeon targets.
- The pigeon targets remain valuable for tagging accuracy — they don't become
  optional just because RTK is on.

**If RTK is NOT active (or RTK module not present):**
- The 1-per-mile rule and all guidance above applies.
- 6–10 GCPs needed; new pigeon targets in the 3-mile middle gap are essential.

**Confirmed:** RTK will be active on official survey control points for today's flight.
The 5 existing US 550 caps are sufficient.  Apply pigeon targets at all 5 monuments
for tagging accuracy; new pigeon-only targets in the middle gap are optional.

---

## The visibility problem

Brass and aluminum caps are approximately 2 inches in diameter.

| Altitude | GSD | 2" monument | Tagging difficulty |
|---|---|---|---|
| 100 ft | 0.033 ft/px | 5 pixels | Hard |
| 150 ft | 0.049 ft/px | 3.4 pixels | Hard |
| 200 ft | 0.066 ft/px | 2.5 pixels | Very hard |
| 300 ft | 0.098 ft/px | 1.7 pixels | Essentially invisible |
| 400 ft | 0.131 ft/px | 1.3 pixels | Essentially invisible |

Tagging error on a 2" cap is the binding accuracy constraint regardless of GCP count.

### Orange clay pigeon targets

Isaiah has been using **8–10 clay pigeons per target** arranged as a cross:
- 3 touching pigeons per arm (a continuous 13" orange bar per arm = ~16 px at 200 ft)
- Touching is better than spaced: continuous bars have cleaner edges for auto-detection
  and a more stable centroid estimate

**At a brass/alum cap:** use 8 pigeons (4 arms × 2 each), **no center pigeon** —
leave the cap exposed in the gap.  The cap is the GCP coordinate; the pigeons frame
it.  The geometric center of the 4 arms = the cap = zero offset error.

**At a standalone new target:** use 9–10 with a center pigeon.  The center pigeon
is the click reference since no monument exists beneath it.

---

## US 550 monuments for today's flight

5 existing survey caps are within the KMZ flight area.  These are the structural
anchors; new pigeon targets must fill the gaps between them.

| ID | Side | Offset | Approx station | Description | Elevation |
|---|---|---|---|---|---|
| 14 | RT | 126 ft | ~249+43 (S end) | NGS VCM 3D Y 430 | 5797.80 ft |
| 15 | LT | 508 ft | N section | ALUM CAP | 5786.20 ft |
| 16 | LT | ~500 ft | N section | ALUM CAP | (similar) |
| 17 | LT | 515 ft | N section | ALUM CAP | 5802.92 ft |
| 18 | RT | 385 ft | N end | ALUM CAP | 5829.44 ft |

**Distribution:** ID 14 is near the southern end; IDs 15–18 are clustered at the northern
end.  With RTK active, the 3-mile middle gap is **not a problem** — RTK constrains position
continuously; GCPs only calibrate boresight tilt, which is satisfied by having anchors near
each end.  5 caps is sufficient.

**Side balance:** 3 LT / 2 RT.  Adding 1–2 pigeon-only targets on the RT (east/southeast)
side in the middle gap would improve cross-track constraint, but is optional with RTK.

KML of all monuments: `results/top10_gcps.kml`
Red stars = RT (east), Blue stars = LT (west), Grey dots = previous-job monuments (out of scope).

---

## Control Sheet (CONTROL SHEET.pdf — Sheet 3 of 0)

The customer-provided control sheet covers **Station 32+80.24 to 330+08.47** along US 550.
Coordinates on the sheet are in the NMDOT raw grid (same as the .dc file — subtract the
empirical offsets to get NM Central state plane ft).  Elevations match the CSV exactly.

### Points with station/offset for field navigation

| Point | Elevation | Description | Station | Offset |
|---|---|---|---|---|
| 4009-37 | 5683.61 ft | Alum. Cap. Rebar | 33+53.80 | 50.68' LT |
| 250557 | 5678.80 ft | Plastic Cap Rebar | 34+18.69 | 72.15' LT |
| 250513 | 5682.38 ft | Brass Cap "3703-38" | 51+38.74 | 54.46' LT |
| 631 | 5710.57 ft | **NGS VCM 3D** | 122+72.23 | 75.56' LT |
| 250171 | 5757.37 ft | Brass Cap | 158+82.25 | 91.33' RT |
| 4009-430 (=ID 14) | 5797.80 ft | **NGS VCM 3D ROD "Y 430"** | 249+43.08 | 170.35' RT |

NGS monuments 631 and Y 430 (ID 14) have published datasheets — independently verifiable.
Y 430 is 170 ft right of the road center and will not be visible from the truck.

### "Out of range" control points (confirmed on NM 516)

Seven points (4009-25 through 4009-36) are flagged "Out of range" on the US 550 sheet —
confirmed to be on the NM 516 corridor.  Valid 3D control, usable as GCPs.

### Sheet coverage

- This is **Sheet 3 of 0** — other sheets exist covering other areas.
- Survey units: US Survey Feet.  Basis of elevations: NMDOT control map (presumably NAVD88).

---

## Recommended approach

### Immediate (day of flight)

1. **Confirm flight scope: US 550 only, or US 550 + NM 516?**
   The KMZ covers US 550.  If NM 516 is in scope, a second flight and KMZ are needed.
   This changes everything: the GCP count, the target placement plan, the flight time.

2. **Confirm flight altitude before launching.**
   - ≤ 200 ft AGL: 0.1 ft H target is achievable with 7+ distributed GCPs
   - 300 ft AGL: 0.1 ft H is not achievable regardless of GCP count or RTK
   - RTK changes GCP count requirements, not GSD — altitude still matters

3. **Clay pigeon target placement at monuments:**
   - 8 pigeons (4 arms × 2 touching), no center pigeon, leave cap exposed in gap
   - Arms push up to the cap edge but do not cover the cap face
   - Prioritize the top-10 monuments in the KML (red and blue stars)

4. **New targets in the gaps: optional with RTK.**
   The 5 caps are sufficient for boresight calibration.  If time allows, 1–2 pigeon-only
   targets on the RT (east/southeast) side mid-corridor improve cross-track constraint —
   but do not skip monument setup time to get there.

5. **Tag all 5 monuments.**
   Get 7+ confirmed images each.  RTK means quality tagging matters more than quantity
   of GCPs — a poorly-tagged RTK GCP is worse than a well-tagged non-RTK one.

### Processing

6. **Flag top-10 as GCP-\*, rest as CHK-\*** in GCPEditorPro.
   Separate ODM run with only top-10 GCPs vs. all GCPs — compare check-point RMSE.

### For future jobs

- **7–10 well-placed colored-X targets** on a site this size match or beat 52 monuments
  at a fraction of tagging effort.
- The GCP placement advisor (in development) generates optimal target locations from the
  flight corridor polygon and exports a KML for field navigation.

---

## Questions to resolve with Isaiah tomorrow

- [x] **Flight scope: US 550 only.**  NM 516 / southern monuments are from a previous job.

- [x] **RTK will be active on official survey control points.**
  GCP count for boresight calibration = 3–5.  The 5 existing US 550 caps are sufficient.
  The 1-per-mile spacing rule does not apply.  New pigeon targets in the middle gap
  are not required for coverage — only needed if side-balance is desired.

- [ ] **What altitude is he planning to fly?**
  Must be ≤ 200 ft AGL to have a realistic path to 0.1 ft horizontal accuracy.

- [ ] **Does the customer's accuracy spec apply to control (GCPs) or check points?**
  These are different numbers.  Control RMSE is always better than check RMSE.

- [ ] **Is spray paint acceptable on pavement adjacent to monuments?**
  If not, pigeons-only targets need to be larger (more arms) for reliable tagging.

- [ ] **Can he get additional control sheets from the customer?**
  Sheet 3 of 0 covers US 550 only.  NM 516 monuments have no station/offset data,
  making field location harder.  Additional sheets would help.

- [ ] **Is WebODM available for processing, or is Pix4D required by the customer?**

- [ ] **What is the customer's deliverable format?**
  Point cloud, ortho, DEM, or all three?  Processing choices differ.
