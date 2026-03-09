# Aztec NM — GCP Analysis and Flight Recommendations

**Job:** F100340 AZTEC
**Site:** ~8.6 × 7.4 mi block (11.31 mi diagonal), Aztec NM
**Customer requirement:** 55 control points + check points
**Accuracy targets:** 0.1 ft horizontal, 0.3 ft vertical

---

## What is in the provided point file

| Category | Count | Notes |
|---|---|---|
| 3D control points | **48** | BRASS CAP, ALUM CAP, NGS VCM — usable as GCPs |
| Road alignment geometry | 32 | PC, PT, PI, EOP — horizontal only, no elevation, not usable as GCPs |
| Crossover centerline | 4 | Elevation = 0.00 — not usable |
| Base station setups | 12 | GNSS antenna heights (2–9 ft), not ground elevation |
| Observed points | 12 | Repeated antenna height measurements — not GCPs |

The 48 three-dimensional control points are the GCPs.  The customer's "55" likely
means they expect a few additional targets to be set in the field.

**Terrain:** 214 ft of relief over 11.31 miles (Z/XY = 0.004) — essentially flat.
No Z-critical GCP slots needed; standard perimeter + center-pin structure applies.

---

## The two things that actually drive RMSE

### 1. Flight altitude — the dominant factor

Achievable RMSE is bounded by ground sample distance (GSD): one image pixel in
real-world units.  The practical floor is 1–2× GSD horizontal, 2–3× GSD vertical.

| GCPs | 100 ft AGL | 150 ft AGL | 200 ft AGL | 300 ft AGL | 400 ft AGL |
|---|---|---|---|---|---|
| 3 | 0.13 / 0.26 | 0.20 / 0.39 | 0.26 / 0.53 | 0.39 / 0.78 | 0.52 / 1.05 |
| 5 | **0.08 / 0.17 ✓** | 0.12 / 0.24 | 0.17 / 0.33 | 0.25 / 0.49 | 0.33 / 0.66 |
| 7 | **0.05 / 0.10 ✓** | **0.07 / 0.15 ✓** | **0.10 / 0.20 ✓** | 0.15 / 0.29 | 0.20 / 0.39 |
| 10 | **✓** | **✓** | **✓** | **0.10 / 0.20 ✓** | 0.13 / 0.26 |
| 15–52 | **✓** | **✓** | **✓** | **✓** | 0.13 / 0.26 |

*Values are horizontal / vertical RMSE in feet.  ✓ = meets 0.1 ft H / 0.3 ft V targets.*

**Critical finding: at 300 ft AGL or higher, no number of GCPs can hit the 0.1 ft
horizontal target.**  At 200 ft AGL, 7+ well-distributed GCPs get there.  The flight
altitude must be confirmed before the flight, not discovered afterward.

Beyond 10–15 GCPs the rows are identical — adding more monuments provides no further
RMSE improvement.  This is a hard limit set by physics (GSD), not by effort.

### 2. GCP distribution — more important than count

The structural priority algorithm selects the best-distributed subset from the 48
monuments.  Coverage gap analysis shows how much of the block is more than X miles
from its nearest GCP:

| GCPs used | Max gap to nearest GCP |
|---|---|
| 5 | 1.69 mi |
| 7 | 1.28 mi |
| **10** | **0.63 mi** |
| 15 | 0.63 mi |
| 20 | 0.63 mi |
| 30 | 0.63 mi |
| **52** | **0.63 mi — same as 10** |

The 10 structurally optimal monuments already define the coverage envelope.  The
remaining 38 are all clustered near those 10 — adding them fills no spatial gaps.
**The customer's 52 monuments cover the block no better than 10 well-chosen ones
from that same set.**

The 0.63 mi residual gap is not a monument problem — it's a gap in the road network
where no monuments exist.  A custom-placed colored-X target in that area would fill
it; an additional monument from the existing 48 cannot.

---

## The visibility problem

This is the issue that will most affect tagging accuracy — and tagging accuracy
directly limits achievable RMSE.

Brass and aluminum caps are approximately 2 inches in diameter.  Their size in drone
imagery:

| Altitude | GSD | 2" monument | Tagging difficulty |
|---|---|---|---|
| 100 ft | 0.033 ft/px | 5 pixels | Hard |
| 150 ft | 0.049 ft/px | 3.4 pixels | Hard |
| 200 ft | 0.066 ft/px | 2.5 pixels | Very hard |
| 300 ft | 0.098 ft/px | 1.7 pixels | Essentially invisible |
| 400 ft | 0.131 ft/px | 1.3 pixels | Essentially invisible |

A correctly placed click needs to land within 1–2 pixels of the true center.  When
the target itself is 2–3 pixels across, that is nearly impossible to do consistently.
Tagging error at this scale propagates directly into model accuracy.

Compare: a **4-foot colored-X target** at 200 ft AGL is **24 pixels** across.
Clicking the center to within 2 pixels is straightforward, even in oblique images.
The colored-X auto-detection pipeline narrows the initial estimate to ±5–30 px
automatically, after which the operator confirms with a single keypress.

**Implication:** even if we use all 52 monuments exactly as the customer specified,
the tagging accuracy on 2" caps at survey altitude will be the binding constraint on
RMSE — not the GCP count.

---

## Structural top 10 — the monuments that matter most

These are selected by the max-insertion algorithm: each one provides the greatest
remaining structural value to the photogrammetric block.

| Priority | ID | Description | Role |
|---|---|---|---|
| 1 | 18 | ALUM CAP | NE distal anchor — sets one corner of the block |
| 2 | 1 | BRASS CAP 3703-211 | SW distal anchor — defines scale and orientation |
| 3 | 2000 | inv pipe | **Center pin — prevents doming; most commonly missed** |
| 4 | 14 | NGS VCM 3D Y 430 | Perimeter fill |
| 5 | 13 | BRASS CAP | Perimeter fill |
| 6 | 2 | ALUM CAP 3703-3011 | Perimeter fill |
| 7 | 7 | ALUM CAP 409-36 | Perimeter fill |
| 8 | 11 | PLASTIC CAP RBR | Perimeter fill |
| 9–10 | (next 2 by geometry) | — | Perimeter fill |

The remaining 38 monuments are effectively check points — they add redundancy and
improve the accuracy report but contribute little additional structural constraint
once the top 10 are in.

Z-critical slots (high/low elevation anchors) are suppressed because the site is
flat (214 ft relief over 11 miles).

---

## Recommended approach

### For this job (do what the customer asked, but work smart)

1. **Confirm flight altitude before flying.**  200 ft AGL is the minimum to have a
   realistic chance at 0.1 ft horizontal.  At 300 ft AGL, the target is out of reach
   regardless of GCPs.  At 150 ft AGL there is comfortable margin.

2. **Supplement monuments with colored-X spray paint.**  On-site tomorrow, spray an
   X on the pavement centered on (or immediately adjacent to) each of the top-10
   structural priority monuments.  The monument coordinates remain the known GCP
   position; the spray paint makes the target visible and accurately taggable from
   altitude.  This costs nothing but a can of marking paint.

3. **Tag all 52 monuments.**  The customer asked for it and it is not much extra
   work once the flight is done.  But prioritize the top 10 — get 7+ confirmed images
   each on those first.

4. **Flag the top 10 as GCP-\* and the rest as CHK-\*** in GCPEditorPro.  This
   automatically exports `gcp_confirmed.txt` (10 control points for ODM) and
   `chk_confirmed.txt` (42 check points for the accuracy report) as separate files.

5. **Process twice in WebODM** — once with all 52 GCPs, once with only the top 10.
   Compare the check-point RMSE between runs.  This is the data that makes the
   argument for the customer on future jobs.

### For future similar jobs

- **7–10 well-placed colored-X targets** on a site this size will match or beat
  the accuracy of 52 monuments, at a fraction of the tagging time.
- The placement advisor tool (in development) generates the optimal target locations
  automatically from the flight corridor polygon, exported as a KML for field
  navigation.
- Processing in **WebODM** (free) eliminates the per-image Pix4D cost entirely.

---

## Control Sheet (CONTROL SHEET.pdf — Sheet 3 of 0)

The customer-provided control sheet covers **Station 32+80.24 to 330+08.47** along US 550.
The coordinate values on the sheet are in the NMDOT raw grid (same as the .dc file — subtract
the empirical offset to get NM Central state plane ft used in the CSV).  Elevations match
the CSV exactly.

### Points with station/offset (field navigation)

| Point | Elevation (ft) | Description | Station | Offset |
|---|---|---|---|---|
| 4009-37 | 5683.61 | Alum. Cap. Rebar | 33+53.80 | 50.68' LT |
| 250557 | 5678.80 | Plastic Cap Rebar | 34+18.69 | 72.15' LT |
| 250513 | 5682.38 | Brass Cap "3703-38" | 51+38.74 | 54.46' LT |
| 631 | 5710.57 | **NGS VCM 3D** | 122+72.23 | 75.56' LT |
| 250171 | 5757.37 | Brass Cap | 158+82.25 | 91.33' RT |
| 4009-430 | 5797.80 | **NGS VCM 3D ROD "Y 430"** | 249+43.08 | 170.35' RT |

NGS monuments (631 and 4009-430 / Y 430) have published datasheets on the NGS website and
can be independently verified before the flight.

### "Out of range" control points

Seven points (4009-25 through 4009-36) are flagged "Out of range" on the sheet — they're
outside the US 550 stationing but are still valid 3D control with full coordinates and
elevations.  These are likely on the NM 516 branch.  They appear correctly in the CSV and
are usable as GCPs.

| Point | Elevation (ft) | Description |
|---|---|---|
| 4009-25 | 5668.37 | Brass Cap "3703-25" |
| 4009-30 | 5640.96 | Alum. Cap. Rebar "3703-30" |
| 4009-32 | 5620.46 | Alum. Cap. Rebar "3703-32" |
| 4009-33 | 5614.53 | Alum. Cap. Rebar "3703-33" |
| 4009-34 | 5624.25 | Alum. Cap. Rebar "3703-34" |
| 4009-35 | 5643.14 | Alum. Cap. "4009-35" |
| 4009-36 | 5687.02 | Alum. Cap. "4009-36" |

### Notes

- This is **Sheet 3 of 0** — there are additional sheets covering other areas or control
  not shown here.  Ask the customer for all sheets if available.
- Survey units: US Survey Feet.  Basis of elevations: NMDOT control map (presumably NAVD88).
- The alignment geometry points (B.O.P. through F.O.P., 19 points) appear in the CSV with
  blank elevation — they are horizontal-only and not usable as GCPs.

---

## Questions to resolve before the flight

- [ ] What altitude is Isaiah planning to fly?  (Must be ≤ 200 ft for 0.1 ft H target)
- [ ] Does the customer's accuracy spec apply to GCPs (control) or check points
  (independent validation)?  These are different numbers.
- [ ] Is spray paint acceptable on the pavement adjacent to monuments, or will the
  customer object?
- [ ] Is WebODM available for processing, or is Pix4D required by the customer?
