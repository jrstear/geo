# Aztec NM — GCP Analysis and Flight Recommendations

**Job:** F100340 AZTEC
**Site:** L-shaped corridor — US 550 (NE of Aztec) + NM 516 (E-W through Aztec)
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

### The project is L-shaped — two corridors

The customer-provided KMZ covers **only US 550** (northeast of Aztec, ~6 miles).
The monument dataset covers **both roads**:

| Corridor | Monuments | Coordinates | KMZ provided? |
|---|---|---|---|
| US 550 | **5** (IDs 11–18) | lat 36.853–36.926, lon -107.967 to -107.904 | ✓ Yes |
| NM 516 | **13** (IDs 1–10) | lat 36.816–36.840, lon -108.056 to -107.977 | **No** |

With only 5 monuments on US 550, the customer's "55 control points" almost certainly
implies setting new targets in the field, not just using the existing caps.

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

Achievable RMSE is bounded by ground sample distance (GSD): one image pixel in
real-world units.  The practical floor is 1–2× GSD horizontal, 2–3× GSD vertical.

| GCPs | 100 ft AGL | 150 ft AGL | 200 ft AGL | 300 ft AGL | 400 ft AGL |
|---|---|---|---|---|---|
| 3 | 0.13 / 0.26 | 0.20 / 0.39 | 0.26 / 0.53 | 0.39 / 0.78 | 0.52 / 1.05 |
| 5 | **0.08 / 0.17 ✓** | 0.12 / 0.24 | 0.17 / 0.33 | 0.25 / 0.49 | 0.33 / 0.66 |
| 7 | **0.05 / 0.10 ✓** | **0.07 / 0.15 ✓** | **0.10 / 0.20 ✓** | 0.15 / 0.29 | 0.20 / 0.39 |
| 10 | **✓** | **✓** | **✓** | **0.10 / 0.20 ✓** | 0.13 / 0.26 |
| 15–18 | **✓** | **✓** | **✓** | **✓** | 0.13 / 0.26 |

*Values are horizontal / vertical RMSE in feet.  ✓ = meets 0.1 ft H / 0.3 ft V targets.*

**Critical finding: at 300 ft AGL or higher, no number of GCPs can hit the 0.1 ft
horizontal target.**  At 200 ft AGL, 7+ well-distributed GCPs get there.  The flight
altitude must be confirmed before the flight, not discovered afterward.

Beyond 10–15 GCPs the rows are identical — adding more monuments provides no further
RMSE improvement.  This is a hard limit set by physics (GSD), not by effort.

### 2. GCP distribution — more important than count

**For a linear corridor project, side-of-road alternation is the critical distribution
property.**  All GCPs on one side allows the block to tilt perpendicular to the road
no matter how many GCPs are used.  A minimum of 2 GCPs on each side per corridor
locks the cross-track axis.

The structural priority algorithm also selects for:
- Distal anchors (lock scale and orientation along the corridor)
- Center pin (prevents doming — one GCP near the midpoint of the block)
- Perimeter fill (minimize maximum gap)

With only 5 US 550 monuments and 13 NM 516 monuments, there is no redundancy —
every monument matters.  The customer's "55 targets" implies supplementing with
newly-placed colored-X targets.

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

## Revised top 10 — from 18 true survey monuments

Generated by `python/top10_gcps.py`.  Structure detail points excluded.
Output files: `results/top10_gcps.csv` and `results/top10_gcps.kml`.

| Priority | ID | Road | Side | Offset | Description | Elevation |
|---|---|---|---|---|---|---|
| 1 | 18 | US550 | RT | 385 ft | ALUM CAP | 5829.44 ft |
| 2 | 1 | NM516 | — | — | BRASS CAP 3703-211 | 5668.37 ft |
| 3 | 11 | NM516 | — | — | PLASTIC CAP RBR | 5695.66 ft |
| 4 | 3 | NM516 | — | — | ALUM CAP 3703-3211 | 5620.46 ft |
| 5 | 14 | US550 | RT | 126 ft | NGS VCM 3D Y 430 | 5797.80 ft |
| 6 | 15 | US550 | LT | 508 ft | ALUM CAP | 5786.20 ft |
| 7 | 8 | NM516 | — | — | ALUM CAP 4009-3711 | 5683.61 ft |
| 8 | 13 | NM516 | — | — | BRASS CAP | 5757.37 ft |
| 9 | 17 | US550 | LT | 515 ft | ALUM CAP | 5802.92 ft |
| 10 | 6 | NM516 | — | — | ALUM CAP 4009-3511 | 5643.14 ft |

KML: Red stars = RT (east/right side of US 550), Blue stars = LT (west/left side).
NM 516 monuments are labelled "—" for side — side relative to US 550 is not meaningful.

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
   - 300 ft AGL: 0.1 ft H is not achievable regardless of GCP count

3. **Clay pigeon target placement at monuments:**
   - 8 pigeons (4 arms × 2 touching), no center pigeon, leave cap exposed in gap
   - Arms push up to the cap edge but do not cover the cap face
   - Prioritize the top-10 monuments in the KML (red and blue stars)

4. **Add new targets for the gaps:**
   Only 5 US 550 monuments exist.  For adequate coverage and side balance, place
   2–4 new targets on the under-represented (northwest/left) side of US 550,
   spaced along the corridor.  Spray-paint X + pigeon arrangement.

5. **Tag all monuments + new targets.**
   Prioritize top-10 KML stars first — get 7+ confirmed images each before moving on.

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

- [ ] **Is the flight US 550 only, or does it include NM 516?**
  The provided KMZ covers only US 550.  The monument dataset covers both corridors.
  This is the most consequential open question.

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
