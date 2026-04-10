# GCP Placement Advisor

**Epic bead:** geo-sge

This document describes the design and implementation plan for the GCP Placement
Advisor — a tool that takes a user-drawn flight corridor and suggests optimal GCP
target locations for field placement before a drone survey.

---

## Problem

Isaiah currently selects GCP target locations by feel: he walks or drives the site,
picks visually open spots near the road, and sets targets.  This works but leaves
structural quality to chance — the targets may not anchor the photogrammetric block
as effectively as a deliberately chosen set would.

The placement advisor automates the geometry that drives `emlid2gcp.py`'s ordering
algorithm, running it *forward* (place N points to maximize structural value) rather
than *backward* (order N existing points by their structural value).  It also layers
in two additional signals: terrain openness (avoid setting targets under tree canopy)
and, later, road accessibility (prefer spots reachable without a long hike).

The output is a KML file the surveyor loads into Google Maps or Emlid Flow before
leaving for the field.

---

## Optimality criteria

In priority order:

1. **Max insertion** — the set of N points that locks down the photogrammetric block
   as quickly as possible.  Defined by the same structural geometry used in
   `emlid2gcp.py`: distal anchors first, elevation extremes if terrain is hilly,
   center pin, then perimeter-fill.
2. **Well-visible** — targets should be on open ground, not under canopy, so they
   are detectable in drone imagery.
3. **Easy to get to** — targets near a driveable road reduce field time.  Less
   critical for road surveys (all points are inherently near the road) but important
   for area surveys.

---

## Relationship to `emlid2gcp.py`

`emlid2gcp.py`'s `_sort_gcps()` solves the *ordering* problem: given a set of
surveyed GCP positions, rank them by structural insertion value.  The placement
advisor solves the dual *placement* problem: given a flight area polygon, generate
the positions themselves.  The underlying geometry is identical — greedy
farthest-point sampling from the polygon — so the two tools share the same slot
definitions and thresholds.

The `--z-threshold` parameter (default 0.05) that controls Z-slot activation in
`emlid2gcp.py` applies here too; the placement advisor uses elevation data to decide
whether to activate Z-critical slots 3 and 4.

---

## Future context: flight path planning

A future workstream (geo-8sk) will address optimal flight path planning, including
varying gimbal angles for oblique capture over areas of significant vertical relief.
This may interact with GCP placement — oblique imagery changes which target locations
provide the most parallax for vertical accuracy — but the placement advisor is
designed to be useful independently of flight path and can be updated later if the
interaction proves significant.

---

## Implementation plan

### geo-gnn — Core placement geometry (P2)

**Status:** ready (no blockers)

The foundational module.  Given a GeoJSON/KML polygon and a target count N, generates
N ranked candidate locations using greedy farthest-point sampling:

| Slot | Selection rule |
|------|---------------|
| 1 | Most distal from polygon centroid |
| 2 | Most distal from slot 1 |
| 3 | Nearest to elevation maximum *(Z-critical; stubbed until geo-lvy)* |
| 4 | Nearest to elevation minimum *(Z-critical; stubbed until geo-lvy)* |
| 5 | Nearest to centroid (center pin) |
| 6–N | Perimeter-fill: maximise minimum distance from all placed points |

Deliverable: Python module (`TargetSighter/placement.py`) + CLI.  No external API
calls in this bead — pure geometry.

Test: apply to a corridor polygon derived from the ghostrider gulch dataset and
compare placement against confirmed GCPs.

---

### geo-lvy — Elevation integration (P2)

**Blocked by:** geo-gnn

Fetches elevation for all placement candidates (and a dense corridor sample for
Z-extremes) using the **Google Elevation API** (simple REST, free tier easily
sufficient for 10–50 points per run).

- Computes Z-span vs. XY diagonal.
- Activates Z-critical slots 3 and 4 when `z_span > z_threshold × xy_diagonal`
  (same logic as `emlid2gcp.py`).
- Relocates slots 3 and 4 to the corridor's local high and low points.

---

### geo-l1w — Visibility scoring (P2)

**Blocked by:** geo-gnn

Scores each candidate by terrain openness so targets are not placed under dense
canopy.

**Phase 1 — US coverage (NLCD):**
Query the USGS National Land Cover Database (free, 30 m resolution) for each
candidate location.  NLCD classes 41, 42, 43 (deciduous/evergreen/mixed forest) are
penalised; shrub, grassland, and developed areas are preferred.  Output: an
`openness_score` in [0, 1].  Candidates below a configurable threshold are flagged
but not auto-moved — the surveyor decides.

**Phase 2 — Global coverage (future):**
Replace or supplement NLCD with Sentinel-2 NDVI via a public tile service.

---

### geo-bo0 — KML export (P2)

**Blocked by:** geo-gnn, geo-lvy, geo-l1w

Exports candidate locations as a KML file for Google Maps or Emlid Flow.

Each placemark includes:
- **Name:** structural role, e.g. `GCP-1 (distal anchor)`, `GCP-5 (center pin)`
- **Description:** elevation (ft/m), openness score, distance to road (once
  geo-o19 is complete)
- **Icon color:** red / amber / green by structural priority (matching GCPEditorPro
  badge scheme)

This is the first deliverable Isaiah can use in the field.

---

### geo-9dk — WebODM UI (P3)

**Blocked by:** geo-bo0

Adds a **Plan GCPs** tab to the auto-gcp WebODM plugin.

**Design principle:** target RMSE is the primary user input — not raw altitude.
Altitude is derived from the RMSE target and camera model (via the GSD chain:
`altitude = RMSE_target × focal_length × image_width_px / (1.5 × sensor_width_mm)`).
This matches how surveyors think ("I need 0.1 ft") rather than requiring them to know
the GSD math.

Primary user inputs:
- **Target RMSE** (ft or m) — drives altitude recommendation
- **Camera model** — determines sensor/focal specs for GSD calculation
- **Max flight time per session** (battery count × endurance)
- **Image count budget** (affects upload time and processing cost)
- **Flight corridor** — drawn on map or uploaded as KML/GeoJSON

Derived and displayed (not entered by user):
- Recommended altitude AGL
- Estimated image count at that altitude + overlap
- Estimated flight time and battery swap count
- Estimated processing cost (WebODM: free; Pix4D: $/image estimate)

Buttons:
- **Suggest GCPs** — runs geo-gnn + geo-lvy + geo-l1w, displays candidates as
  color-coded map pins; geo-5fh flags any with < 7 estimated images
- **Export KML** — triggers geo-bo0 output for download

See also: geo-8sk (flight path planning) for the fuller optimization framing.

---

### geo-o19 — Road accessibility scoring (P3)

**Blocked by:** geo-9dk

For each candidate, computes distance to the nearest driveable road using the
**OSM Overpass API** (free, no API key required).  Surfaces the distance in the
WebODM UI map tooltip and in the KML placemark description.

In a future iteration this score could be used to nudge candidates toward the road
when a nearby open-ground alternative exists.

---

### geo-5fh — Flight coverage filter (P3)

**Blocked by:** geo-9dk

Using the altitude derived from the RMSE target (see geo-9dk), front/side overlap,
and camera FOV, estimates the number of images that will cover each GCP candidate
location.

- Candidates near the corridor ends (fewer overlapping swaths) may fall below the
  7-image target.
- Computes a recommended **end-anchor setback** distance so slots 1 and 2 land where
  coverage is adequate.
- Surfaces the RMSE/altitude trade-off explicitly: if the required altitude to hit
  RMSE target causes coverage shortfall, the tool shows whether the fix is (a) setback
  the end anchor (free), (b) increase overlap at corridor ends (more images), or
  (c) accept a slight altitude increase (relaxes RMSE, reduces images).
- Flags under-covered candidates in the UI; does not auto-move them.

---

## Dependency chain

```
geo-gnn  (core geometry)
├── geo-lvy  (elevation)     ─┐
├── geo-l1w  (visibility)     ├── geo-bo0  (KML export)
└────────────────────────────┘       │
                                     └── geo-9dk  (WebODM UI)
                                              ├── geo-o19  (roads)
                                              └── geo-5fh  (coverage)
```

Also related: **geo-8sk** (flight path planning, placeholder P4).

```
geo-8sk  (flight path planning — RMSE → altitude → overlap → image count → cost)
         feeds altitude recommendation into:
         └── geo-9dk  (WebODM UI — RMSE is primary input, altitude is derived)
                  └── geo-5fh  (coverage filter — altitude drives image count per GCP)
```

---

## Data sources

| Signal | Source | Cost | Coverage |
|--------|---------|------|----------|
| Elevation | Google Elevation API | Free tier (2 500 req/day) | Global |
| Vegetation / openness | USGS NLCD | Free | US only |
| Vegetation / openness (phase 2) | Sentinel-2 NDVI | Free | Global |
| Road network | OSM Overpass API | Free | Global |
| Map display | Google Maps JS API | Free tier | Global |
