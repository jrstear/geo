# Coordinate Systems in the Survey-to-Delivery Pipeline

This guide explains which coordinate system is in use at every step, why, and how
to load data correctly in QGIS.  It is the authoritative reference for resolving
"why don't these layers align?" questions.

---

## 1. The family tree

Coordinate systems have two independent dimensions: the **datum** (a model of the
Earth's shape and orientation) and the **coordinate representation** (how you
express a location — angles vs. flat-plane distances).

```
Datum
├── WGS84  ─────────────────────────────────────────────────────────────────┐
│   Used by GPS satellites and most global software.                        │
│                                                                           │
├── NAD83(2011)   ──────────────────────────────────────────────────────────┤
│   North American Datum 1983, 2011 realization (most current).             │
│   Used by Emlid RS3 with NTRIP and by NGS control monuments.              │
│                                                                           │
└── NAD83(NSRS2007)  ────── agrees w/2011 to ~1cm in NM  ───────────────────┘
    Earlier realization of NAD83.  Coordinates differ from NAD83(2011)
    by millimetres to a few centimetres.  Treated as identical in practice.

Coordinate representation (how a location is expressed)
├── Geographic — angles (degrees)
│   └── EPSG:4326   WGS84 lat/lon               e.g. (-107.934°, 36.881°)
│       Emlid CSV "Longitude" / "Latitude" columns are in this form.
│       NOT directly usable in QGIS as a projected layer; needs reprojection.
│
└── Projected — flat plane, linear units
    ├── UTM Zone 13N — metres
    │   └── EPSG:32613/WGS 84/UTM zone 13N e.g. (237,992 m E, 4 085,125 m N)
    │       ODM control files, rmse_calc, gdalwarp intermediate.
    │       Central meridian 105 °W.  Covers ~103–111 °W.
    │
    └── New Mexico Central (LCC) — US survey feet
        ├── EPSG:6529   NAD83(2011)   e.g. (1,147,722 ft E, 2,144,276 ft N)
        │   Emlid RS3 native output ("Easting" / "Northing" columns).
        │   CS name in Emlid CSV: "NAD83(2011) / New Mexico Central (ftUS) (5)"
        │
        └── EPSG:3618   NAD83(NSRS2007)
            Customer control monuments after offset removal, QGIS review layer,
            reprojected orthophoto for review and delivery.
            *** For this site, EPSG:6529 and EPSG:3618 give the same numbers
                to within centimetres.  Treat them as interchangeable. ***

Special: Customer Design Grid
    Not an EPSG code.  It is EPSG:3618 with a constant translation applied:
        Design E = state_plane_E + 1,546,702.929 ft
        Design N = state_plane_N −     3,567.471 ft
    Appears only in the .dc file (raw 69KI/81CB records) and in
    Customer-delivery outputs.  Never used inside the processing pipeline.
```

### Quick recognition guide

| Numbers look like | Units | What it is |
|---|---|---|
| −107.9, 36.9 | degrees | Geographic WGS84/NAD83 (EPSG:4326) — Emlid Lat/Lon cols |
| 237,000 – 260,000 E, 4,080,000 – 4,100,000 N | metres | UTM 13N EPSG:32613 — ODM files |
| 1,100,000 – 1,200,000 E, 2,120,000 – 2,170,000 N | US survey ft | NM Central state plane EPSG:6529/3618 — Emlid E/N cols, DC-corrected points |
| 2,600,000 – 2,750,000 E, 2,140,000 – 2,170,000 N | US survey ft | Customer design grid (raw .dc) — NOT state plane |
---

## 2. CRS at each pipeline stage

### Stage 1 — Receive from customer (.dc file)

| Item | CRS | Notes |
|---|---|---|
| `.dc` 69KI / 81CB records (control points) | Customer design grid | Raw easting ~2.6–2.75 M ft; apply offset to get state plane |
| `.dc` 66KI / 66FD records (base stations) | EPSG:4326 (lat/lon) | Converted to state plane by `transform.py dc` |
| `{job}_points.csv` (output of `transform.py dc`) | EPSG:6529 | Design-grid offset removed; ready for Emlid and QGIS |
| `transform.yaml` (output of `transform.py dc`) | — | Job CRS + design-grid shift params |

**To load in QGIS**: Delimited Text, X = `easting_ft`, Y = `northing_ft`,
Z = `elevation_ft`, CRS = **EPSG:6529** (or EPSG:3618 — same numbers for this site).

---

### Stage 2 — Field survey (Emlid RS3)

| Item | CRS | Notes |
|---|---|---|
| Emlid `Easting` / `Northing` columns | EPSG:6529 | State plane ft — use these in QGIS |
| Emlid `Longitude` / `Latitude` columns | EPSG:4326 (degrees) | Geographic — do NOT use as QGIS X/Y with a projected CRS |
| Emlid `Elevation` column | NAVD88 (ft, GEOID18) | Orthometric height |
| Emlid `Ellipsoidal height` column | WGS84 ellipsoid (ft) | Ellipsoidal height — used by `convert_coords.py` |

**To load in QGIS**: Delimited Text, X = `Easting`, Y = `Northing`,
Z = `Elevation`, CRS = **EPSG:6529**.

> **Common mistake**: QGIS defaults to importing Latitude/Longitude as X/Y because
> those column names look like coordinates.  This places all points near the state
> plane origin (~0 ft, ~0 ft) — miles from the highway.  Always verify you are
> using Easting/Northing.

The Emlid localisation check: the `m`-suffix points (e.g. `12m`, `15m`) are
re-occupations of Customer monuments taken after localisation is applied.  Their
residuals vs `F100340_{job}_points.csv` should be < 0.2 ft; this confirms the
Emlid is in the same coordinate system as the DC file.

---

### Stage 3 — GCP tagging (sight.py → GCPEditorPro → transform.py split)

| Item | CRS | Notes |
|---|---|---|
| `{job}.csv` (filtered Emlid survey) | EPSG:6529 | Input to `sight.py` |
| `{job}.txt` (output of `sight.py`) | EPSG:6529 | GCPEditorPro input |
| `{job}_confirmed.txt` (GCPEditorPro export) | EPSG:6529 | Z is NAVD88 feet |
| `transform.yaml` (output of `transform.py dc`) | — | CRS and shift params for the job |
| `gcp_list.txt` (output of `transform.py split`) | **EPSG:32613** | X/Y metres, Z ellipsoidal metres |
| `chk_list.txt` (output of `transform.py split`) | **EPSG:32613** | Same as above |

`transform.py split` performs three conversions in one step:
1. EPSG:6529 (ft) → EPSG:32613 (m) via pyproj (source CRS read from `transform.yaml`)
2. NAVD88 orthometric height (ft) → ellipsoidal height (m)
3. Splits GCP- and CHK- prefixed points into separate files

**Why EPSG:32613 for ODM?**  The NM Central CRS (EPSG:3618/6529) defines only
horizontal units (US survey feet) and leaves vertical units ambiguous.  ODM
assumes Z is in metres for any 2D CRS, causing a ~3.28× vertical scale error when
Z is in feet.  EPSG:32613 is a 3D CRS: all three axes are in metres with no
ambiguity.

---

### Stage 4 — ODM processing

| Item | CRS | Notes |
|---|---|---|
| `gcp_list.txt` | EPSG:32613 | Passed to ODM |
| `opensfm/reconstruction.json` | ODM local frame | local_X = UTM_E − ref_UTM_E (metre offsets from UTM origin) |
| `odm_orthophoto/odm_orthophoto.original.tif` | EPSG:32613 | Real output; `.tif` alone is a stub |
| `odm_georeferencing/coords.txt` | — | Contains UTM origin used by reconstruction.json |

> **Note on `odm_orthophoto.tif` vs `.original.tif`**: when `--optimize-disk-space`
> is used, ODM replaces the working `.tif` with a 429 KB stub.  Always use
> `odm_orthophoto.original.tif` (the real file, ~1 GB).

---

### Stage 5 — Accuracy QC (rmse_calc.py)

| Item | CRS | Notes |
|---|---|---|
| `chk_list.txt` | EPSG:32613 | Ground truth X/Y/Z |
| `reconstruction.json` positions | ODM local frame → EPSG:32613 via `ref_UTM + local_offset` | `rmse_calc.py` handles this conversion |
| RMSE output | metres | Compare to GSD (~0.03 m at 250 ft AGL) |

The conversion from ODM local frame to EPSG:32613 is a direct addition:
`UTM_E = ref_UTM_E + local_X`, **not** a flat-earth ENU formula.  See
`accuracy_study/rmse_calc.py` for details and `stratus/aztec3/rmse_results.md`
for the bug that was fixed when the flat-earth formula was used.

---

### Stage 6 — QGIS review and delivery

| Item | CRS | Notes |
|---|---|---|
| `odm_orthophoto.original_3618.tif` | EPSG:3618 | Output of `packager/reproject_deliverable.py`; for review only |
| `odm_orthophoto.original_3618_cog.tif` | EPSG:3618 | COG version; fast QGIS loading |
| `F100340_{job}_points.csv` | EPSG:3618 | DC control monuments — load X=easting_ft, Y=northing_ft |
| Emlid `{job}.csv` | EPSG:6529 | Survey points — load X=Easting, Y=Northing |
| Deliverable orthophoto (Customer) | Customer design grid | Apply +1 546 702.929 ft E / −3 567.471 ft N via `package.py --shift-x/y` |
| Deliverable contours / TIN (Customer) | Customer design grid | Same shift, applied by `packager/package.py` |

**To load ortho in QGIS**: drag-and-drop or Layer → Add Raster.  CRS is embedded
in the GeoTIFF header.  Set the **project** CRS to EPSG:3618 for all layers to
display without on-the-fly reprojection overhead.

---

## 3. Full CRS flow (summary)

```
.dc file (design grid ft)
    ↓ transform.py dc  [remove Customer offset; auto-detect CRS from .dc]
{job}_points.csv  (EPSG:6529 ft)  +  transform.yaml
    ↓ Emlid RS3 field survey  [localize to monuments]
{job}.csv Easting/Northing  (EPSG:6529 ft  ≈  EPSG:3618)
    ↓ sight.py + GCPEditorPro
{job}_confirmed.txt  (EPSG:6529 ft, NAVD88 Z ft)
    ↓ transform.py split  [project + geoid; reads transform.yaml for field_crs]
gcp_list.txt / chk_list.txt  (EPSG:32613 m, ellipsoidal Z m)
    ↓ ODM
odm_orthophoto.original.tif  (EPSG:32613 m)
    ↓ packager/reproject_deliverable.py  [gdalwarp, pixel resample]
odm_orthophoto.original_3618.tif  (EPSG:3618 ft)  ← QGIS review
    ↓ package.py --shift-x +1546702.929 --shift-y -3567.471
deliverable orthophoto  (Customer design grid ft)  ← customer delivery
```

---

## 4. QGIS cheat sheet

| Layer | Add as | X field | Y field | CRS |
|---|---|---|---|---|
| `F100340_*_points.csv` | Delimited Text | `easting_ft` | `northing_ft` | EPSG:3618 |
| Emlid `*.csv` | Delimited Text | `Easting` | `Northing` | EPSG:6529 |
| `odm_orthophoto.original_3618_cog.tif` | Raster | — | — | embedded (EPSG:3618) |
| `aztec_control.txt` / `gcp_list.txt` | Delimited Text | col 1 (X m) | col 2 (Y m) | EPSG:32613 |

**Project CRS**: set to EPSG:3618 for review sessions.  All layers except the
ODM control files are already in EPSG:3618/6529 (identical numbers); the COG
ortho is reprojected to EPSG:3618.  QGIS on-the-fly reprojection from EPSG:6529
to EPSG:3618 is a sub-centimetre datum shift — visually invisible.

**If layers do not align**: check the displayed coordinates when you click a
point.  They should be in the range E = 1,100,000 – 1,200,000 ft,
N = 2,100,000 – 2,200,000 ft for the Aztec site.  If you see degree-sized
numbers (−107, 36) you are using the Longitude/Latitude columns instead of
Easting/Northing.
