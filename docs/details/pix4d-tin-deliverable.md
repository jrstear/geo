# Pix4D TIN Deliverable Format Reference

Reference for what Pix4D matic exports as a TIN deliverable, derived
from the aztec highway job's actual output. Used to size the geo
pipeline's CAD-deliverable plan (bead [`geo-ctr`](../../).

Source: `/Volumes/Stratus Files/survey/BSN/aztec highway/aztec highway matic/exports/`
(Pix4Dmatic 2.0.2, exported 2026-03-17).

## File inventory

| File | Size | What it is |
|------|-----:|------------|
| `aztec-TIN.xml` | 43 MB | **LandXML 1.2 TIN surface** — 223,092 points + 446,143 faces |
| `aztec-contour_lines.dxf` | 20 MB | Contour lines as CAD polylines |
| `aztec-contour_lines.prj` | 1.6 KB | WKT projection sidecar |
| `aztec highway matic-orthomosaic.tiff` | 7.9 GB | Full-res ortho |
| `aztec highway matic-orthomosaic-lores.tiff` | 591 MB | Preview ortho |
| `aztec highway matic-quality_report.pdf` | 872 KB | Pix4D's QA report |
| `areas-of-concern.dxf` (+`.prj`) | 26 KB | Surveyor annotation overlay |

The two formats civil clients consume directly are **`.xml` (TIN)** and
**`.dxf` (contour lines)**. The orthomosaic + quality report are
auxiliary.

## TIN format: LandXML 1.2

The TIN is **not** a raw DXF mesh — it's
[LandXML 1.2](http://www.landxml.org/), the open civil-engineering
interchange format. AutoCAD Civil 3D, Bentley OpenRoads / MicroStation,
Trimble Business Center all import LandXML natively as a `Surface`
object (not as a generic mesh).

### Document structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2"
         version="1.2" date="2026-03-17" time="13:59:19">
  <Application name="PIX4Dmatic" version="2.0.2"/>
  <Project name="aztec highway matic"/>
  <Units>
    <Imperial areaUnit="squareFoot" linearUnit="USSurveyFoot"
              volumeUnit="cubicYard" temperatureUnit="fahrenheit"
              pressureUnit="inchHG"/>
  </Units>
  <CoordinateSystem ogcWktCode="<full WKT compound CRS in one line>"/>
  <Surfaces>
    <Surface name="TIN">
      <Definition surfType="TIN">
        <Pnts>
          <P id="1">2142250.716796875 1145547.845703125 5765.978576660156</P>
          <!-- ... 223,092 points total ... -->
        </Pnts>
        <Faces>
          <F>p1 p2 p3</F>
          <!-- ... 446,143 triangles total ... -->
        </Faces>
      </Definition>
    </Surface>
  </Surfaces>
</LandXML>
```

### CRS encoding

The full PROJ-style WKT is embedded in the `CoordinateSystem.ogcWktCode`
attribute. For aztec, this is a **compound CRS**:

- Horizontal: `EPSG:2258` — NAD83 / New Mexico Central (US survey foot)
- Vertical: `EPSG:6360` — NAVD88 height (US survey foot), GEOID18

> Note: Pix4D emits the **NSRS2007** realization (2258), not the
> **NAD83(2011)** version (6529) we use in the geo pipeline. The
> horizontal coordinates differ by a fraction of a foot regionally and
> are interchangeable for survey-grade work — but emitters that need to
> match Pix4D's CRS code exactly should write 2258, not 6529.

The `<Units>` block is redundant with the WKT but is what older importers
read; emit both.

### Geometry conventions

- **Points** are XYZ in projected coordinates, all three values in survey
  feet (matching the WKT). Each point has an integer `id` attribute
  referenced by face triangles.
- **Faces** are triangles with three space-separated point IDs. The TIN
  is **2.5D Delaunay over (X, Y)**, not a 3D mesh — one Z per (X, Y)
  position. Faces can be omitted if the consumer software can
  retriangulate, but providing both is the proper format and is what
  Pix4D does.
- **Face/point ratio**: 446,143 / 223,092 ≈ 2.0 — exactly the ratio
  Delaunay triangulation produces on a planar Euler-characteristic surface.

### Density and simplification

- **38 points/acre** ≈ 1 point per 1100 sq ft.
- **Z range**: 5744–5893 ft (149 ft span over a ~6 km corridor); p1=5749,
  p99=5850.
- No outliers consistent with tree canopies — the surface is bare-earth
  (or near-bare-earth on this near-treeless desert site).
- The TIN is **heavily simplified** vs the upstream dense point cloud
  (which is 100–1000× denser). Pix4D adaptively decimates flat areas and
  preserves detail in changing-elevation areas. Civil clients want clean
  TINs, not noise — match this density target rather than dumping every
  ground-classified point.

## Contour format: AutoCAD DXF

The contour deliverable is a vanilla DXF (AutoCAD 2010 dialect, header
`AC1021`) holding contour polylines. Sidecar `.prj` carries the same
compound CRS WKT as the LandXML TIN.

The bbox declared in the DXF header (`$EXTMIN`/`$EXTMAX`) matches the
TIN extent: easting 1,145,478 — 1,158,226 ft, northing 2,141,938 —
2,161,996 ft (note Pix4D writes coords in `northing, easting` order in
the DXF — opposite of the LandXML — verify when emitting).

A `.prj` sidecar in the same directory as the `.dxf` is conventional for
GIS interop and is what Pix4D emits. Generate one alongside the DXF.

## Implications for the geo pipeline

The current `geo-ctr` bead description called for `.dxf for CAD
delivery` — that's correct for **contours** but **wrong for the TIN**.
The TIN deliverable must be **LandXML 1.2** if it's to land naturally in
the customer's Civil 3D / OpenRoads / Trimble workflow.

### TIN emitter

No off-the-shelf tool emits LandXML directly:

- GDAL: no LandXML writer.
- PDAL: no LandXML writer.
- ODM: no LandXML writer.

A small Python emitter (~50 lines) is the right move. Inputs:
ground-classified point set + Delaunay-triangulated faces (numpy/scipy or
`pdal writers.tin`). Output: the LandXML structure shown above with the
correct `<CoordinateSystem ogcWktCode>` for the job.

### Contour emitter

Standard tooling chain:

```bash
# 1. raster contours (GeoPackage, ESRI Shapefile, or directly to DXF)
gdal_contour -i 1.0 -a ELEV input.dtm.tif contours.gpkg

# 2. convert to AutoCAD DXF
ogr2ogr -f DXF contours.dxf contours.gpkg

# 3. emit a .prj sidecar matching contours.dxf
gdalsrsinfo -o wkt input.dtm.tif > contours.prj
```

ODM has had a `--contours` flag since v3.5 that does this internally
(emitting `odm_dem/dtm_contours.gpkg`). Adding `--contours` to
`ODM_FLAGS` and post-converting the `.gpkg` to `.dxf` in `package.py` is
the lowest-cost first step toward the deliverable.

### Density target

Aim for ~30–40 ground points/acre after simplification, matching what
Pix4D produces. Dumping all ODM ground-classified points (~100s of
thousands per corridor) into a TIN would yield a file far too large and
detailed for civil-engineering consumption.

## Cross-references

- bead [`geo-ctr`](../) — Contour + TIN export (this is the implementation work)
- bead [`geo-h6t`](../) — PDAL pipeline wrapper (ground classification → DTM raster)
- bead [`geo-utg`](../) — research: ground classification tool comparison
- [`odm-output-options.md`](odm-output-options.md) — ODM flags including `--contours`
