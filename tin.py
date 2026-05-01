#!/usr/bin/env python3
"""tin.py — parse a LandXML TIN and sample Z at (X, Y) queries.

Used by rmse.py via --tin PATH for a third accuracy axis: TIN-sampled Z at
each surveyed target's XY, residual against surveyed Z. Toolchain-agnostic:
works on Pix4Dmatic-emitted LandXML, ODM-future LandXML (geo-btcl), or any
LandXML 1.2 surface conforming to <Surface>/<Definition>/<Pnts><P>+<Faces><F>.

Auto-detects the producing toolchain from the <Application> element so the
rmse.html report places the tin_dZ column under the right tool's sub-section
in the ODM | Pix4D hierarchy.

Coordinate convention note (Pix4Dmatic-confirmed on aztec):
LandXML 1.2 point text is 'Northing Easting Elevation' even when the
declared CoordinateSystem axes are 'easting=ORDER[1]'. We swap to (E, N, Z)
so the (X, Y) returned matches what callers expect for projected coords.

CLI:
    python tin.py path/to/surface.xml [--sample X Y]
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Triangle:
    """A single TIN triangle. Vertices stored as (E, N, Z) tuples."""
    v1: Tuple[float, float, float]
    v2: Tuple[float, float, float]
    v3: Tuple[float, float, float]
    min_x: float = field(init=False)
    max_x: float = field(init=False)
    min_y: float = field(init=False)
    max_y: float = field(init=False)

    def __post_init__(self):
        self.min_x = min(self.v1[0], self.v2[0], self.v3[0])
        self.max_x = max(self.v1[0], self.v2[0], self.v3[0])
        self.min_y = min(self.v1[1], self.v2[1], self.v3[1])
        self.max_y = max(self.v1[1], self.v2[1], self.v3[1])


@dataclass
class TIN:
    triangles:           List[Triangle]
    application:         Optional[str]  # e.g. 'PIX4Dmatic'
    application_version: Optional[str]
    crs_wkt:             Optional[str]
    crs_epsg_horizontal: Optional[int]
    crs_epsg_vertical:   Optional[int]
    units:               str            # 'ft' (US survey foot) or 'm'
    source_path:         str
    bbox:                Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


def _strip_namespaces(root: ET.Element) -> None:
    """Strip XML namespace prefixes in-place so element queries don't need them."""
    for elem in root.iter():
        if elem.tag.startswith('{'):
            elem.tag = elem.tag.split('}', 1)[1]


def _parse_crs(wkt: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Pull horizontal (PROJCRS) and vertical (VERTCRS) EPSG codes from a
    LandXML compound-CRS WKT string. Uses pyproj when available (correctly
    handles nested CRS lookups); falls back to regex picking the LAST EPSG
    in the PROJCRS section (skips the inner BASEGEOGCRS/datum code).
    Returns (horiz, vert)."""
    if not wkt:
        return (None, None)
    try:
        from pyproj import CRS as _CRS
        crs = _CRS.from_wkt(wkt)
        if crs.is_compound:
            subs = crs.sub_crs_list
            h = subs[0].to_epsg() if subs else None
            v = subs[1].to_epsg() if len(subs) > 1 else None
            return (h, v)
        return (crs.to_epsg(), None)
    except Exception:
        pass
    # Fallback: split into PROJCRS and VERTCRS halves; LAST EPSG ID in each
    # is the right one (skips inner method/parameter EPSG codes).
    parts = re.split(r'\bVERTCRS\[', wkt, maxsplit=1)
    proj_section = parts[0]
    vert_section = parts[1] if len(parts) > 1 else ''
    h_ids = re.findall(r'ID\["EPSG",(\d+)\]', proj_section)
    v_ids = re.findall(r'ID\["EPSG",(\d+)\]', vert_section)
    return (int(h_ids[-1]) if h_ids else None,
            int(v_ids[-1]) if v_ids else None)


def parse_landxml_tin(path: Path) -> TIN:
    """Parse a LandXML 1.2 TIN surface from `path`.

    Validates expected structure and raises ValueError with a clear message on
    malformed input. Handles namespaced and bare-tag variants.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    _strip_namespaces(root)

    app = root.find('Application')
    application = app.get('name') if app is not None else None
    application_version = app.get('version') if app is not None else None

    cs = root.find('CoordinateSystem')
    crs_wkt = cs.get('ogcWktCode') if cs is not None else None
    crs_h, crs_v = _parse_crs(crs_wkt)

    units = 'ft' if root.find('.//Imperial') is not None else 'm'

    pnts_elem = root.find('.//Pnts')
    if pnts_elem is None:
        raise ValueError(f"{path}: no <Pnts> element found in LandXML")
    points: dict = {}
    for p in pnts_elem.findall('P'):
        pid = p.get('id')
        coords = (p.text or '').strip().split()
        if not pid or len(coords) < 3:
            continue
        try:
            n, e, z = float(coords[0]), float(coords[1]), float(coords[2])
        except ValueError:
            continue
        # LandXML 1.2 default order is N E Z; convert to (E, N, Z).
        points[pid] = (e, n, z)

    if not points:
        raise ValueError(f"{path}: no usable <P> entries found")

    faces_elem = root.find('.//Faces')
    if faces_elem is None:
        raise ValueError(f"{path}: no <Faces> element found")
    triangles: List[Triangle] = []
    for f in faces_elem.findall('F'):
        ids_text = (f.text or '').strip()
        if not ids_text:
            continue
        ids = ids_text.replace(',', ' ').split()
        if len(ids) < 3:
            continue
        v1 = points.get(ids[0])
        v2 = points.get(ids[1])
        v3 = points.get(ids[2])
        if v1 is None or v2 is None or v3 is None:
            continue
        triangles.append(Triangle(v1, v2, v3))

    if not triangles:
        raise ValueError(f"{path}: no usable <F> faces found")

    xs = [t.min_x for t in triangles] + [t.max_x for t in triangles]
    ys = [t.min_y for t in triangles] + [t.max_y for t in triangles]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    return TIN(
        triangles=triangles,
        application=application,
        application_version=application_version,
        crs_wkt=crs_wkt,
        crs_epsg_horizontal=crs_h,
        crs_epsg_vertical=crs_v,
        units=units,
        source_path=str(path),
        bbox=bbox,
    )


def sample_z(tin: TIN, x: float, y: float) -> Optional[float]:
    """TIN-interpolated Z at (x, y) in the TIN's native CRS, or None if outside.

    Linear scan with axis-aligned bbox pre-filter; barycentric interpolation
    inside the matching triangle. For ~10^5 triangles and ~10^2 sample points
    this completes in seconds without a spatial index. Add a grid bucket if
    larger TINs become routine.
    """
    if x < tin.bbox[0] or x > tin.bbox[2] or y < tin.bbox[1] or y > tin.bbox[3]:
        return None
    for t in tin.triangles:
        if x < t.min_x or x > t.max_x or y < t.min_y or y > t.max_y:
            continue
        x1, y1, z1 = t.v1
        x2, y2, z2 = t.v2
        x3, y3, z3 = t.v3
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-15:
            continue  # degenerate triangle
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        c = 1.0 - a - b
        eps = 1e-9
        if -eps <= a <= 1.0 + eps and -eps <= b <= 1.0 + eps and -eps <= c <= 1.0 + eps:
            return a * z1 + b * z2 + c * z3
    return None


def detect_tool(tin: TIN) -> str:
    """Identify the producing toolchain from <Application name="...">.

    Returns 'pix4d', 'odm', or 'unknown'. Used by rmse.py to place the
    tin_dZ column under the right tool's sub-section in the report.
    """
    name = (tin.application or '').lower()
    if 'pix4d' in name:
        return 'pix4d'
    if 'odm' in name or 'opendrone' in name:
        return 'odm'
    return 'unknown'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('tin', type=Path, help='LandXML TIN file')
    ap.add_argument('--sample', type=float, nargs=2, metavar=('X', 'Y'),
                    help='If provided, also print the TIN-interpolated Z at (X, Y).')
    args = ap.parse_args()

    if not args.tin.exists():
        sys.exit(f"ERROR: file not found: {args.tin}")

    tin = parse_landxml_tin(args.tin)
    print(f"TIN: {args.tin.name}")
    print(f"  Application: {tin.application} v{tin.application_version} "
          f"(detected as: {detect_tool(tin)})")
    print(f"  CRS: horizontal=EPSG:{tin.crs_epsg_horizontal}  "
          f"vertical=EPSG:{tin.crs_epsg_vertical}  units={tin.units}")
    print(f"  Triangles: {len(tin.triangles):,}")
    print(f"  Bbox: ({tin.bbox[0]:.2f}, {tin.bbox[1]:.2f}) -> "
          f"({tin.bbox[2]:.2f}, {tin.bbox[3]:.2f})  ({tin.units})")
    if args.sample:
        x, y = args.sample
        z = sample_z(tin, x, y)
        if z is None:
            print(f"  Sample at ({x:.3f}, {y:.3f}): outside TIN")
        else:
            print(f"  Sample at ({x:.3f}, {y:.3f}): Z = {z:.3f} ({tin.units})")


if __name__ == '__main__':
    main()
