#!/usr/bin/env python3
"""transform.py — Coordinate transformation tool for the survey-to-delivery pipeline.

Subcommands
-----------
dc   Parse a Trimble .dc file → {job}_points.csv (state-plane) + transform.yaml
     transform.yaml captures the job's CRS and design-grid shift for downstream use.

split  Reproject + split GCPEditorPro {job}_confirmed.txt → gcp_list.txt + chk_list.txt (EPSG:32613)
       Reads transform.yaml (written by dc) to get field_crs and odm_crs automatically.

Typical sequence
----------------
  # 1. Before field survey — extract control monuments and record job CRS:
  python transform.py dc {customer}_{job}.dc --shift-x X --shift-y Y

  # 2. After GCPEditorPro tagging — reproject for ODM:
  python transform.py split {job}_confirmed.txt
  #    transform.yaml is found automatically in the same directory

transform.yaml — written by dc, read by split (and future sight.py and package.py --transform-yaml)
---------------------------------------------------------------------------
job: <job_name>
field_crs: EPSG:XXXX      # Emlid RS3 output CRS; also the GCPEditorPro header CRS
odm_crs: EPSG:32613       # ODM gcp/chk input CRS (always UTM 13N metres)
delivery_crs: EPSG:XXXX   # State-plane CRS for QGIS review and delivery (auto-detected)
design_grid:
  shift_x: <float>        # ft: add to state-plane E to get design-grid E
  shift_y: <float>        # ft: add to state-plane N to get design-grid N
  scale: 1.0              # Helmert scale (1.0 = pure translation)
  rotation_deg: 0.0       # Helmert rotation (0.0 = pure translation)
  helmert_residual_ft:    # max residual after fit (null until validated)
  anchor_x: <float>       # state-plane E of control centroid
  anchor_y: <float>       # state-plane N of control centroid

Shift convention (consistent with package.py --shift-x/y):
  state_E = design_E - shift_x     (dc subcommand, parsing .dc records)
  design_E = state_E + shift_x     (package.py, delivery output)

Example (Customer/Aztec job):
  python transform.py dc "F100340 AZTEC.dc" --shift-x 1546702.929 --shift-y -3567.471
  python transform.py split ~/stratus/aztec3/aztec3_confirmed.txt
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

FT_TO_M = 0.3048006096012192   # US survey foot (exact 1200/3937)
ODM_CRS  = "EPSG:32613"        # UTM 13N metres — always used for ODM

# FIPS State Plane zone → EPSG (feet units, NAD83(2011) or NSRS2007 variants)
# Keys: FIPS code.  Values: dict keyed on datum string found in C8NM line.
FIPS_TO_EPSG: dict[int, dict[str, int]] = {
    3001: {"NAD83(2011)": 6527, "default": 3619},  # NM East
    3002: {"NAD83(2011)": 6529, "default": 3618},  # NM Central
    3003: {"NAD83(2011)": 6528, "default": 3617},  # NM West
    # Extend as new job sites are encountered
}


# ---------------------------------------------------------------------------
# Minimal YAML writer/reader (no PyYAML dependency)
# ---------------------------------------------------------------------------

def _yaml_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        # Enough precision without scientific notation
        s = f"{v:.10g}"
        return s
    if isinstance(v, int):
        return str(v)
    # String: quote if it contains YAML special chars
    if any(c in str(v) for c in ': #{}[]|>&*!,'):
        return f'"{v}"'
    return str(v)


def write_yaml(path: Path, data: dict) -> None:
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for k2, v2 in v.items():
                lines.append(f"  {k2}: {_yaml_value(v2)}")
        else:
            lines.append(f"{k}: {_yaml_value(v)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict:
    """Read a simple two-level YAML file (no lists, no complex types)."""
    try:
        import yaml  # use PyYAML if available
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        pass

    data: dict = {}
    section: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  "):                # indented → sub-key
            if section is None:
                continue
            key, _, val = line.strip().partition(": ")
            val = val.strip().strip('"')
            if val == "null":
                data[section][key] = None
            elif val in ("true", "false"):
                data[section][key] = val == "true"
            elif "." in val:
                try:
                    data[section][key] = float(val)
                except ValueError:
                    data[section][key] = val
            else:
                try:
                    data[section][key] = int(val)
                except ValueError:
                    data[section][key] = val
        else:                                    # top-level key
            key, _, val = line.partition(": ")
            val = val.strip().strip('"')
            if not val:                          # section header (dict value)
                section = key.rstrip(":")
                data[section] = {}
            else:
                section = None
                data[key] = val
    return data


# ---------------------------------------------------------------------------
# .dc file helpers
# ---------------------------------------------------------------------------

def detect_crs_from_dc(dc_path: Path) -> Optional[str]:
    """Return EPSG string from the first C8NM line, or None if unrecognised."""
    with open(dc_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("C8NM"):
                continue
            # FIPS zone codes we care about are 4-digit numbers starting with 3
            m = re.search(r'\b(3\d{3})\b', line)
            if not m:
                continue
            fips = int(m.group(1))
            if fips not in FIPS_TO_EPSG:
                continue
            datum = "NAD83(2011)" if "NAD83(2011)" in line else "default"
            epsg = FIPS_TO_EPSG[fips].get(datum, FIPS_TO_EPSG[fips]["default"])
            return f"EPSG:{epsg}"
    return None


def extract_job_name(dc_path: Path) -> str:
    """Extract job name from 10NM record, falling back to filename."""
    with open(dc_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("10NM"):
                name = line[4:].strip()
                name = re.sub(r'\s+\d+\s*$', '', name).strip()
                if name:
                    return name.replace(" ", "_")
    return dc_path.stem.replace(" ", "_")


# ---------------------------------------------------------------------------
# .dc record parsers
# ---------------------------------------------------------------------------

def _parse_69ki(line: str, shift_x: float, shift_y: float) -> Optional[dict]:
    """Control monuments: design-grid coords, northing stored first."""
    if not line.startswith("69KI"):
        return None
    content = line[4:].lstrip()
    parts = content.split(None, 1)
    if len(parts) < 2:
        return None
    pid, rest = parts[0], parts[1]
    if len(rest) < 48:
        return None
    first_str  = rest[ 0:16].strip()   # northing first in .dc
    second_str = rest[16:32].strip()   # easting second
    elev_str   = rest[32:48].strip()
    desc       = rest[48:].strip() or ""
    try:
        raw_northing = float(first_str)
        raw_easting  = float(second_str)
        v    = float(elev_str)
        elev = v if elev_str and v != 0.0 and v > -99999 else None
    except ValueError:
        return None
    return {
        "point_id":    pid,
        "easting_ft":  round(raw_easting  - shift_x, 3),
        "northing_ft": round(raw_northing - shift_y, 3),
        "elevation_ft": round(elev, 2) if elev is not None else "",
        "description": desc,
        "point_type":  "control",
    }


def _parse_66(line: str, geo_to_sp) -> Optional[dict]:
    """Base stations stored as lat/lon (EPSG:4326); convert to state-plane."""
    if not re.match(r"^66(FD|KI|SI)\s", line):
        return None
    content = line[6:].lstrip()
    m = re.search(r"(\d+\.\d+)(-\d+\.\d+)(\d+\.\d+)(.*)", content)
    if not m:
        return None
    lat_s, lon_s, elev_s, tail = m.group(1), m.group(2), m.group(3), m.group(4)
    name_end = content.find(lat_s)
    name = content[:name_end].strip() if name_end > 0 else ""
    try:
        lat  = float(lat_s)
        lon  = float(lon_s)
        elev = float(elev_s)
        east, north = geo_to_sp.transform(lon, lat)
    except Exception:
        return None
    desc = (tail.strip() + " " if tail.strip() else "") + "[elevation_ft=antenna_ht]"
    return {
        "point_id":    name or f"base_{lat:.4f}_{lon:.4f}",
        "easting_ft":  round(east,  3),
        "northing_ft": round(north, 3),
        "elevation_ft": round(elev, 2),
        "description": desc,
        "point_type":  "base",
    }


def _parse_81(line: str, shift_x: float, shift_y: float) -> Optional[dict]:
    """Observed grid points; same design-grid frame as 69KI."""
    if not (line.startswith("81CB") or line.startswith("81KI")):
        return None
    rest = line[4:].strip()
    if len(rest) < 1 + 48:
        return None
    rest = rest[1:]    # skip 1-char code byte
    try:
        raw_northing = float(rest[ 0:16].strip())
        raw_easting  = float(rest[16:32].strip())
        elev         = float(rest[32:48].strip())
    except ValueError:
        return None
    return {
        "easting_ft":  round(raw_easting  - shift_x, 3),
        "northing_ft": round(raw_northing - shift_y, 3),
        "elevation_ft": round(elev, 2),
        "point_type":  "observed",
    }


def parse_dc(dc_path: Path, shift_x: float, shift_y: float, delivery_crs: str) -> list[dict]:
    """Parse all relevant records from .dc file; return list of point dicts."""
    from pyproj import Transformer
    geo_to_sp = Transformer.from_crs("EPSG:4326", delivery_crs, always_xy=True)

    rows: list[dict] = []
    seen_69: set[str] = set()
    seen_66: set[tuple] = set()
    obs_n = 0

    with open(dc_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line:
                continue

            p = _parse_69ki(line, shift_x, shift_y)
            if p and p["point_id"] not in seen_69:
                seen_69.add(p["point_id"])
                rows.append(p)
                continue

            p = _parse_66(line, geo_to_sp)
            if p:
                key = (p["point_id"], round(p["easting_ft"], 1), round(p["northing_ft"], 1))
                if key not in seen_66:
                    seen_66.add(key)
                    rows.append(p)
                continue

            p = _parse_81(line, shift_x, shift_y)
            if p and p["easting_ft"] > 100_000 and p["northing_ft"] > 100_000:
                obs_n += 1
                p["point_id"]   = f"obs_{obs_n}"
                p["description"] = "81CB/81KI"
                rows.append(p)

    return rows


# ---------------------------------------------------------------------------
# dc subcommand
# ---------------------------------------------------------------------------

def cmd_dc(args) -> int:
    dc_path = Path(args.dc_file)
    if not dc_path.exists():
        sys.exit(f"ERROR: .dc file not found: {dc_path}")

    out_dir = Path(args.out_dir) if args.out_dir else dc_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # CRS detection
    delivery_crs = detect_crs_from_dc(dc_path)
    if delivery_crs is None:
        if not args.delivery_crs:
            sys.exit(
                "ERROR: Could not auto-detect CRS from .dc file.\n"
                "       Provide --delivery-crs EPSG:XXXX"
            )
        delivery_crs = args.delivery_crs
        print(f"CRS: {delivery_crs} (from --delivery-crs)")
    else:
        if args.delivery_crs and args.delivery_crs.upper() != delivery_crs.upper():
            print(f"WARNING: auto-detected {delivery_crs} but --delivery-crs overrides → {args.delivery_crs}")
            delivery_crs = args.delivery_crs
        else:
            print(f"CRS: {delivery_crs} (auto-detected from .dc)")

    if args.shift_x is None or args.shift_y is None:
        sys.exit(
            "ERROR: --shift-x and --shift-y are required.\n"
            "\n"
            "These are the design-grid offsets (in state-plane ft) to ADD to\n"
            "state-plane coordinates to produce design-grid coordinates:\n"
            "  design_E = state_E + shift_x\n"
            "  design_N = state_N + shift_y\n"
            "\n"
            "Derivation: survey at least one monument whose true state-plane\n"
            "coordinates are known (e.g. from NGS datasheet), then:\n"
            "  shift_x = design_E_from_dc - state_E_from_survey\n"
            "  shift_y = design_N_from_dc - state_N_from_survey\n"
            "\n"
            "Customer/Aztec job: --shift-x 1546702.929 --shift-y -3567.471"
        )
    shift_x = args.shift_x
    shift_y = args.shift_y

    job_name = args.job or extract_job_name(dc_path)
    print(f"Job:   {job_name}")
    print(f"Shift: design_E = state_E + {shift_x:+.3f} ft,  design_N = state_N + {shift_y:+.3f} ft")

    rows = parse_dc(dc_path, shift_x, shift_y, delivery_crs)

    # Write CSV
    csv_path = out_dir / f"{job_name}_points.csv"
    fieldnames = ["point_id", "easting_ft", "northing_ft", "elevation_ft", "description", "point_type"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items() if k in fieldnames})

    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["point_type"]] = by_type.get(r["point_type"], 0) + 1
    print(f"\nWrote {len(rows)} points → {csv_path.name}")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")
    print(f"QGIS: Delimited Text, X=easting_ft, Y=northing_ft, CRS={delivery_crs}")

    # Write transform.yaml
    transform = {
        "job":          job_name,
        "field_crs":    delivery_crs,   # Emlid uses same zone; override if different
        "odm_crs":      ODM_CRS,
        "delivery_crs": delivery_crs,
        "design_grid": {
            "shift_x":              shift_x,
            "shift_y":              shift_y,
            "scale":                1.0,
            "rotation_deg":         0.0,
            "helmert_residual_ft":  None,
            "anchor_x":             None,
            "anchor_y":             None,
        },
    }
    yaml_path = out_dir / "transform.yaml"
    write_yaml(yaml_path, transform)
    print(f"\nWrote {yaml_path.name}")
    print("  Review field_crs — it should match your Emlid RS3 output CRS.")
    print("  If the Emlid is in a different zone, update field_crs manually.")

    return 0


# ---------------------------------------------------------------------------
# split subcommand
# ---------------------------------------------------------------------------

def _locate_transform_yaml(input_path: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"ERROR: transform.yaml not found: {p}")
        return p
    # Auto-locate: same dir as input, then cwd
    for candidate in [input_path.parent / "transform.yaml", Path.cwd() / "transform.yaml"]:
        if candidate.exists():
            return candidate
    return None


def _crs_is_feet(epsg_str: str) -> bool:
    try:
        from pyproj import CRS
        crs = CRS.from_user_input(epsg_str)
        return "foot" in crs.axis_info[0].unit_name.lower()
    except Exception:
        return False


def cmd_split(args) -> int:
    in_path = Path(args.confirmed)
    if not in_path.exists():
        sys.exit(f"ERROR: confirmed file not found: {in_path}")

    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load transform.yaml if available
    yaml_path = _locate_transform_yaml(in_path, args.transform_yaml)
    transform = read_yaml(yaml_path) if yaml_path else {}
    if yaml_path:
        print(f"Loaded: {yaml_path}")

    # Parse confirmed file
    with open(in_path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        sys.exit(f"ERROR: empty file: {in_path}")

    file_crs_header = lines[0].strip()   # e.g. "EPSG:6529"

    # Determine source CRS: prefer transform.yaml field_crs; fall back to file header
    src_crs = transform.get("field_crs") or file_crs_header
    if transform.get("field_crs") and transform["field_crs"].upper() != file_crs_header.upper():
        print(f"INFO: field_crs in transform.yaml ({src_crs}) differs from file header ({file_crs_header}); using transform.yaml")

    dst_crs = transform.get("odm_crs") or args.target_crs
    print(f"Input:  {in_path.name}  (CRS: {src_crs})")
    print(f"Output: {dst_crs}")

    # Parse rows (filter to 'confirmed' if column 8 present)
    raw_rows = []
    for raw in lines[1:]:
        line = raw.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        if len(fields) >= 8 and fields[7] != "confirmed":
            continue
        raw_rows.append(fields)
    print(f"Rows:   {len(raw_rows)} confirmed observations")

    # Coordinate conversion
    needs_xy = src_crs.upper() != dst_crs.upper()
    needs_z  = _crs_is_feet(src_crs)

    xfm = None
    if needs_xy:
        try:
            from pyproj import Transformer
            xfm = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            print(f"Reprojecting XY: {src_crs} → {dst_crs}")
        except Exception as e:
            sys.exit(f"ERROR: cannot build transformer {src_crs} → {dst_crs}: {e}")
    if needs_z:
        print(f"Converting Z: US survey feet → metres (× {FT_TO_M})")

    converted = []
    for fields in raw_rows:
        try:
            x, y, z = float(fields[0]), float(fields[1]), float(fields[2])
        except ValueError:
            continue
        if xfm:
            x, y = xfm.transform(x, y)
        if needs_z:
            z *= FT_TO_M
        converted.append([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"] + fields[3:])

    # Split by label prefix
    gcp_rows, chk_rows, skipped = [], [], 0
    for fields in converted:
        label = fields[6] if len(fields) > 6 else ""
        if label.startswith("GCP-"):
            gcp_rows.append(fields)
        elif label.startswith("CHK-"):
            chk_rows.append(fields)
        else:
            skipped += 1

    if skipped:
        bad_labels = sorted({f[6] for f in converted if len(f) > 6
                             and not f[6].startswith("GCP-") and not f[6].startswith("CHK-")})
        print(f"WARNING: skipped {skipped} observations with unrecognised label prefix: {bad_labels}",
              file=sys.stderr)

    unique_gcp = len({f[6] for f in gcp_rows})
    unique_chk = len({f[6] for f in chk_rows})
    print(f"\n  GCP- points: {unique_gcp} unique, {len(gcp_rows)} observations")
    print(f"  CHK- points: {unique_chk} unique, {len(chk_rows)} observations")

    def _write(path: Path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(dst_crs + "\n")
            for fields in rows:
                f.write("\t".join(fields[:7]) + "\n")
        print(f"  wrote {path}  ({len(rows)} observations)")

    print()
    _write(out_dir / "gcp_list.txt", gcp_rows)
    _write(out_dir / "chk_list.txt", chk_rows)

    gcp_path = out_dir / "gcp_list.txt"
    chk_path = out_dir / "chk_list.txt"
    print(f"\nDone.  Run ODM with:  --gcp {gcp_path}")
    print(f"       Run RMSE with: {chk_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- dc subcommand ---
    dc = sub.add_parser("dc", help="Parse Trimble .dc → {job}_points.csv + transform.yaml")
    dc.add_argument("dc_file", metavar="{customer}_{job}.dc",
                    help="Input Trimble .dc data-collector file")
    dc.add_argument("--shift-x", type=float, default=None,
                    help="Design-grid offset: state_E + shift_x = design_E (ft)")
    dc.add_argument("--shift-y", type=float, default=None,
                    help="Design-grid offset: state_N + shift_y = design_N (ft)")
    dc.add_argument("--delivery-crs", default=None, metavar="EPSG:XXXX",
                    help="Override auto-detected state-plane CRS for deliverables")
    dc.add_argument("--job", default=None, help="Job name (default: from 10NM record or filename)")
    dc.add_argument("--out-dir", default=None, help="Output directory (default: same as .dc file)")

    # --- split subcommand ---
    gp = sub.add_parser("split", help="Reproject + split {job}_confirmed.txt → gcp_list.txt + chk_list.txt")
    gp.add_argument("confirmed", metavar="{job}_confirmed.txt",
                    help="GCPEditorPro confirmed file (tab-separated)")
    gp.add_argument("--transform-yaml", default=None, metavar="FILE",
                    help="Path to transform.yaml (auto-located in input dir or cwd if omitted)")
    gp.add_argument("--target-crs", default=ODM_CRS, metavar="EPSG:XXXX",
                    help=f"Output CRS (default: {ODM_CRS}); overridden by transform.yaml odm_crs")
    gp.add_argument("--out-dir", default=None, help="Output directory (default: same as input)")

    args = p.parse_args()
    if args.cmd == "dc":
        return cmd_dc(args)
    if args.cmd == "split":
        return cmd_split(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
