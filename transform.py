#!/usr/bin/env python3
"""transform.py — Coordinate transformation tool for the survey-to-delivery pipeline.

Subcommands
-----------
dc   Parse a Trimble .dc file → {job}_{epsg}.csv + {job}_design.csv + transform.yaml
     transform.yaml captures the job's CRS and design-grid shift for downstream use.

split  Split GCPEditorPro {job}_tagged.txt → gcp_list.txt (GCP- tagged, for ODM)
       + chk_list.txt (CHK- tagged, for rmse_calc.py)
       + {job}_targets.csv (one row/target, EPSG:32613, tagged=GCP-/CHK- prefix, untagged=bare label)
       + {job}_targets_design.csv (same, design-grid ft, if transform.yaml present).
       Reads transform.yaml (written by dc) to get field_crs and odm_crs automatically.

Typical sequence
----------------
  # 1. Before field survey — extract control monuments and record job CRS:
  #    Preferred: provide one anchor monument whose state-plane coords are known
  python transform.py dc {customer}_{job}.dc --anchor <id> <state_E_ft> <state_N_ft>
  #    Alternative: provide shift directly if already known
  python transform.py dc {customer}_{job}.dc --shift-x X --shift-y Y

  # 2. After GCPEditorPro tagging — split for ODM:
  python transform.py split {job}_tagged.txt
  #    transform.yaml is found automatically in the same directory

transform.yaml — written by dc, read by split (and sight.py and package.py --transform-yaml)
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
  anchor_x: <float>       # state-plane E of anchor monument (null if --shift-x/y used directly)
  anchor_y: <float>       # state-plane N of anchor monument (null if --shift-x/y used directly)

Shift convention (consistent with package.py --shift-x/y):
  state_E = design_E - shift_x     (dc subcommand, parsing .dc records)
  design_E = state_E + shift_x     (package.py, delivery output)

Example (Customer/Aztec job — anchor on NGS monument 14, state-plane from NGS datasheet):
  python transform.py dc "F100340 AZTEC.dc" --anchor 14 1147722.527 2144275.554
  python transform.py split ~/stratus/aztec3/aztec3_confirmed.txt
"""

import argparse
import csv
import json
import math
import re
import sys
import urllib.request
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


def _read_raw_69ki(dc_path: Path) -> dict:
    """Return {point_id: (raw_easting_ft, raw_northing_ft)} for all 69KI records.

    Raw = design-grid coordinates as stored in the .dc file (northing first,
    easting second).  No shift is applied.  Used to compute the shift when
    --anchor is provided.
    """
    result: dict = {}
    with open(dc_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("69KI"):
                continue
            content = line[4:].lstrip()
            parts = content.split(None, 1)
            if len(parts) < 2:
                continue
            pid, rest = parts[0], parts[1]
            if len(rest) < 32:
                continue
            try:
                raw_northing = float(rest[ 0:16].strip())
                raw_easting  = float(rest[16:32].strip())
            except ValueError:
                continue
            if pid not in result:
                result[pid] = (raw_easting, raw_northing)
    return result


def _ngs_lookup(dc_path: Path, delivery_crs: str) -> list:
    """Query the NGS NDE API to find state-plane coordinates for NGS monuments
    listed in the dc file.

    Strategy:
    1. Extract base-station lat/lon from 66FD records → search center
    2. Query NGS radial API within 10 miles
    3. Match results to dc-file NGS monuments by designation tokens
    4. Convert matched lat/lon → state-plane feet via pyproj

    Returns list of dicts, one per matched dc monument:
        dc_pid, dc_desc, ngs_pid, ngs_name, lat, lon, state_e_ft, state_n_ft
    Empty list if API unavailable, no base-station coordinates, or no NGS monuments.
    """
    # --- Collect base-station lat/lon for search center ---
    lats, lons = [], []
    with open(dc_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not re.match(r"^66(FD|KI|SI)\s", line):
                continue
            m = re.search(r"(\d+\.\d+)(-\d+\.\d+)", line[6:].lstrip())
            if m:
                lats.append(float(m.group(1)))
                lons.append(float(m.group(2)))
    if not lats:
        return []

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # --- Collect NGS-labeled monuments from dc file ---
    raw_pts = _read_raw_69ki(dc_path)
    descs: dict = {}
    with open(dc_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("69KI"):
                continue
            content = line[4:].lstrip()
            parts = content.split(None, 1)
            if len(parts) < 2:
                continue
            pid, rest = parts[0], parts[1]
            desc = rest[48:].strip() if len(rest) >= 48 else ""
            if pid not in descs:
                descs[pid] = desc
    ngs_monuments = {pid: descs[pid] for pid in raw_pts
                     if "NGS" in descs.get(pid, "").upper()}
    if not ngs_monuments:
        return []

    # --- Query NGS radial API ---
    url = (f"https://geodesy.noaa.gov/api/nde/radial?"
           f"lat={center_lat:.6f}&lon={center_lon:.6f}&radius=10&units=MILE")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "transform.py/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            ngs_results = json.loads(resp.read())
    except Exception as e:
        print(f"  WARNING: NGS API unavailable ({e})", file=sys.stderr)
        return []

    # Determine FIPS zone code and zone label for SPC lookup.
    # Zone labels match NGS datasheet SPC lines, e.g. "SPC NM C", "SPC NM W".
    _FIPS_TO_ZONE_LABEL: dict[int, str] = {
        3001: "NM E", 3002: "NM C", 3003: "NM W",
    }
    fips_code: Optional[int] = None
    zone_label: Optional[str] = None
    for fips, epsg_map in FIPS_TO_EPSG.items():
        if any(f"EPSG:{v}" == delivery_crs.upper() for v in epsg_map.values()):
            fips_code = fips
            zone_label = _FIPS_TO_ZONE_LABEL.get(fips)
            break

    def _spc_from_datasheet(pid: str) -> Optional[tuple]:
        """Fetch NGS datasheet text and parse published SPC in US survey feet.
        Returns (e_ft, n_ft) from the matching SPC line, or None if not found."""
        if zone_label is None:
            return None
        ds_url = f"https://www.ngs.noaa.gov/cgi-bin/ds_mark.prl?PidBox={pid}"
        try:
            req_ds = urllib.request.Request(ds_url, headers={"User-Agent": "transform.py/1.0"})
            with urllib.request.urlopen(req_ds, timeout=15) as resp_ds:
                text = resp_ds.read().decode("utf-8", errors="replace")
        except Exception:
            return None
        # Match lines like:  PID;SPC NM C     - 2,144,275.554  1,147,722.527   sFT
        # North comes first in the datasheet, then East.
        pattern = (r";SPC\s+" + re.escape(zone_label) +
                   r"\s*-\s*([\d,]+\.\d+)\s+([\d,]+\.\d+)\s+sFT")
        m = re.search(pattern, text)
        if not m:
            return None
        n_ft = float(m.group(1).replace(",", ""))
        e_ft = float(m.group(2).replace(",", ""))
        return e_ft, n_ft

    def _spc_from_ncat(lat: float, lon: float) -> Optional[tuple]:
        """Call NCAT llh → SPC in US survey feet (fallback when datasheet lacks our zone).
        Accuracy is limited by the NDE lat/lon precision (~20 ft typical)."""
        if fips_code is None:
            return None
        ncat_url = (
            f"https://geodesy.noaa.gov/api/ncat/llh?"
            f"lat={lat:.10f}&lon={lon:.10f}&eht=0"
            f"&inDatum=nad83%282011%29&outDatum=nad83%282011%29"
            f"&spcZone={fips_code}&units=usft"
        )
        try:
            req2 = urllib.request.Request(ncat_url, headers={"User-Agent": "transform.py/1.0"})
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                ncat = json.loads(resp2.read())
            e_ft = float(ncat["spcEasting_usft"].replace(",", ""))
            n_ft = float(ncat["spcNorthing_usft"].replace(",", ""))
            return e_ft, n_ft
        except Exception:
            return None

    # --- Match dc NGS monuments to NGS API results ---
    # Strip common non-identifying tokens; what remains is the designation (e.g. "Y 430")
    _NOISE = {"NGS", "VCM", "3D", "ROD", "DISK", "CAP", "MARK", "BM", "RM", ""}
    matches = []
    for dc_pid, dc_desc in ngs_monuments.items():
        dc_tokens = {t.strip("'\",.") for t in dc_desc.upper().split()} - _NOISE
        best = None
        best_score = 0
        for rec in ngs_results:
            ngs_name = rec.get("name", "")
            ngs_tokens = {t.strip("'\",.") for t in ngs_name.upper().split()} - _NOISE
            score = len(dc_tokens & ngs_tokens)
            if score > best_score:
                best_score = score
                best = rec
        if best and best_score > 0:
            try:
                lat = float(best["lat"])
                lon = float(best["lon"])
            except (KeyError, ValueError):
                continue
            pid_str = best.get("pid", "")
            # Try exact SPC from datasheet first; fall back to NCAT, then pyproj
            spc = _spc_from_datasheet(pid_str)
            spc_source = "NGS datasheet (exact)"
            if spc is None:
                spc = _spc_from_ncat(lat, lon)
                spc_source = "NCAT lat/lon (~20 ft)"
            if spc is None:
                try:
                    from pyproj import Transformer
                    geo_to_sp = Transformer.from_crs("EPSG:4326", delivery_crs, always_xy=True)
                    spc = geo_to_sp.transform(lon, lat)
                    spc_source = "pyproj lat/lon (~20 ft)"
                except Exception:
                    continue
            sp_e, sp_n = spc
            matches.append({
                "dc_pid":      dc_pid,
                "dc_desc":     dc_desc,
                "ngs_pid":     pid_str,
                "ngs_name":    best.get("name", ""),
                "lat":         lat,
                "lon":         lon,
                "state_e_ft":  sp_e,
                "state_n_ft":  sp_n,
                "spc_source":  spc_source,
            })

    return matches


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

    # --- Resolve design-grid shift ---
    if args.shift_x is not None and args.shift_y is not None:
        shift_x = args.shift_x
        shift_y = args.shift_y
        anchor_used = None
    elif args.anchor:
        anchor_id = args.anchor[0]
        try:
            anchor_state_e = float(args.anchor[1])
            anchor_state_n = float(args.anchor[2])
        except ValueError:
            sys.exit(f"ERROR: --anchor STATE_E_FT and STATE_N_FT must be numbers, got: {args.anchor[1]!r} {args.anchor[2]!r}")
        raw_pts = _read_raw_69ki(dc_path)
        if anchor_id not in raw_pts:
            ids_found = ", ".join(sorted(raw_pts.keys())) or "(none)"
            sys.exit(
                f"ERROR: anchor monument '{anchor_id}' not found in 69KI records.\n"
                f"       Monument IDs in .dc file: {ids_found}"
            )
        raw_e, raw_n = raw_pts[anchor_id]
        shift_x = raw_e - anchor_state_e
        shift_y = raw_n - anchor_state_n
        anchor_used = (anchor_id, anchor_state_e, anchor_state_n, raw_e, raw_n)
    else:
        # Neither shift nor anchor provided — try NGS API auto-lookup first.
        print("No --anchor or --shift-x/y provided — querying NGS database...")
        ngs_matches = _ngs_lookup(dc_path, delivery_crs)

        if ngs_matches:
            raw_pts = _read_raw_69ki(dc_path)
            shifts = []
            for m in ngs_matches:
                raw_e, raw_n = raw_pts[m["dc_pid"]]
                sx = raw_e - m["state_e_ft"]
                sy = raw_n - m["state_n_ft"]
                shifts.append((sx, sy, m))

            shift_x, shift_y, primary = shifts[0]
            anchor_used = (
                primary["dc_pid"],
                primary["state_e_ft"],
                primary["state_n_ft"],
                raw_pts[primary["dc_pid"]][0],
                raw_pts[primary["dc_pid"]][1],
            )
            print(f"  Found {len(ngs_matches)} NGS monument(s) via API:")
            for sx, sy, m in shifts:
                print(f"    dc:{m['dc_pid']} ({m['dc_desc'].strip()}) "
                      f"→ NGS {m['ngs_pid']} \"{m['ngs_name']}\"  "
                      f"state-plane ({m['state_e_ft']:.3f}, {m['state_n_ft']:.3f}) ft"
                      f"  [{m.get('spc_source', 'unknown')}]")
            all_exact = all(m.get("spc_source", "").startswith("NGS datasheet") for m in ngs_matches)
            if not all_exact:
                print(f"  NOTE: Some coordinates derived from lat/lon conversion — typical accuracy "
                      f"±20 ft. For survey-quality work, verify with --anchor <id> <exact_E> <exact_N>.")
            if len(shifts) > 1:
                max_dev = max(
                    math.hypot(sx - shift_x, sy - shift_y)
                    for sx, sy, _ in shifts[1:]
                )
                if max_dev > 1.0:
                    print(f"  WARNING: shifts from multiple anchors disagree by up to "
                          f"{max_dev:.2f} ft — using dc:{primary['dc_pid']} as primary.")
                else:
                    print(f"  Cross-check: {len(shifts)} anchors agree within {max_dev:.3f} ft ✓")
        else:
            # NGS lookup failed or no matches — print diagnostic table and exit.
            raw_pts = _read_raw_69ki(dc_path)
            if not raw_pts:
                sys.exit(
                    "ERROR: No 69KI control monuments found in .dc file.\n"
                    "       Check that the file is a valid Trimble data-collector export."
                )
            descs: dict = {}
            with open(dc_path, encoding="utf-8", errors="replace") as _f:
                for _line in _f:
                    if not _line.startswith("69KI"):
                        continue
                    _content = _line[4:].lstrip()
                    _parts = _content.split(None, 1)
                    if len(_parts) < 2:
                        continue
                    _pid, _rest = _parts[0], _parts[1]
                    _desc = _rest[48:].strip() if len(_rest) >= 48 else ""
                    if _pid not in descs:
                        descs[_pid] = _desc
            ngs_ids = [pid for pid in raw_pts
                       if "NGS" in descs.get(pid, "").upper()]
            lines_out = [
                "ERROR: NGS API lookup returned no matches. Provide --anchor manually.",
                "",
                "  Control monuments (69KI) in this .dc file:",
                "    {:>12}  {:>18}  {:>18}  {}".format(
                    "id", "design_E_raw_ft", "design_N_raw_ft", "description"),
            ]
            for pid, (re_, rn) in sorted(raw_pts.items()):
                marker = "  ← NGS" if pid in ngs_ids else ""
                lines_out.append("    {:>12}  {:>18.3f}  {:>18.3f}  {}{}".format(
                    pid, re_, rn, descs.get(pid, ""), marker))
            lines_out += [
                "",
                "  Look up NGS state-plane coordinates at https://www.ngs.noaa.gov/datasheets/",
                "  then: --anchor <monument_id> <state_E_ft> <state_N_ft>",
            ]
            sys.exit("\n".join(lines_out))

    job_name = args.job or dc_path.stem.replace(" ", "_")
    print(f"Job:   {job_name}")
    if anchor_used:
        pid, sp_e, sp_n, raw_e, raw_n = anchor_used
        print(f"Anchor: monument {pid}  state-plane ({sp_e:.3f}, {sp_n:.3f})  design-raw ({raw_e:.3f}, {raw_n:.3f})")
    print(f"Shift: design_E = state_E + {shift_x:+.3f} ft,  design_N = state_N + {shift_y:+.3f} ft")

    rows = parse_dc(dc_path, shift_x, shift_y, delivery_crs)

    # Write state-plane CSV (delivery CRS, ft) — for Emlid RS3 localization
    epsg_suffix = delivery_crs.split(":")[-1]
    csv_path = out_dir / f"{job_name}_{epsg_suffix}.csv"
    fieldnames = ["point_id", "easting_ft", "northing_ft", "elevation_ft", "description", "point_type"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items() if k in fieldnames})

    # Write design-grid CSV (delivery_crs + shift, ft) — for QGIS design review
    design_csv_path = out_dir / f"{job_name}_design.csv"
    with open(design_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            design_r = dict(r)
            try:
                design_r["easting_ft"]  = round(float(r["easting_ft"])  + shift_x, 3)
                design_r["northing_ft"] = round(float(r["northing_ft"]) + shift_y, 3)
            except (ValueError, TypeError):
                pass
            w.writerow({k: ("" if v is None else v) for k, v in design_r.items() if k in fieldnames})

    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["point_type"]] = by_type.get(r["point_type"], 0) + 1
    print(f"\nWrote {len(rows)} points:")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")
    print(f"  → {csv_path.name}  (state-plane {delivery_crs}, ft; for Emlid localization)")
    print(f"  → {design_csv_path.name}  (design-grid ft; for QGIS design review)")

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
            "anchor_x":             anchor_used[1] if anchor_used else None,
            "anchor_y":             anchor_used[2] if anchor_used else None,
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
    import csv as _csv
    in_path = Path(args.confirmed)
    if not in_path.exists():
        sys.exit(f"ERROR: file not found: {in_path}")

    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load transform.yaml if available
    yaml_path = _locate_transform_yaml(in_path, args.transform_yaml)
    transform = read_yaml(yaml_path) if yaml_path else {}
    if yaml_path:
        print(f"Loaded: {yaml_path}")

    # Parse tagged file
    with open(in_path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        sys.exit(f"ERROR: empty file: {in_path}")

    file_crs_header = lines[0].strip()   # e.g. "EPSG:6529"

    # Determine source CRS: file header is authoritative; fall back to transform.yaml field_crs
    src_crs = file_crs_header or transform.get("field_crs")
    if transform.get("field_crs") and file_crs_header and transform["field_crs"].upper() != file_crs_header.upper():
        print(f"INFO: file header CRS ({file_crs_header}) differs from transform.yaml field_crs "
              f"({transform.get('field_crs')}); using file header")

    dst_crs = transform.get("odm_crs") or args.target_crs
    print(f"Input:  {in_path.name}  (CRS: {src_crs})")
    print(f"Output: {dst_crs}")

    # Parse all rows; track tagged status (col 8 == "tagged")
    raw_rows = []
    for raw in lines[1:]:
        line = raw.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        raw_rows.append(fields)
    print(f"Rows:   {len(raw_rows)} observations total")

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

    # Split by label prefix and tagged status
    # all_targets: label → (fields, is_tagged) — first occurrence wins for XYZ;
    #              upgraded to tagged=True if any tagged observation found later.
    gcp_rows, chk_rows, skipped = [], [], 0
    all_targets: dict = {}

    for fields in converted:
        label = fields[6] if len(fields) > 6 else ""
        is_tagged = len(fields) > 7 and fields[7] == "tagged"

        # Track one entry per target for targets CSV (first occurrence for XYZ)
        if label not in all_targets:
            all_targets[label] = (fields, is_tagged)
        elif is_tagged and not all_targets[label][1]:
            all_targets[label] = (fields, True)

        if not is_tagged:
            continue

        if label.startswith("GCP-"):
            gcp_rows.append(fields)
        elif label.startswith("CHK-"):
            chk_rows.append(fields)
        else:
            skipped += 1

    if skipped:
        bad = sorted({f[6] for f in converted
                      if len(f) > 7 and f[7] == "tagged"
                      and len(f) > 6
                      and not f[6].startswith("GCP-") and not f[6].startswith("CHK-")})
        print(f"WARNING: skipped {skipped} tagged observations with unrecognised label prefix: {bad}",
              file=sys.stderr)

    unique_gcp  = len({f[6] for f in gcp_rows})
    unique_chk  = len({f[6] for f in chk_rows})
    untagged_ct = sum(1 for _, (_, tagged) in all_targets.items() if not tagged)
    print(f"\n  GCP- points: {unique_gcp} unique, {len(gcp_rows)} observations  → gcp_list.txt")
    print(f"  CHK- points: {unique_chk} unique, {len(chk_rows)} observations  → chk_list.txt")
    print(f"  Untagged:    {untagged_ct} targets (prefix stripped in targets CSV)")

    def _write_odm(path: Path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(dst_crs + "\n")
            for fields in rows:
                f.write("\t".join(fields[:7]) + "\n")
        print(f"  wrote {path}  ({len(rows)} observations)")

    print()
    _write_odm(out_dir / "gcp_list.txt", gcp_rows)
    _write_odm(out_dir / "chk_list.txt", chk_rows)

    # Derive job name: from transform.yaml, or strip "_tagged" suffix from input stem
    stem = in_path.stem
    job_name = transform.get("job") or (stem[:-7] if stem.endswith("_tagged") else stem)

    def _display_label(label: str, is_tagged: bool) -> str:
        """Keep prefix for tagged targets; strip any XXX- prefix for untagged.

        Note: 'XXX-' here is regex shorthand for 'any uppercase prefix followed
        by a hyphen' (sight.py assigns GCP-/CHK- prefixes to all targets, plus
        a legacy DUP- prefix on older files). The regex strips the leading
        prefix back to bare monument IDs for untagged rows so the QGIS targets
        layer shows the surveyor's original labels. Not a TODO.

        Trailing '-dup'/'-dup2'/etc. suffixes (the new near-duplicate marker)
        are NOT stripped, so untagged 'GCP-104-dup' becomes '104-dup' — the
        duplicate is still visually distinguishable from the primary in QGIS.
        """
        if is_tagged:
            return label
        return re.sub(r'^[A-Z]+-', '', label)

    # Write {job}_targets.csv — one row per distinct target, EPSG:32613
    targets_path = out_dir / f"{job_name}_targets.csv"
    with open(targets_path, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["label", "X", "Y", "Z"])
        for label in sorted(all_targets):
            fields, is_tagged = all_targets[label]
            w.writerow([_display_label(label, is_tagged), fields[0], fields[1], fields[2]])
    print(f"  wrote {targets_path}  ({len(all_targets)} targets)")

    # Write {job}_targets_design.csv — same but in design-grid coordinates
    delivery_crs = transform.get("delivery_crs")
    dg = transform.get("design_grid") if isinstance(transform.get("design_grid"), dict) else {}
    shift_x = dg.get("shift_x")
    shift_y = dg.get("shift_y")

    if delivery_crs and shift_x is not None and shift_y is not None:
        try:
            from pyproj import Transformer as _Transformer
            xfm_back = _Transformer.from_crs(dst_crs, delivery_crs, always_xy=True)
            design_path = out_dir / f"{job_name}_targets_design.csv"
            with open(design_path, "w", encoding="utf-8", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["label", "X", "Y", "Z"])
                for label in sorted(all_targets):
                    fields, is_tagged = all_targets[label]
                    try:
                        x_m, y_m, z_m = float(fields[0]), float(fields[1]), float(fields[2])
                    except ValueError:
                        continue
                    x_sp, y_sp = xfm_back.transform(x_m, y_m)
                    z_ft = z_m / FT_TO_M
                    x_d = x_sp + float(shift_x)
                    y_d = y_sp + float(shift_y)
                    w.writerow([_display_label(label, is_tagged),
                                f"{x_d:.3f}", f"{y_d:.3f}", f"{z_ft:.3f}"])
            print(f"  wrote {design_path}  ({len(all_targets)} targets, design-grid)")
        except Exception as e:
            print(f"WARNING: could not write targets_design.csv: {e}", file=sys.stderr)
    else:
        missing = []
        if not delivery_crs:  missing.append("delivery_crs")
        if shift_x is None:   missing.append("design_grid.shift_x")
        if shift_y is None:   missing.append("design_grid.shift_y")
        print(f"  skipped targets_design.csv — transform.yaml missing: {missing}")

    gcp_path = out_dir / "gcp_list.txt"
    chk_path = out_dir / "chk_list.txt"
    print(f"\nDone.  Run ODM with:  --gcp {gcp_path}")
    print(f"       Run RMSE with: rmse.py <reconstruction.topocentric.json> "
          f"{gcp_path} {chk_path}")
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
    dc.add_argument("--anchor", nargs=3, metavar=("MONUMENT_ID", "STATE_E_FT", "STATE_N_FT"),
                    type=lambda v: v,   # keep as strings; convert below
                    help="Compute shift from one known monument: ID + published state-plane E/N (ft). "
                         "Run without --anchor or --shift-x/y to see monuments in the .dc file.")
    dc.add_argument("--shift-x", type=float, default=None,
                    help="Design-grid offset: state_E + shift_x = design_E (ft) — use if shift already known")
    dc.add_argument("--shift-y", type=float, default=None,
                    help="Design-grid offset: state_N + shift_y = design_N (ft) — use if shift already known")
    dc.add_argument("--delivery-crs", default=None, metavar="EPSG:XXXX",
                    help="Override auto-detected state-plane CRS for deliverables")
    dc.add_argument("--job", default=None, help="Job name (default: from 10NM record or filename)")
    dc.add_argument("--out-dir", default=None, help="Output directory (default: same as .dc file)")

    # --- split subcommand ---
    gp = sub.add_parser("split",
                        help="Split {job}_tagged.txt → gcp_list.txt + chk_list.txt + "
                             "{job}_targets.csv + {job}_targets_design.csv")
    gp.add_argument("confirmed", metavar="{job}_tagged.txt",
                    help="GCPEditorPro tagged file (tab-separated; col 8 == 'tagged' "
                         "for tagged rows, empty for untagged)")
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
