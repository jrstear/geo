#!/usr/bin/env python3
"""pix4d_qr.py — parse a Pix4Dmatic quality_report.pdf into a structured dict.

Pix4Dmatic v2.0.x does not export a JSON quality report; only the PDF.
This module uses pdfplumber to extract the high-level Quality check table
(page 1), the per-target tie-point tables (pages 7-8 typically; the GCP
table can flow across pages), and the project CRS info (page 12).

Library use:

    from pix4d_qr import parse_quality_report
    qr = parse_quality_report(Path("aztec_quality_report.pdf"))
    print(qr['quality_checks']['gcps']['detail'])
    for g in qr['gcps']:
        if g['marked'] > 0 and g['verified'] < g['marked']:
            print(f"GCP {g['label']}: {g['verified']}/{g['marked']} — solver rejected marks")

CLI:

    python pix4d_qr.py path/to/quality_report.pdf

Prints a brief summary plus any actionable findings.

Why this lives outside rmse.py:
This module is a pure parser, returning structured data. Wiring its
output into rmse.py for cross-validation columns (qr_dH, qr_dZ,
qr_marked) is a separate small piece (geo-6gni step 2) so other tools
can consume Pix4D's report independently.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float(s) -> Optional[float]:
    """Parse a cell as float; '' or None or non-numeric → None."""
    if s is None:
        return None
    s = s.strip() if isinstance(s, str) else s
    if s in ('', None):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _to_int(s) -> Optional[int]:
    if s is None:
        return None
    s = s.strip() if isinstance(s, str) else s
    if s in ('', None):
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _is_tiepoint_header(table_row) -> bool:
    """First row of a tie-points table contains 'Position error' merged-cell."""
    if not table_row:
        return False
    return any('Position error' in (c or '') for c in table_row)


def _is_tiepoint_table(table) -> bool:
    """True if the table looks like a tie-points table (8 columns, has header
    or rows shaped like data rows)."""
    if not table or len(table[0]) != 8:
        return False
    if _is_tiepoint_header(table[0]):
        return True
    # Continuation table (no header). Check if first row's label is a
    # plausible tie-point label (numeric or 'Min'/'Max'/'Mean'/...).
    first_label = (table[0][0] or '').strip() if table[0] else ''
    if first_label in ('Min', 'Max', 'Mean', 'Median', 'Sigma'):
        return True
    if re.match(r'^[A-Za-z0-9][A-Za-z0-9 \-_]*$', first_label):
        return True
    return False


def _parse_data_row(row) -> Optional[dict]:
    """Parse a per-target data row from the tie-points table.

    Schema: [label, dx, dy, dz, reproj_px, acc_xy, acc_z, verified/marked]

    Returns None if the row doesn't look like a target data row.
    """
    if not row or len(row) < 8:
        return None
    label = (row[0] or '').strip()
    if not label or label in ('Min', 'Max', 'Mean', 'Median', 'Sigma'):
        return None
    vm = (row[7] or '').strip()
    if '/' in vm:
        try:
            v_str, m_str = vm.split('/', 1)
            verified = int(v_str)
            marked = int(m_str)
        except (ValueError, TypeError):
            verified, marked = None, None
    else:
        verified, marked = None, None
    return {
        'label':     label,
        'dx':        _to_float(row[1]),
        'dy':        _to_float(row[2]),
        'dz':        _to_float(row[3]),
        'reproj_px': _to_float(row[4]),
        'acc_xy':    _to_float(row[5]),
        'acc_z':     _to_float(row[6]),
        'verified':  verified,
        'marked':    marked,
    }


def _parse_summary_row(row) -> Optional[dict]:
    """Parse a Min/Max/Mean/Median/Sigma row (label is in row[0])."""
    if not row or len(row) < 5:
        return None
    return {
        'dx':        _to_float(row[1]),
        'dy':        _to_float(row[2]),
        'dz':        _to_float(row[3]),
        'reproj_px': _to_float(row[4]),
    }


def _looks_like_rms_row(row) -> bool:
    """RMS row often has a None label and numeric values in dx/dy/dz/reproj
    (this is a pdfplumber artifact — the visible text 'RMS' is split into
    a different cell or merged into adjacent one)."""
    if not row or len(row) < 5:
        return False
    if (row[0] or '').strip() != '':
        return False
    return all(_to_float(c) is not None for c in row[1:5])


# ---------------------------------------------------------------------------
# Page-level parsers
# ---------------------------------------------------------------------------

# Severity heuristic: Pix4D shows ✓ ⚠ ❗ icons that are images (not in tables).
# We classify by the explanatory text Pix4D includes when status isn't OK.
_WARN_MARKERS = (
    'percentage of difference between',  # 5-20% camera optimization
    'Median of fewer than',               # low matches
)
_FAIL_MARKERS = (
    'more than 2.5 times the average GSD',  # CHK error too large
    'more than 20%',                         # camera opt severe
    'less than 100%',                        # uncalibrated cameras
)


def _classify_severity(detail: str) -> Optional[str]:
    if not detail:
        return None
    if any(m in detail for m in _FAIL_MARKERS):
        return 'fail'
    if any(m in detail for m in _WARN_MARKERS):
        return 'warn'
    return 'ok'


# Quality-check row label normalization (Page 1)
_CHECK_LABEL_MAP = {
    'matches':              'matches',
    'dataset':              'dataset',
    'camera optimization':  'camera_opt',
    'gcps':                 'gcps',
    'checkpoints':          'checkpoints',
    'atps':                 'atps',
}


def _parse_page1(page, out: dict) -> None:
    """Page 1: project header + metadata + Quality check.

    pdfplumber's table extractor is unreliable on this page across Matic
    versions — it sometimes drops rows like 'Camera', 'Dense point count',
    or 'Matches'. Use page text for metadata and metadata-style entries; use
    table extraction for the multi-line Quality check rows where text-order
    flips.
    """
    text = page.extract_text() or ''
    lines = text.split('\n')

    # Project name from header: "QR-{date}/ {project}"
    m = re.search(r'^QR[-/\d ]+/\s*(.+?)\s*$', lines[0]) if lines else None
    if m:
        out['project_name'] = m.group(1).strip()
    m = re.search(r'PIX4Dmatic\s+v([\d.]+)', text)
    if m:
        out['pix4d_version'] = m.group(1)

    # --- Metadata: parse from text, line by line ---
    for line in lines:
        for prefix, dst in (('Camera ', 'camera'),
                            ('Project CRS ', 'project_crs')):
            if line.startswith(prefix) and not out.get(dst):
                out[dst] = line[len(prefix):].strip()
                break
        m = re.match(r'^Area covered\s+([\d.]+)\s*ac', line)
        if m and out.get('area_acus') is None:
            out['area_acus'] = float(m.group(1))
        m = re.match(r'^Average GSD\s+([\d.]+)\s*ftUS', line)
        if m and out.get('gsd_ftus') is None:
            out['gsd_ftus'] = float(m.group(1))
        m = re.match(r'^Dense point count\s+([\d,]+)', line)
        if m and out.get('dense_point_count') is None:
            try:
                out['dense_point_count'] = int(m.group(1).replace(',', ''))
            except ValueError:
                pass

    # --- Quality check rows: prefer tables (handle multi-line cells correctly) ---
    for table in page.extract_tables():
        for row in table:
            if not row or len(row) < 2:
                continue
            key = (row[0] or '').strip()
            val = (row[1] or '').strip()
            key_lower = key.lower()
            if key_lower in _CHECK_LABEL_MAP:
                normalized = _CHECK_LABEL_MAP[key_lower]
                detail = val.replace('\n', ' ').strip()
                out['quality_checks'][normalized] = {
                    'detail':   detail,
                    'severity': _classify_severity(detail),
                }

    # 'Matches' row is often missed by table extraction. Pull from text.
    if 'matches' not in out['quality_checks']:
        for line in lines:
            m = re.match(r'^Matches\s+(.+)$', line)
            if m:
                detail = m.group(1).strip()
                out['quality_checks']['matches'] = {
                    'detail':   detail,
                    'severity': _classify_severity(detail),
                }
                break


def _parse_tiepoint_tables(pdf, out: dict) -> None:
    """Walk pages, extract tie-points tables, assign data rows to gcps or cps.

    Section detection: a table starting with a 'Position error' header begins
    a new section. The first such table is GCPs; the second is CPs. Tables
    without the header are continuations of the most recent section.
    """
    section = None  # 'gcps' or 'cps'
    sections_seen = 0

    for page in pdf.pages:
        for table in page.extract_tables():
            if not _is_tiepoint_table(table):
                continue

            if _is_tiepoint_header(table[0]):
                # New section begins
                sections_seen += 1
                section = 'gcps' if sections_seen == 1 else 'cps'
                rows = table[2:]   # skip 2 header rows
            else:
                # Continuation
                if section is None:
                    continue
                rows = table

            saw_summary = False
            for row in rows:
                # Skip header rows that occasionally appear in continuations
                if _is_tiepoint_header(row):
                    continue
                label = (row[0] or '').strip() if row and row[0] else ''
                summary_key = f'gcp_summary' if section == 'gcps' else 'cp_summary'
                # Summary row?
                if label in ('Min', 'Max', 'Mean', 'Median', 'Sigma'):
                    summary = _parse_summary_row(row)
                    if summary:
                        out[summary_key][label.lower()] = summary
                    saw_summary = True
                    continue
                # RMS row often has empty label (pdfplumber artifact)
                if saw_summary and _looks_like_rms_row(row):
                    summary = _parse_summary_row(row)
                    if summary:
                        out[summary_key]['rms'] = summary
                    continue
                # Data row
                data = _parse_data_row(row)
                if data:
                    out[section].append(data)


def _parse_crs_page(pdf, out: dict) -> None:
    """Find the Hardware & Settings page (typically last) and pull CRS info."""
    for page in pdf.pages:
        text = page.extract_text() or ''
        if 'Coordinate reference systems' not in text:
            continue
        for table in page.extract_tables():
            for row in table:
                if not row or len(row) < 2:
                    continue
                key = (row[0] or '').replace('\n', ' ').lower()
                val = (row[1] or '').replace('\n', ' ').strip()
                if 'image coordinate' in key:
                    out['image_crs'] = val
                elif 'ground control' in key or 'gcp' in key:
                    out['gcp_crs'] = val
                elif 'project coordinate' in key:
                    if not out.get('project_crs'):
                        out['project_crs'] = val
        return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_quality_report(path: Path) -> dict:
    """Parse a Pix4Dmatic quality_report.pdf and return a structured dict.

    See module docstring for the schema.
    """
    try:
        import pdfplumber
    except ImportError:
        sys.exit("ERROR: pdfplumber not installed. Run setup.sh or "
                 "'pip install pdfplumber' in the geo conda env.")

    out = {
        'format':            None,    # 'matic_2.0.x' | 'unsupported'
        'pix4d_version':     None,
        'project_name':      None,
        'camera':            None,
        'project_crs':       None,
        'gcp_crs':           None,
        'image_crs':         None,
        'area_acus':         None,
        'gsd_ftus':          None,
        'dense_point_count': None,
        'quality_checks':    {},
        'gcps':              [],
        'gcp_summary':       {},
        'cps':               [],
        'cp_summary':        {},
        'source_pdf':        str(path),
    }

    with pdfplumber.open(path) as pdf:
        # Format detection — the parser is calibrated to Pix4Dmatic 2.0.x.
        # Other Pix4D products (Mapper, Matic 2.15.x cloud-style) emit
        # different page layouts that pdfplumber would silently mis-parse.
        page1_text = pdf.pages[0].extract_text() if pdf.pages else ''
        if 'PIX4Dmatic v2.0' in (page1_text or ''):
            out['format'] = 'matic_2.0.x'
        else:
            out['format'] = 'unsupported'
            print(f"WARNING: {path.name} does not match Pix4Dmatic v2.0.x layout. "
                  f"Parser is calibrated to Matic 2.0.x. Older/cloud Pix4D "
                  f"variants need a separate format handler — file an issue if "
                  f"you hit this.",
                  file=sys.stderr)
            return out

        if len(pdf.pages) >= 1:
            _parse_page1(pdf.pages[0], out)
        _parse_tiepoint_tables(pdf, out)
        _parse_crs_page(pdf, out)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_actionable(qr: dict) -> List[Tuple[str, str]]:
    """Return (severity, message) tuples for things worth flagging to the user."""
    findings = []

    # Quality-check fails / warns from the page-1 summary
    for key, info in qr.get('quality_checks', {}).items():
        sev = info.get('severity')
        if sev in ('fail', 'warn'):
            findings.append((sev, f"{key}: {info['detail']}"))

    # Per-GCP: marked but verified < marked (solver rejected marks)
    for g in qr.get('gcps', []):
        v, m = g.get('verified'), g.get('marked')
        if v is not None and m is not None and m > 0 and v < m:
            findings.append(('warn',
                f"GCP {g['label']}: solver verified {v}/{m} marks "
                f"(rejected {m - v} as outliers)"))
        if m == 0:
            findings.append(('warn',
                f"GCP {g['label']}: 0 marks placed (untagged)"))

    return findings


def _format_summary(qr: dict) -> str:
    L = []
    proj = qr.get('project_name') or '(unknown project)'
    ver  = qr.get('pix4d_version') or '?'
    L.append(f"Project: {proj}  (Pix4Dmatic v{ver})")
    if qr.get('camera'):
        L.append(f"  Camera: {qr['camera']}")
    if qr.get('gsd_ftus') is not None:
        L.append(f"  GSD: {qr['gsd_ftus']:.3f} ftUS  "
                 f"({qr.get('area_acus','?')} acUS)")
    if qr.get('project_crs'):
        L.append(f"  Project CRS: {qr['project_crs']}")
    L.append("")

    L.append("Quality checks (from Pix4D's report):")
    for k in ('matches', 'dataset', 'camera_opt', 'gcps', 'checkpoints', 'atps'):
        info = qr.get('quality_checks', {}).get(k)
        if not info:
            continue
        sev = info.get('severity') or '-'
        sev_marker = {'ok': '[OK]  ', 'warn': '[WARN]', 'fail': '[FAIL]', '-': '[--]  '}.get(sev, '[--]  ')
        L.append(f"  {sev_marker} {k:<14s}{info['detail']}")

    L.append("")
    L.append(f"GCPs:        {len(qr['gcps'])} target(s)")
    s = qr.get('gcp_summary', {})
    if 'rms' in s:
        rms = s['rms']
        L.append(f"  RMS: dX={rms['dx']:.4f}  dY={rms['dy']:.4f}  "
                 f"dZ={rms['dz']:.4f}  reproj={rms['reproj_px']:.2f} px")
    L.append(f"Checkpoints: {len(qr['cps'])} target(s)")
    s = qr.get('cp_summary', {})
    if 'rms' in s:
        rms = s['rms']
        L.append(f"  RMS: dX={rms['dx']:.4f}  dY={rms['dy']:.4f}  "
                 f"dZ={rms['dz']:.4f}  reproj={rms['reproj_px']:.2f} px")

    findings = _find_actionable(qr)
    if findings:
        L.append("")
        L.append(f"Actionable findings ({len(findings)}):")
        for sev, msg in findings:
            sev_marker = '!' if sev == 'fail' else '?'
            L.append(f"  [{sev_marker}] {msg}")

    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('pdf', type=Path, help='Path to *_quality_report.pdf')
    ap.add_argument('--json', action='store_true',
                    help='Emit the parsed structure as JSON instead of '
                         'human-readable summary.')
    args = ap.parse_args()

    if not args.pdf.exists():
        sys.exit(f"ERROR: file not found: {args.pdf}")

    qr = parse_quality_report(args.pdf)

    if args.json:
        import json
        print(json.dumps(qr, indent=2, default=str))
    else:
        print(_format_summary(qr))


if __name__ == '__main__':
    main()
