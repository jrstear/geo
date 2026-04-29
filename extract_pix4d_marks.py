#!/usr/bin/env python3
"""
extract_pix4d_marks.py — extract per-image GCP marks from a Pix4D project history log.

Pix4D does not export marks to a standalone file in its standard outputs;
the per-image pixel observations of GCP/CHK targets live inside the project's
operation log `history.p4mpl` as a chronological sequence of operation blocks.
The relevant operations for current marks state:

    MarkTiePoint     — adds (or replaces, if same image+label) a single mark
    RemoveTiePoints  — removes every mark whose label is in the names list

This module replays both operation types in file order and returns the
current state. Used by:

  - `check_tags.py` (geo-y3j3): library import — `parse_marks(path)` returns
    the live marks list, no intermediate file needed.
  - This script's CLI: writes a CSV snapshot. Useful as a test fixture,
    for cross-machine handoff, and for committing reference data to git
    alongside the geo native `_tagged.txt`.

Usage:
    python extract_pix4d_marks.py <history.p4mpl> [-o <output.csv>] [-v]

If -o is omitted, the output is written next to the input as
`<history_stem>_marks.csv`. Default stdout is two lines: the write
confirmation and the GCPEditorPro-tier summary. Pass -v for a full
per-label table.

Output CSV columns:
    image_name,gcp_label,px,py

  image_name : basename of the imagePath in the log (Windows abs path → name)
  gcp_label  : the `name` field from the MarkTiePoint block
  px, py     : settings.pixelPosition values

MarkTiePoint block format (Pix4DMatic 2.0.2):

    MarkTiePoint {
        imagePath : "C:/path/to/DJI_..._V.JPG"
        name : '131 2'
        settings.clickAccuracy : 1
        settings.pixelPosition : [2520.808, 3266.690]
        inputs : [InputCameras{ },TiePointSet{ }]
        outputs : [TiePointSet{ }]
    }

RemoveTiePoints block format:

    RemoveTiePoints {
        names : ['68']         # or multi-line, e.g. ['1','2','3',...]
        inputs : [...]
        outputs : [...]
    }

Older Pix4D versions may differ. Extend the regexes or add a --version flag
if/when we encounter Survey 1.86.0 or other variants.

Pure stdlib — no conda env needed:
    python extract_pix4d_marks.py path/to/history.p4mpl -o marks.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple


class Mark(NamedTuple):
    image: str   # basename of imagePath
    label: str   # GCP/CHK target name
    px:    float
    py:    float


# Top-level operation blocks: `^OpName { ... ^}` spanning multiple lines.
# Non-greedy `.*?` with re.DOTALL stops at the first `\n}` (closing brace
# on its own line, which we anchor with re.MULTILINE).
_OP_RE = re.compile(
    r"^(MarkTiePoint|RemoveTiePoints)\s*\{(.*?)^\}",
    re.MULTILINE | re.DOTALL,
)

# Field regexes inside a block body.
_IMAGE_PATH_RE   = re.compile(r'imagePath\s*:\s*"([^"]+)"')
_NAME_RE         = re.compile(r"name\s*:\s*'([^']+)'")
_PIXEL_RE        = re.compile(
    r"settings\.pixelPosition\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]"
)
_NAMES_LIST_RE   = re.compile(r"names\s*:\s*\[(.*?)\]", re.DOTALL)
_NAME_IN_LIST_RE = re.compile(r"'([^']+)'")


def parse_marks(history_path: Path) -> List[Mark]:
    """Replay history.p4mpl to compute the current marks state.

    Walks every MarkTiePoint and RemoveTiePoints block in file order. A
    MarkTiePoint adds (or replaces) a mark for (image, label). A
    RemoveTiePoints drops every mark whose label is in the names list,
    across all images.

    Returns one Mark per (image, label) currently in effect, in arbitrary
    order. Caller may sort.
    """
    text = history_path.read_text(encoding="utf-8", errors="replace")

    state: Dict[Tuple[str, str], Mark] = {}

    for op_match in _OP_RE.finditer(text):
        op   = op_match.group(1)
        body = op_match.group(2)

        if op == "MarkTiePoint":
            ip = _IMAGE_PATH_RE.search(body)
            nm = _NAME_RE.search(body)
            pp = _PIXEL_RE.search(body)
            if not (ip and nm and pp):
                continue
            image = Path(ip.group(1)).name   # basename, drops Windows abs path
            label = nm.group(1)
            px    = float(pp.group(1))
            py    = float(pp.group(2))
            state[(image, label)] = Mark(image, label, px, py)

        else:  # RemoveTiePoints
            names_match = _NAMES_LIST_RE.search(body)
            if not names_match:
                continue
            names = set(_NAME_IN_LIST_RE.findall(names_match.group(1)))
            if not names:
                continue
            state = {k: v for k, v in state.items() if v.label not in names}

    return list(state.values())


def write_csv(marks: List[Mark], out_path: Path) -> None:
    """Write the marks list to CSV with header `image_name,gcp_label,px,py`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(marks, key=lambda m: (m.label, m.image))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "gcp_label", "px", "py"])
        for m in rows:
            w.writerow([m.image, m.label, f"{m.px:.3f}", f"{m.py:.3f}"])


def tier_counts(marks: List[Mark]) -> Tuple[int, int, int, Counter]:
    """Return (n_green, n_amber, n_red, label_counts).

    Tiers match GCPEditorPro/src/app/gcps-map/gcps-map.component.ts:
        green : >= 7 marks
        amber : 3..6 marks
        red   : < 3 marks
    """
    counts = Counter(m.label for m in marks)
    g = a = r = 0
    for n in counts.values():
        if   n >= 7: g += 1
        elif n >= 3: a += 1
        else:        r += 1
    return g, a, r, counts


def _print_table(counts: Counter) -> None:
    """Verbose per-label table, sorted by mark count desc then label asc."""
    print(f"{'label':<14s} {'marks':>6s}  tier")
    for lbl, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        tier = "green" if n >= 7 else ("amber" if n >= 3 else "RED")
        print(f"{lbl:<14s} {n:>6d}  {tier}")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="extract_pix4d_marks.py",
        description=(
            "Extract per-image GCP marks from a Pix4D project history.p4mpl "
            "log and write them to CSV. Output columns: "
            "image_name,gcp_label,px,py."
        ),
        epilog=(
            "Default output path is <history_stem>_marks.csv next to the "
            "input file. GCPEditorPro tier guideline applied to the summary: "
            "green ≥ 7 marks per label, amber 3-6, red < 3."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("history", type=Path, metavar="history.p4mpl",
                    help="path to the project's history.p4mpl")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="destination CSV path (default: <history_stem>_marks.csv "
                         "alongside the input)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print a per-label mark-count table")

    args = ap.parse_args()

    if not args.history.exists():
        print(f"ERROR: history file not found: {args.history}", file=sys.stderr)
        return 1

    out_path = args.output or args.history.with_name(args.history.stem + "_marks.csv")

    marks = parse_marks(args.history)
    write_csv(marks, out_path)

    g, a, r, counts = tier_counts(marks)
    print(f"wrote {len(marks)} marks ({len(counts)} labels) to {out_path}")
    print(f"GCPEditorPro tiers: {g} green, {a} amber, {r} red"
          + (f"  ← {r} label(s) below the < 3 hard-stop floor" if r else ""))
    if args.verbose:
        _print_table(counts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
