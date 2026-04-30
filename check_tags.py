#!/usr/bin/env python3
"""
check_tags.py — pre-bundle tagging consistency check.

Toolchain-agnostic: works on sight.py / GCPEditorPro / ODM tagging output,
on Pix4D Matic project history (history.p4mpl), and on Pix4D-format marks
CSVs (Filename,Label,PixelX,PixelY).

Two checks run, in order:

  1. MARKS-PER-TARGET tier (gate) — mirrors GCPEditorPro's badge thresholds:
       red   (< 3 marks/target)  → FAIL  exit 1   (insufficient for bundle)
       amber (3-6 marks/target)  → WARN  pass
       green (≥ 7 marks/target)  → PASS

     Fairness clamp when total visible images per target is known (sight
     mode): green threshold drops to min(7, visible) so a target visible in
     only 4 images can still pass at 4 marks.

  2. TAG-VS-ESTIMATE consensus (reviewer aid; sight mode only) — identifies
     targets whose user tags are inconsistent with sight's color/tri_color
     estimates. Informational — does NOT affect exit code. (Earlier
     iterations of this script gated on a composite suspicion score; that
     gate proved too noisy and has been retired in favour of the 3-tier
     mark-count rule above. The consensus table is preserved because it
     surfaces useful "go look at this target" signal that the simple
     mark-count rule misses.)

Usage:
    # SIGHT / GCPEditorPro / ODM mode:
    check_tags.py {job}.txt {job}_tagged.txt
    check_tags.py {job}_tagged.txt        # auto-locates {job}.txt sibling

    # PIX4D mode:
    check_tags.py path/to/history.p4mpl
    check_tags.py path/to/{job}_marks.csv  # extracted marks CSV

Distinct from:
  - sight.py pre-tagging quality checks (catch survey-side: FLOAT shots,
    datum mismatches, control residuals — geo-40vs)
  - rmse.py (post-bundle, against reconstruction — different stage)

See docs/plans/tag-quality-consistency.md.
"""
import argparse
import csv as _csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# GCPEditorPro 3-tier thresholds (gcps-map.component.ts):
#   green ≥ CONFIRMED_GREEN, amber ≥ CONFIRMED_AMBER, red < CONFIRMED_AMBER
TIER_GREEN_FLOOR = 7
TIER_AMBER_FLOOR = 3


def _load_rows(path: Path) -> Tuple[str, List[dict]]:
    """Load a sight.py-format file. Returns (crs_header, rows)."""
    rows: List[dict] = []
    with open(path) as f:
        crs = next(f).strip()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            try:
                rows.append({
                    "x":   float(p[0]),
                    "y":   float(p[1]),
                    "z":   float(p[2]),
                    "px":  float(p[3]),
                    "py":  float(p[4]),
                    "img": p[5],
                    "lbl": p[6],
                    "src": p[7] if len(p) > 7 else "",
                })
            except ValueError:
                continue
    return crs, rows


# ---------------------------------------------------------------------------
# Pix4D inputs (history.p4mpl + extracted marks CSV)
# ---------------------------------------------------------------------------

def _load_pix4d_history(path: Path) -> List[dict]:
    """Replay history.p4mpl via extract_pix4d_marks.parse_marks().

    Returns row dicts in the same shape as _load_rows (subset of keys —
    only img/lbl/px/py are populated; the rest are absent because the
    .p4mpl carries no ground coords).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from extract_pix4d_marks import parse_marks
    except ImportError as e:
        sys.exit(f"ERROR: cannot import extract_pix4d_marks ({e}); "
                 f"required for .p4mpl input")
    return [{"img": m.image, "lbl": m.label, "px": m.px, "py": m.py}
            for m in parse_marks(path)]


def _load_pix4d_marks_csv(path: Path) -> List[dict]:
    """Read a Pix4D-format marks CSV (Filename,Label,PixelX,PixelY).

    Tolerant header detection — accepts any case + 'PixelX'/'X'/'px'.
    """
    rows: List[dict] = []
    with open(path) as f:
        r = _csv.DictReader(f)
        # Map columns case-insensitively
        cols = {h.strip().lower(): h for h in (r.fieldnames or [])}
        col_img = cols.get('filename') or cols.get('image') or cols.get('image_name')
        col_lbl = cols.get('label')    or cols.get('gcp_label') or cols.get('name')
        col_px  = cols.get('pixelx')   or cols.get('px')         or cols.get('x')
        col_py  = cols.get('pixely')   or cols.get('py')         or cols.get('y')
        if not all([col_img, col_lbl, col_px, col_py]):
            sys.exit(f"ERROR: marks CSV {path} missing one of "
                     f"Filename/Label/PixelX/PixelY columns "
                     f"(found: {list(r.fieldnames or [])})")
        for row in r:
            try:
                rows.append({
                    "img": row[col_img],
                    "lbl": row[col_lbl],
                    "px":  float(row[col_px]),
                    "py":  float(row[col_py]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _looks_like_marks_csv(path: Path) -> bool:
    """Header sniff: Pix4D-format marks CSV.

    Accepts both sight.py's _write_pix4d output (Filename,Label,PixelX,PixelY)
    and extract_pix4d_marks.py's output (image_name,gcp_label,px,py) — must
    have at least one synonym for each of the four required columns.
    """
    try:
        with open(path) as f:
            header_cols = [h.strip().lower() for h in f.readline().strip().split(',')]
    except Exception:
        return False
    img_aliases = {'filename', 'image', 'image_name'}
    lbl_aliases = {'label', 'gcp_label', 'name'}
    px_aliases  = {'pixelx', 'px', 'x'}
    py_aliases  = {'pixely', 'py', 'y'}
    cols = set(header_cols)
    return (cols & img_aliases) and (cols & lbl_aliases) and \
           (cols & px_aliases)  and (cols & py_aliases)


# ---------------------------------------------------------------------------
# Mark-count tier check (the gate)
# ---------------------------------------------------------------------------

def count_marks_per_target(rows: List[dict],
                            tagged_only: bool = False) -> Dict[str, int]:
    """Count marks per target label.

    `tagged_only`: when True (sight {job}_tagged.txt input), restricts to
    rows with src=='tagged'. Pix4D marks have no src column; pass False.
    """
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        if tagged_only and r.get('src') != 'tagged':
            continue
        counts[r['lbl']] += 1
    return dict(counts)


def count_visible_images_per_target(estimate_rows: List[dict]) -> Dict[str, int]:
    """From sight estimates ({job}.txt), count distinct images per target.

    Used as the fairness-clamp upper bound — a target visible in only 4
    images can't possibly receive 7 marks.
    """
    by_lbl: Dict[str, set] = defaultdict(set)
    for r in estimate_rows:
        by_lbl[r['lbl']].add(r['img'])
    return {lbl: len(imgs) for lbl, imgs in by_lbl.items()}


def tier_classify(marks: int, visible: Optional[int] = None) -> str:
    """Return 'green' | 'amber' | 'red' for one target.

    When `visible` is known (sight mode), apply fairness clamp: green
    threshold = min(TIER_GREEN_FLOOR, visible). Without `visible` (Pix4D
    mode), use raw thresholds — over-strict for targets visible in <7
    images, but the alternative requires Pix4D's image-coverage list
    which we don't yet parse.
    """
    if visible is not None:
        green_t = min(TIER_GREEN_FLOOR, visible)
        amber_t = min(TIER_AMBER_FLOOR, visible)
    else:
        green_t = TIER_GREEN_FLOOR
        amber_t = TIER_AMBER_FLOOR
    if marks >= green_t: return 'green'
    if marks >= amber_t: return 'amber'
    return 'red'


def run_tier_check(marks_counts: Dict[str, int],
                    visible_counts: Optional[Dict[str, int]] = None) -> Tuple[int, int, int, List[Tuple[str, int, Optional[int], str]]]:
    """Apply the 3-tier rule across all targets.

    Returns (n_red, n_amber, n_green, per_target_rows) where each row is
    (label, marks, visible_or_None, tier).
    """
    rows: List[Tuple[str, int, Optional[int], str]] = []
    for lbl in sorted(marks_counts):
        m = marks_counts[lbl]
        v = visible_counts.get(lbl) if visible_counts else None
        rows.append((lbl, m, v, tier_classify(m, v)))
    n_red   = sum(1 for _, _, _, t in rows if t == 'red')
    n_amber = sum(1 for _, _, _, t in rows if t == 'amber')
    n_green = sum(1 for _, _, _, t in rows if t == 'green')
    return n_red, n_amber, n_green, rows


_TIER_ORDER = {'red': 0, 'amber': 1, 'green': 2}


def _short_image_id(img_name: str) -> str:
    """Return the distinguishing tail of an image filename.

    Picks the last digit run in the name (typically the frame number for
    DJI captures: DJI_20260309171745_0197_V.JPG -> '0197'). Falls back to
    the full basename when no digits are present.
    """
    digits = _re.findall(r'\d+', img_name)
    return digits[-1] if digits else Path(img_name).stem


def _format_suspects(suspects: List[dict], max_show: int = 5) -> str:
    """Comma-separated list of suspect 'frame-id/residual' pairs, worst-first."""
    if not suspects:
        return ''
    ranked = sorted(suspects, key=lambda x: -x['residual'])
    shown = ranked[:max_show]
    items = [f"{_short_image_id(s['img'])}/{s['residual']:.0f}" for s in shown]
    extra = len(suspects) - len(shown)
    s = ', '.join(items)
    if extra > 0:
        s += f' +{extra}'
    return s


def _build_legend(anchor_px: float, suspect_px: float) -> str:
    return (
        "Columns:\n"
        "  est        # of distinct images sight identified as containing this target\n"
        "             (only when sight estimates are loaded; '-' otherwise).\n"
        "  marks      # of (image, target) tags placed for this target.\n"
        "  tier       red(<3) | amber(3-6) | green(>=7) marks. Fairness clamp:\n"
        "             green threshold = min(7, est) when est is known.\n"
        f"  anc        Anchors: marks within {anchor_px:.0f} px of a color/tri_color estimate\n"
        "             (= you accepted sight's pixel). Tune with --anchor-px.\n"
        "  col        # of marks whose sight estimate was 'color' or 'tri_color'\n"
        "             (sub-pixel-accurate). Higher = stronger consensus signal.\n"
        "  frac       anc / col. High = mostly accepted sight; low = mostly disagreed.\n"
        "  cons_src   How consensus offset was computed:\n"
        "               anchors  mean of anchor offsets (~= 0 by construction)\n"
        "               color    median of color/tri_color tag offsets — fires when too few\n"
        "                        anchors; magnitude reveals systematic disagreement between\n"
        "                        marks and sight color refinement\n"
        "               all      median of all marks (any source) — last fallback\n"
        "  cons_off   |consensus offset| in pixels. ~= 0 when you and sight agree;\n"
        "             larger values mean look at why they disagree (could be your\n"
        "             tagging OR sight's color refinement at fault).\n"
        f"  image/offset  Color/tri_color marks whose residual from consensus exceeds\n"
        f"             {suspect_px:.0f} px (tune with --suspect-px). Worst-first, formatted as\n"
        "             frame-id/residual_px, comma-separated. '+N' = N more not shown.\n"
        "             Each entry points at one image to review for this target.\n"
        "\n"
        "Sort: tier (red -> amber -> green), then by suspiciousness within tier.\n"
    )


def _print_consolidated_table(tier_rows: List[Tuple[str, int, Optional[int], str]],
                                consensus_by_norm: Dict[str, dict],
                                normalizer,
                                top: Optional[int],
                                have_estimates: bool) -> None:
    """Single sorted table merging tier classification and consensus stats.

    tier_rows: (display_label, marks, est_or_None, tier) per target.
    consensus_by_norm: norm_label -> consensus result dict; empty if no estimates.
    """
    def score_for(disp_label: str) -> float:
        c = consensus_by_norm.get(normalizer(disp_label))
        return c['score'] if c else 0.0

    sorted_rows = sorted(tier_rows,
                         key=lambda r: (_TIER_ORDER[r[3]], -score_for(r[0]), r[0]))
    if top is not None:
        sorted_rows = sorted_rows[:top]

    have_consensus = bool(consensus_by_norm)

    # Auto-size label column to longest label seen, with min 5 (header "label").
    label_w = max([len('label')] + [len(r[0]) for r in sorted_rows]) if sorted_rows else len('label')

    if have_consensus:
        print(f"{'rank':>4s}  {'label':<{label_w}s}  {'est':>3s} {'marks':>5s} {'tier':>5s}  "
              f"{'anc':>3s} {'col':>3s} {'frac':>4s}  {'cons_src':>8s} {'cons_off':>8s}  "
              f"image/offset (worst first)")
    elif have_estimates:
        print(f"{'rank':>4s}  {'label':<{label_w}s}  {'est':>3s} {'marks':>5s} {'tier':>5s}")
    else:
        print(f"{'rank':>4s}  {'label':<{label_w}s}  {'marks':>5s} {'tier':>5s}")

    for i, (lbl, marks, est, tier) in enumerate(sorted_rows, 1):
        est_str = f"{est}" if est is not None else "-"
        c = consensus_by_norm.get(normalizer(lbl))
        if have_consensus and c:
            suspects_str = _format_suspects(c['suspects'])
            print(f"{i:4d}  {lbl:<{label_w}s}  {est_str:>3s} {marks:5d} {tier:>5s}  "
                  f"{c['n_anchor']:3d} {c['n_color']:3d} {c['frac_anchor']:4.2f}  "
                  f"{c['consensus_src']:>8s} {c['consensus_mag']:8.1f}  "
                  f"{suspects_str}")
        elif have_consensus:
            print(f"{i:4d}  {lbl:<{label_w}s}  {est_str:>3s} {marks:5d} {tier:>5s}  "
                  f"{'-':>3s} {'-':>3s} {'-':>4s}  {'-':>8s} {'-':>8s}  -")
        elif have_estimates:
            print(f"{i:4d}  {lbl:<{label_w}s}  {est_str:>3s} {marks:5d} {tier:>5s}")
        else:
            print(f"{i:4d}  {lbl:<{label_w}s}  {marks:5d} {tier:>5s}")


def _composite_score(
    n_color_tags: int,
    n_color_suspect: int,
    frac_anchor_of_color: float,
    consensus_mag: float,
) -> float:
    """
    Suspicion score: higher = more suspect.  Restricted to color/tri_color
    tag inconsistency because projection-source tags have an expected
    pixel-space offset distribution driven by per-camera EXIF noise (50-
    200 px), which is irrelevant to whether the user tagged the right
    feature.

    Components, each contributing ~[0, 1]:
      - color_suspect_frac: n_color_suspect / max(n_color_tags, 1) * 2.0
                            primary signal; user disagreed with color refinement
      - low_color_anchor:   max(0, 0.3 - frac_anchor_of_color) * 3.33
                            1.0 if no color tag matches estimate; 0 if >= 30 %
      - high_consensus:     min(1.0, max(0, consensus_mag - 20) / 80)
                            fires when *all* color tags disagree with sight by
                            > 20 px in a consistent direction
                            (sight color refinement found the wrong feature)
    """
    color_suspect_frac = (n_color_suspect / n_color_tags * 2.0) if n_color_tags else 0.0
    low_color_anchor = max(0.0, 0.3 - frac_anchor_of_color) * 3.33
    high_consensus = min(1.0, max(0.0, consensus_mag - 20.0) / 80.0)
    return color_suspect_frac + low_color_anchor + high_consensus


import re as _re


def _normalize_label(lbl: str) -> str:
    """Normalize a target label for cross-toolchain matching.

    Strips sight's CHK-/GCP-/DUP- prefix and any -dup\\d* suffix, then keeps
    only alphanumerics. So 'CHK-131-2', 'GCP-131-2', '131-2', '131 2', and
    'CHK-131-2-dup' all normalize to '1312'. Lets a Pix4D tag named
    '131 2' match a sight estimate named 'CHK-131-2'.
    """
    s = lbl
    for pfx in ('CHK-', 'GCP-', 'DUP-'):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    s = _re.sub(r'-dup\d*$', '', s)
    return _re.sub(r'[^A-Za-z0-9]', '', s).upper()


def _find_sight_estimates_sibling(marks_path: Path) -> Optional[Path]:
    """Look in marks_path.parent for a sight pre-tagging estimates file
    matching the marks-file stem.

    Strips known marks-file suffixes from the filename to derive the job
    stem, then looks for `{stem}.txt` adjacent. Filename-stem match avoids
    accidentally picking up rmse.py ortho-tagging files (which also parse
    as sight format but have one row per target, not per (image, target)).

    Returns None if no stem-matched file exists or the candidate doesn't
    parse as sight format with non-zero rows.

    Strip patterns (in order):
        {stem}_<digits>_<conf>_marks.csv  →  {stem}        (sight._write_pix4d output)
        {stem}_marks.csv                  →  {stem}        (extract_pix4d_marks output)
        {stem}.p4mpl                      →  {stem}        (Pix4D Matic history)
        {stem}.csv                        →  {stem}        (generic)
    """
    name = marks_path.name
    candidates_stems: List[str] = []
    # Specific marks-file naming patterns first
    m = _re.match(r'^(.+?)_\d+_[A-Za-z_]+_marks\.csv$', name)
    if m: candidates_stems.append(m.group(1))
    m = _re.match(r'^(.+?)_marks\.csv$', name)
    if m: candidates_stems.append(m.group(1))
    # .p4mpl: take the basename stem
    if marks_path.suffix.lower() == '.p4mpl':
        candidates_stems.append(marks_path.stem)
    # Fallback: bare stem
    candidates_stems.append(marks_path.stem)

    for stem in candidates_stems:
        cand = marks_path.parent / f"{stem}.txt"
        if not cand.exists() or cand.name.endswith('_tagged.txt'):
            continue
        try:
            _, rows = _load_rows(cand)
        except Exception:
            continue
        if rows:
            return cand
    return None


def analyse(
    est_rows: List[dict],
    tag_rows: List[dict],
    *,
    anchor_px: float,
    suspect_px: float,
    min_anchors: int,
    tagged_only: bool = True,
    label_normalizer=None,
) -> List[dict]:
    """Per-target consensus / anchor analysis.

    est_rows, tag_rows: pre-loaded row lists with keys img, lbl, px, py
    (sight rows also carry src; Pix4D rows omit it).

    label_normalizer: callable lbl→canonical for matching across toolchains.
    Default: identity (sight labels match sight labels exactly).

    Returns per-target dicts sorted by descending suspicion score.
    """
    if label_normalizer is None:
        label_normalizer = lambda x: x

    est_by_li = {(label_normalizer(r['lbl']), r['img']): r for r in est_rows}
    tagged = [r for r in tag_rows
              if (not tagged_only) or r.get('src') == 'tagged']

    by_norm: Dict[str, List[dict]] = defaultdict(list)
    display_labels: Dict[str, str] = {}
    for t in tagged:
        norm = label_normalizer(t['lbl'])
        e = est_by_li.get((norm, t['img']))
        if e is None:
            continue
        if norm not in display_labels:
            display_labels[norm] = t['lbl']
        dx = t['px'] - e['px']
        dy = t['py'] - e['py']
        by_norm[norm].append({
            "img":    t['img'],
            "tag_px": t['px'], "tag_py": t['py'],
            "est_px": e['px'], "est_py": e['py'],
            "src":    e.get('src', ''),
            "dx":     dx, "dy": dy,
            "offset_mag": math.hypot(dx, dy),
        })

    out: List[dict] = []
    for norm, tags in by_norm.items():
        n = len(tags)
        # Restrict consensus + suspect detection to tags whose estimate came
        # from color refinement (color or tri_color sources). Projection-
        # source tags have an expected pixel offset driven by per-camera
        # EXIF pose noise (~50-200 px) that is unrelated to whether the
        # user tagged the right feature.
        color_tags = [t for t in tags if t["src"] in ("color", "tri_color")]
        n_color = len(color_tags)
        anchors = [t for t in color_tags if t["offset_mag"] < anchor_px]
        n_anc = len(anchors)

        if n_anc >= min_anchors:
            cdx = statistics.mean([t["dx"] for t in anchors])
            cdy = statistics.mean([t["dy"] for t in anchors])
            consensus_src = "anchors"
        elif n_color >= 1:
            cdx = statistics.median([t["dx"] for t in color_tags])
            cdy = statistics.median([t["dy"] for t in color_tags])
            consensus_src = "color"
        elif n >= 1:
            cdx = statistics.median([t["dx"] for t in tags])
            cdy = statistics.median([t["dy"] for t in tags])
            consensus_src = "all"
        else:
            continue

        for t in tags:
            t["residual"] = math.hypot(t["dx"] - cdx, t["dy"] - cdy)
        color_suspects = [t for t in color_tags if t["residual"] > suspect_px]
        consensus_mag = math.hypot(cdx, cdy)
        frac_anchor_of_color = (n_anc / n_color) if n_color else 0.0

        score = _composite_score(
            n_color_tags=n_color,
            n_color_suspect=len(color_suspects),
            frac_anchor_of_color=frac_anchor_of_color,
            consensus_mag=consensus_mag,
        )

        out.append({
            "label":          display_labels[norm],
            "norm_label":     norm,
            "n_tagged":       n,
            "n_color":        n_color,
            "n_anchor":       n_anc,
            "frac_anchor":    frac_anchor_of_color,
            "consensus_dx":   cdx,
            "consensus_dy":   cdy,
            "consensus_mag":  consensus_mag,
            "consensus_src":  consensus_src,
            "n_suspect":      len(color_suspects),
            "max_resid":      max((t["residual"] for t in color_tags), default=0.0),
            "med_resid":      statistics.median([t["residual"] for t in color_tags]) if color_tags else 0.0,
            "score":          score,
            "tags":           tags,
            "suspects":       color_suspects,
        })

    out.sort(key=lambda r: -r["score"])
    return out


def _resolve_inputs(args) -> Tuple[str, List[Path]]:
    """Detect input mode from CLI args.

    Returns (mode, paths) where mode is one of:
      'sight'       — paths = [estimates ({job}.txt), tagged ({job}_tagged.txt)]
      'pix4d_p4mpl' — paths = [history.p4mpl]
      'pix4d_csv'   — paths = [marks.csv]
    """
    paths = [Path(p) for p in args.input]
    for p in paths:
        if not p.exists():
            sys.exit(f"ERROR: input file not found: {p}")

    if len(paths) == 2:
        return 'sight', paths

    p = paths[0]
    if p.suffix.lower() == '.p4mpl':
        return 'pix4d_p4mpl', paths
    if p.suffix.lower() == '.csv' and _looks_like_marks_csv(p):
        return 'pix4d_csv', paths
    if p.name.endswith('_tagged.txt'):
        # Auto-locate sibling estimates file: strip "_tagged" from stem
        est = p.parent / (p.name[:-len('_tagged.txt')] + '.txt')
        if not est.exists():
            sys.exit(f"ERROR: cannot auto-locate estimates file. Expected: {est}\n"
                     f"       Pass it explicitly: check_tags.py {est} {p}")
        return 'sight', [est, p]
    sys.exit(f"ERROR: cannot determine input mode for {p}.\n"
             f"       Sight mode:  check_tags.py {{job}}.txt {{job}}_tagged.txt\n"
             f"       Pix4D mode:  check_tags.py {{path/to/history.p4mpl|marks.csv}}")


def _load_marks(mode: str, marks_path: Path) -> List[dict]:
    """Load tag/marks rows in the right shape for the given mode."""
    if mode == 'sight':
        _, rows = _load_rows(marks_path)
        return rows
    if mode == 'pix4d_p4mpl':
        return _load_pix4d_history(marks_path)
    if mode == 'pix4d_csv':
        return _load_pix4d_marks_csv(marks_path)
    sys.exit(f"ERROR: unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  SIGHT (2 args, or 1 ending in _tagged.txt):\n"
            "      check_tags.py {job}.txt {job}_tagged.txt\n"
            "      check_tags.py {job}_tagged.txt   # auto-locates {job}.txt\n"
            "  PIX4D (1 arg, .p4mpl or marks.csv):\n"
            "      check_tags.py path/to/history.p4mpl\n"
            "      check_tags.py path/to/{job}_marks.csv\n"
            "      # Auto-locates {job}.txt sibling for consensus analysis\n"
            "      # (override with --estimates PATH)\n"
        ),
    )
    ap.add_argument("input", nargs='+', type=str,
                    help="Input file(s). See modes in epilog.")
    ap.add_argument("--estimates", type=Path, default=None,
                    help="Sight estimates file ({job}.txt) for consensus "
                         "analysis. Auto-located when omitted (Pix4D modes).")
    ap.add_argument("--anchor-px", type=float, default=10.0,
                    help="Mark within this many px of a 'color'/'tri_color' "
                         "estimate is treated as anchor (default 10).")
    ap.add_argument("--suspect-px", type=float, default=50.0,
                    help="Mark residual from consensus that flags it as "
                         "informational suspect (default 50).")
    ap.add_argument("--min-anchors", type=int, default=3,
                    help="Minimum anchors for anchor-based consensus instead "
                         "of median consensus (default 3).")
    ap.add_argument("--top", type=int, default=None,
                    help="Limit table to this many top-ranked targets "
                         "(default: show all).")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optionally write per-target consensus metrics to "
                         "this TSV.")
    ap.add_argument("--no-consensus", action='store_true',
                    help="Skip the consensus/anchor analysis. Useful for "
                         "fast tier-only checks.")
    ap.add_argument("--no-legend", action='store_true',
                    help="Skip the column-legend block before the table.")
    args = ap.parse_args()

    # ------- Resolve inputs -------
    mode, paths = _resolve_inputs(args)
    marks_path = paths[-1]
    if mode == 'sight':
        est_path = paths[0]
        tagged_only = True
    else:
        # Pix4D mode: try --estimates, then auto-locate sibling.
        if args.estimates:
            est_path = args.estimates
            if not est_path.exists():
                sys.exit(f"ERROR: --estimates file not found: {est_path}")
        else:
            est_path = _find_sight_estimates_sibling(marks_path)
        tagged_only = False

    # ------- Load marks + estimates -------
    tag_rows = _load_marks(mode, marks_path)
    print(f"Input: {mode.upper()} — marks: {marks_path.name} ({len(tag_rows)} row(s))")

    est_rows: List[dict] = []
    if est_path is not None:
        try:
            _, est_rows = _load_rows(est_path)
            print(f"       estimates: {est_path.name} ({len(est_rows)} row(s))")
        except Exception as e:
            print(f"       WARNING: could not load estimates ({e})", file=sys.stderr)
            est_rows = []
    elif mode != 'sight':
        print("       estimates: (none — pass --estimates PATH for consensus + fairness clamp)")

    # Label normalizer: identity in sight mode (labels match exactly);
    # cross-toolchain normalizer in Pix4D modes (e.g. '131 2' <-> 'CHK-131-2').
    normalizer = (lambda x: x) if mode == 'sight' else _normalize_label

    if not tag_rows:
        print("No marks found in input. Nothing to check.")
        return 0

    # ------- Build per-target marks + estimates counts (normalized keys) -------
    marks_by_norm: Dict[str, int] = defaultdict(int)
    display_labels: Dict[str, str] = {}
    for r in tag_rows:
        if tagged_only and r.get('src') != 'tagged':
            continue
        norm = normalizer(r['lbl'])
        marks_by_norm[norm] += 1
        display_labels.setdefault(norm, r['lbl'])

    if not marks_by_norm:
        print("No marks found in input (filtering by tagged confidence). Nothing to check.")
        return 0

    est_by_norm: Dict[str, set] = defaultdict(set)
    for r in est_rows:
        est_by_norm[normalizer(r['lbl'])].add(r['img'])

    # ------- Tier check (the gate) -------
    tier_rows: List[Tuple[str, int, Optional[int], str]] = []
    for norm, count in marks_by_norm.items():
        est_count = len(est_by_norm[norm]) if norm in est_by_norm else None
        tier = tier_classify(count, est_count if est_rows else None)
        tier_rows.append((display_labels[norm], count, est_count, tier))

    n_red   = sum(1 for r in tier_rows if r[3] == 'red')
    n_amber = sum(1 for r in tier_rows if r[3] == 'amber')
    n_green = sum(1 for r in tier_rows if r[3] == 'green')

    print()
    print(f"GCPEditorPro 3-tier check:  "
          f"{n_green} green, {n_amber} amber, {n_red} red  of {len(tier_rows)} target(s)")
    if est_rows:
        print(f"  Fairness clamp ON: green threshold = "
              f"min({TIER_GREEN_FLOOR}, est) per target.")
    else:
        print(f"  Fairness clamp OFF (no estimates): green threshold = "
              f"{TIER_GREEN_FLOOR} regardless of visibility.")

    # ------- Consensus analysis (reviewer aid) -------
    consensus_results: List[dict] = []
    consensus_by_norm: Dict[str, dict] = {}
    if est_rows and not args.no_consensus:
        consensus_results = analyse(
            est_rows, tag_rows,
            anchor_px=args.anchor_px,
            suspect_px=args.suspect_px,
            min_anchors=args.min_anchors,
            tagged_only=tagged_only,
            label_normalizer=normalizer,
        )
        consensus_by_norm = {r['norm_label']: r for r in consensus_results}

    # ------- Print legend + consolidated table -------
    print()
    if not args.no_legend:
        print(_build_legend(args.anchor_px, args.suspect_px))
    _print_consolidated_table(
        tier_rows=tier_rows,
        consensus_by_norm=consensus_by_norm,
        normalizer=normalizer,
        top=args.top,
        have_estimates=bool(est_rows),
    )

    # ------- Optional report sidecar -------
    if args.report is not None and consensus_results:
        try:
            with open(args.report, "w") as f:
                f.write("label\tnorm_label\tn_tagged\tn_color\tn_anchor\tfrac_anchor\t"
                        "consensus_mag_px\tconsensus_src\tn_suspect\t"
                        "med_resid_px\tmax_resid_px\tscore\n")
                for r in consensus_results:
                    f.write(f"{r['label']}\t{r['norm_label']}\t{r['n_tagged']}\t{r['n_color']}\t"
                            f"{r['n_anchor']}\t{r['frac_anchor']:.4f}\t"
                            f"{r['consensus_mag']:.3f}\t{r['consensus_src']}\t"
                            f"{r['n_suspect']}\t{r['med_resid']:.3f}\t"
                            f"{r['max_resid']:.3f}\t{r['score']:.4f}\n")
            print(f"\nWrote per-target consensus report: {args.report}")
        except Exception as e:
            print(f"\nWARNING: could not write report ({e})", file=sys.stderr)

    # ------- Exit -------
    if n_red > 0:
        print(file=sys.stderr)
        print(f"FAIL: {n_red} target(s) have < {TIER_AMBER_FLOOR} marks "
              f"(insufficient for bundle adjustment).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
