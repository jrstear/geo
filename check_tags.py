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


def _print_tier_table(rows: List[Tuple[str, int, Optional[int], str]],
                       visible_known: bool) -> None:
    """Pretty-print the per-target tier table, worst-tier first."""
    tier_order = {'red': 0, 'amber': 1, 'green': 2}
    rows_sorted = sorted(rows, key=lambda r: (tier_order[r[3]], -r[1] if r[3] == 'green' else r[1], r[0]))
    if visible_known:
        print(f"{'label':<14s} {'marks':>5s} {'visible':>7s} {'tier':>6s}")
        for lbl, m, v, t in rows_sorted:
            v_str = f"{v}" if v is not None else "-"
            print(f"{lbl:<14s} {m:5d} {v_str:>7s} {t:>6s}")
    else:
        print(f"{'label':<14s} {'marks':>5s} {'tier':>6s}")
        for lbl, m, _, t in rows_sorted:
            print(f"{lbl:<14s} {m:5d} {t:>6s}")


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


def analyse(
    estimates_path: Path,
    tagged_path: Path,
    *,
    anchor_px: float,
    suspect_px: float,
    min_anchors: int,
) -> List[dict]:
    """
    Returns a list of per-target dicts, one per target with >= 1 tagged row.
    Each dict has the keys printed by main().
    """
    _, est_rows = _load_rows(estimates_path)
    _, tag_rows = _load_rows(tagged_path)

    est_by_li = {(r["lbl"], r["img"]): r for r in est_rows}
    tagged = [r for r in tag_rows if r["src"] == "tagged"]

    by_label: Dict[str, List[dict]] = defaultdict(list)
    for t in tagged:
        e = est_by_li.get((t["lbl"], t["img"]))
        if e is None:
            continue
        dx = t["px"] - e["px"]
        dy = t["py"] - e["py"]
        by_label[t["lbl"]].append({
            "img":    t["img"],
            "tag_px": t["px"], "tag_py": t["py"],
            "est_px": e["px"], "est_py": e["py"],
            "src":    e["src"],
            "dx":     dx, "dy": dy,
            "offset_mag": math.hypot(dx, dy),
        })

    out: List[dict] = []
    for lbl, tags in by_label.items():
        n = len(tags)
        # Restrict consensus + suspect detection to tags whose estimate came
        # from color refinement (color or tri_color sources).  Projection-
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
            consensus_src = "color_median"
        elif n >= 1:
            # Fallback when no color hits at all — use median of all tags.
            cdx = statistics.median([t["dx"] for t in tags])
            cdy = statistics.median([t["dy"] for t in tags])
            consensus_src = "all_median"
        else:
            continue

        # Per-tag residual from consensus (computed for all tags so the
        # report can show projection-tag residuals too if useful).
        for t in tags:
            t["residual"] = math.hypot(t["dx"] - cdx, t["dy"] - cdy)
        # Suspect = color/tri_color tag whose residual exceeds the threshold.
        # Projection tags are reported but never marked suspect themselves.
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
            "label":          lbl,
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
        ),
    )
    ap.add_argument("input", nargs='+', type=str,
                    help="Input file(s). See modes in epilog.")
    ap.add_argument("--anchor-px", type=float, default=10.0,
                    help="Tag within this many px of a 'color'/'tri_color' "
                         "estimate is treated as anchor for consensus analysis "
                         "(default 10). Sight mode only.")
    ap.add_argument("--suspect-px", type=float, default=50.0,
                    help="Tag residual from consensus that flags it as "
                         "informational suspect (default 50). Sight mode only.")
    ap.add_argument("--min-anchors", type=int, default=3,
                    help="Minimum anchors for anchor-based consensus instead "
                         "of median consensus (default 3). Sight mode only.")
    ap.add_argument("--top", type=int, default=15,
                    help="Show this many top-suspect targets in the consensus "
                         "table (default 15). Sight mode only.")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optionally write per-target consensus metrics to "
                         "this TSV. Sight mode only.")
    ap.add_argument("--no-consensus", action='store_true',
                    help="Skip the consensus/anchor reviewer-aid analysis "
                         "(sight mode only). Useful for fast tier-only checks.")
    args = ap.parse_args()

    mode, paths = _resolve_inputs(args)

    # ------- Load marks + (sight mode) estimates -------
    if mode == 'sight':
        est_path, tag_path = paths
        _, est_rows = _load_rows(est_path)
        _, tag_rows = _load_rows(tag_path)
        marks_counts   = count_marks_per_target(tag_rows, tagged_only=True)
        visible_counts = count_visible_images_per_target(est_rows)
        print(f"Input: SIGHT mode — estimates={est_path.name}, tagged={tag_path.name}")
    elif mode == 'pix4d_p4mpl':
        rows = _load_pix4d_history(paths[0])
        marks_counts   = count_marks_per_target(rows, tagged_only=False)
        visible_counts = None
        print(f"Input: PIX4D mode — history={paths[0].name} ({len(rows)} marks)")
    else:  # pix4d_csv
        rows = _load_pix4d_marks_csv(paths[0])
        marks_counts   = count_marks_per_target(rows, tagged_only=False)
        visible_counts = None
        print(f"Input: PIX4D mode — marks csv={paths[0].name} ({len(rows)} marks)")

    if not marks_counts:
        print("No marks found in input. Nothing to check.")
        return 0

    # ------- CHECK 1: marks-per-target tier (the gate) -------
    n_red, n_amber, n_green, tier_rows = run_tier_check(marks_counts, visible_counts)
    n_total = len(tier_rows)

    print()
    print(f"Marks-per-target check (GCPEditorPro 3-tier):  "
          f"{n_green} green, {n_amber} amber, {n_red} red  "
          f"of {n_total} target(s)")
    if visible_counts is not None:
        print(f"  Fairness clamp ON (sight mode): green threshold = "
              f"min({TIER_GREEN_FLOOR}, visible_images_per_target)")
    else:
        print(f"  Fairness clamp OFF (Pix4D mode, no visibility data): "
              f"green threshold = {TIER_GREEN_FLOOR} regardless of visibility")
    print()
    _print_tier_table(tier_rows, visible_known=visible_counts is not None)

    # ------- CHECK 2: consensus / anchor analysis (reviewer aid) -------
    if mode == 'sight' and not args.no_consensus:
        results = analyse(
            paths[0], paths[1],
            anchor_px=args.anchor_px,
            suspect_px=args.suspect_px,
            min_anchors=args.min_anchors,
        )
        if results:
            print()
            print(f"Tag-vs-estimate consensus (reviewer aid; not gating):")
            print(f"  Anchor < {args.anchor_px:.0f} px from color/tri_color estimate;  "
                  f"suspect tag > {args.suspect_px:.0f} px from consensus.")
            print()
            print(f"{'rank':>4s}  {'label':<14s} {'n_tag':>5s} {'n_col':>5s} {'anc':>4s} "
                  f"{'frac':>5s} {'cons':>6s} {'#susp':>5s} {'med_r':>6s} {'max_r':>6s} "
                  f"{'cons_src':>11s} {'score':>5s}")
            for i, r in enumerate(results[:args.top], 1):
                print(f"{i:4d}  {r['label']:<14s} "
                      f"{r['n_tagged']:5d} {r['n_color']:5d} {r['n_anchor']:4d} "
                      f"{r['frac_anchor']:5.2f} {r['consensus_mag']:6.1f} {r['n_suspect']:5d} "
                      f"{r['med_resid']:6.1f} {r['max_resid']:6.1f} "
                      f"{r['consensus_src']:>11s} {r['score']:5.2f}")
            if args.report is not None:
                try:
                    with open(args.report, "w") as f:
                        f.write("label\tn_tagged\tn_color\tn_anchor\tfrac_anchor\t"
                                "consensus_mag_px\tconsensus_src\tn_suspect\t"
                                "med_resid_px\tmax_resid_px\tscore\n")
                        for r in results:
                            f.write(f"{r['label']}\t{r['n_tagged']}\t{r['n_color']}\t"
                                    f"{r['n_anchor']}\t{r['frac_anchor']:.4f}\t"
                                    f"{r['consensus_mag']:.3f}\t{r['consensus_src']}\t"
                                    f"{r['n_suspect']}\t{r['med_resid']:.3f}\t"
                                    f"{r['max_resid']:.3f}\t{r['score']:.4f}\n")
                    print(f"\nWrote per-target consensus report: {args.report}")
                except Exception as e:
                    print(f"\nWARNING: could not write report ({e})", file=sys.stderr)

    # ------- Exit code: red count from tier check -------
    if n_red > 0:
        print(file=sys.stderr)
        print(f"FAIL: {n_red} target(s) have < {TIER_AMBER_FLOOR} marks "
              f"(insufficient for bundle adjustment).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
