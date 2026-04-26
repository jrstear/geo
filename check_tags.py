#!/usr/bin/env python3
"""
check_tags.py — pre-ODM bad-tag detector.

Compares user-confirmed pixel tags in {job}_tagged.txt against sight.py's
estimates in {job}.txt, ranks targets by suspicion, flags individual tags
that diverge from the per-target anchor consensus, and exits non-zero if
any target's suspicion score exceeds a configurable gate.

Use as a checklist before launching ODM on EC2 — catastrophic per-target
mistakes (e.g. user clicked a base station instead of the target) typically
rank at the top of the suspect list and would have produced a failed ODM
run had they reached it.

Usage:
    conda run -n geo python check_tags.py \\
        {job}.txt {job}_tagged.txt

    # Optional: write per-target metrics to a TSV alongside stdout
    conda run -n geo python check_tags.py {job}.txt {job}_tagged.txt \\
        --report report.tsv

    # Tune thresholds (defaults derived from aztec analysis):
    #   --anchor-px N    distance from estimate where a tag is treated as
    #                    confirming the estimate (default 10 px)
    #   --suspect-px N   per-tag residual from consensus that flags a tag
    #                    (default 50 px)
    #   --min-anchors N  minimum anchors needed to use anchor-based consensus
    #                    instead of median consensus (default 3)
    #   --gate-score X   exit code 1 if any target's suspicion score exceeds X
    #                    (default 0.7; set 0 to never gate)

Inputs are tab-separated with an EPSG header line followed by data rows:
    geo_x geo_y geo_z px py image_name gcp_label confidence [marker_bbox]

The detector reads:
    - estimates: rows from {job}.txt, where confidence is 'color',
      'tri_color', 'tri_proj', or 'projection'
    - tags:      rows from {job}_tagged.txt with confidence == 'tagged'

Algorithm:
    For each target with >= 1 tagged row:
      1. Pair tagged rows with their estimate (by (image, label))
      2. Compute per-tag pixel offset = tag - estimate
      3. Identify anchors = tags whose estimate is 'color' or 'tri_color'
         AND whose |offset| < ANCHOR_PX (the user accepted the high-
         confidence estimate sub-pixel)
      4. Consensus offset = mean of anchors (if >= MIN_ANCHORS) else
         median of all offsets
      5. Per-tag residual = |offset - consensus|
      6. Per-target metrics: frac_anchor, n_suspect, max_residual,
         consensus magnitude, bimodality flag
      7. Composite suspicion score; rank targets by it

Output:
    Ranked target table (most suspect first) printed to stdout, with the
    individual flagged tags listed under each suspect target. Returns
    non-zero exit code if any target exceeds --gate-score (default 0.7).

Distinct from rmse.py (post-ODM, against reconstruction) and from
geo-aw0 (also post-ODM, triangulation spread). This runs pre-ODM and
needs no reconstruction.

See docs/plans/tag-quality-consistency.md.
"""
import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("estimates", type=Path,
                    help="sight.py output: {job}.txt")
    ap.add_argument("tagged", type=Path,
                    help="GCPEditorPro output: {job}_tagged.txt")
    ap.add_argument("--anchor-px", type=float, default=10.0,
                    help="Tag within this many px of a 'color'/'tri_color' "
                         "estimate is treated as anchor (default 10).")
    ap.add_argument("--suspect-px", type=float, default=50.0,
                    help="Tags whose residual from consensus exceeds this "
                         "are flagged (default 50).")
    ap.add_argument("--min-anchors", type=int, default=3,
                    help="Minimum anchors needed for anchor-based consensus "
                         "instead of median consensus (default 3).")
    ap.add_argument("--gate-score", type=float, default=0.7,
                    help="Exit non-zero if any target's suspicion score "
                         "exceeds this (default 0.7; set 0 to disable gate).")
    ap.add_argument("--top", type=int, default=15,
                    help="Show this many top-suspect targets in stdout "
                         "(default 15).")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optionally write per-target metrics to this TSV.")
    args = ap.parse_args()

    if not args.estimates.exists():
        sys.exit(f"ERROR: estimates file not found: {args.estimates}")
    if not args.tagged.exists():
        sys.exit(f"ERROR: tagged file not found: {args.tagged}")

    results = analyse(
        args.estimates, args.tagged,
        anchor_px=args.anchor_px,
        suspect_px=args.suspect_px,
        min_anchors=args.min_anchors,
    )

    if not results:
        print("No targets with both tagged rows and matching estimates were found.")
        return 0

    n_total = len(results)
    n_gated = sum(1 for r in results if r["score"] > args.gate_score) if args.gate_score > 0 else 0

    print(f"Analysed {n_total} target(s) with tagged data.  "
          f"Anchor < {args.anchor_px:.0f} px from color/tri_color estimate.  "
          f"Suspect tag > {args.suspect_px:.0f} px from consensus.")
    if args.gate_score > 0:
        print(f"Gate score = {args.gate_score:.2f}; {n_gated} target(s) exceed it.")
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

    flagged_targets = [r for r in results if r["score"] > args.gate_score] if args.gate_score > 0 else []
    if flagged_targets:
        print()
        print(f"Suspect tags in {len(flagged_targets)} flagged target(s) — review in GCPEditorPro:")
        for r in flagged_targets:
            print(f"  {r['label']} (score={r['score']:.2f}):")
            for t in sorted(r["suspects"], key=lambda x: -x["residual"])[:10]:
                print(f"    {t['img']:40s}  tag=({t['tag_px']:7.1f},{t['tag_py']:7.1f})  "
                      f"est=({t['est_px']:7.1f},{t['est_py']:7.1f})  "
                      f"src={t['src']:<11s} residual={t['residual']:6.1f} px")
            extra = len(r["suspects"]) - 10
            if extra > 0:
                print(f"    ... and {extra} more suspect tag(s)")

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
            print(f"\nWrote per-target report: {args.report}")
        except Exception as e:
            print(f"\nWARNING: could not write report ({e})")

    if args.gate_score > 0 and n_gated > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
