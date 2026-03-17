#!/usr/bin/env python3
"""
Color consistency analysis for GCP marker detections.

Analyzes LAB color properties of detected markers in confirmed-good vs
confirmed-bad pairs to determine if color consensus can be used as a
post-pass filter (R4).

Reads:
  - gcp_confirmed.txt (ground truth)
  - gcp_list-r3.txt (current refinement output with marker_bbox)
  - raw photos/ (drone images)

Outputs analysis to stdout.
"""
import math
import os
import sys
import json

import cv2
import numpy as np


def parse_gcp(path):
    rows = {}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            label = parts[6]
            key = (label, parts[5])
            bbox = None
            if len(parts) > 8 and parts[8]:
                b = parts[8].split(',')
                bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            rows[key] = {
                'px': float(parts[3]), 'py': float(parts[4]),
                'conf': parts[7], 'bbox': bbox,
            }
    return rows


def extract_marker_lab(img_dir, img_name, bbox):
    """Extract median LAB color of the marker bbox region."""
    path = os.path.join(img_dir, img_name)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    x1, y1, x2, y2 = bbox
    # Clamp to image bounds
    ih, iw = img.shape[:2]
    x1 = max(0, min(x1, iw - 1))
    x2 = max(0, min(x2, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    y2 = max(0, min(y2, ih - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    # Return median L, a, b
    med = np.median(lab.reshape(-1, 3), axis=0)
    std = np.std(lab.reshape(-1, 3), axis=0)
    return {
        'L': float(med[0]), 'a': float(med[1]), 'b': float(med[2]),
        'L_std': float(std[0]), 'a_std': float(std[1]), 'b_std': float(std[2]),
        'n_pixels': crop.shape[0] * crop.shape[1],
    }


def main():
    data_dir = "/Users/jrstear/stratus/ghostrider gulch"
    img_dir = os.path.join(data_dir, "raw photos")

    human = parse_gcp(os.path.join(data_dir, "gcp_confirmed.txt"))
    algo = parse_gcp(os.path.join(data_dir, "gcp_list-r3.txt"))

    # Compute errors and classify as good/bad
    records = []
    for key in sorted(human):
        gcp, img = key
        h = human[key]
        if key not in algo:
            continue
        a = algo[key]
        dx = a['px'] - h['px']
        dy = a['py'] - h['py']
        dist = math.sqrt(dx * dx + dy * dy)

        records.append({
            'gcp': gcp, 'img': img, 'dist': dist,
            'conf': a['conf'],
            'algo_bbox': a['bbox'],
            'human_bbox': h['bbox'],
        })

    dists = [r['dist'] for r in records]
    n = len(dists)
    good10 = sum(1 for d in dists if d < 10)
    bad50 = sum(1 for d in dists if d >= 50)
    print(f"=== R3 Baseline: N={n}, Mean={sum(dists)/n:.1f}, Good(<10)={good10}, Bad(>=50)={bad50} ===")
    print()

    # Extract colors from algo bboxes (what the algorithm detected)
    print("Extracting marker colors from detected bboxes...")
    for r in records:
        r['algo_color'] = None
        r['human_color'] = None
        if r['algo_bbox'] and r['conf'] == 'color_refined':
            r['algo_color'] = extract_marker_lab(img_dir, r['img'], r['algo_bbox'])
        if r['human_bbox']:
            r['human_color'] = extract_marker_lab(img_dir, r['img'], r['human_bbox'])

    # Classify
    good_records = [r for r in records if r['dist'] < 10 and r['algo_color'] is not None]
    bad_records = [r for r in records if r['dist'] >= 50 and r['algo_color'] is not None]
    mid_records = [r for r in records if 10 <= r['dist'] < 50 and r['algo_color'] is not None]

    print(f"\nWith color data: good={len(good_records)}, bad={len(bad_records)}, mid={len(mid_records)}")

    # ---- Analysis 1: Overall color distributions ----
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Overall LAB color distributions")
    print("=" * 70)

    for label, recs in [("GOOD (<10px)", good_records), ("BAD (>=50px)", bad_records)]:
        if not recs:
            continue
        L_vals = [r['algo_color']['L'] for r in recs]
        a_vals = [r['algo_color']['a'] for r in recs]
        b_vals = [r['algo_color']['b'] for r in recs]
        print(f"\n  {label} (n={len(recs)}):")
        print(f"    L*: mean={np.mean(L_vals):.1f}, std={np.std(L_vals):.1f}, "
              f"range=[{np.min(L_vals):.1f}, {np.max(L_vals):.1f}]")
        print(f"    a*: mean={np.mean(a_vals):.1f}, std={np.std(a_vals):.1f}, "
              f"range=[{np.min(a_vals):.1f}, {np.max(a_vals):.1f}]")
        print(f"    b*: mean={np.mean(b_vals):.1f}, std={np.std(b_vals):.1f}, "
              f"range=[{np.min(b_vals):.1f}, {np.max(b_vals):.1f}]")

    # ---- Analysis 2: Intra-GCP variance ----
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Intra-GCP color consistency")
    print("=" * 70)

    # Group by GCP
    from collections import defaultdict
    gcp_good = defaultdict(list)
    gcp_bad = defaultdict(list)
    gcp_all = defaultdict(list)

    for r in good_records:
        gcp_good[r['gcp']].append(r)
        gcp_all[r['gcp']].append(r)
    for r in bad_records:
        gcp_bad[r['gcp']].append(r)
        gcp_all[r['gcp']].append(r)
    for r in mid_records:
        gcp_all[r['gcp']].append(r)

    print(f"\n  Per-GCP color stats (using GOOD detections only as ground truth):")
    print(f"  {'GCP':<10} {'n_good':>6} {'n_bad':>6} {'L_std':>6} {'a_std':>6} {'b_std':>6} "
          f"{'a_med':>6} {'b_med':>6}")
    print("  " + "-" * 60)

    gcp_consensus = {}  # Store consensus colors for each GCP
    for gcp in sorted(gcp_all.keys()):
        good = gcp_good.get(gcp, [])
        bad = gcp_bad.get(gcp, [])
        if good:
            L_vals = [r['algo_color']['L'] for r in good]
            a_vals = [r['algo_color']['a'] for r in good]
            b_vals = [r['algo_color']['b'] for r in good]
            gcp_consensus[gcp] = {
                'L': np.median(L_vals), 'a': np.median(a_vals), 'b': np.median(b_vals),
                'L_std': np.std(L_vals), 'a_std': np.std(a_vals), 'b_std': np.std(b_vals),
            }
            print(f"  {gcp:<10} {len(good):>6} {len(bad):>6} "
                  f"{np.std(L_vals):>6.1f} {np.std(a_vals):>6.1f} {np.std(b_vals):>6.1f} "
                  f"{np.median(a_vals):>6.1f} {np.median(b_vals):>6.1f}")
        else:
            print(f"  {gcp:<10} {0:>6} {len(bad):>6}   (no good detections for consensus)")

    # ---- Analysis 3: Can color distance distinguish good from bad? ----
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Color distance from GCP consensus (chrominance only: a*, b*)")
    print("=" * 70)

    # For each detection, compute distance from its GCP's consensus color
    color_dists_good = []
    color_dists_bad = []
    color_dists_mid = []

    detail_lines = []

    for r in records:
        if r['algo_color'] is None:
            continue
        gcp = r['gcp']
        if gcp not in gcp_consensus:
            continue
        cons = gcp_consensus[gcp]
        # Chrominance distance (a*, b* only — ignore L* which varies with lighting)
        da = r['algo_color']['a'] - cons['a']
        db = r['algo_color']['b'] - cons['b']
        cdist = math.sqrt(da * da + db * db)

        # Also compute full LAB distance
        dL = r['algo_color']['L'] - cons['L']
        lab_dist = math.sqrt(dL * dL + da * da + db * db)

        r['color_dist_ab'] = cdist
        r['color_dist_lab'] = lab_dist

        if r['dist'] < 10:
            color_dists_good.append(cdist)
        elif r['dist'] >= 50:
            color_dists_bad.append(cdist)
        else:
            color_dists_mid.append(cdist)

        detail_lines.append(
            f"  {r['dist']:7.1f}px  ab_dist={cdist:5.1f}  lab_dist={lab_dist:5.1f}  "
            f"a={r['algo_color']['a']:5.1f} b={r['algo_color']['b']:5.1f}  "
            f"L={r['algo_color']['L']:5.1f}  {r['gcp']:<10} {r['img']}"
        )

    print(f"\n  Chrominance distance (a*, b*) from GCP consensus:")
    for label, vals in [("GOOD (<10px)", color_dists_good),
                         ("BAD (>=50px)", color_dists_bad),
                         ("MID (10-50px)", color_dists_mid)]:
        if vals:
            print(f"    {label}: n={len(vals)}, mean={np.mean(vals):.1f}, "
                  f"std={np.std(vals):.1f}, median={np.median(vals):.1f}, "
                  f"max={np.max(vals):.1f}, p90={np.percentile(vals, 90):.1f}")

    # ---- Analysis 4: What threshold would separate good from bad? ----
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: Threshold analysis for color consensus filter")
    print("=" * 70)

    if color_dists_good and color_dists_bad:
        for thresh in [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]:
            good_rejected = sum(1 for d in color_dists_good if d > thresh)
            bad_rejected = sum(1 for d in color_dists_bad if d > thresh)
            print(f"    Threshold {thresh:>2}: "
                  f"good rejected={good_rejected}/{len(color_dists_good)} "
                  f"({100*good_rejected/len(color_dists_good):.0f}%), "
                  f"bad rejected={bad_rejected}/{len(color_dists_bad)} "
                  f"({100*bad_rejected/len(color_dists_bad):.0f}%)")

    # ---- Analysis 5: Individual record details ----
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: Per-detection detail (sorted by pixel error)")
    print("=" * 70)
    detail_lines.sort(key=lambda x: float(x.split('px')[0].strip()))
    for line in detail_lines:
        print(line)

    # ---- Analysis 6: Bbox consistency ----
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: Bbox size consistency per GCP")
    print("=" * 70)

    for gcp in sorted(gcp_all.keys()):
        all_recs = gcp_all[gcp]
        refined = [r for r in all_recs if r['algo_bbox'] is not None and r['conf'] == 'color_refined']
        if len(refined) < 2:
            continue
        sizes = []
        for r in refined:
            bw = r['algo_bbox'][2] - r['algo_bbox'][0]
            bh = r['algo_bbox'][3] - r['algo_bbox'][1]
            sizes.append((bw + bh) / 2.0)
        med_size = np.median(sizes)

        print(f"\n  {gcp}: median_bbox_size={med_size:.1f}")
        for r in sorted(refined, key=lambda x: x['dist']):
            bw = r['algo_bbox'][2] - r['algo_bbox'][0]
            bh = r['algo_bbox'][3] - r['algo_bbox'][1]
            bavg = (bw + bh) / 2.0
            ratio = bavg / med_size if med_size > 0 else 0
            print(f"    err={r['dist']:7.1f}px  bbox_avg={bavg:5.1f}  "
                  f"ratio={ratio:.2f}  {r['img']}")

    # ---- Analysis 7: Human bbox colors (what the marker ACTUALLY looks like) ----
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: Human-confirmed marker colors")
    print("=" * 70)

    human_with_color = [r for r in records if r['human_color'] is not None]
    if human_with_color:
        a_vals = [r['human_color']['a'] for r in human_with_color]
        b_vals = [r['human_color']['b'] for r in human_with_color]
        L_vals = [r['human_color']['L'] for r in human_with_color]
        print(f"\n  All human-confirmed markers (n={len(human_with_color)}):")
        print(f"    L*: mean={np.mean(L_vals):.1f}, std={np.std(L_vals):.1f}")
        print(f"    a*: mean={np.mean(a_vals):.1f}, std={np.std(a_vals):.1f}")
        print(f"    b*: mean={np.mean(b_vals):.1f}, std={np.std(b_vals):.1f}")

    # For bad cases, compare algo color vs human color at same location
    print("\n  BAD cases: algo vs human color comparison:")
    for r in sorted(bad_records, key=lambda x: -x['dist']):
        ac = r['algo_color']
        hc = r['human_color']
        if ac and hc:
            da = ac['a'] - hc['a']
            db = ac['b'] - hc['b']
            print(f"    err={r['dist']:7.1f}  algo(a={ac['a']:.0f},b={ac['b']:.0f})  "
                  f"human(a={hc['a']:.0f},b={hc['b']:.0f})  "
                  f"delta_a={da:+.0f} delta_b={db:+.0f}  {r['gcp']} {r['img']}")
        elif ac:
            print(f"    err={r['dist']:7.1f}  algo(a={ac['a']:.0f},b={ac['b']:.0f})  "
                  f"human=N/A  {r['gcp']} {r['img']}")


if __name__ == '__main__':
    main()
