#!/usr/bin/env python3
"""Compare refinement algorithm output(s) against human-confirmed ground truth.

Usage:
    python compare_refinement.py <confirmed> <baseline> [variant1 [variant2 ...]]

Example:
    python compare_refinement.py gcp_confirmed.txt gcp_list-before.txt gcp_list-k015.txt gcp_list-k03.txt
"""
import math
import sys
import os


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


def compute_errors(algo, human):
    """Return list of (dist, gcp, img, conf, algo_px, algo_py, human_px, human_py, bbox_w, bbox_h)."""
    results = []
    for key in sorted(human):
        gcp, img = key
        h = human[key]
        if key in algo:
            a = algo[key]
            dx = a['px'] - h['px']
            dy = a['py'] - h['py']
            dist = math.sqrt(dx * dx + dy * dy)
            bw = (a['bbox'][2] - a['bbox'][0]) if a['bbox'] else 0
            bh = (a['bbox'][3] - a['bbox'][1]) if a['bbox'] else 0
            results.append((dist, gcp, img, a['conf'], a['px'], a['py'],
                            h['px'], h['py'], bw, bh))
        else:
            results.append((None, gcp, img, 'MISSING', None, None,
                            h['px'], h['py'], 0, 0))
    return results


def summarize(name, results):
    dists = [r[0] for r in results if r[0] is not None]
    n = len(dists)
    if not n:
        return {}
    good5  = sum(1 for d in dists if d < 5)
    good10 = sum(1 for d in dists if d < 10)
    bad50  = sum(1 for d in dists if d >= 50)
    return {
        'name': name, 'n': n,
        'mean': sum(dists) / n,
        'median': sorted(dists)[n // 2],
        'max': max(dists),
        'good5': good5, 'good10': good10, 'bad50': bad50,
    }


def per_gcp(results):
    gcps = {}
    for r in results:
        g = r[1]
        if g not in gcps:
            gcps[g] = []
        gcps[g].append(r)
    out = {}
    for g in sorted(gcps):
        ds = [r[0] for r in gcps[g] if r[0] is not None]
        if ds:
            out[g] = {
                'n': len(ds),
                'mean': sum(ds) / len(ds),
                'max': max(ds),
                'good10': sum(1 for d in ds if d < 10),
                'bad50': sum(1 for d in ds if d >= 50),
            }
    return out


def print_full_report(name, results):
    results_sorted = sorted(results, key=lambda r: (r[0] is None, -(r[0] or 0)))
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"\n{'dist':>8}  {'conf':>15}  {'GCP':<10}  {'image':<42}")
    print('-' * 85)
    for r in results_sorted:
        dist, gcp, img, conf = r[0], r[1], r[2], r[3]
        if dist is not None:
            print(f'{dist:8.1f}  {conf:>15}  {gcp:<10}  {img:<42}')
        else:
            print(f"{'MISSING':>8}  {conf:>15}  {gcp:<10}  {img:<42}")

    s = summarize(name, results)
    print(f"\n--- Summary ---")
    print(f"Total: {len(results)}, Matched: {s['n']}")
    print(f"Mean:   {s['mean']:7.1f} px")
    print(f"Median: {s['median']:7.1f} px")
    print(f"Max:    {s['max']:7.1f} px")
    for thresh in [5, 10, 20, 50]:
        c = sum(1 for r in results if r[0] is not None and r[0] < thresh)
        print(f"  <{thresh:2d} px: {c:3d} ({100 * c / s['n']:4.0f}%)")
    print(f"  >=50 px: {s['bad50']:3d} ({100 * s['bad50'] / s['n']:4.0f}%)")

    print(f"\n--- Per-GCP ---")
    for g, v in per_gcp(results).items():
        print(f"  {g:<10}  n={v['n']:2d}  mean={v['mean']:7.1f}  "
              f"max={v['max']:7.1f}  good(<10)={v['good10']:2d}  bad(>=50)={v['bad50']:2d}")


def print_comparison(base_name, base_results, variants):
    """Print side-by-side delta table and summary comparison."""
    base_by_key = {(r[1], r[2]): r[0] for r in base_results}

    # Summary table header
    all_names = [base_name] + [v[0] for v in variants]
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'metric':<14}"
    for nm in all_names:
        short = os.path.basename(nm)[:20]
        header += f"  {short:>12}"
    print(header)
    print('-' * len(header))

    all_results = [base_results] + [v[1] for v in variants]
    all_summaries = [summarize(nm, res) for nm, res in zip(all_names, all_results)]
    for metric, fmt in [('mean', '{:.1f}'), ('median', '{:.1f}'),
                        ('max', '{:.1f}'), ('good5', '{}'),
                        ('good10', '{}'), ('bad50', '{}')]:
        row = f'{metric:<14}'
        for s in all_summaries:
            row += f'  {fmt.format(s[metric]):>12}'
        print(row)

    # Per-variant delta detail
    for var_name, var_results in variants:
        var_by_key = {(r[1], r[2]): r[0] for r in var_results}
        diffs = []
        for key in base_by_key:
            b = base_by_key.get(key)
            a = var_by_key.get(key)
            if b is not None and a is not None:
                diffs.append((a - b, b, a, key[0], key[1]))
        diffs.sort(key=lambda x: x[0])

        improved = sum(1 for d in diffs if d[0] < -5)
        worsened = sum(1 for d in diffs if d[0] > 5)
        unchanged = sum(1 for d in diffs if abs(d[0]) <= 5)

        print(f"\n--- {os.path.basename(var_name)} vs {os.path.basename(base_name)} ---")

        # Only show rows that changed
        changed = [d for d in diffs if abs(d[0]) > 0.05]
        if changed:
            print(f"{'delta':>8}  {'before':>8}  {'after':>8}  "
                  f"{'GCP':<10}  {'image':<42}")
            print('-' * 85)
            for delta, b, a, gcp, img in changed:
                marker = ' <<<' if abs(delta) > 10 else ''
                print(f'{delta:+8.1f}  {b:8.1f}  {a:8.1f}  '
                      f'{gcp:<10}  {img:<42}{marker}')
        else:
            print('  (no changes)')

        print(f"\nImproved (>5px): {improved}   "
              f"Worsened (>5px): {worsened}   Unchanged: {unchanged}")

        bs = summarize(base_name, base_results)
        vs = summarize(var_name, var_results)
        print(f"Good (<10px): {bs['good10']} -> {vs['good10']}  "
              f"({vs['good10'] - bs['good10']:+d})")
        print(f"Bad  (>=50px): {bs['bad50']} -> {vs['bad50']}  "
              f"({vs['bad50'] - bs['bad50']:+d})")
        print(f"Mean error: {bs['mean']:.1f} -> {vs['mean']:.1f}  "
              f"({vs['mean'] - bs['mean']:+.1f})")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    confirmed_path = sys.argv[1]
    baseline_path = sys.argv[2]
    variant_paths = sys.argv[3:]

    human = parse_gcp(confirmed_path)
    base = parse_gcp(baseline_path)
    base_results = compute_errors(base, human)

    print_full_report(baseline_path, base_results)

    variants = []
    for vp in variant_paths:
        vdata = parse_gcp(vp)
        vresults = compute_errors(vdata, human)
        print_full_report(vp, vresults)
        variants.append((vp, vresults))

    if variants:
        print_comparison(baseline_path, base_results, variants)


if __name__ == '__main__':
    main()
