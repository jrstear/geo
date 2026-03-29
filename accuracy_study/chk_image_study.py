"""
CHK RMSE mini-study: vary image count per CHK label and exclude outlier labels.

Usage (run from the job directory, e.g. ~/stratus/aztec7/):
    conda run -n geo python ~/git/geo/accuracy_study/chk_image_study.py \\
        ../aztec6/opensfm/reconstruction.topocentric.json \\
        gcp_list.txt chk_list.txt

The script generates variants of chk_list.txt and runs rmse.py on each,
printing a comparison table of RMS_H / RMS_Z / RMS_3D.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

RMSE_PY = Path(__file__).resolve().parent.parent / "rmse.py"

# Points identified as base-station monuments: coordinate accuracy may differ
# from aerial targets, and camera geometry at those locations is typically poor.
DEFAULT_OUTLIERS = {"CHK-14", "CHK-18"}


def read_chk_list(path: Path):
    with open(path) as f:
        lines = f.readlines()
    header = lines[0]
    by_label = defaultdict(list)
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 7:
            by_label[parts[6]].append(line)
    return header, by_label


def make_variant(header, by_label, max_images=None, exclude=None):
    exclude = exclude or set()
    out = [header]
    for label in sorted(by_label):
        if label in exclude:
            continue
        rows = by_label[label]
        if max_images is not None:
            rows = rows[:max_images]
        out.extend(rows)
    return "".join(out)


def run_rmse(reconstruction, gcp_list, chk_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(chk_content)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, str(RMSE_PY), reconstruction, gcp_list, tmp],
            capture_output=True, text=True
        )
        output = result.stdout + result.stderr
    finally:
        os.unlink(tmp)

    # Isolate the CHK section so we don't accidentally match GCP values.
    chk_section = ""
    m = re.search(r"CHK accuracy.*", output, re.DOTALL)
    if m:
        chk_section = m.group(0)

    def _ft(section, pattern):
        m = re.search(pattern + r"\s*=\s*[\d.]+\s*m\s+([\d.]+)\s*ft", section)
        return float(m.group(1)) if m else None

    n_m = re.search(r"N=(\d+)", chk_section)
    return {
        "n": int(n_m.group(1)) if n_m else None,
        "rms_h_ft":  _ft(chk_section, r"RMS_H"),
        "rms_z_ft":  _ft(chk_section, r"RMS_Z"),
        "rms_3d_ft": _ft(chk_section, r"RMS_3D"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reconstruction", help="reconstruction.topocentric.json")
    ap.add_argument("gcp_list", help="gcp_list.txt (GCP control file)")
    ap.add_argument("chk_list", help="chk_list.txt (full CHK file to vary)")
    ap.add_argument("--outliers", nargs="*", default=sorted(DEFAULT_OUTLIERS),
                    metavar="LABEL",
                    help=f"Labels to treat as outliers (default: {sorted(DEFAULT_OUTLIERS)})")
    args = ap.parse_args()

    outliers = set(args.outliers)
    header, by_label = read_chk_list(Path(args.chk_list))
    all_labels = sorted(by_label)
    max_per_label = max(len(v) for v in by_label.values())

    print(f"CHK labels: {len(all_labels)}  |  max images/label: {max_per_label}")
    print(f"Outlier labels (excluded in *_excl variants): {sorted(outliers)}\n")

    image_counts = [1, 3, 5, 7, max_per_label]
    image_counts = sorted(set(image_counts))  # deduplicate if max_per_label ≤ 7

    variants = []
    for n in image_counts:
        label = "all" if n == max_per_label else f"top{n}"
        variants.append((f"{label} imgs, all CHKs",   n, set()))
        variants.append((f"{label} imgs, excl outliers", n, outliers))

    # Header
    col_w = 32
    print(f"{'Variant':<{col_w}}  {'N':>4}  {'RMS_H ft':>10}  {'RMS_Z ft':>10}  {'RMS_3D ft':>10}")
    print("-" * (col_w + 44))

    for name, max_img, excl in variants:
        max_arg = None if max_img == max_per_label else max_img
        content = make_variant(header, by_label, max_images=max_arg, exclude=excl)
        r = run_rmse(args.reconstruction, args.gcp_list, content)
        n    = f"{r['n']}"    if r['n']    is not None else "?"
        rh   = f"{r['rms_h_ft']:.4f}"  if r['rms_h_ft']  is not None else "?"
        rz   = f"{r['rms_z_ft']:.4f}"  if r['rms_z_ft']  is not None else "?"
        r3d  = f"{r['rms_3d_ft']:.4f}" if r['rms_3d_ft'] is not None else "?"
        print(f"{name:<{col_w}}  {n:>4}  {rh:>10}  {rz:>10}  {r3d:>10}")


if __name__ == "__main__":
    main()
