#!/usr/bin/env python3
"""
experiment_gen.py — GCP file variant generator for ablation experiments.

Reads a master_tags.txt (all confirmed GCP-* and CHK-* rows in pipeline
priority order) and an experiment config JSON, then writes a trimmed
gcp_experiment.txt containing only the selected control labels and the
requested number of images per label.

CLI:
    python experiment_gen.py master_tags.txt \\
        --config config.json \\
        --out gcp_experiment.txt

    python experiment_gen.py master_tags.txt \\
        --control-labels GCP-1 GCP-2 GCP-3 \\
        --images-per-label 7 \\
        --out gcp_experiment.txt

Config schema (config.json):
    {
        "control_labels": ["GCP-1", "GCP-2", "GCP-3"],
        "images_per_label": 7,
        "description": "top3-7imgs"
    }

    images_per_label may also be a dict keyed by label with a "default" fallback:
        "images_per_label": {"GCP-1": 5, "GCP-2": 10, "default": 7}

Output format: ODM-compatible gcp_list.txt (CRS header + tab-separated rows).
The description is NOT written into the file body (ODM treats line 1 as the CRS
string and rejects comments). Store description in results_table.csv instead.
"""

import argparse
import json
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_master_tags(master_path: str):
    """
    Parse master_tags.txt.

    Returns:
        crs_header  : str              — line 1, CRS string (verbatim, strip \\n)
        rows_by_label : dict[str, list[str]]  — label → list of raw lines (no \\n)
        label_order : list[str]        — labels in the order first encountered
    """
    with open(master_path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"master_tags.txt is empty: {master_path}")

    crs_header = lines[0].rstrip('\n')
    rows_by_label = defaultdict(list)
    label_order = []

    for raw in lines[1:]:
        line = raw.rstrip('\n')
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) < 7:
            # Malformed row — skip silently (could be a blank/header artifact)
            continue
        label = fields[6]
        if label not in rows_by_label:
            label_order.append(label)
        rows_by_label[label].append(line)

    return crs_header, rows_by_label, label_order


def resolve_n_images(images_per_label, label: str) -> int:
    """
    Return the number of images to select for *label*.

    images_per_label may be:
      - int   → used for every label
      - dict  → label-specific; falls back to images_per_label.get('default', 7)
    """
    if isinstance(images_per_label, int):
        return images_per_label
    if isinstance(images_per_label, dict):
        return int(images_per_label.get(label, images_per_label.get('default', 7)))
    raise TypeError(f"images_per_label must be int or dict, got {type(images_per_label)}")


def generate_experiment(
    crs_header: str,
    rows_by_label: dict,
    control_labels: list,
    images_per_label,
) -> list:
    """
    Select rows for each label in control_labels (in the given order).

    Returns a flat list of raw row strings (no trailing newline).
    Emits warnings to stderr for labels not present in rows_by_label.
    """
    output_rows = []
    for label in control_labels:
        if label not in rows_by_label:
            print(f"WARNING: label {label!r} not in master_tags.txt — skipping",
                  file=sys.stderr)
            continue
        n = resolve_n_images(images_per_label, label)
        output_rows.extend(rows_by_label[label][:n])
    return output_rows


def write_output(crs_header: str, output_rows: list, out_path=None):
    """
    Write CRS header + rows to out_path (or stdout if None).

    The description is intentionally omitted from the file body so ODM can
    parse line 1 as the CRS string without error.
    """
    if out_path:
        with open(out_path, 'w') as f:
            f.write(crs_header + '\n')
            for row in output_rows:
                f.write(row + '\n')
    else:
        sys.stdout.write(crs_header + '\n')
        for row in output_rows:
            sys.stdout.write(row + '\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Generate a trimmed ODM GCP file from master_tags.txt.",
    )
    p.add_argument('master_tags', nargs='?',
                   help="Path to master_tags.txt (CRS header + tab-separated rows).")
    p.add_argument('--config', metavar='CONFIG_JSON',
                   help="Path to experiment config JSON.")
    p.add_argument('--control-labels', nargs='+', metavar='LABEL',
                   help="Labels to include (overrides --config control_labels).")
    p.add_argument('--images-per-label', type=int, metavar='N',
                   help="Images per label (overrides --config images_per_label).")
    p.add_argument('--out', metavar='OUTPUT_PATH',
                   help="Output path. If omitted, writes to stdout.")
    p.add_argument('--test', action='store_true',
                   help="Run self-contained acceptance test and exit.")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.test:
        run_tests()
        return

    if not args.master_tags:
        parser.error("master_tags positional argument is required (unless --test).")

    # Load config from JSON if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Merge: CLI flags override config values
    control_labels = args.control_labels or config.get('control_labels')
    if not control_labels:
        parser.error("Provide --control-labels or a config JSON with 'control_labels'.")

    images_per_label = (
        args.images_per_label
        if args.images_per_label is not None
        else config.get('images_per_label', 7)
    )

    description = config.get('description', '')
    if description:
        print(f"INFO: experiment description: {description!r}", file=sys.stderr)

    # Load and process
    crs_header, rows_by_label, _ = load_master_tags(args.master_tags)
    output_rows = generate_experiment(crs_header, rows_by_label, control_labels,
                                      images_per_label)

    write_output(crs_header, output_rows, args.out)

    n_written = len(output_rows)
    print(f"INFO: wrote {n_written} rows to {args.out or 'stdout'}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Self-contained acceptance test
# ---------------------------------------------------------------------------

def run_tests():
    """
    Synthetic acceptance test — no external files required.
    Exits 0 on success, 1 on failure.
    """
    import io
    import traceback

    FAILURES = []

    def check(name, condition, msg=""):
        if not condition:
            FAILURES.append(f"FAIL [{name}] {msg}")
            print(f"  FAIL: {name} — {msg}", file=sys.stderr)
        else:
            print(f"  ok:   {name}", file=sys.stderr)

    print("Running experiment_gen acceptance tests...", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build synthetic master_tags.txt content (3 labels × 10 images each)
    # ------------------------------------------------------------------
    CRS = "EPSG:6529"
    labels = ["GCP-1", "GCP-2", "CHK-1"]
    n_imgs = 10
    lines = [CRS]
    for lbl in labels:
        for i in range(n_imgs):
            # geo_x  geo_y  geo_z  px  py  image_name  gcp_label  confidence
            row = "\t".join([
                "1234567.89",
                "987654.32",
                "5678.0",
                str(100 + i),
                str(200 + i),
                f"img_{lbl}_{i:02d}.JPG",
                lbl,
                "projection",
            ])
            lines.append(row)
    master_content = "\n".join(lines) + "\n"

    # Parse via load_master_tags using a StringIO-like approach (write to /tmp)
    import tempfile, os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
        tf.write(master_content)
        tmp_master = tf.name

    try:
        crs_header, rows_by_label, label_order = load_master_tags(tmp_master)

        # --- Test 1: CRS header parsed correctly
        check("crs_header", crs_header == CRS,
              f"expected {CRS!r}, got {crs_header!r}")

        # --- Test 2: label_order matches input order
        check("label_order", label_order == labels,
              f"expected {labels}, got {label_order}")

        # --- Test 3: row counts per label
        for lbl in labels:
            check(f"row_count_{lbl}",
                  len(rows_by_label[lbl]) == n_imgs,
                  f"expected {n_imgs}, got {len(rows_by_label[lbl])}")

        # --- Test 4: generate 2 labels × 5 images
        selected_labels = ["GCP-1", "GCP-2"]
        n_select = 5
        output_rows = generate_experiment(
            crs_header, rows_by_label, selected_labels, n_select
        )
        expected_count = len(selected_labels) * n_select
        check("row_count_total",
              len(output_rows) == expected_count,
              f"expected {expected_count} rows, got {len(output_rows)}")

        # --- Test 5: no CHK rows in output
        chk_rows = [r for r in output_rows if "\tCHK-" in r]
        check("no_chk_rows", len(chk_rows) == 0,
              f"found {len(chk_rows)} CHK rows in output")

        # --- Test 6: correct labels only
        for row in output_rows:
            fields = row.split('\t')
            lbl = fields[6]
            check(f"label_is_selected_{lbl}",
                  lbl in selected_labels,
                  f"unexpected label {lbl!r}")

        # --- Test 7: image ordering preserved (first n_select images for GCP-1)
        gcp1_rows = [r for r in output_rows if r.split('\t')[6] == "GCP-1"]
        for idx, row in enumerate(gcp1_rows):
            expected_img = f"img_GCP-1_{idx:02d}.JPG"
            actual_img = row.split('\t')[5]
            check(f"image_order_GCP1_{idx}",
                  actual_img == expected_img,
                  f"expected {expected_img!r}, got {actual_img!r}")

        # --- Test 8: write output and verify format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as out_tf:
            tmp_out = out_tf.name

        write_output(crs_header, output_rows, tmp_out)
        with open(tmp_out) as f:
            out_lines = f.readlines()

        check("output_first_line_is_crs",
              out_lines[0].rstrip('\n') == CRS,
              f"line 1 = {out_lines[0]!r}, expected {CRS!r}")
        check("output_row_count",
              len(out_lines) == 1 + expected_count,
              f"expected {1 + expected_count} lines, got {len(out_lines)}")

        # Verify each data row has at least 7 tab-separated columns
        for i, line in enumerate(out_lines[1:], start=2):
            fields = line.rstrip('\n').split('\t')
            check(f"output_col_count_line{i}",
                  len(fields) >= 7,
                  f"line {i} has only {len(fields)} columns: {line!r}")

        os.unlink(tmp_out)

        # --- Test 9: dict images_per_label
        per_label_dict = {"GCP-1": 3, "GCP-2": 8, "default": 7}
        mixed_rows = generate_experiment(
            crs_header, rows_by_label, ["GCP-1", "GCP-2"], per_label_dict
        )
        gcp1_count = sum(1 for r in mixed_rows if r.split('\t')[6] == "GCP-1")
        gcp2_count = sum(1 for r in mixed_rows if r.split('\t')[6] == "GCP-2")
        check("dict_images_per_label_GCP1", gcp1_count == 3,
              f"expected 3, got {gcp1_count}")
        check("dict_images_per_label_GCP2", gcp2_count == 8,
              f"expected 8, got {gcp2_count}")

        # --- Test 10: missing label emits warning but does not crash
        import io as _io
        old_stderr = sys.stderr
        sys.stderr = _io.StringIO()
        warn_rows = generate_experiment(
            crs_header, rows_by_label, ["GCP-1", "GCP-MISSING"], 3
        )
        stderr_out = sys.stderr.getvalue()
        sys.stderr = old_stderr
        check("missing_label_no_crash", len(warn_rows) == 3,
              f"expected 3 rows (only GCP-1), got {len(warn_rows)}")
        check("missing_label_warning",
              "GCP-MISSING" in stderr_out and "WARNING" in stderr_out,
              f"expected WARNING for GCP-MISSING in stderr, got: {stderr_out!r}")

        # --- Test 11: CRS header byte-for-byte identity
        check("crs_byte_identity", crs_header == CRS,
              "CRS header must be verbatim from master_tags.txt")

    except Exception:
        traceback.print_exc(file=sys.stderr)
        FAILURES.append("EXCEPTION during tests")
    finally:
        os.unlink(tmp_master)

    print(file=sys.stderr)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s):", file=sys.stderr)
        for f in FAILURES:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"All tests passed.", file=sys.stderr)
        sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
