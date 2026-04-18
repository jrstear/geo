#!/usr/bin/env python3
"""Export Grafana Cloud Prometheus + Loki data for an ODM run.

Per job, saves to the chosen output dir (default: ~/stratus/grafana-export/):
  {job}_prom_{metric}.json   — raw query_range response per metric
  {job}_loki_bootstrap.json  — bootstrap log lines (if Loki has them)
  {job}_summary.json         — aggregate stats (min/p50/p95/max per metric)

Usage:
  # Export a single named job (hardcoded JOBS entry OR ad-hoc via CLI):
  python3 export.py aztec10
  python3 export.py aztec12 \\
      --project bsn_aztec12 \\
      --start 2026-05-01T09:00:00Z \\
      --end   2026-05-01T12:00:00Z

  # Export all known JOBS at once:
  python3 export.py --all

Credentials: reads GRAFANA_API_KEY from env or ~/.odium/env. Uses the
Prometheus and Loki user IDs from infra/grafana/ (kept in memory/reference
for now; promote to repo-tracked config when we have more than one).

Known-jobs entries are kept in the JOBS dict at the top of this file —
append new surveys there for one-command re-export.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# ── Fixed config ────────────────────────────────────────────────────────────
PROM_BASE = "https://prometheus-prod-67-prod-us-west-0.grafana.net/api/prom"
LOKI_BASE = "https://logs-prod-021.grafana.net/loki/api/v1"
PROM_USER = "3068736"
LOKI_USER = "1530043"
DEFAULT_OUT = Path.home() / "stratus" / "grafana-export"

# Known jobs — append new surveys here. Times are UTC, bracket the active
# pipeline window by ~5 minutes on each side.
JOBS = {
    "aztec10": {
        "project": "bsn_aztec10",
        "start":   "2026-04-17T15:53:00Z",
        "end":     "2026-04-17T18:00:00Z",
        "label":   "baseline (v3.6.0 unpatched)",
    },
    "aztec11": {
        "project": "bsn_aztec11",
        "start":   "2026-04-18T03:36:00Z",
        "end":     "2026-04-18T06:05:00Z",
        "label":   "patched (v3.6.0 + PRs #48 + #2008)",
    },
}

# PromQL queries. Each template takes {project}. Returns time series stream(s)
# the summary code aggregates into min/p50/p95/max.
PROM_QUERIES = {
    "cpu_busy_pct":   '100 - avg(rate(node_cpu_seconds_total{{project="{project}",mode="idle"}}[1m])) * 100',
    "mem_used_bytes": 'node_memory_MemTotal_bytes{{project="{project}"}} - node_memory_MemAvailable_bytes{{project="{project}"}}',
    "mem_total_bytes":'node_memory_MemTotal_bytes{{project="{project}"}}',
    "disk_io_pct":    'rate(node_disk_io_time_seconds_total{{project="{project}",device=~"nvme.*|xvda.*"}}[1m]) * 100',
    "disk_read_mbps": 'rate(node_disk_read_bytes_total{{project="{project}",device=~"nvme.*|xvda.*"}}[1m]) / 1048576',
    "disk_write_mbps":'rate(node_disk_written_bytes_total{{project="{project}",device=~"nvme.*|xvda.*"}}[1m]) / 1048576',
    "net_rx_mbps":    'rate(node_network_receive_bytes_total{{project="{project}",device!~"lo|docker.*"}}[1m]) * 8 / 1048576',
    "net_tx_mbps":    'rate(node_network_transmit_bytes_total{{project="{project}",device!~"lo|docker.*"}}[1m]) * 8 / 1048576',
    "load1":          'node_load1{{project="{project}"}}',
    "procs_running":  'node_procs_running{{project="{project}"}}',
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("GRAFANA_API_KEY")
    if key:
        return key
    env_file = Path.home() / ".odium" / "env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GRAFANA_API_KEY="):
                return line.split("=", 1)[1].strip()
    sys.exit("Missing GRAFANA_API_KEY (env var or ~/.odium/env).")


def auth_header(user: str, key: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def http_get(url: str, auth: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": auth})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def prom_query_range(query: str, start: str, end: str, step: str, auth: str) -> dict:
    params = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": step})
    return http_get(f"{PROM_BASE}/api/v1/query_range?{params}", auth)


def loki_query_range(query: str, start: str, end: str, auth: str, limit: int = 5000) -> dict:
    params = urllib.parse.urlencode({"query": query, "start": start, "end": end,
                                     "limit": str(limit), "direction": "forward"})
    return http_get(f"{LOKI_BASE}/query_range?{params}", auth)


def summarise_series(values: list) -> dict:
    nums = []
    for _, v in values:
        try:
            f = float(v)
            if f == f:  # exclude NaN
                nums.append(f)
        except (ValueError, TypeError):
            pass
    if not nums:
        return {"n": 0}
    nums_sorted = sorted(nums)
    return {
        "n":    len(nums),
        "min":  round(min(nums), 3),
        "p50":  round(statistics.median(nums), 3),
        "p95":  round(nums_sorted[int(0.95 * (len(nums) - 1))], 3),
        "max":  round(max(nums), 3),
        "mean": round(statistics.fmean(nums), 3),
    }


# ── Core export ─────────────────────────────────────────────────────────────

def export_one(job_key: str, meta: dict, out_dir: Path,
               prom_auth: str, loki_auth: str) -> None:
    project = meta["project"]
    start, end = meta["start"], meta["end"]
    step = meta.get("step", "15s")

    summary: dict = {"job": job_key, "project": project, "start": start, "end": end,
                     "label": meta.get("label", ""), "metrics": {}}

    # Prometheus metrics
    for name, tmpl in PROM_QUERIES.items():
        q = tmpl.format(project=project)
        try:
            data = prom_query_range(q, start, end, step, prom_auth)
        except Exception as e:
            print(f"  ⚠ {job_key}/{name}: {e}", file=sys.stderr)
            continue
        (out_dir / f"{job_key}_prom_{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        all_values = []
        for ser in data.get("data", {}).get("result", []):
            all_values.extend(ser.get("values", []))
        summary["metrics"][name] = summarise_series(all_values)
        print(f"  ✓ {job_key}/{name}: {summary['metrics'][name].get('n', 0)} samples")

    # Loki: bootstrap log (best-effort; some deployments use different filename labels)
    for loki_q in (
        f'{{project="{project}",filename="/var/log/odm-bootstrap.log"}}',
        f'{{project="{project}"}} |~ "bootstrap|stage"',
    ):
        try:
            data = loki_query_range(loki_q, start, end, loki_auth)
            n = sum(len(s.get("values", [])) for s in data.get("data", {}).get("result", []))
            if n:
                (out_dir / f"{job_key}_loki_bootstrap.json").write_text(json.dumps(data, indent=2) + "\n")
                summary["loki_bootstrap_entries"] = n
                print(f"  ✓ {job_key}/loki: {n} lines (query {loki_q[:40]}…)")
                break
        except Exception as e:
            print(f"  ⚠ {job_key}/loki ({loki_q[:40]}…): {e}", file=sys.stderr)

    (out_dir / f"{job_key}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  ✓ {job_key}/summary written")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", nargs="?", help="Job name (key in JOBS, or ad-hoc with --project/--start/--end).")
    p.add_argument("--all", action="store_true", help="Export all known JOBS.")
    p.add_argument("--project", help="Project label (e.g. bsn_aztec12); overrides JOBS lookup.")
    p.add_argument("--start", help="Window start (RFC3339 UTC).")
    p.add_argument("--end",   help="Window end (RFC3339 UTC).")
    p.add_argument("--label", default="", help="Human-readable label for the summary.")
    p.add_argument("--step",  default="15s", help="Prom sample step (default 15s).")
    p.add_argument("--out",   default=str(DEFAULT_OUT), help=f"Output dir (default {DEFAULT_OUT}).")
    args = p.parse_args()

    if not (args.all or args.job):
        p.error("Specify a job name, or --all.")

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    key = get_api_key()
    prom_auth = auth_header(PROM_USER, key)
    loki_auth = auth_header(LOKI_USER, key)

    targets = list(JOBS.items()) if args.all else [(args.job, None)]
    for job_key, known in targets:
        if known is not None:
            meta = known
        elif args.project and args.start and args.end:
            meta = {"project": args.project, "start": args.start, "end": args.end,
                    "label": args.label, "step": args.step}
        elif job_key in JOBS:
            meta = JOBS[job_key]
        else:
            sys.exit(f"Unknown job {job_key!r} and --project/--start/--end not given.")
        print(f"\n=== {job_key} — {meta.get('label', '')} ===")
        export_one(job_key, meta, out_dir, prom_auth, loki_auth)


if __name__ == "__main__":
    main()
