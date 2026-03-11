#!/bin/bash
set -euo pipefail

# ── 1. Download inputs from S3 ────────────────────────────────────────────────
echo "[run.sh] Syncing images from ${S3_PROJECT}/inputs/images/ ..."
aws s3 sync "${S3_PROJECT}/inputs/images/" /data/images/

echo "[run.sh] Downloading emlid.csv ..."
aws s3 cp "${S3_PROJECT}/inputs/emlid.csv" /data/emlid.csv

echo "[run.sh] Downloading gcp_experiment.txt for run ${RUN_ID} ..."
aws s3 cp "${S3_PROJECT}/runs/${RUN_ID}/gcp_experiment.txt" /data/gcp/geo_cps.txt

# ── 2. Run ODM ────────────────────────────────────────────────────────────────
echo "[run.sh] Starting ODM ..."
python3 /code/run.py /data \
  --gcp /data/gcp/geo_cps.txt \
  --orthophoto-resolution 2 \
  --pc-quality high \
  ${ODM_OPTIONS:-}

# ── 3. Upload outputs to S3 ───────────────────────────────────────────────────
echo "[run.sh] Uploading reconstruction.json ..."
aws s3 cp /data/odm_opensfm/reconstruction.json \
  "${S3_PROJECT}/runs/${RUN_ID}/outputs/reconstruction.json" \
  --tagging "Expiry=run-output"

echo "[run.sh] Uploading orthophoto ..."
aws s3 sync /data/odm_orthophoto/ \
  "${S3_PROJECT}/runs/${RUN_ID}/outputs/odm_orthophoto/" \
  --metadata-directive REPLACE \
  --tagging "Expiry=run-output"

echo "[run.sh] Uploading DEM ..."
aws s3 sync /data/odm_dem/ \
  "${S3_PROJECT}/runs/${RUN_ID}/outputs/odm_dem/" \
  --metadata-directive REPLACE \
  --tagging "Expiry=run-output"

echo "[run.sh] Done."
