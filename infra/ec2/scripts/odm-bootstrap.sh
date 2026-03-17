#!/bin/bash
# Bootstrap script — runs on every boot via @reboot cron.
# Sources /etc/odm-env for config (written once by Terraform user_data).
#
# New instance  — images/ empty: syncs project from S3, runs pipeline, uploads outputs.
# Spot resume   — images/ present: skips sync, resumes pipeline from last complete stage.
# Already done  — .odm-complete marker: waits 5min then shuts down (inspection window).
#
# Cancelling auto-shutdown:
#   touch /data/project/.no-autoshutdown    # before or during the 5-min countdown
#   (delete it when done to re-enable auto-shutdown on next boot)
#
# To iterate: scp this file to /usr/local/bin/odm-bootstrap.sh and re-run directly.
exec >> /var/log/odm-bootstrap.log 2>&1
set -euo pipefail
source /etc/odm-env

PROJECT_DIR=/data/project
DONE_MARKER="${PROJECT_DIR}/.odm-complete"

notify() {
  aws sns publish \
    --topic-arn "${SNS_TOPIC}" \
    --subject "$1" \
    --message "$2" \
    --region "${REGION}" || true
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-bootstrap starting (project: ${PROJECT})"

# Pipeline already complete — shut down again.
# Persistent spot will keep restarting the instance after shutdown; we shut down
# each time until the user runs 'terraform destroy' to cancel the spot request.
if [ -f "${DONE_MARKER}" ]; then
  NO_SHUTDOWN_FLAG="${PROJECT_DIR}/.no-autoshutdown"
  if [ -f "${NO_SHUTDOWN_FLAG}" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline already complete — auto-shutdown DISABLED (.no-autoshutdown present). Instance will remain up."
    notify "ODM idle restart: ${PROJECT}" \
      "Instance restarted after completed pipeline. Auto-shutdown disabled by .no-autoshutdown — instance will remain up."
    exit 0
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline already complete — shutting down in 5 minutes. SSH in and touch /data/project/.no-autoshutdown to cancel."
  notify "ODM idle restart: ${PROJECT}" \
    "Instance restarted after completed pipeline. Shutting down in 5 minutes. Touch /data/project/.no-autoshutdown to cancel. Run 'terraform destroy' to cancel the spot request."
  for i in $(seq 1 30); do   # 30 × 10s = 5 min
    sleep 10
    if [ -f "${NO_SHUTDOWN_FLAG}" ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Auto-shutdown cancelled (.no-autoshutdown detected). Instance will remain up."
      notify "ODM idle restart: ${PROJECT}" "Auto-shutdown cancelled. Instance will remain up."
      exit 0
    fi
  done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Shutting down now. Run 'terraform destroy' to fully clean up."
  /sbin/shutdown -h +1
  exit 0
fi

# Wait for Docker to be ready (critical on reboot before dockerd is fully up).
until docker info &>/dev/null 2>&1; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Waiting for Docker..."
  sleep 5
done

# Wait for ODM image (may still be pulling on first boot).
until docker image inspect opendronemap/odm:3.3.0 &>/dev/null 2>&1; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Waiting for ODM image pull to complete..."
  sleep 10
done

mkdir -p "${PROJECT_DIR}/images"

# Sync full project state from S3 if images are not on EBS.
# - New instance, fresh job:  downloads images/ + gcp_list.txt only
# - New instance, resuming:   downloads images/ + prior stage outputs; is_done() detects them
# - Spot resume (same EBS):   images already present → entire block skipped
IMAGE_COUNT=$(find "${PROJECT_DIR}/images" -maxdepth 1 -type f | wc -l)
if [ "${IMAGE_COUNT}" -eq 0 ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  No images on EBS — syncing project from s3://${BUCKET}/${PROJECT}/"
  notify "ODM starting: ${PROJECT}" \
    "Syncing project from S3 (images + any prior stage outputs). Pipeline will begin automatically."
  aws s3 sync "s3://${BUCKET}/${PROJECT}/" "${PROJECT_DIR}/" \
    --exclude "*.MRK" --exclude "*.nav" --exclude "*.obs" --exclude "*.bin" \
    --region "${REGION}"
  IMAGE_COUNT=$(find "${PROJECT_DIR}/images" -maxdepth 1 -type f | wc -l)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  S3 sync complete (${IMAGE_COUNT} images)"
else
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${IMAGE_COUNT} images on EBS — skipping S3 sync (spot resume)"
fi

# Verify GCP file is present (should have come down with the sync above).
if [ ! -f "${PROJECT_DIR}/gcp_list.txt" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Fetching GCP file from s3://${BUCKET}/${PROJECT}/gcp_list.txt"
  aws s3 cp "s3://${BUCKET}/${PROJECT}/gcp_list.txt" "${PROJECT_DIR}/gcp_list.txt" \
    --region "${REGION}"
fi

# Run the pipeline.
if /usr/local/bin/odm-run.sh; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline complete — syncing outputs to s3://${BUCKET}/${PROJECT}/"
  aws s3 sync "${PROJECT_DIR}/" "s3://${BUCKET}/${PROJECT}/" \
    --exclude "images/*" \
    --region "${REGION}"
  touch "${DONE_MARKER}"
  notify "ODM complete: ${PROJECT}" \
    "Outputs synced to s3://${BUCKET}/${PROJECT}/. Run 'terraform destroy' to cancel the spot request and delete EBS."
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Done. Shutting down in 2 minutes. Run 'terraform destroy' to fully clean up."
  /sbin/shutdown -h +2
else
  # odm-run.sh already sent the failure SNS; leave instance up for debugging.
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline failed — instance remains up. SSH in to investigate."
fi
