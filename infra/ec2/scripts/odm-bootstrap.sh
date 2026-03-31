#!/bin/bash
# Bootstrap script — runs on every boot via @reboot cron.
# Sources /etc/odm-env for config (written once by Terraform user_data).
#
# New instance  — images/ empty: syncs project from S3, runs pipeline, uploads outputs.
# Spot resume   — images/ present: skips sync, resumes pipeline from last complete stage.
# Already done  — .odm-complete marker: 5-min window then shuts down.
# Failed        — .odm-failed marker: 15-min window then shuts down (logs synced to S3).
#                 On subsequent reboots: 5-min window then shuts down again.
#
# Cancelling auto-shutdown (any path):
#   touch /data/project/.no-autoshutdown    # before or during countdown
#   (delete it when done to re-enable auto-shutdown on next boot)
#
# Re-running after failure:
#   rm /data/project/.odm-failed            # clears the failure marker; pipeline re-runs on next boot
#
# To iterate: scp this file to /usr/local/bin/odm-bootstrap.sh and re-run directly.
exec >> /var/log/odm-bootstrap.log 2>&1
set -euo pipefail
source /etc/odm-env

PROJECT_DIR=/data/project
DONE_MARKER="${PROJECT_DIR}/.odm-complete"
FAILED_MARKER="${PROJECT_DIR}/.odm-failed"
NO_SHUTDOWN_FLAG="${PROJECT_DIR}/.no-autoshutdown"

notify() {
  aws sns publish \
    --topic-arn "${SNS_TOPIC}" \
    --subject "$1" \
    --message "$2" \
    --region "${REGION}" || true
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  odm-bootstrap starting (project: ${PROJECT})"

# ── Pipeline already complete ──────────────────────────────────────────────────
# Persistent spot will keep restarting the instance after shutdown; we shut down
# each time until the user runs 'terraform destroy' to cancel the spot request.
if [ -f "${DONE_MARKER}" ]; then
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

# ── Pipeline previously failed ─────────────────────────────────────────────────
# On any reboot after a failure shutdown, don't re-run the pipeline.
# Give the same 5-minute inspection window then shut down again.
if [ -f "${FAILED_MARKER}" ]; then
  if [ -f "${NO_SHUTDOWN_FLAG}" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline previously failed — auto-shutdown DISABLED (.no-autoshutdown present). Instance will remain up."
    notify "ODM failed restart: ${PROJECT}" \
      "Instance restarted after failed pipeline. Auto-shutdown disabled by .no-autoshutdown — instance will remain up for investigation."
    exit 0
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline previously failed — shutting down in 5 minutes. SSH in and touch /data/project/.no-autoshutdown to cancel."
  notify "ODM failed restart: ${PROJECT}" \
    "Instance restarted after previous failure. Shutting down in 5 minutes. Touch /data/project/.no-autoshutdown to cancel and investigate. Delete .odm-failed to re-run the pipeline."
  for i in $(seq 1 30); do   # 30 × 10s = 5 min
    sleep 10
    if [ -f "${NO_SHUTDOWN_FLAG}" ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Auto-shutdown cancelled (.no-autoshutdown detected). Instance will remain up."
      notify "ODM failed restart: ${PROJECT}" "Auto-shutdown cancelled. Instance will remain up for investigation."
      exit 0
    fi
  done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Shutting down now."
  /sbin/shutdown -h +1
  exit 0
fi

# ── Live instance pricing ──────────────────────────────────────────────────────
# Queries IMDS for instance type + lifecycle, then:
#   spot       → describe-spot-instance-requests for actual bid price
#   on-demand  → AWS Pricing API (us-east-1 endpoint) for list price
# Writes INSTANCE_COST_PER_HOUR to /etc/odm-env for downstream scripts.
lookup_instance_cost() {
  local token instance_id instance_type lifecycle region cost

  token=$(curl -sf -X PUT \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
    http://169.254.169.254/latest/api/token 2>/dev/null || true)

  _imds() { curl -sf -H "X-aws-ec2-metadata-token: ${token}" \
    "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || true; }

  instance_id=$(_imds instance-id)
  instance_type=$(_imds instance-type)
  lifecycle=$(_imds instance-life-cycle)   # "spot" or "on-demand"
  region=$(_imds placement/region)
  region=${region:-${REGION}}

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pricing lookup: ${instance_type}, lifecycle=${lifecycle}, region=${region}"

  if [ "${lifecycle}" = "spot" ] && [ -n "${instance_id}" ]; then
    cost=$(aws ec2 describe-spot-instance-requests \
      --filters "Name=instance-id,Values=${instance_id}" \
      --query 'SpotInstanceRequests[0].SpotPrice' \
      --output text --region "${region}" 2>/dev/null || true)
    [ "${cost}" = "None" ] && cost=""
    [ -n "${cost}" ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Spot price: \$${cost}/hr"
  fi

  if [ -z "${cost}" ] && [ -n "${instance_type}" ]; then
    # Pricing API is only available in us-east-1 / ap-south-1
    local location
    case "${region}" in
      us-east-1)      location="US East (N. Virginia)" ;;
      us-east-2)      location="US East (Ohio)" ;;
      us-west-1)      location="US West (N. California)" ;;
      us-west-2)      location="US West (Oregon)" ;;
      eu-west-1)      location="Europe (Ireland)" ;;
      eu-west-2)      location="Europe (London)" ;;
      eu-central-1)   location="Europe (Frankfurt)" ;;
      ap-northeast-1) location="Asia Pacific (Tokyo)" ;;
      ap-southeast-1) location="Asia Pacific (Singapore)" ;;
      ap-southeast-2) location="Asia Pacific (Sydney)" ;;
      ap-south-1)     location="Asia Pacific (Mumbai)" ;;
      sa-east-1)      location="South America (Sao Paulo)" ;;
      *)              location="" ;;
    esac

    if [ -n "${location}" ]; then
      cost=$(aws pricing get-products \
        --service-code AmazonEC2 \
        --region us-east-1 \
        --filters \
          "Type=TERM_MATCH,Field=instanceType,Value=${instance_type}" \
          "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
          "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
          "Type=TERM_MATCH,Field=location,Value=${location}" \
          "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
          "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
        --query 'PriceList[0]' \
        --output text 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    terms = data.get('terms', {}).get('OnDemand', {})
    for sku_term in terms.values():
        for dim in sku_term.get('priceDimensions', {}).values():
            price = dim.get('pricePerUnit', {}).get('USD', '')
            if price and float(price) > 0:
                print(price)
                break
        break
except Exception:
    pass
" 2>/dev/null || true)
      [ -n "${cost}" ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  On-demand price: \$${cost}/hr (${instance_type}, ${location})"
    fi
  fi

  if [ -n "${cost}" ] && python3 -c "v=float('${cost}'); assert v > 0" 2>/dev/null; then
    printf 'export INSTANCE_COST_PER_HOUR="%s"\n' "${cost}" >> /etc/odm-env
    source /etc/odm-env
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  WARNING: could not determine instance cost per hour"
  fi
}

lookup_instance_cost

# Wait for Docker to be ready (critical on reboot before dockerd is fully up).
until docker info &>/dev/null 2>&1; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Waiting for Docker..."
  sleep 5
done

# Wait for ODM image (may still be pulling on first boot).
until docker image inspect "${ODM_IMAGE:-opendronemap/odm:3.6.0}" &>/dev/null 2>&1; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Waiting for ODM image pull to complete..."
  sleep 10
done

# Patch exifread 3.x IndexError on empty DJI MakerNote tag values.
# exifread's _get_printable_for_field does str(values[0]) without guarding
# for an empty list, crashing on some DJI images.  Unfixed as of ExifRead 3.5.1.
# Creates a locally-tagged patched image so odm-run.sh containers use the fix.
ODM_BASE="${ODM_IMAGE:-opendronemap/odm:3.6.0}"
ODM_PATCHED="${ODM_BASE}-patched"
if ! docker image inspect "${ODM_PATCHED}" &>/dev/null 2>&1; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Patching exifread DJI MakerNote bug..."
  docker run --name odm-patch --entrypoint bash "${ODM_BASE}" -c \
    "sed -i \"s/printable = str(values\[0\])/printable = str(values[0]) if values else \\\"\\\"/\" \
     /code/venv/lib/python*/site-packages/exifread/core/exif_header.py"
  docker commit odm-patch "${ODM_PATCHED}"
  docker rm odm-patch
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Created patched image: ${ODM_PATCHED}"
fi
export ODM_IMAGE="${ODM_PATCHED}"

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

# Seed BA with a prior camera calibration if available.
# cameras.json is written by ODM after the dataset stage and synced back to S3,
# so subsequent runs (same or new job with same drone) pick it up automatically.
# Operator workflow for a new job: copy cameras.json from a prior job's S3 prefix
# to the new project prefix before terraform apply.
if [ -f "${PROJECT_DIR}/cameras.json" ] && ! grep -q "\-\-cameras" /etc/odm-env; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  cameras.json found — appending --cameras flag to ODM_FLAGS"
  echo 'export ODM_FLAGS="${ODM_FLAGS} --cameras /datasets/project/cameras.json"' >> /etc/odm-env
  source /etc/odm-env
fi

# Run the pipeline.
if /usr/local/bin/odm-run.sh; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline complete — syncing outputs to s3://${BUCKET}/${PROJECT}/"
  aws s3 sync "${PROJECT_DIR}/" "s3://${BUCKET}/${PROJECT}/" \
    --exclude "images/*" \
    --region "${REGION}"
  touch "${DONE_MARKER}"

  # Cancel the spot request so AWS doesn't relaunch a new instance after shutdown.
  SPOT_REQUEST_ID=$(curl -s http://169.254.169.254/latest/meta-data/spot/spot-instance-request-id 2>/dev/null || true)
  if [ -n "${SPOT_REQUEST_ID}" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Cancelling spot request ${SPOT_REQUEST_ID}"
    aws ec2 cancel-spot-instance-requests \
      --spot-instance-request-ids "${SPOT_REQUEST_ID}" \
      --region "${REGION}" || true
  fi

  notify "ODM complete: ${PROJECT}" \
    "Outputs synced to s3://${BUCKET}/${PROJECT}/. Spot request cancelled. Shutting down."
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Done. Shutting down in 2 minutes."
  /sbin/shutdown -h +2
else
  touch "${FAILED_MARKER}"

  # Cancel the spot request so AWS doesn't relaunch after shutdown (same as success path).
  SPOT_REQUEST_ID=$(curl -s http://169.254.169.254/latest/meta-data/spot/spot-instance-request-id 2>/dev/null || true)
  if [ -n "${SPOT_REQUEST_ID}" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Cancelling spot request ${SPOT_REQUEST_ID}"
    aws ec2 cancel-spot-instance-requests \
      --spot-instance-request-ids "${SPOT_REQUEST_ID}" \
      --region "${REGION}" || true
  fi

  # Sync whatever outputs and logs exist so they're accessible without the instance.
  aws s3 sync "${PROJECT_DIR}/" "s3://${BUCKET}/${PROJECT}/" \
    --exclude "images/*" \
    --region "${REGION}" || true
  aws s3 cp /var/log/odm-bootstrap.log \
    "s3://${BUCKET}/${PROJECT}/logs/odm-bootstrap.log" \
    --region "${REGION}" || true

  notify "ODM FAILED: ${PROJECT}" \
    "Pipeline failed. Logs synced to s3://${BUCKET}/${PROJECT}/logs/. Instance shutting down in 15 minutes — SSH in to investigate, or touch /data/project/.no-autoshutdown to cancel shutdown. Delete /data/project/.odm-failed to re-run pipeline on next boot."
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Pipeline failed — shutting down in 15 minutes. SSH in to investigate or touch /data/project/.no-autoshutdown to cancel."

  for i in $(seq 1 90); do   # 90 × 10s = 15 min
    sleep 10
    if [ -f "${NO_SHUTDOWN_FLAG}" ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Auto-shutdown cancelled (.no-autoshutdown detected). Instance will remain up."
      notify "ODM FAILED: ${PROJECT}" "Shutdown cancelled. Instance will remain up for investigation. Delete .odm-failed to re-run pipeline."
      exit 0
    fi
  done

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Shutting down now."
  /sbin/shutdown -h +1
fi
