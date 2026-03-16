#!/bin/bash
# Polls IMDS for spot interruption notice; sends SNS alert on 2-min warning.
# EBS is preserved on stop — AWS auto-restarts (persistent spot) and
# odm-bootstrap.sh resumes automatically on reboot. No action needed.
source /etc/odm-env

META_URL="http://169.254.169.254/latest/meta-data/spot/termination-time"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

while true; do
  HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "${META_URL}" || echo 000)
  if [ "${HTTP}" = "200" ]; then
    echo "$(date -u) Spot termination imminent"
    aws sns publish \
      --topic-arn "${SNS_TOPIC}" \
      --subject "ODM spot: termination imminent" \
      --message "Instance ${INSTANCE_ID} interrupted. EBS preserved. AWS will auto-restart the instance (persistent spot) and resume the pipeline. No action needed." \
      --region "${REGION}" || true
    exit 0
  fi
  sleep 5
done
