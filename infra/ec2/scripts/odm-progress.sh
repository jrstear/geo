#!/bin/bash
# ODM progress reporter — runs every minute via cron.
# Identifies the current running ODM stage/substage, computes progress indicators,
# and writes Prometheus textfile metrics for node_exporter to scrape → Grafana Cloud.
#
# Metrics written:
#   odm_stage_active{stage,substage,project}  1 while running, 0 otherwise
#   odm_progress_ratio{stage,substage,project} 0.0–1.0  (NaN when unknown)
#   odm_progress_done{stage,substage,project}  items completed
#   odm_progress_total{stage,substage,project} items total (0 = unknown)
#   odm_elapsed_seconds{stage,substage,project} wall-clock since container start
#
# Requires: node_exporter started with
#   --collector.textfile.directory=/var/lib/node_exporter/textfile_collector

set -uo pipefail

source /etc/odm-env 2>/dev/null || true

PROJECT_DIR=/data/project
TEXTFILE_DIR=/var/lib/node_exporter/textfile_collector
OUTFILE="${TEXTFILE_DIR}/odm_progress.prom"
PROJECT_LABEL="${PROJECT//[\/ ]/_}"
TOTAL_IMAGES=$(ls "${PROJECT_DIR}/images/" 2>/dev/null | wc -l | tr -d ' \n')

mkdir -p "${TEXTFILE_DIR}"

# ── Find running ODM container ─────────────────────────────────────────────────
CURRENT_STAGE=""
for s in dataset opensfm openmvs odm_filterpoints odm_meshing mvs_texturing \
          odm_georeferencing odm_dem odm_orthophoto odm_report; do
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^odm-${s}$"; then
    CURRENT_STAGE="${s}"
    break
  fi
done

# If nothing is running, write zeroed metrics and exit
if [ -z "${CURRENT_STAGE}" ]; then
  cat > "${OUTFILE}" << PROM
# HELP odm_stage_active 1 if an ODM stage container is currently running
# TYPE odm_stage_active gauge
odm_stage_active{stage="none",substage="none",project="${PROJECT_LABEL}"} 0
PROM
  exit 0
fi

# ── Container elapsed time ─────────────────────────────────────────────────────
STARTED_AT=$(docker inspect --format '{{.State.StartedAt}}' "odm-${CURRENT_STAGE}" 2>/dev/null || echo "")
ELAPSED_SECONDS=0
if [ -n "${STARTED_AT}" ]; then
  START_EPOCH=$(date -d "${STARTED_AT}" +%s 2>/dev/null || \
                python3 -c "from dateutil.parser import parse; import calendar; print(int(calendar.timegm(parse('${STARTED_AT}').timetuple())))" 2>/dev/null || echo 0)
  NOW_EPOCH=$(date +%s)
  ELAPSED_SECONDS=$(( NOW_EPOCH - START_EPOCH ))
fi

# ── Stage-specific progress ────────────────────────────────────────────────────
SUBSTAGE="unknown"
DONE=0
TOTAL=0

case "${CURRENT_STAGE}" in

  opensfm)
    FEATURES_DONE=$(ls "${PROJECT_DIR}/opensfm/features/" 2>/dev/null | wc -l | tr -d ' \n')
    MATCHES_DONE=$(ls "${PROJECT_DIR}/opensfm/matches/" 2>/dev/null | wc -l | tr -d ' \n')
    UNDISTORTED=$(ls "${PROJECT_DIR}/opensfm/undistorted/images/" 2>/dev/null | wc -l | tr -d ' \n')

    if [ "${FEATURES_DONE:-0}" -lt "${TOTAL_IMAGES:-0}" ] 2>/dev/null; then
      SUBSTAGE="detect_features"
      DONE=${FEATURES_DONE}
      TOTAL=${TOTAL_IMAGES}

    elif [ ! -f "${PROJECT_DIR}/opensfm/reconstruction.json" ]; then
      # Either match_features or reconstruct — distinguish by log content
      FULL_LOGS=$(docker logs "odm-${CURRENT_STAGE}" 2>&1)
      SHOTS_ADDED=$(echo "${FULL_LOGS}" | grep -c "Adding .* to the reconstruction" 2>/dev/null || true)

      if [ "${SHOTS_ADDED}" -gt 0 ]; then
        SUBSTAGE="reconstruct"
        DONE=${SHOTS_ADDED}
        TOTAL=${TOTAL_IMAGES}
      else
        SUBSTAGE="match_features"
        DONE=${MATCHES_DONE}
        TOTAL=0   # total pairs depends on matching strategy — unknown
      fi

    elif [ "${UNDISTORTED:-0}" -lt "${TOTAL_IMAGES:-0}" ] 2>/dev/null; then
      SUBSTAGE="undistort"
      DONE=${UNDISTORTED}
      TOTAL=${TOTAL_IMAGES}

    else
      SUBSTAGE="finishing"
      DONE=${TOTAL_IMAGES}
      TOTAL=${TOTAL_IMAGES}
    fi
    ;;

  dataset)
    # dataset stage reads all images; track via log "Reading data for image X (queue-size=N)"
    # queue-size tells us how many are queued/remaining — extract the max seen as a proxy
    FULL_LOGS=$(docker logs "odm-${CURRENT_STAGE}" 2>&1)
    SUBSTAGE="load_images"
    DONE=$(echo "${FULL_LOGS}" | grep -c "Reading data for image\|Extracting.*features" 2>/dev/null || true)
    TOTAL=${TOTAL_IMAGES}
    ;;

  openmvs)
    SUBSTAGE="dense_matching"
    # openmvs logs "Processed X/Y images" or similar — grab last progress line
    FULL_LOGS=$(docker logs "odm-${CURRENT_STAGE}" 2>&1)
    PROGRESS_LINE=$(echo "${FULL_LOGS}" | grep -oE "[0-9]+/[0-9]+" | tail -1)
    if [ -n "${PROGRESS_LINE}" ]; then
      DONE=$(echo "${PROGRESS_LINE}" | cut -d'/' -f1)
      TOTAL=$(echo "${PROGRESS_LINE}" | cut -d'/' -f2)
    else
      DONE=0
      TOTAL=0
    fi
    ;;

  odm_filterpoints|odm_meshing|mvs_texturing|odm_georeferencing|odm_dem|odm_orthophoto|odm_report)
    SUBSTAGE="processing"
    DONE=0
    TOTAL=0
    ;;

esac

# ── Compute ratio ──────────────────────────────────────────────────────────────
RATIO="NaN"
DONE=${DONE:-0}
TOTAL=${TOTAL:-0}
if [ "${TOTAL}" -gt 0 ] 2>/dev/null; then
  RATIO=$(awk "BEGIN {printf \"%.4f\", ${DONE}/${TOTAL}}")
fi

# ── Write textfile metrics ─────────────────────────────────────────────────────
LABELS="stage=\"${CURRENT_STAGE}\",substage=\"${SUBSTAGE}\",project=\"${PROJECT_LABEL}\""

cat > "${OUTFILE}.tmp" << PROM
# HELP odm_stage_active 1 if an ODM stage container is currently running
# TYPE odm_stage_active gauge
odm_stage_active{${LABELS}} 1

# HELP odm_progress_ratio Fraction of current substage complete (0.0-1.0, NaN if unknown)
# TYPE odm_progress_ratio gauge
odm_progress_ratio{${LABELS}} ${RATIO}

# HELP odm_progress_done Items completed in current substage
# TYPE odm_progress_done gauge
odm_progress_done{${LABELS}} ${DONE}

# HELP odm_progress_total Total items in current substage (0 if unknown)
# TYPE odm_progress_total gauge
odm_progress_total{${LABELS}} ${TOTAL}

# HELP odm_elapsed_seconds Seconds since current stage container started
# TYPE odm_elapsed_seconds gauge
odm_elapsed_seconds{${LABELS}} ${ELAPSED_SECONDS}
PROM

if [ -n "${INSTANCE_COST_PER_HOUR:-}" ]; then
  cat >> "${OUTFILE}.tmp" << PROM

# HELP odm_instance_cost_per_hour On-demand or spot price for this instance (USD/hour)
# TYPE odm_instance_cost_per_hour gauge
odm_instance_cost_per_hour{project="${PROJECT_LABEL}"} ${INSTANCE_COST_PER_HOUR}
PROM
fi

mv "${OUTFILE}.tmp" "${OUTFILE}"
