#!/bin/bash
# Per-stage ODM runner. Sources /etc/odm-env for config.
# Skips already-complete stages; sends SNS on each completion or failure.
#
# Hang detection:
#   Primary:  CPU idle watchdog — kills container if CPU < 10% for 45min
#             (ODM is always CPU-active when working; quiet CPU = hung)
#   Backstop: 12h absolute timeout per stage via `timeout`
#             (catches infinite loops where CPU spins but nothing progresses)
#
# Usage: /usr/local/bin/odm-run.sh
#   Called by odm-bootstrap.sh; can also be run directly for debugging.
set -euo pipefail
source /etc/odm-env

PROJECT_DIR=/data/project
IDLE_CPU_THRESHOLD=10      # percent — below this is considered idle
IDLE_CHECK_INTERVAL=300    # seconds between CPU checks (5 min)
IDLE_KILL_CHECKS=9         # kill after this many consecutive idle checks (9×5min = 45min)
STAGE_TIMEOUT=12h          # absolute backstop per stage

SPOT_FLAG=/tmp/odm-spot-interrupted          # written by poller on 2-min warning
INTERRUPTION_COUNT_FILE=/var/log/odm-interruption-count

threads_for() {
  case "$1" in
    dataset|opensfm|odm_filterpoints|odm_georeferencing|odm_dem|odm_report) echo 16 ;;
    openmvs|odm_meshing|mvs_texturing)                                       echo 8  ;;
    odm_orthophoto)                                                           echo 4  ;;
    *) echo 8 ;;
  esac
}

# Returns 0 if stage outputs are present (safe to skip).
# opensfm also checks undistorted images — their absence (from --optimize-disk-space
# on a prior run) forces opensfm to rerun rather than silently skipping.
is_done() {
  case "$1" in
    dataset)            [ -f "${PROJECT_DIR}/opensfm/camera_models.json" ] ;;
    opensfm)            [ -f "${PROJECT_DIR}/opensfm/reconstruction.json" ] &&
                        ls "${PROJECT_DIR}"/opensfm/undistorted/images/*.* &>/dev/null 2>&1 ;;
    openmvs)            [ -f "${PROJECT_DIR}/opensfm/undistorted/openmvs/scene_dense.ply" ] ;;
    odm_filterpoints)   ls "${PROJECT_DIR}"/odm_filterpoints/*.ply &>/dev/null 2>&1 ;;
    odm_meshing)        [ -f "${PROJECT_DIR}/odm_meshing/odm_mesh.ply" ] ;;
    mvs_texturing)      ls "${PROJECT_DIR}"/odm_texturing/*.obj &>/dev/null 2>&1 ;;
    odm_georeferencing) [ -f "${PROJECT_DIR}/odm_georeferencing/odm_georeferenced_model.las" ] ;;
    odm_dem)            ls "${PROJECT_DIR}"/odm_dem/*.tif &>/dev/null 2>&1 ;;
    odm_orthophoto)     [ -f "${PROJECT_DIR}/odm_orthophoto/odm_orthophoto.tif" ] ;;
    odm_report)         ls "${PROJECT_DIR}"/odm_report/*.pdf &>/dev/null 2>&1 ;;
    *) return 1 ;;
  esac
}

notify() {
  aws sns publish \
    --topic-arn "${SNS_TOPIC}" \
    --subject "$1" \
    --message "$2" \
    --region "${REGION}" || true
}

# Post a vertical marker annotation to Grafana Cloud at the current timestamp.
# Silent no-op when GRAFANA_STACK_URL or GRAFANA_API_KEY are absent.
# Runs fire-and-forget (&) so it can never block or fail the pipeline.
# Tags are comma-separated: "odm,stage_start,opensfm"
annotate_grafana() {
  local text="$1" tags_csv="${2:-odm}"
  [ -n "${GRAFANA_STACK_URL:-}" ] && [ -n "${GRAFANA_API_KEY:-}" ] || return 0
  local ts_ms tags_json text_esc
  ts_ms=$(date +%s%3N)
  # "a,b,c" → ["a","b","c"]
  tags_json='["'"$(echo "$tags_csv" | sed 's/,/","/g')"'"]'
  text_esc="${text//\"/\\\"}"
  curl -s -o /dev/null --max-time 5 \
    -X POST "${GRAFANA_STACK_URL}/api/annotations" \
    -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"time\":${ts_ms},\"text\":\"${text_esc}\",\"tags\":${tags_json}}" &
}

# ── Spot interruption: resume detection ───────────────────────────────────────
# On every boot, check whether we were stopped by a spot interruption.
# The poller (below) writes SPOT_FLAG when it catches the 2-min warning.
# EBS persists across stop/start so the flag survives.
if [ -f "${SPOT_FLAG}" ]; then
  INTERRUPTED_AT=$(cat "${SPOT_FLAG}")
  INTERRUPTION_COUNT=$(( $(cat "${INTERRUPTION_COUNT_FILE}" 2>/dev/null || echo 0) + 1 ))
  echo "${INTERRUPTION_COUNT}" > "${INTERRUPTION_COUNT_FILE}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ↩ resumed after spot interruption #${INTERRUPTION_COUNT} (stopped at ${INTERRUPTED_AT})"
  annotate_grafana "↩ resumed after spot interruption #${INTERRUPTION_COUNT} (stopped at ${INTERRUPTED_AT})" "odm,spot,spot_resume"
  notify "ODM ${PROJECT}" \
    "Resumed after spot interruption #${INTERRUPTION_COUNT} on ${PROJECT} (stopped at ${INTERRUPTED_AT}). Continuing pipeline."
  rm -f "${SPOT_FLAG}"
fi

# ── Spot interruption: background poller ──────────────────────────────────────
# Polls EC2 metadata every 5s for the 2-min termination warning.
# On warning: writes SPOT_FLAG, fires Grafana annotation, logs to stdout
# (captured by odm-bootstrap.log → Loki).  Silent no-op on on-demand instances
# (endpoint always 404s).
(
  while true; do
    sleep 5
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
      "http://169.254.169.254/latest/meta-data/spot/termination-time" 2>/dev/null || echo 000)
    if [ "${HTTP}" = "200" ]; then
      TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      echo "${TS}" > "${SPOT_FLAG}"
      echo "${TS}  ⚡ spot interruption warning — instance stopping in ~2min"
      annotate_grafana "⚡ spot interruption warning — ${PROJECT}" "odm,spot,spot_warning"
      break
    fi
  done
) &
SPOT_POLLER_PID=$!
trap 'kill "${SPOT_POLLER_PID}" 2>/dev/null || true' EXIT

# ── Run a single stage with CPU idle watchdog + absolute timeout backstop.
# Container is named odm-<stage> so it can be killed by name from the watchdog.
run_stage() {
  local stage=$1 threads=$2 force_rerun=${3:-true}
  local container="odm-${stage}"
  local rerun_flag=""
  [ "${force_rerun}" = "true" ] && rerun_flag="--rerun ${stage}"

  # Remove any stale container from a prior interrupted run (would cause exit 125).
  docker rm -f "${container}" 2>/dev/null || true

  # Absolute backstop: kill if stage exceeds STAGE_TIMEOUT regardless of CPU
  timeout "${STAGE_TIMEOUT}" \
    docker run --rm --name "${container}" \
      -v "${PROJECT_DIR}":/datasets/project \
      "${ODM_IMAGE:-opendronemap/odm:3.6.0}" \
      --project-path /datasets project \
      ${ODM_FLAGS} \
      --max-concurrency "${threads}" \
      ${rerun_flag} &
  local run_pid=$!

  # Background CPU idle watchdog.
  # Polls docker stats every IDLE_CHECK_INTERVAL seconds.
  # Kills the container (and thus the timeout+docker process) after
  # IDLE_KILL_CHECKS consecutive checks below IDLE_CPU_THRESHOLD.
  (
    local idle_checks=0
    while kill -0 "${run_pid}" 2>/dev/null; do
      sleep "${IDLE_CHECK_INTERVAL}"
      kill -0 "${run_pid}" 2>/dev/null || break  # finished normally while sleeping

      # docker stats --no-stream returns one row; strip % and decimal part
      local cpu_raw cpu_int
      cpu_raw=$(docker stats --no-stream --format "{{.CPUPerc}}" "${container}" 2>/dev/null || echo "100%")
      cpu_int=$(echo "${cpu_raw}" | tr -d '%' | cut -d'.' -f1)
      cpu_int=${cpu_int:-100}

      if [ "${cpu_int}" -lt "${IDLE_CPU_THRESHOLD}" ]; then
        idle_checks=$(( idle_checks + 1 ))
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${stage} idle ${idle_checks}/${IDLE_KILL_CHECKS} (CPU ${cpu_raw})"
      else
        [ "${idle_checks}" -gt 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${stage} active again (CPU ${cpu_raw})"
        idle_checks=0
      fi

      if [ "${idle_checks}" -ge "${IDLE_KILL_CHECKS}" ]; then
        local idle_min=$(( IDLE_KILL_CHECKS * IDLE_CHECK_INTERVAL / 60 ))
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✗ ${stage} hung — CPU idle ${idle_min}min, killing"
        docker stop "${container}" 2>/dev/null || kill "${run_pid}" 2>/dev/null || true
        break
      fi
    done
  ) &
  local watchdog_pid=$!

  wait "${run_pid}"
  local exit_code=$?
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true

  # Translate timeout exit code to something more readable in logs
  if [ "${exit_code}" -eq 124 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✗ ${stage} killed by ${STAGE_TIMEOUT} absolute timeout"
  fi
  return "${exit_code}"
}

STAGES=(dataset opensfm openmvs odm_filterpoints odm_meshing mvs_texturing
        odm_georeferencing odm_dem odm_orthophoto odm_report)

for stage in "${STAGES[@]}"; do
  if is_done "${stage}"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✓ ${stage} (complete, skipping)"
    annotate_grafana "✓ ${stage} (cached — skipped)" "odm,stage_complete,${stage}"
    notify "ODM ${PROJECT}" "Stage ${stage} already complete on ${PROJECT} — skipping (resume after interruption)."
    continue
  fi

  THREADS=$(threads_for "${stage}")

  # opensfm: if reconstruction.json already exists (reconstruct substage complete)
  # but undistorted images don't (undistort substage incomplete), run WITHOUT
  # --rerun so ODM's internal substage cache skips reconstruct and resumes at
  # undistort — preserving similarity_transform.json and hours of work.
  FORCE_RERUN=true
  if [ "${stage}" = "opensfm" ] && \
     [ -f "${PROJECT_DIR}/opensfm/reconstruction.json" ] && \
     ! ls "${PROJECT_DIR}"/opensfm/undistorted/images/*.* &>/dev/null 2>&1; then
    FORCE_RERUN=false
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ▶ ${stage}  [threads=${THREADS}] (resuming from undistort — reconstruction cached)"
    annotate_grafana "▶ ${stage} [threads=${THREADS}] (resuming from undistort)" "odm,stage_start,${stage}"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ▶ ${stage}  [threads=${THREADS}]"
    annotate_grafana "▶ ${stage} [threads=${THREADS}]" "odm,stage_start,${stage}"
  fi

  if run_stage "${stage}" "${THREADS}" "${FORCE_RERUN}"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✓ ${stage} complete"
    annotate_grafana "✓ ${stage} complete" "odm,stage_complete,${stage}"
    notify "ODM ${PROJECT}" "Stage ${stage} complete on ${PROJECT}."
    # Sync outputs to S3 so a terraform destroy+apply can resume from this point.
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Syncing to s3://${BUCKET}/${PROJECT}/ ..."
    aws s3 sync "${PROJECT_DIR}/" "s3://${BUCKET}/${PROJECT}/" \
      --exclude "images/*" \
      --region "${REGION}"
  else
    EXIT=$?
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✗ ${stage} FAILED (exit ${EXIT})"
    annotate_grafana "✗ ${stage} FAILED (exit ${EXIT})" "odm,stage_failed,${stage}"
    # Check if failure was caused by a spot interruption.
    # Prefer the flag written by the background poller (caught 2-min warning);
    # fall back to querying the metadata endpoint directly if the poller missed it.
    SPOT_HTTP=000
    if [ ! -f "${SPOT_FLAG}" ]; then
      SPOT_HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
        "http://169.254.169.254/latest/meta-data/spot/termination-time" 2>/dev/null || echo 000)
      if [ "${SPOT_HTTP}" = "200" ]; then
        # Poller didn't catch the warning (e.g. very short notice) — write flag now
        date -u +%Y-%m-%dT%H:%M:%SZ > "${SPOT_FLAG}"
        annotate_grafana "⚡ spot interruption (late detection) — ${PROJECT}" "odm,spot,spot_warning"
      fi
    fi
    if [ -f "${SPOT_FLAG}" ] || [ "${SPOT_HTTP}" = "200" ]; then
      notify "ODM ${PROJECT}" \
        "Stage ${stage} interrupted by spot termination on ${PROJECT}. EBS preserved — will auto-restart and resume."
    else
      notify "ODM ${PROJECT}" \
        "Stage ${stage} failed on ${PROJECT} (exit ${EXIT}). SSH in to investigate."
    fi
    exit 1
  fi
done

# ── Post-processing: true orthophoto ──────────────────────────────────────────
# Reproject camera imagery with visibility-aware nadir camera selection.
# Requires --dtm in ODM_FLAGS (needs DTM for ground Z).
# Runs inside the ODM container which has numpy/scipy/gdal/cv2.

TRUE_ORTHO_OUT="${PROJECT_DIR}/odm_orthophoto/true_orthophoto.tif"
if [ -f "${TRUE_ORTHO_OUT}" ] || [ -f "${TRUE_ORTHO_OUT%.tif}_cog.tif" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✓ true_ortho (complete, skipping)"
elif [ ! -f "${PROJECT_DIR}/odm_dem/dtm.tif" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ⚠ true_ortho skipped — no DTM (add --dtm to ODM_FLAGS)"
elif [ ! -f "${PROJECT_DIR}/odm_orthophoto/odm_orthophoto.original.tif" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ⚠ true_ortho skipped — no orthophoto"
else
  # Update phase metric for Grafana
  PROM_DIR=/var/lib/node_exporter/textfile_collector
  cat > "${PROM_DIR}/odm_phase.prom" << PROM
# HELP odm_pipeline_phase Current pipeline phase (1=pulling, 2=patching, 3=syncing, 4=running, 5=true_ortho, 8=complete, 9=failed)
# TYPE odm_pipeline_phase gauge
odm_pipeline_phase{project="${PROJECT}"} 5
# HELP odm_pipeline_phase_start_time Unix timestamp when this phase began
# TYPE odm_pipeline_phase_start_time gauge
odm_pipeline_phase_start_time{project="${PROJECT}"} $(date +%s)
PROM
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ▶ true_ortho"
  annotate_grafana "▶ true_ortho" "odm,stage_start,true_ortho"
  notify "ODM ${PROJECT}" "Starting true ortho post-processing on ${PROJECT}."

  NCPU=$(nproc)
  WORKERS=$(( NCPU > 2 ? NCPU - 2 : 1 ))

  DSM_FLAG=""
  if [ -f "${PROJECT_DIR}/odm_dem/dsm.tif" ]; then
    DSM_FLAG="--dsm /data/odm_dem/dsm.tif"
  fi

  docker run --rm --name odm-true-ortho \
    -v "${PROJECT_DIR}":/data \
    -v /usr/local/bin/true_ortho.py:/scripts/true_ortho.py \
    --entrypoint python3 \
    "${ODM_IMAGE:-opendronemap/odm:3.6.0}" \
    /scripts/true_ortho.py \
    /data/opensfm/reconstruction.topocentric.json \
    /data/odm_orthophoto/odm_orthophoto.original.tif \
    /data/images/ \
    --dtm /data/odm_dem/dtm.tif \
    ${DSM_FLAG} \
    --workers "${WORKERS}" \
    -o /data/odm_orthophoto/true_orthophoto.tif

  if [ $? -eq 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✓ true_ortho complete"
    annotate_grafana "✓ true_ortho complete" "odm,stage_complete,true_ortho"
    notify "ODM ${PROJECT}" "True ortho complete on ${PROJECT}."

    # Sync true ortho to S3
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  Syncing true ortho to S3..."
    aws s3 sync "${PROJECT_DIR}/odm_orthophoto/" "s3://${BUCKET}/${PROJECT}/odm_orthophoto/" \
      --exclude "*.tif" --include "true_orthophoto*" \
      --region "${REGION}"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ⚠ true_ortho FAILED (non-fatal, continuing)"
    annotate_grafana "⚠ true_ortho FAILED" "odm,stage_failed,true_ortho"
    notify "ODM ${PROJECT}" "True ortho failed on ${PROJECT} (non-fatal)."
  fi
fi

# Disable strict mode for cleanup — any error here must not poison the exit code
# and prevent bootstrap from triggering the shutdown path.
set +euo pipefail

kill "${SPOT_POLLER_PID:-}" 2>/dev/null || true
wait "${SPOT_POLLER_PID:-}" 2>/dev/null || true

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ═══ All stages complete ═══"
annotate_grafana "═══ ${PROJECT} pipeline complete ═══" "odm,pipeline_complete"
