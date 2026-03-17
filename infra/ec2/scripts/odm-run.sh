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

# Run a single stage with CPU idle watchdog + absolute timeout backstop.
# Container is named odm-<stage> so it can be killed by name from the watchdog.
run_stage() {
  local stage=$1 threads=$2
  local container="odm-${stage}"

  # Absolute backstop: kill if stage exceeds STAGE_TIMEOUT regardless of CPU
  timeout "${STAGE_TIMEOUT}" \
    docker run --rm --name "${container}" \
      -v "${PROJECT_DIR}":/datasets/project \
      opendronemap/odm:3.3.0 \
      --project-path /datasets project \
      ${ODM_FLAGS} \
      --max-concurrency "${threads}" \
      --rerun "${stage}" &
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
    continue
  fi

  THREADS=$(threads_for "${stage}")
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ▶ ${stage}  [threads=${THREADS}]"

  if run_stage "${stage}" "${THREADS}"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✓ ${stage} complete"
    notify "ODM ${PROJECT}" "Stage ${stage} complete on ${PROJECT}."
  else
    EXIT=$?
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ✗ ${stage} FAILED (exit ${EXIT})"
    notify "ODM ${PROJECT}" \
      "Stage ${stage} failed on ${PROJECT} (exit ${EXIT}). SSH in to investigate."
    exit 1
  fi
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ═══ All stages complete ═══"
