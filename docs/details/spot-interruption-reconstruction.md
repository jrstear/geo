# Spot Interruption During OpenSFM Reconstruction: Cascading Failure

**Date:** 2026-03-31 (aztec7 run)
**Related beads:** geo-8fg (stage switching), geo-c3q (pause gate), geo-5sx (corruption detection)

## Summary

A spot interruption during the OpenSFM incremental reconstruction phase caused
a cascading failure requiring manual intervention and ~4 hours of wasted compute.

## Timeline

| Time (UTC) | Event |
|---|---|
| 03:42 | Dataset stage starts |
| 03:44 | OpenSFM starts (Run 1) |
| 04:14 | Run 1 stopped manually (debugging bootstrap issues) |
| 04:17 | Bootstrap restarts → OpenSFM Run 2 starts from scratch |
| 05:14 | Reconstruction begins |
| 06:43 | 1168/1385 cameras added, alignment phase begins |
| ~06:50 | **Spot interruption kills instance** during alignment |
| 06:51 | Instance reboots, bootstrap restarts → OpenSFM Run 3 |
| 06:52 | Re-extracts features from scratch (doesn't reuse existing) |
| 07:38 | Matching completes |
| 07:47 | Reconstruction starts |
| 07:48 | **Crash — 90 seconds into reconstruction** (corrupted state) |
| 07:48 | Pipeline marks as failed, shuts down |

## Root Cause

OpenSFM's incremental reconstruction writes state (reconstruction.json, tracks)
incrementally without atomic commits.  A kill during the alignment phase leaves
invalid intermediate state.  On resume, OpenSFM detects existing features/matches
and skips to reconstruction, but the corrupted reconstruction state causes an
immediate crash.

## Impact

- ~4 hours wasted compute on r5.4xlarge spot ($1.008/hr) = ~$4
- Manual intervention required: SSH in, clean state, restart
- The manual fix was: delete corrupted opensfm outputs (keep features/matches),
  copy clean features/matches from a prior run (aztec6), restart

## Key Findings

1. **OpenSFM reconstruction is non-resumable.** There is no checkpoint mechanism
   during the incremental camera addition (2+ hours). A kill at any point requires
   a full restart of reconstruction.

2. **Features and matches ARE reusable.** They are per-image files written atomically.
   A partial feature extraction can be completed by re-running (idempotent).
   Complete features/matches from a prior run with the same images can be reused.

3. **The reconstruction substage is the most vulnerable to spot interruption.**
   It's the longest-running non-resumable phase.  All other stages either checkpoint
   internally or produce atomic outputs.

## Recommendations

### Immediate (geo-5sx)
Detect corrupted opensfm state on resume: check reconstruction.json validity
before allowing opensfm to proceed.  If corrupted, delete reconstruction outputs
(keep features/matches) and re-run cleanly.

### Medium-term (geo-8fg)
Run the reconstruction substage on on-demand instances, not spot.  Feature
extraction, matching, and all post-reconstruction stages can stay on spot.

### Long-term
Contribute an opensfm checkpoint mechanism upstream — save reconstruction state
every N cameras so it can resume from the last checkpoint after interruption.

## Artifacts and Logs

| Artifact | Location |
|---|---|
| Bootstrap log (all runs) | `s3://stratus-jrstear/bsn/aztec7/logs/odm-bootstrap.log` |
| Bootstrap log (on-instance) | EC2 `/var/log/odm-bootstrap.log` (instance `i-0e296195bf0f5182c`, r5.4xlarge spot) |
| GCP file used | `s3://stratus-jrstear/bsn/aztec7/gcp_list.txt` and local `~/stratus/aztec7/gcp_list.txt` (10 GCPs, 162 obs) |
| CHK file | `s3://stratus-jrstear/bsn/aztec7/chk_list.txt` and local `~/stratus/aztec7/chk_list.txt` (31 CHKs, 515 obs) |
| Clean features/matches | `s3://stratus-jrstear/bsn/aztec7/opensfm/{features,matches,exif}/` (moved from aztec6 at ~04:15 UTC) |
| Corrupted reconstruction (Run 3) | Overwritten by Run 4; no copy preserved |
| Successful manual reconstruction (Run 4) | Started ~13:00 UTC via `docker run --rm --entrypoint bash opendronemap/odm:3.5.6-patched -c "opensfm reconstruct"` |
| Terraform config | `geo` repo commit `cace54d` (ODM 3.5.6, exifread patch, `--dtm --dsm --cog --build-overviews`) |
| EC2 instance | `i-0e296195bf0f5182c`, r5.4xlarge spot, $1.008/hr, us-west-2, 124 GB RAM, 16 vCPU |
| ODM image | `opendronemap/odm:3.5.6` (patched to `3.5.6-patched` at boot) |
| Claude Code session | `~/.claude/projects/-Users-jrstear-git-geo/55893a1a-2b7a-4371-bfe5-2404ee325a34.jsonl` |

## Relevance to Paper

This finding demonstrates a practical limitation of using spot instances for
survey-quality photogrammetry pipelines.  The cost savings of spot (~60-70% vs
on-demand) must be weighed against the risk of non-resumable computation loss.
A hybrid approach (spot for parallelizable stages, on-demand for the critical
reconstruction phase) optimizes both cost and reliability.
