# Permissions
- All local file reads are pre-approved (Read, Glob, Grep, cat, head, etc.)
- All git read commands are pre-approved (log, diff, show, blame, status, branch, etc.)
- All read-only Bash commands are pre-approved (ls, cat, head, tail, wc, find, grep, awk, sed for reading, python one-liners for data inspection, etc.)
- `bd` (beads) commands are pre-approved: list, show, ready, stats, sync, create, update, close, dep, etc.

# Workflow

## Issue Tracking
- Track multi-session work in beads (`bd`). Check `bd ready` before starting.
- Use `TodoWrite` for single-session execution tasks only.
- Set dependencies with `bd dep add` — don't close a bead that blocks others.
- Put detailed design/specs in `docs/` and reference from bead notes.

## Testing & Committing
- Do NOT commit until testing is confirmed.
- Prefer agent-driven testing against ground truth data:
  - Ground truth: `~/stratus/ghostrider gulch/gcp_confirmed.txt`
  - Test script: `accuracy_study/compare_refinement.py <confirmed> <baseline> [variants...]`
  - Run in conda geo env: `conda run -n geo python ...`
- If no automated test is possible, ask the user to test and confirm first.
- After confirmation, in this order:
  1. `bd sync` — commit bead state
  2. `git commit` — commit code changes
  3. `git push` — push everything

## Repositories
- `geo` (this repo) — push to `main`
- `webodm/coreplugins/auto_gcp` — push to fork (origin = user's fork)
- `GCPEditorPro/src/app` — push to fork (origin = user's fork)

## EC2 ODM Instances
- SSH to ODM EC2 instances is pre-approved (read logs, check status, restart services,
  push scripts, run diagnostics). These are ephemeral compute — data is synced to S3.
  - Key: `~/.ssh/geo-odm-ec2.pem`, user: `ec2-user`
  - IP: `cd ~/git/geo/infra/ec2 && terraform output -raw public_ip`
  - All commands require `sudo` on the instance
- AWS CLI operations on the ODM infrastructure are pre-approved:
  - S3 reads/writes within `s3://stratus-jrstear/`
  - EC2 describe/start/stop for ODM instances
  - Do NOT terminate instances or destroy terraform without explicit approval

## Environment
- Pipeline requires the `geo` conda env (cv2, numpy, pyproj):
  `conda run -n geo python TargetSighter/sight.py ...`
- WebODM Docker: coreplugins are NOT volume-mounted — use `docker cp` after editing.
