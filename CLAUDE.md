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
  - Test script: `compare_refinement.py <confirmed> <baseline> [variants...]`
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

## Environment
- Pipeline requires the `geo` conda env (cv2, numpy, pyproj):
  `conda run -n geo python TargetSighter/sight.py ...`
- WebODM Docker: coreplugins are NOT volume-mounted — use `docker cp` after editing.
