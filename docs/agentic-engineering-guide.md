# Agentic Engineering with Claude Code — Guide for This Project

This document covers the Claude Code features and practices used (or planned) in
building the geo / GCPEditorPro / WebODM pipeline.  It is both a reference and a
learning log — reading it alongside the beads and commit history gives a full
picture of how agentic engineering works in practice.

---

## What "agentic engineering" means here

Claude Code agents are not just autocomplete — they can autonomously read files,
run shell commands, edit code, run tests, and manage state across sessions.  The
skill is **designing work so agents can execute it independently and in parallel**,
with clear interfaces between tasks and reliable verification steps.

The geo project is a concrete exercise in this: the experiment framework has four
independent components (rmse_calc, experiment_gen, Terraform, ODM Docker) that can
be built by separate agents simultaneously, coordinated through beads and shared
spec docs.

---

## Claude Code features to learn in this project

### 1. Beads (`bd`) — cross-session issue tracking

The built-in TodoWrite tool is session-only (lost on context clear).  Beads persist
across sessions in git-committed JSONL.

```bash
bd ready                          # what can I work on right now?
bd show geo-rv4                   # full spec before starting
bd update geo-rv4 --status=in_progress   # claim the task
bd close geo-rv4                  # mark complete after verification
bd list --label rmse-exp          # filter to this project
bd list --label rmse-exp --long   # with status
```

**Best practice for agent-ready bead descriptions:**
- State the exact deliverable (file path, function name, CLI interface)
- List inputs and outputs with formats and example paths
- Include the acceptance/verification command an agent can run
- Reference spec docs rather than duplicating spec in the description
- A well-written bead is a complete agent prompt — the agent reads it, reads the
  referenced doc, implements, runs the acceptance test, closes the bead.

### 2. Task tool — spawning parallel subagents

The `Task` tool launches a sub-process agent.  Key patterns:

```
run_in_background: true   ← fire and forget; you get notified on completion
subagent_type: Explore    ← read-only codebase search (no edits)
subagent_type: Plan       ← architecture planning (no edits)
subagent_type: general-purpose  ← full capability including edits + bash
isolation: "worktree"     ← gives the agent its own git branch
```

**When to parallelize:**
- Independent bead implementations (rmse_calc + experiment_gen + Terraform)
- Research tasks (reading docs, searching code) that don't need each other's output
- Document writing (all three spec docs in this session were written in parallel)

**When NOT to parallelize:**
- Sequential dependencies (ODM Docker needs Terraform outputs)
- Tasks that edit the same files (race condition on git state)

**Prompt quality for subagents:**
The agent has no conversation history.  Write prompts that include: what the agent
is, what it should produce, exact file paths, relevant existing code to reference,
and what done looks like.  Treat the prompt like a bead description.

### 3. CLAUDE.md — persistent project instructions

`/Users/jrstear/git/geo/CLAUDE.md` is loaded at the start of every session.  It
overrides Claude's defaults.  Current contents cover: issue tracking workflow,
testing protocol, repository locations, environment.

**When to add to CLAUDE.md:**
- Workflow rules that apply to every session ("do not commit until user confirms")
- Environment facts an agent would otherwise discover by trial and error
- Repo-specific conventions (branch names, docker commands)

**Do not put in CLAUDE.md:**  current task state, in-progress work, anything
session-specific.  That belongs in beads or memory.

### 4. Memory files — cross-session agent context

`~/.claude/projects/*/memory/MEMORY.md` is automatically loaded each session.
It holds stable architectural knowledge: key file paths, design decisions, patterns
that came up more than once.

In this project, MEMORY.md captures: pipeline architecture, GCPEditorPro behaviors,
open bead priorities, user preferences.  An agent starting a new session on a
rmse-exp bead would find the relevant context already there.

**When to update MEMORY.md:**  after implementing something non-obvious; after
discovering a gotcha (e.g. exifr needs `{xmp: true}` + File object, not blob URL);
after a design decision that will affect future sessions.

### 5. Plan mode — get alignment before writing code

For any non-trivial change, `EnterPlanMode` lets the agent explore the codebase
and write a plan for review before touching any files.  The plan file lives in
`~/.claude/plans/`.

This is valuable for: multi-file refactors, new features with architectural
choices, anything where doing it wrong wastes significant time.  For this project,
plan mode is appropriate before implementing rmse_calc.py (coordinate transform
decisions) and the Terraform modules (account-specific choices).

### 6. Worktrees — parallel agents on the same repo

```
isolation: "worktree"
```

When set on a Task invocation, the agent gets a temporary git worktree — its own
copy of the repo on a new branch.  Changes are isolated.  The main session can
merge when done.

**Use this when:** two agents need to edit different files in the same repo at the
same time (e.g. rmse_calc.py agent and experiment_gen.py agent both modifying
files in GCPSighter/).

Without worktrees, parallel agents on the same repo will conflict.  The fix is
either worktrees or sequencing.

### 7. Model selection — cost vs capability

| Task | Best model | Why |
|------|-----------|-----|
| Reading files, searching code | `haiku` | Fast and cheap |
| Writing a single known function | `haiku` | Well-defined, bounded |
| Implementing a new component from spec | `sonnet` | Needs reasoning |
| Architecture planning, complex debugging | `sonnet` or `opus` | High complexity |

Specify `model: "haiku"` on Task invocations for simple research/search tasks.
Default (sonnet) for implementation.  Opus only when you're genuinely stuck.

### 8. Background agents + notification

```python
run_in_background=True
```

The main session continues while the agent works.  You get a system notification
on completion.  Used in this session for the three parallel doc-writing agents.

**Do not poll** — the notification is automatic.  Do not `sleep` and check.

### 9. Hooks — automate session hygiene

Hooks run shell commands on tool events.  The `SessionStart` hook in this project
runs `bd prime` automatically.  You can add hooks for:
- Auto-running linting after file edits
- Auto-committing bead state after `bd close`
- Running tests after edits to specific files

See `~/.claude/settings.json` for hook configuration.

---

## Multi-agent workflow for the rmse-exp beads

The experiment framework has a clear parallel-then-sequential structure:

```
Phase 1 (parallel — start immediately):
  Agent A: geo-rv4  rmse_calc.py
  Agent B: geo-gmp  experiment_gen.py
  Agent C: geo-49z  Terraform modules
  Human:   geo-2zx  Tag CHK-* in GCPEditorPro

Phase 2 (after Phase 1):
  Agent D: geo-12e  ODM Docker adapter  (needs Terraform outputs for S3 paths)

Phase 3 (after Phase 2):
  Agent E: geo-dk5  Experiment driver   (needs all three code components + infra)

Phase 4 (after Phase 3 + human data):
  Agent F: geo-g27  Run experiments + analyze results
```

**Starting a phase:**
```bash
# Check what's ready
bd ready --label rmse-exp

# Assign and start
bd update geo-rv4 --status=in_progress

# Give the agent context
# Prompt: "Work on geo-rv4. Read the bead: bd show geo-rv4.
#          Read the spec: docs/experiment-framework-spec.md.
#          Implement, run acceptance tests, close the bead when done."
```

**Estimated token cost per agent:**
- geo-rv4 (rmse_calc.py): ~3 sessions × $5 = $15
- geo-gmp (experiment_gen.py): ~2 sessions × $4 = $8
- geo-49z (Terraform): ~3 sessions × $5 = $15
- geo-12e (ODM Docker): ~2 sessions × $4 = $8
- geo-dk5 (driver): ~3 sessions × $5 = $15
- Total implementation: ~$60–$80

---

## Portfolio and publication strategy

### Public GitHub repos

The three repos are already public.  To maximize their signal value:

1. **geo** — add a top-level `README.md` that tells the story:
   "Automated GCP pipeline: Emlid rover CSV + drone images → GCPEditorPro
   confirmation → WebODM reconstruction → RMSE accuracy report.  Built
   agentic-ally with Claude Code."  Include a diagram of the pipeline.
   Link to the three spec docs.

2. **GCPEditorPro fork** — the `feature/auto-gcp-pipeline` branch has substantial
   Angular work (zoom view, compass, tilt indicator, shift-click untag, scroll zoom,
   confidence column, progress badges).  A PR description or branch README
   would make this visible.

3. **WebODM plugin** — same branch.  The GCPSighter plugin is production-quality.
   Opening a WebODM upstream PR once the accuracy report is integrated would be
   the highest-visibility contribution.

### WebODM contracting

WebODM's open issues and roadmap are public.  The accuracy report (RMSE per axis
for check points, using reconstruction.json) is a feature WebODM does not have and
that their professional users want.  Concrete path:

1. Implement rmse_calc.py (geo-rv4) and validate against the ghostrider gulch data
2. Integrate it into the WebODM task output (a new "Accuracy Report" tab)
3. Open a PR to ODM/WebODM with the feature + tests
4. That PR is your contract application — it demonstrates the capability directly

The plugin architecture (`geo/GCPSighter/plugin.py`) already shows you know how
WebODM's plugin system works.  The accuracy report integration goes one level
deeper (into core ODM output), which is the level where contracting engagement
typically happens.

### Technical paper

Working title: *"Empirical Optimization of Ground Control Point Selection and
Image Weighting for UAV Photogrammetric Accuracy"*

**Contributions:**
1. A structural priority algorithm for GCP ordering (max-insertion sort on
   convex hull + Z extremes + centroid anti-doming) and its geometric rationale
2. A per-image confidence scoring function (centredness + nadir-weight) and
   evidence for the optimal nadir-weight parameter
3. Ablation experiments: accuracy vs. (GCP count, GCP geometry, images/GCP,
   nadir-weight) on a real survey dataset
4. Open-source implementation in csv2gcp.py + GCPEditorPro + rmse_calc.py

**Structure:** Introduction, Related Work (photogrammetric accuracy standards,
GCP placement guidelines), Methods (pipeline + algorithms), Experiments (setup +
matrix), Results (RMSE tables + figures), Discussion (practical recommendations),
Conclusion.

**Target venues:**
- *UAV-g 2025/2026* (ISPRS working group on UAVs in Geomatics) — most directly
  relevant audience
- *ASPRS Annual Conference* — US-focused, strong surveying community
- *Remote Sensing* journal (MDPI) — open access, reasonable review time
- *ISPRS Journal of Photogrammetry and Remote Sensing* — highest impact, longer review

**The experiments are the paper.**  geo-g27 produces the result tables.
Everything else (algorithm descriptions, open-source code) is already in place.

### Isaiah's marketing report

A 3–4 page condensed version of the paper:
- "What is photogrammetric accuracy and why it matters for your survey"
- "How we validate: independent check points vs. control points"
- "Results: our pipeline achieves X ft RMSE horizontal / Y ft RMSE vertical on
  [survey name] dataset, comparable to Pix4D at 1/50th the software cost"
- One figure: RMSE vs. GCP count (from the experiments)
- Appendix: methodology note (for clients who want the technical detail)

This gives Isaiah a leave-behind that demonstrates rigor without requiring the
client to understand bundle adjustment.  It answers the implicit question
"how do I know your drone survey is accurate?" with a real number and a clear
methodology.

---

## Agentic engineering lessons as they happen

*This section should be updated as the project progresses — add gotchas,
patterns that worked well, cost surprises, and things that required human
intervention.*

| Lesson | Context |
|--------|---------|
| exifr needs `{xmp: true}` AND a File object (not blob URL) for DJI XMP tags | geo-rrn compass debugging |
| Parallel doc-writing agents (3 simultaneous) finished in ~3 min vs ~15 min sequential | rmse-exp doc sprint |
| Bead descriptions that include exact file paths + acceptance commands need zero follow-up questions from agents | geo-rv4, geo-gmp descriptions |
| nadir_weight=0.4 is insufficient to get obliques in top 7 when ≥13 well-centered nadirs exist; 0.2 achieves the interleave | geo nadir-weight experiments |
