com# Documentation index

## Start here

- **[odm-workflow.md](odm-workflow.md)** — end-to-end survey-quality ODM workflow
  (Trimble `.dc` → Emlid → sight.py → GCPEditorPro → ODM on EC2 → rmse.py → packager).
  This is the canonical entry point.

## Top-level reference

- **[gcpeditorpro-changes.md](gcpeditorpro-changes.md)** — GCPEditorPro modifications
  on the `feature/auto-gcp-pipeline` branch (zoom view, compass/tilt, tag status,
  spacebar confirm, etc.) with file-change summary.
- **[cloud-infra-spec.md](cloud-infra-spec.md)** — AWS infrastructure spec for the
  ODM EC2 pipeline (terraform module, S3 layout, SNS notifications, spot strategy).
- **[coordinate-flow.md](coordinate-flow.md)** — which CRS is in use at each step
  and why; complements the "CRS notes" section of odm-workflow.md.
- **[rmse-explained.md](rmse-explained.md)** — accuracy concepts for surveyors
  reading rmse.py output.

## [details/](details/) — long-tail references

Format references, hardware specs, design background, historical Q&A, and the
agentic engineering meta-guide. Read on demand, not in normal flow.

## [plans/](plans/) — proposals and not-yet-built specs

Design specs for work that has not yet been implemented (or only partially):
GCP placement advisor, experiment framework.
