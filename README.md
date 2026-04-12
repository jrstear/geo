# geo

A set of tools to make open-source drone mapping easy and accurate.
Designed for a surveyor using Trimble control points, collecting field data
with Emlid and DJI, tagging images, running ODM in AWS, and producing accuracy
reports including control and check point RMSE and orthophoto crops.

See [`docs/odm-workflow.md`](docs/odm-workflow.md) for the workflow.

## Components

| Component | What it does |
|---|---|
| [`transform.py`](transform.py) | Trimble `.dc` → state-plane CSV + design-grid CSV + `transform.yaml`; auto-NGS lookup; post-tagging split into `gcp_list.txt` / `chk_list.txt` / target CSVs |
| [`TargetSighter/sight.py`](TargetSighter/sight.py) | Emlid CSV + drone images → tagging file with EXIF-projected pixel estimates, structural ordering (most-distal anchors first), and optional color-X marker refinement |
| [GCPEditorPro fork](https://github.com/jrstear/GCPEditorPro/tree/feature/auto-gcp-pipeline) | Pixel tagging UI with zoom view, compass/tilt overlays, spacebar confirm, progress badges. The list of modifications relative to upstream uav4geo/GCPEditorPro lives in the fork itself: [`CHANGES-fork.md`](https://github.com/jrstear/GCPEditorPro/blob/feature/auto-gcp-pipeline/CHANGES-fork.md) |
| [`infra/ec2/`](infra/ec2/) | Terraform module that provisions an EC2 ODM instance, runs the pipeline via `odm-bootstrap.sh`, syncs to/from S3, sends SNS stage notifications, and shuts itself down on completion. See [`docs/cloud-infra-spec.md`](docs/cloud-infra-spec.md) |
| [`rmse.py`](rmse.py) | Independent RMSE accuracy assessment from `reconstruction.topocentric.json` + GCP/CHK lists. Generates an HTML report with annotated ortho crops, outlier detection, and optional uncertainty overlay ([example](https://jrstear.github.io/geo-samples/examples/rmse_report.html)) |
| [`packager/`](packager/) | GDAL wrappers for reprojecting + shifting + tiling deliverables into proper form (design-grid CRS, COG, optional downsizing) |
| [`accuracy_study/`](accuracy_study/) | Research scripts: refinement comparison, image-count ablations, projection mode validation, ortho uncertainty overlay generation. See [`accuracy_study/README.md`](accuracy_study/README.md) |
| [`experimental/`](experimental/) | Validated-but-shelved code kept for reference (currently `true_ortho.py`). See [`experimental/README.md`](experimental/README.md) |

## Documentation

| Doc | When to read |
|---|---|
| [`docs/odm-workflow.md`](docs/odm-workflow.md) | Canonical end-to-end workflow |
| [GCPEditorPro `CHANGES-fork.md`](https://github.com/jrstear/GCPEditorPro/blob/feature/auto-gcp-pipeline/CHANGES-fork.md) | Description of GCPEditorPro fork modifications (lives in the fork repo, not here) |
| [`docs/cloud-infra-spec.md`](docs/cloud-infra-spec.md) | AWS infrastructure spec for the ODM EC2 pipeline |
| [`docs/coordinate-flow.md`](docs/coordinate-flow.md) | CRS reference: which coordinate system is used at each step and why |
| [`docs/rmse-explained.md`](docs/rmse-explained.md) | Accuracy concepts for surveyors reading rmse.py output |
| [`docs/details/`](docs/details/) | Long-tail references: camera model, CSV formats, ODM flags, etc. |
| [`docs/plans/`](docs/plans/) | Design specs for not-yet-built features |

## Setup

```bash
bash setup.sh           # creates the conda 'geo' env (gdal, pyproj, opencv, numpy)
conda activate geo
```

All Python pipeline scripts are designed to run inside the `geo` env, e.g.:

```bash
conda run -n geo python TargetSighter/sight.py ...
conda run -n geo python transform.py dc ...
conda run -n geo python rmse.py ...
```

## Issue tracking and development workflow

Multi-session work is tracked in [**bd (beads)**](https://github.com/steveyegge/beads).
Use `bd ready` to find unblocked work, `bd show <id>` to read an issue, and
`bd prime` for full workflow context. First-time setup: `bd hooks install`.

Claude Code-specific permissions and operational conventions live in
[`CLAUDE.md`](CLAUDE.md). The session-close commit/sync/push protocol is
auto-injected by a SessionStart hook.

## License

This repository is currently licensed under the **GNU Affero General Public
License v3** (see [`LICENSE`](LICENSE)). The [GCPEditorPro fork](https://github.com/jrstear/GCPEditorPro/tree/feature/auto-gcp-pipeline)
is governed by its own license (Fair Source, inherited from upstream uav4geo).
