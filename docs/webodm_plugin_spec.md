# WebODM Auto-GCP Plugin — Specification

## Overview

A WebODM coreplugin that adds a "Generate GCP Estimates" button to the task
detail view. When clicked, it runs the Python GCP estimation pipeline and
returns a downloadable `gcpeditpro.txt` + `.estimates.json` sidecar.

Repo: `~/git/webodm`
Plugin path: `coreplugins/auto-gcp/`
Branch: `feature/auto-gcp-pipeline`

## Established Plugin Pattern

WebODM's plugin system supports injecting React buttons into the task view.
Reference implementation: `coreplugins/cesiumion/` and `coreplugins/cloudimport/`.

```
auto-gcp/
├── __init__.py
├── manifest.json
├── plugin.py              # PluginBase subclass
├── requirements.txt       # pyproj, exifread, etc.
├── pipeline.py            # GCP estimation pipeline (B1+B2+B3)
├── api_views.py           # DRF API endpoint
├── templates/
│   └── load_buttons.js    # Calls PluginsAPI.Dashboard.addTaskActionButton()
└── public/
    └── TaskView.jsx       # React: button + upload dialog + progress
```

## plugin.py

```python
class Plugin(PluginBase):
    def build_jsx_components(self):
        return ["TaskView.jsx"]

    def include_js_files(self):
        return ["load_buttons.js"]

    def api_mount_points(self):
        return [
            MountPoint("task/(?P<pk>[^/.]+)/generate", GenerateGCPView.as_view()),
        ]
```

## Frontend (TaskView.jsx)

Button "Generate GCP Estimates" appears in task action buttons area.

Clicking opens a modal dialog:
1. File input: "Upload Emlid CSV" (required)
2. Checkbox: "Use reconstruction.json (if available)" — checked by default
3. [Generate] button

On submit:
- POST to `/api/plugins/auto-gcp/task/<task_id>/generate`
  with multipart: `emlid_csv` file + `use_reconstruction` bool
- Show progress spinner
- On success: show download link for `gcpeditpro.txt` (and `.estimates.json`)

## Backend (api_views.py)

```python
class GenerateGCPView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, pk):
        # 1. Get the task and its directories (confirmed from app/models/task.py)
        task = get_object_or_404(Task, pk=pk, project__owner=request.user)
        images_dir = task.task_path("images")          # task.task_path(*args) is correct
        assets_dir = task.assets_path()                 # output assets directory

        # 2. Save uploaded Emlid CSV to temp file
        emlid_csv = request.FILES['emlid_csv']
        ...

        # 3. Find reconstruction.json if requested
        reconstruction_path = None
        if request.data.get('use_reconstruction'):
            candidate = task.assets_path("opensfm", "reconstruction.json")
            if os.path.exists(candidate):
                reconstruction_path = candidate

        # 4. Run pipeline asynchronously via Celery (run_function_async pattern)
        from app.plugins.worker import run_function_async
        from .pipeline import run_pipeline
        result = run_function_async(run_pipeline,
            images_dir=str(images_dir),
            emlid_csv_path=tmp_csv_path,
            reconstruction_path=reconstruction_path,
        )
        # Poll result.task_id for completion; return task_id immediately

        # 5. Return task ID for polling (or download URLs on sync fallback)
        return Response({'task_id': result.task_id})
```

**R1 Findings (confirmed from WebODM source)**:
- `task.task_path(*args)` — correct method for task working directory (NOT `get_task_folder()`)
- `task.assets_path(*args)` — correct method for output assets (replaces `gcp_directory_path`)
- `run_function_async(fn, **kwargs)` from `app.plugins.worker` — Celery-backed async dispatch
- Plugin API: `build_jsx_components()`, `include_js_files()`, `api_mount_points()` all confirmed
- GCP files live in `task.assets_path()` (not a separate gcp directory)

## Pipeline Module (pipeline.py)

The Python backend of the plugin. Wraps the geo-repo pipeline (B1+B2+B3) or
can be a copy/import if the geo repo is not on the WebODM Python path.

Inputs:
- `images_dir`: path to task images directory
- `emlid_csv_path`: path to uploaded Emlid CSV
- `reconstruction_path`: optional path to `opensfm/reconstruction.json`

Outputs: `(gcpeditpro_txt_content: str, estimates_json_content: str)`

This module is the primary thing that replaces `gcp.py`. The geo-repo pipeline
(gcp_pipeline.py) and this module should share code or this one should import
from there (if the geo repo is importable).

## Async Considerations

For flights with 500+ images, the pipeline may take 1-3 minutes. Options:
1. **Synchronous** (simpler): block the request; set a long timeout. Okay for
   small to medium datasets.
2. **Celery task** (robust): use WebODM's existing Celery worker infrastructure
   (see `app/worker.py`). Return a task ID and poll for completion.

The confirmed async pattern (from `tasknotification` and other plugin analysis):
use `run_function_async()` from `app.plugins.worker`. This dispatches to Celery.
Return a task ID to the frontend; frontend polls a status endpoint.

## manifest.json

```json
{
    "name": "Auto-GCP Estimator",
    "description": "Automatically estimates GCP pixel locations from Emlid CSV and drone imagery.",
    "version": "0.1.0",
    "author": "jrstear",
    "permissions": []
}
```
