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
        # 1. Get the task and its output directory
        task = get_object_or_404(Task, pk=pk, project__owner=request.user)
        task_output_dir = task.get_task_folder()   # check Task model for method name

        # 2. Save uploaded Emlid CSV to temp file
        emlid_csv = request.FILES['emlid_csv']
        ...

        # 3. Find reconstruction.json if requested
        reconstruction_path = None
        if request.data.get('use_reconstruction'):
            candidate = os.path.join(task_output_dir, 'opensfm', 'reconstruction.json')
            if os.path.exists(candidate):
                reconstruction_path = candidate

        # 4. Run pipeline (B1 + B2 + B3) — synchronously (may need Celery for large datasets)
        from .pipeline import run_pipeline
        gcpeditpro_txt, estimates_json = run_pipeline(
            images_dir=os.path.join(task_output_dir, 'images'),
            emlid_csv_path=tmp_csv_path,
            reconstruction_path=reconstruction_path,
        )

        # 5. Return download URLs
        return Response({
            'gcpeditpro_txt': <url>,
            'estimates_json': <url>,
        })
```

**Research needed (R1)**:
- What is the correct method/attribute for `task.get_task_folder()`? See `app/models/task.py`.
- Is there an async job queue (Celery) already wired for long-running plugin tasks?
  Check `tasknotification` plugin and `worker.py` for patterns.
- How does the existing GCP upload flow work (for reference)?

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

R1 should determine which pattern existing action-button plugins use for
long-running operations.

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
