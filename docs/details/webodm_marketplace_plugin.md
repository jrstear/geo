# WebODM Marketplace Plugin: Auto-GCP Estimator

## Current State

The plugin lives at `webodm/coreplugins/auto_gcp/` — a **coreplugin** embedded in the
WebODM source tree. This means:

- Ships only with installations that have our custom WebODM fork
- No install path for anyone else
- Cannot appear in the WebODM Plugin Manager UI
- Changes require modifying the WebODM source (a fork of OpenDroneMap/WebODM)

## Recommended: Separate Repo + Marketplace Plugin

### Should it be a separate repo?

**Yes.** The standard WebODM community plugin pattern is one plugin per repo, e.g.
`WebODM-Plugin-AutoGCP` or `webodm-plugin-auto-gcp`. Reasons:

- WebODM's plugin manager installs from GitHub repos directly
- Independent versioning and releases
- Installable without touching WebODM source
- Clear ownership boundary from WebODM core
- Community can contribute without needing our full WebODM fork

### Repo structure (proposed)

```
webodm-plugin-auto-gcp/
├── plugin.py
├── manifest.json
├── api_views.py
├── pipeline.py          ← copy of emlid2gcp.py (see sync strategy below)
├── requirements.txt
├── public/
│   ├── TaskView.jsx
│   ├── load_buttons.js
│   └── webpack.config.js
└── README.md
```

### Pipeline sync strategy

`emlid2gcp.py` lives canonically in the `geo` repo. The plugin needs a copy.
Options (in order of recommendation):

1. **Publish `geo` as a pip package** (`pip install geo-gcp-pipeline` or similar).
   Plugin does `pip install git+https://github.com/user/geo.git`. Clean, versioned.
   Requires making `geo` a proper package (setup.py / pyproject.toml).

2. **Keep a copy in the plugin repo**, note in both READMEs that they must be kept
   in sync. Simple but fragile. Current approach.

3. **Git submodule**: plugin repo includes geo as a submodule. More complex.

For now, option 2 is pragmatic. Option 1 is the right long-term answer if the
pipeline becomes stable and reused.

## Install Flow (after conversion)

### Via WebODM Plugin Manager (once registered in marketplace)
1. WebODM Admin → Plugins → Browse
2. Find "Auto-GCP Estimator"
3. Click Install → WebODM downloads from GitHub, places in `media/plugins/auto_gcp/`
4. Restart not required (hot-load)

### Manual install
```bash
cd /path/to/webodm/media/plugins
git clone https://github.com/user/webodm-plugin-auto-gcp auto_gcp
# Then in WebODM Admin → Plugins → enable it
```

### Docker manual install
```bash
docker compose exec webapp bash -c "
  cd /webodm/media/plugins &&
  git clone https://github.com/user/webodm-plugin-auto-gcp auto_gcp
"
# WebODM detects and loads plugins from media/plugins at startup
docker compose restart webapp
```

## Dependency Gap (System + Python Packages)

### Current gaps

| Dependency | Type | Status |
|---|---|---|
| `exiftool` | System binary | Not installed in WebODM Docker image by default |
| `pyproj` | Python/conda | In WebODM's Python env (already a WebODM dependency) |
| `scipy` | Python | NOT in WebODM's Python env — Mode B will fail silently |
| `numpy` | Python | Likely present (WebODM uses it), but not guaranteed |

### Fixes needed

1. **requirements.txt** in the plugin should list `scipy` (and `numpy` as needed).
   WebODM's plugin loader runs `pip install -r requirements.txt` automatically when
   a plugin is installed.

2. **exiftool** cannot be pip-installed. Options:
   a. Add a startup check in `plugin.py` that calls `shutil.which('exiftool')` and
      disables the button / shows a warning if not found.
   b. Document in README that the WebODM Docker image must be extended:
      ```dockerfile
      FROM opendronemap/webodm_webapp
      RUN apt-get update && apt-get install -y libimage-exiftool-perl
      ```
   c. Replace exiftool with a pure-Python EXIF library (e.g. `piexif`, `exifread`).
      This is the cleanest solution but requires pipeline changes.

3. **Plugin health check**: Add a `GET /api/plugins/auto_gcp/health` endpoint that
   checks for exiftool, pyproj, scipy presence and returns JSON status. The
   TaskView button can call this on mount and show a disabled state + tooltip if
   dependencies are missing.

## Marketplace Registration

To appear in the official WebODM plugin browser, submit a PR to:
https://github.com/OpenDroneMap/webodm-plugins-registry

(or equivalent registry repo — check current OpenDroneMap community process)
