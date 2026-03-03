# GCPSighter

WebODM community plugin that automatically estimates GCP pixel locations, via line-of-sight
calculations using an emlid.csv and raw images as inputs.  

## Requirements

- WebODM (any recent version)
- `exiftool` installed in the WebODM container (the standard `opendronemap/webodm_webapp` image includes it)
- Python packages listed in `requirements.txt` are installed automatically by WebODM on first load

## Installing

### 1. Create the zip

```bash
cd /path/to/geo
zip -r GCPSighter.zip GCPSighter/
```

### 2. Upload via WebODM Admin

1. Open **http://your-webodm-host/admin/app/plugin/**
2. Click **Upload Plugin**
3. Select `GCPSighter.zip` and submit

WebODM extracts the zip, installs Python dependencies (`pyproj`, `scipy`, `numpy`), and compiles the UI component.

### 3. Restart the webapp container

```bash
docker compose restart webapp
```

### 4. Verify

The plugin appears in **Admin → Plugins** as **GCPSighter**.
Open any task's expanded panel — a **GCPSighter** button will appear in the action bar.

## Usage

1. Process your drone images in WebODM as usual (with or without GCPs)
2. Click the **GCPSighter** button on any task
3. Upload your Emlid CSV file
4. Optionally enable **Use reconstruction.json** for more accurate pixel estimates (requires a completed reconstruction)
5. Click **Generate** — the pipeline runs and produces a `gcpeditpro.txt` file
6. Download `gcpeditpro.txt` and open it in [GCPEditorPro](https://uav4geo.com/software/gcpeditorpro) to review and confirm the estimates

## Output

`gcpeditpro.txt` — GCPEditorPro / OpenDroneMap `gcp_list.txt` format with an extra `confidence` column:

```
EPSG:6529
431000.123  3760000.456  1234.5  1024.00  768.00  DJI_0001.JPG  GCP-1  projection
...
```

Confidence values: `projection` (pipeline estimate), `mouse_click` (user-confirmed in GCPEditorPro).

## Updating

Re-zip the `GCPSighter/` directory, delete the existing plugin in Admin, and re-upload.
