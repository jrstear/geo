# Emlid CSV Format Reference

Source: Emlid Reach RS3, RTK mode, exported from Emlid Flow app.
Example file: `~/stratus/ghostrider/emlid.csv`

## Column Reference

| Column | Example | Notes |
|--------|---------|-------|
| `Name` | `1`, `2`, `rbr7719` | GCP label. Use as the GCP ID in gcpeditpro.txt. |
| `Code` | (empty) | Optional; ignore. |
| `Code description` | (empty) | Optional; ignore. |
| `Easting` | `1666848.163` | Projected X in CRS units (ftUS here). Use in gcpeditpro.txt. |
| `Northing` | `1597697.37` | Projected Y in CRS units (ftUS). Use in gcpeditpro.txt. |
| `Elevation` | `6198.961` | Projected Z (height above geoid, ftUS). Use in gcpeditpro.txt. |
| `Description` | (empty) | Optional; ignore. |
| `Longitude` | `-106.1613153` | WGS84 decimal degrees. **Use for footprint matching.** |
| `Latitude` | `35.39123866` | WGS84 decimal degrees. **Use for footprint matching.** |
| `Ellipsoidal height` | `6134.472` | Ellipsoidal (not geoid) height. Ignore for most uses. |
| `Origin` | `Global` | `Global` = RTK w/ network correction. |
| `Tilt angle` | `0.4` | Receiver tilt in degrees; ignore. |
| `Easting RMS` | `0.038` | Horizontal precision (ftUS). |
| `Northing RMS` | `0.038` | Horizontal precision (ftUS). |
| `Elevation RMS` | `0.034` | Vertical precision (ftUS). |
| `Solution status` | `FIX` | **Filter: only use `FIX` rows.** `FLOAT` is low accuracy. |
| `CS name` | `NAD83(2011) / New Mexico Central (ftUS) + NAVD88(GEOID18) height (ftUS)` | The projected CRS description. Parse for proj4/EPSG. |
| ... | ... | Remaining columns (device, satellites, etc.) can be ignored. |

## Parsing Notes

1. **Filter by `Solution status == FIX`** before using any point.
2. Use `Latitude` + `Longitude` for footprint matching (no CRS transform needed).
3. Use `Easting`, `Northing`, `Elevation` for the gcpeditpro.txt output lines.
4. **CRS**: The `CS name` column gives the human-readable CRS. The known EPSG for
   "NAD83(2011) / New Mexico Central (ftUS)" is **EPSG:6529** (or EPSG:2258 for
   NAD83 classic). Parse this string to select the proj4 string for the header.
   Preferred: use pyproj `CRS.from_user_input(cs_name_string)` which handles the
   compound CRS. Fallback: run the existing CRS inference logic from gcp.py.
5. The `Name` field may be a number (`1`, `2`) or a string (`rbr7719`). Always
   treat as string / label.

## Example Row (parsed)

```python
{
    "label": "1",
    "lat": 35.39123866,
    "lon": -106.1613153,
    "easting": 1666848.163,
    "northing": 1597697.37,
    "elevation": 6198.961,
    "fix": True,
}
```

## Multiple CSV Variants

Different Emlid firmware versions may produce slightly different column names
(e.g., `Point Name` vs `Name`). The parser should handle column matching
by substring (e.g., `east` → Easting, `north` → Northing) as gcp.py does,
and fall back to positional parsing if header is absent.
