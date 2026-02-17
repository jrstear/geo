#!/usr/bin/env python3
import csv
import subprocess
import sys
import os
import json
from pathlib import Path
from pyproj import Transformer, CRS

# Candidate CRS systems for New Mexico and UTM
CANDIDATES = [
    ("EPSG:2257", "NAD83 / New Mexico East (ftUS)"),
    ("EPSG:2258", "NAD83 / New Mexico Central (ftUS)"),
    ("EPSG:2259", "NAD83 / New Mexico West (ftUS)"),
    ("EPSG:32613", "WGS84 / UTM zone 13N"),
    ("EPSG:32612", "WGS84 / UTM zone 12N"),
]

def get_image_centroid(image_dir):
    """Get average Lat/Lon from images using exiftool."""
    cmd = ["exiftool", "-json", "-n", "-GPSLatitude", "-GPSLongitude", str(image_dir)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        lats = [item['GPSLatitude'] for item in data if 'GPSLatitude' in item]
        lons = [item['GPSLongitude'] for item in data if 'GPSLongitude' in item]
        
        if not lats:
            return None
        
        return sum(lats)/len(lats), sum(lons)/len(lons)
    except Exception as e:
        print(f"Error reading EXIF: {e}")
        return None

def get_gcp_centroid(csv_path):
    """Read GCP CSV and get average Easting/Northing."""
    points = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Try to find Easting/Northing columns
        x_idx, y_idx, z_idx, label_idx = None, None, None, None
        
        lower_header = [h.lower().strip() for h in header]
        for idx, h in enumerate(lower_header):
            if 'east' in h or h == 'x': x_idx = idx
            if 'north' in h or h == 'y': y_idx = idx
            if 'elev' in h or h == 'z' or h == 'alt': z_idx = idx
            if 'label' in h or h == 'name' or h == 'id': label_idx = idx

        # Default fallback if no header matches well (Matic format: ID, X, Y, Z, Label)
        if x_idx is None: x_idx, y_idx, z_idx, label_idx = 1, 2, 3, 4
        
        for row in reader:
            if not row: continue
            try:
                points.append({
                    'x': float(row[x_idx]),
                    'y': float(row[y_idx]),
                    'z': float(row[z_idx]) if z_idx < len(row) else 0.0,
                    'label': row[label_idx] if label_idx < len(row) else f"pt_{len(points)}"
                })
            except ValueError:
                continue
                
    if not points:
        return None, None
        
    avg_x = sum(p['x'] for p in points) / len(points)
    avg_y = sum(p['y'] for p in points) / len(points)
    
    return (avg_x, avg_y), points

def infer_crs(img_centroid, gcp_centroid):
    """Find the CRS that maps img_centroid closest to gcp_centroid."""
    lat, lon = img_centroid
    best_epsg = None
    best_error = float('inf')
    
    print(f"Image Centroid: {lat:.6f}, {lon:.6f}")
    print(f"GCP Centroid: {gcp_centroid[0]:.3f}, {gcp_centroid[1]:.3f}")
    print("\nTesting candidates...")

    for epsg, name in CANDIDATES:
        try:
            transformer = Transformer.from_crs("EPSG:4324", epsg, always_xy=True)
            proj_x, proj_y = transformer.transform(lon, lat)
            
            error = ((proj_x - gcp_centroid[0])**2 + (proj_y - gcp_centroid[1])**2)**0.5
            print(f" - {epsg} ({name}): Error = {error:.2f} units")
            
            if error < best_error:
                best_error = error
                best_epsg = epsg
        except:
            continue
            
    return best_epsg

def main():
    if len(sys.argv) < 3:
        print("Usage: gcp_infer.py <gcp.csv> <image_dir>")
        sys.exit(1)
        
    csv_path = Path(sys.argv[1])
    img_dir = Path(sys.argv[2])
    output_path = csv_path.with_suffix('.txt')
    
    img_centroid = get_image_centroid(img_dir)
    if not img_centroid:
        print("No GPS data found in images.")
        sys.exit(1)
        
    gcp_centroid, gcp_points = get_gcp_centroid(csv_path)
    if not gcp_centroid:
        print("No GCP data found in CSV.")
        sys.exit(1)
        
    best_epsg = infer_crs(img_centroid, gcp_centroid)
    
    if not best_epsg:
        print("Could not infer CRS.")
        sys.exit(1)
        
    print(f"\nWinner: {best_epsg}")
    
    # Get the Proj.4 string
    crs = CRS.from_user_input(best_epsg)
    proj4 = crs.to_proj4()
    
    # Generate WebODM gcp_list.txt
    # Format: 
    # CRS
    # (blank line)
    # Label,Easting,Northing,Elevation
    with open(output_path, 'w') as f:
        f.write(f"{proj4}\n\n")
        f.write("Label,Easting,Northing,Elevation\n")
        for p in gcp_points:
            f.write(f"{p['label']},{p['x']:.4f},{p['y']:.4f},{p['z']:.4f}\n")
            
    print(f"\nGenerated: {output_path}")
    print("You can now load this .txt file into the WebODM GCP Interface.")

if __name__ == "__main__":
    main()
