#!/usr/bin/env python3
# NOTE: WebODM's interactive GCP interface currently only supports a simplified 
# 4-column CSV/TXT format (Label, Easting, Northing, Elevation) for initial 
# loading. Providing additional columns like 'im_x', 'im_y', or 'image_name' 
# during the initial import may result in a "No points" error in the UI.
import csv
import subprocess
import sys
import os
import json
import math
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from pyproj import Transformer, CRS

# Candidate CRS systems for New Mexico and UTM
CANDIDATES = [
    ("EPSG:2257", "NAD83 / New Mexico East (ftUS)"),
    ("EPSG:2258", "NAD83 / New Mexico Central (ftUS)"),
    ("EPSG:2259", "NAD83 / New Mexico West (ftUS)"),
    ("EPSG:32613", "WGS84 / UTM zone 13N"),
    ("EPSG:32612", "WGS84 / UTM zone 12N"),
]

DEFAULT_FOV = 73.7

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_image_metadata(img_path):
    """Extract GPS and DJI specific metadata using exiftool."""
    cmd = [
        "exiftool", "-json", "-n", 
        "-GPSLatitude", "-GPSLongitude", "-RelativeAltitude", "-FieldOfView",
        str(img_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)[0]
        return {
            'path': img_path,
            'lat': data.get('GPSLatitude'),
            'lon': data.get('GPSLongitude'),
            'alt': data.get('RelativeAltitude'),
            'fov': data.get('FieldOfView', DEFAULT_FOV)
        }
    except Exception:
        return {'path': img_path, 'lat': None, 'lon': None, 'alt': None, 'fov': DEFAULT_FOV}

# ------------------------------------------------------------------------------
# Parallel Worker
# ------------------------------------------------------------------------------

def check_image_relevance(args):
    """
    Worker function to check if an image contains any GCPs based on footprint.
    args: (img_path, gcp_list_gps, fallback_radius)
    """
    img_path, gcp_gps, fallback_radius = args
    meta = get_image_metadata(img_path)
    
    if meta['lat'] is None or meta['lon'] is None:
        return None
    
    # Calculate footprint radius
    # Radius = Height * tan(FOV/2)
    radius = fallback_radius
    if meta['alt'] is not None:
        # FieldOfView in EXIF is often diagonal. We use it as a safe upper bound.
        half_fov_rad = math.radians(meta['fov'] / 2)
        radius = abs(float(meta['alt'])) * math.tan(half_fov_rad)
    
    # Check distance to each GCP
    for gcp_lat, gcp_lon in gcp_gps:
        dist = haversine(meta['lat'], meta['lon'], gcp_lat, gcp_lon)
        if dist <= radius:
            return meta['path']
            
    return None

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------

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

        # Default fallback if no header matches well
        if x_idx is None: x_idx, y_idx, z_idx, label_idx = 1, 2, 3, 0
        
        for row in reader:
            if not row: continue
            try:
                points.append({
                    'x': float(row[x_idx]),
                    'y': float(row[y_idx]),
                    'z': float(row[z_idx]) if z_idx < len(row) else 0.0,
                    'label': row[label_idx] if label_idx < len(row) else f"pt_{len(points)}"
                })
            except (ValueError, IndexError):
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
            transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
            proj_x, proj_y = transformer.transform(lon, lat)
            
            error = ((proj_x - gcp_centroid[0])**2 + (proj_y - gcp_centroid[1])**2)**0.5
            print(f" - {epsg} ({name}): Error = {error:.2f} units")
            
            if error < best_error:
                best_error = error
                best_epsg = epsg
        except:
            continue
            
    return best_epsg

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GCP Inference and Image Selector for WebODM.")
    parser.add_argument("gcp_csv", help="Input GCP CSV file.")
    parser.add_argument("image_dir", help="Directory containing raw images.")
    parser.add_argument("--radius", type=float, default=50.0, help="Fallback radius in meters (default 50.0)")
    parser.add_argument("--no-symlinks", action="store_true", help="Do not create the gcp_images directory")
    parser.add_argument("--threads", type=int, default=cpu_count(), help="Number of parallel threads")
    args = parser.parse_args()

    csv_path = Path(args.gcp_csv)
    img_dir = Path(args.image_dir)
    output_txt = csv_path.with_suffix('.txt')
    
    if not img_dir.exists():
        print(f"Error: Image directory {img_dir} not found.")
        sys.exit(1)

    print(f"🔍 Analyzing images in {img_dir}...")
    
    # 1. Get Centroid for CRS Inference (Quick batch)
    cmd = ["exiftool", "-json", "-n", "-GPSLatitude", "-GPSLongitude", str(img_dir)]
    try:
        exif_out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        exif_data = json.loads(exif_out.stdout)
        lats = [item['GPSLatitude'] for item in exif_data if 'GPSLatitude' in item]
        lons = [item['GPSLongitude'] for item in exif_data if 'GPSLongitude' in item]
        if not lats:
            print("No GPS data found in images.")
            sys.exit(1)
        img_centroid = (sum(lats)/len(lats), sum(lons)/len(lons))
    except Exception as e:
        print(f"Error reading initial EXIF: {e}")
        sys.exit(1)

    # 2. GCP Centroid and CRS Inference
    gcp_centroid, gcp_points = get_gcp_centroid(csv_path)
    if not gcp_centroid:
        print("No GCP data found in CSV.")
        sys.exit(1)
        
    best_epsg = infer_crs(img_centroid, gcp_centroid)
    if not best_epsg:
        print("Could not infer CRS.")
        sys.exit(1)
        
    print(f"\n✅ Winner: {best_epsg}")
    crs = CRS.from_user_input(best_epsg)
    proj4 = crs.to_proj4()
    
    # 3. Project GCPs to GPS for Image Selection
    transformer_to_gps = Transformer.from_crs(best_epsg, "EPSG:4326", always_xy=True)
    gcp_gps_list = []
    for p in gcp_points:
        lon, lat = transformer_to_gps.transform(p['x'], p['y'])
        gcp_gps_list.append((lat, lon))

    # 4. Parallel Image Selection
    images = [img_dir / f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    print(f"\n🚀 Selecting images containing GCPs using {args.threads} threads...")
    
    relevant_images = []
    total = len(images)
    task_args = [(str(img), gcp_gps_list, args.radius) for img in images]
    
    with Pool(args.threads) as p:
        completed = 0
        for result in p.imap_unordered(check_image_relevance, task_args):
            if result:
                relevant_images.append(result)
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                print(f"   Progress: {(completed/total)*100:6.1f}% ({completed}/{total} images processed)")

    print(f"\n✨ Found {len(relevant_images)} relevant images out of {total}.")

    # 5. Generate WebODM gcp_list.txt
    with open(output_txt, 'w') as f:
        f.write(f"{proj4}\n\n")
        f.write("Label,Easting,Northing,Elevation\n")
        for p in gcp_points:
            f.write(f"{p['label']},{p['x']:.4f},{p['y']:.4f},{p['z']:.4f}\n")
            
    print(f"📝 Generated GCP file: {output_txt}")

    # 6. Create Symlinks
    if not args.no_symlinks and relevant_images:
        gcp_img_dir = img_dir.parent / "gcp_images"
        if gcp_img_dir.exists():
            shutil.rmtree(gcp_img_dir)
        gcp_img_dir.mkdir()
        
        for img_path in relevant_images:
            src = Path(img_path).absolute()
            dst = gcp_img_dir / src.name
            os.symlink(src, dst)
            
        print(f"📂 Created symbolic links in: {gcp_img_dir}")

if __name__ == "__main__":
    main()
