import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import sys
from typing import List, Dict, Tuple, Optional


class Triangle:
    """Represents a triangle in the TIN"""
    def __init__(self, v1: Tuple[float, float, float], 
                 v2: Tuple[float, float, float], 
                 v3: Tuple[float, float, float]):
        self.v1 = v1  # (x, y, z)
        self.v2 = v2
        self.v3 = v3
        self.min_x = min(v1[0], v2[0], v3[0])
        self.max_x = max(v1[0], v2[0], v3[0])
        self.min_y = min(v1[1], v2[1], v3[1])
        self.max_y = max(v1[1], v2[1], v3[1])


def parse_landxml_tin(file_path: str) -> Tuple[List[Triangle], str]:
    """
    Parse LandXML TIN file and extract triangles.
    
    LandXML TIN structure typically has:
    - Surface/Definition/Faces with P attribute (point indices)
    - Surface/Definition/Pnts with point coordinates
    
    Returns:
        Tuple of (triangles list, unit string)
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle namespace - try multiple approaches
        namespaces = {}
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0][1:]
            namespaces = {'landxml': ns}
            # Also register default namespace
            ET.register_namespace('', ns)
        
        # Parse units from LandXML
        unit = ''  # Default to no units if detection fails
        units_elem = None
        
        # Try multiple ways to find Units element (handle namespaces)
        # Method 1: Direct iteration
        for elem in root:
            if 'Units' in elem.tag:
                units_elem = elem
                break
        
        # Method 2: Search with namespace
        if units_elem is None:
            try:
                units_elem = root.find('.//Units')
            except:
                pass
        
        # Method 3: Search all elements
        if units_elem is None:
            for elem in root.iter():
                if 'Units' in elem.tag:
                    units_elem = elem
                    break
        
        if units_elem:
            # Check for Imperial units
            imperial = None
            for child in units_elem:
                if 'Imperial' in child.tag:
                    imperial = child
                    break
            
            if imperial:
                linear_unit = imperial.get('linearUnit', '')
                linear_unit_lower = linear_unit.lower()
                # Check for US Survey Foot (most specific first)
                # Must check foot-related terms BEFORE checking for 'm' to avoid false matches
                if 'ussurveyfoot' in linear_unit_lower or 'ussurvey' in linear_unit_lower:
                    unit = 'ft'
                elif 'surveyfoot' in linear_unit_lower or ('survey' in linear_unit_lower and 'foot' in linear_unit_lower):
                    unit = 'ft'
                elif 'foot' in linear_unit_lower or 'ft' in linear_unit_lower:
                    unit = 'ft'
                elif 'meter' in linear_unit_lower or 'metre' in linear_unit_lower:
                    unit = 'm'
                # Don't check for just 'm' as it might match 'USSurveyFoot' or other words
            else:
                # Check for Metric units
                metric = None
                for child in units_elem:
                    if 'Metric' in child.tag:
                        metric = child
                        break
                
                if metric:
                    linear_unit = metric.get('linearUnit', '')
                    linear_unit_lower = linear_unit.lower()
                    if 'foot' in linear_unit_lower or 'ft' in linear_unit_lower:
                        unit = 'ft'
                    elif 'meter' in linear_unit_lower or 'metre' in linear_unit_lower:
                        unit = 'm'
        
        # Debug: log the detected unit and what we found
        if units_elem:
            # Re-find imperial for debug (we already found it above, but this is cleaner)
            imperial_debug = None
            for child in units_elem:
                if 'Imperial' in child.tag:
                    imperial_debug = child
                    break
            if imperial_debug:
                linear_unit_debug = imperial_debug.get('linearUnit', '')
                print(f"DEBUG: Found Imperial units with linearUnit='{linear_unit_debug}', detected unit='{unit}'", file=sys.stderr)
            else:
                metric_debug = None
                for child in units_elem:
                    if 'Metric' in child.tag:
                        metric_debug = child
                        break
                if metric_debug:
                    linear_unit_debug = metric_debug.get('linearUnit', '')
                    print(f"DEBUG: Found Metric units with linearUnit='{linear_unit_debug}', detected unit='{unit}'", file=sys.stderr)
                else:
                    print(f"DEBUG: Found Units element but no Imperial/Metric child, detected unit='{unit}'", file=sys.stderr)
        else:
            print(f"DEBUG: No Units element found, using default unit='{unit}'", file=sys.stderr)
        
        # Try to find Surface element with various namespace approaches
        surface = None
        search_paths = [
            ('.//Surface', {}),
            ('.//{*}Surface', {}),
            ('.//landxml:Surface', namespaces),
        ]
        
        for path, ns_dict in search_paths:
            try:
                surface = root.find(path, ns_dict if ns_dict else None)
                if surface is not None:
                    break
            except:
                continue
        
        if surface is None:
            # Last resort: search all elements
            for elem in root.iter():
                if 'Surface' in elem.tag:
                    surface = elem
                    break
        
        if surface is None:
            raise ValueError("Could not find Surface element in LandXML file")
        
        # Find Definition element
        definition = None
        for child in surface:
            if 'Definition' in child.tag:
                definition = child
                break
        
        if definition is None:
            raise ValueError("Could not find Definition element in Surface")
        
        # Extract points - try multiple approaches
        pnts_elem = None
        for child in definition:
            if 'Pnts' in child.tag:
                pnts_elem = child
                break
        
        if pnts_elem is None:
            raise ValueError("Could not find Pnts element in Definition")
        
        # Build points list and dictionary
        points_list = []  # Ordered list for index-based access
        points_dict = {}  # ID-based lookup
        
        for pnt in pnts_elem:
            if 'P' not in pnt.tag:
                continue
                
            point_id = pnt.get('id')
            coords_text = pnt.text
            if coords_text:
                coords = coords_text.strip().split()
                if len(coords) >= 3:
                    try:
                        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                        point_coord = (x, y, z)
                        points_list.append(point_coord)
                        
                        # Store by ID if available
                        if point_id:
                            points_dict[point_id] = point_coord
                            # Also store by string and int versions of ID
                            points_dict[str(point_id)] = point_coord
                            try:
                                points_dict[int(point_id)] = point_coord
                            except:
                                pass
                    except ValueError:
                        continue
        
        if len(points_list) == 0:
            raise ValueError("No points found in Pnts element")
        
        # Extract faces (triangles) - try multiple approaches
        faces_elem = None
        for child in definition:
            if 'Faces' in child.tag:
                faces_elem = child
                break
        
        if faces_elem is None:
            raise ValueError("Could not find Faces element in Definition")
        
        triangles = []
        failed_count = 0
        sample_errors = []
        
        for face in faces_elem:
            if 'F' not in face.tag:
                continue
            
            success = False
            point_ids = None
            
            # Try to get point indices from P attribute first
            p_attr = face.get('P') or face.get('p')
            if p_attr:
                point_ids = p_attr.strip().replace(',', ' ').split()
            # If no P attribute, try face text content (common format: <F>1 2 3</F>)
            elif face.text:
                point_ids = face.text.strip().replace(',', ' ').split()
            
            if point_ids:
                # Filter out empty strings
                point_ids = [pid for pid in point_ids if pid]
                
                if len(point_ids) >= 3:
                    # FIRST: Try as point IDs (most common - direct ID lookup)
                    # This handles cases where face text contains point IDs like "1 2 3"
                    try:
                        id1 = str(point_ids[0])
                        id2 = str(point_ids[1])
                        id3 = str(point_ids[2])
                        
                        if id1 in points_dict and id2 in points_dict and id3 in points_dict:
                            v1 = points_dict[id1]
                            v2 = points_dict[id2]
                            v3 = points_dict[id3]
                            triangle = Triangle(v1, v2, v3)
                            triangles.append(triangle)
                            success = True
                    except (KeyError, ValueError) as e:
                        if failed_count < 3:
                            sample_errors.append(f"ID-based: {point_ids[:3]} -> {str(e)}")
                    
                    if not success:
                        # SECOND: Try as 1-based indices (subtract 1 to get 0-based array index)
                        try:
                            idx1 = int(point_ids[0]) - 1
                            idx2 = int(point_ids[1]) - 1
                            idx3 = int(point_ids[2]) - 1
                            
                            if 0 <= idx1 < len(points_list) and 0 <= idx2 < len(points_list) and 0 <= idx3 < len(points_list):
                                v1 = points_list[idx1]
                                v2 = points_list[idx2]
                                v3 = points_list[idx3]
                                triangle = Triangle(v1, v2, v3)
                                triangles.append(triangle)
                                success = True
                        except (ValueError, IndexError) as e:
                            if failed_count < 3:
                                sample_errors.append(f"1-based: {point_ids[:3]} -> {str(e)}")
                    
                    if not success:
                        # THIRD: Try as 0-based indices
                        try:
                            idx1 = int(point_ids[0])
                            idx2 = int(point_ids[1])
                            idx3 = int(point_ids[2])
                            
                            if 0 <= idx1 < len(points_list) and 0 <= idx2 < len(points_list) and 0 <= idx3 < len(points_list):
                                v1 = points_list[idx1]
                                v2 = points_list[idx2]
                                v3 = points_list[idx3]
                                triangle = Triangle(v1, v2, v3)
                                triangles.append(triangle)
                                success = True
                        except (ValueError, IndexError) as e:
                            if failed_count < 3:
                                sample_errors.append(f"0-based: {point_ids[:3]} -> {str(e)}")
                    
                    if not success:
                        failed_count += 1
                        if failed_count <= 5:
                            # Store sample of failed attempts for debugging
                            try:
                                idx_vals = [int(pid) for pid in point_ids[:3]]
                                max_idx = max(idx_vals) if idx_vals else 0
                                min_idx = min(idx_vals) if idx_vals else 0
                                sample_errors.append(
                                    f"Face P='{p_attr[:50]}' -> parsed indices: {point_ids[:3]}, "
                                    f"as ints: {idx_vals}, range: [{min_idx}, {max_idx}], "
                                    f"points_list length: {len(points_list)}"
                                )
                            except:
                                sample_errors.append(
                                    f"Face P='{p_attr[:50]}' -> indices: {point_ids[:3]}, "
                                    f"points_list length: {len(points_list)}"
                                )
            
            # Fallback: try to parse coordinates directly from face text
            if not success and face.text:
                coords = face.text.strip().split()
                if len(coords) >= 9:
                    try:
                        v1 = (float(coords[0]), float(coords[1]), float(coords[2]))
                        v2 = (float(coords[3]), float(coords[4]), float(coords[5]))
                        v3 = (float(coords[6]), float(coords[7]), float(coords[8]))
                        triangle = Triangle(v1, v2, v3)
                        triangles.append(triangle)
                        success = True
                    except (ValueError, IndexError):
                        pass
        
        if len(triangles) == 0:
            # Provide diagnostic information
            num_points = len(points_list)
            num_faces = len(list(faces_elem)) if faces_elem is not None else 0
            error_msg = (
                f"No triangles found in TIN file. "
                f"Found {num_points} points and {num_faces} face elements. "
                f"Failed to match {failed_count} faces. "
            )
            if sample_errors:
                error_msg += f"Sample errors: {'; '.join(sample_errors[:3])}"
            raise ValueError(error_msg)
        
        return triangles, unit
    
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error: {str(e)}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error parsing LandXML TIN: {str(e)}")


def parse_gcp_csv(file_path: str) -> pd.DataFrame:
    """
    Parse GCP CSV file with columns: id, Easting, Northing, Elevation, type
    """
    try:
        df = pd.read_csv(file_path)
        
        # Normalize column names (case-insensitive, strip whitespace)
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip().str.lower()
        
        # Required columns
        required_cols = ['id', 'easting', 'northing', 'elevation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            found_cols = ', '.join(original_columns) if len(original_columns) > 0 else 'none'
            raise ValueError(
                f"Missing required columns: {', '.join(missing_cols)}. "
                f"Found columns: {found_cols}. "
                f"Note: Column names are case-insensitive (e.g., 'Easting', 'EASTING', 'easting' are all valid)."
            )
        
        # Ensure numeric columns are numeric
        df['easting'] = pd.to_numeric(df['easting'], errors='coerce')
        df['northing'] = pd.to_numeric(df['northing'], errors='coerce')
        df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
        
        # Check for NaN values
        if df[['easting', 'northing', 'elevation']].isna().any().any():
            raise ValueError("GCP file contains invalid numeric values")
        
        # Type column is optional
        if 'type' not in df.columns:
            df['type'] = ''
        
        return df
    
    except pd.errors.EmptyDataError:
        raise ValueError("GCP CSV file is empty")
    except Exception as e:
        raise ValueError(f"Error parsing GCP CSV: {str(e)}")


def point_in_triangle(triangle: Triangle, x: float, y: float, tolerance: float = 1e-6) -> bool:
    """
    Check if a point (x, y) is inside a triangle using barycentric coordinates.
    Uses a small tolerance to handle points on edges.
    """
    v1, v2, v3 = triangle.v1, triangle.v2, triangle.v3
    
    # Quick bounding box check with tolerance
    if x < triangle.min_x - tolerance or x > triangle.max_x + tolerance or \
       y < triangle.min_y - tolerance or y > triangle.max_y + tolerance:
        return False
    
    # Barycentric coordinate method
    denom = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
    
    if abs(denom) < 1e-10:
        return False  # Degenerate triangle
    
    a = ((v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])) / denom
    b = ((v3[1] - v1[1]) * (x - v3[0]) + (v1[0] - v3[0]) * (y - v3[1])) / denom
    c = 1 - a - b
    
    # Use tolerance for edge cases - increased tolerance for numerical precision issues
    return a >= -tolerance and b >= -tolerance and c >= -tolerance


def barycentric_interpolation(triangle: Triangle, x: float, y: float) -> float:
    """
    Interpolate Z value at point (x, y) using barycentric coordinates.
    """
    v1, v2, v3 = triangle.v1, triangle.v2, triangle.v3
    
    denom = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
    
    if abs(denom) < 1e-10:
        # Degenerate triangle, return average Z
        return (v1[2] + v2[2] + v3[2]) / 3.0
    
    a = ((v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])) / denom
    b = ((v3[1] - v1[1]) * (x - v3[0]) + (v1[0] - v3[0]) * (y - v3[1])) / denom
    c = 1 - a - b
    
    # Interpolate Z
    z = a * v1[2] + b * v2[2] + c * v3[2]
    return z


def distance_to_triangle(triangle: Triangle, x: float, y: float) -> float:
    """Calculate minimum distance from point to triangle (2D) - distance to closest point on triangle"""
    v1, v2, v3 = triangle.v1, triangle.v2, triangle.v3
    
    # Check if point is inside triangle (distance = 0)
    if point_in_triangle(triangle, x, y):
        return 0.0
    
    # Calculate distance to each edge
    def point_to_line_segment(px, py, x1, y1, x2, y2):
        """Distance from point to line segment"""
        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1
        # Vector from line start to point
        px_dx = px - x1
        px_dy = py - y1
        
        # Project point onto line
        t = max(0, min(1, (px_dx * dx + px_dy * dy) / (dx * dx + dy * dy) if (dx * dx + dy * dy) > 0 else 0))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    # Distance to each edge
    dist1 = point_to_line_segment(x, y, v1[0], v1[1], v2[0], v2[1])
    dist2 = point_to_line_segment(x, y, v2[0], v2[1], v3[0], v3[1])
    dist3 = point_to_line_segment(x, y, v3[0], v3[1], v1[0], v1[1])
    
    return min(dist1, dist2, dist3)


def interpolate_elevation(triangles: List[Triangle], x: float, y: float, debug: bool = False) -> Optional[float]:
    """
    Find the triangle containing point (x, y) and interpolate elevation.
    If no triangle contains the point, use nearest neighbor fallback.
    
    Args:
        triangles: List of Triangle objects
        x: Easting coordinate
        y: Northing coordinate
        debug: If True, return diagnostic info as tuple (elevation, debug_dict)
    """
    if not triangles:
        return None
    
    debug_info = {
        'point': (x, y),
        'num_triangles': len(triangles),
        'found_in_triangle': False,
        'used_fallback': False,
        'triangle_index': None,
        'method': None
    }
    
    # First, try to find a triangle containing the point
    # Use bounding box pre-filtering to speed up search
    candidates = []
    for triangle in triangles:
        # Quick bounding box check first
        if triangle.min_x <= x <= triangle.max_x and triangle.min_y <= y <= triangle.max_y:
            candidates.append(triangle)
    
    debug_info['bounding_box_candidates'] = len(candidates)
    
    # If we have candidates, check them for point-in-triangle
    # Otherwise check all triangles (point might be outside all bounding boxes)
    search_list = candidates if candidates else triangles
    
    # Check point-in-triangle for candidates
    for idx, triangle in enumerate(search_list):
        if point_in_triangle(triangle, x, y):
            z = barycentric_interpolation(triangle, x, y)
            debug_info['found_in_triangle'] = True
            debug_info['triangle_index'] = idx
            debug_info['method'] = 'barycentric_interpolation'
            debug_info['elevation'] = z
            if debug:
                return z, debug_info
            return z
    
    # If no triangle contains the point, find the closest triangle
    # Use bounding box filtering for efficiency
    min_dist = float('inf')
    closest_triangle = None
    closest_idx = None
    
    # First try candidates from bounding box check
    for idx, triangle in enumerate(search_list):
        dist = distance_to_triangle(triangle, x, y)
        if dist < min_dist:
            min_dist = dist
            closest_triangle = triangle
            closest_idx = idx
    
    # If no good candidate found, search all triangles
    if closest_triangle is None or min_dist > 1000:  # If very far, search all
        for idx, triangle in enumerate(triangles):
            dist = distance_to_triangle(triangle, x, y)
            if dist < min_dist:
                min_dist = dist
                closest_triangle = triangle
                closest_idx = idx
    
    debug_info['used_fallback'] = True
    debug_info['closest_distance'] = min_dist
    debug_info['triangle_index'] = closest_idx
    
    if closest_triangle:
        v1, v2, v3 = closest_triangle.v1, closest_triangle.v2, closest_triangle.v3
        
        # If very close to triangle, use barycentric interpolation
        if min_dist < 1e-3:  # Within 0.001 units
            # Try barycentric interpolation - it should work if we're close enough
            try:
                z = barycentric_interpolation(closest_triangle, x, y)
                return z
            except:
                pass
        
        # Otherwise, use inverse distance weighting to triangle vertices
        # Calculate distances to vertices
        d1 = np.sqrt((x - v1[0])**2 + (y - v1[1])**2)
        d2 = np.sqrt((x - v2[0])**2 + (y - v2[1])**2)
        d3 = np.sqrt((x - v3[0])**2 + (y - v3[1])**2)
        
        # Avoid division by zero
        if d1 < 1e-10:
            return v1[2]
        if d2 < 1e-10:
            return v2[2]
        if d3 < 1e-10:
            return v3[2]
        
        # Use inverse distance weighting (higher power for more localized effect)
        power = 2.0
        w1 = 1.0 / ((d1 + 1e-10) ** power)
        w2 = 1.0 / ((d2 + 1e-10) ** power)
        w3 = 1.0 / ((d3 + 1e-10) ** power)
        total_weight = w1 + w2 + w3
        
        if total_weight > 0:
            z = (w1 * v1[2] + w2 * v2[2] + w3 * v3[2]) / total_weight
            debug_info['method'] = 'inverse_distance_weighting'
            debug_info['elevation'] = z
            debug_info['triangle_z_values'] = [v1[2], v2[2], v3[2]]
            if debug:
                return z, debug_info
            return z
        else:
            # Fallback to average
            z = (v1[2] + v2[2] + v3[2]) / 3.0
            debug_info['method'] = 'triangle_average'
            debug_info['elevation'] = z
            debug_info['triangle_z_values'] = [v1[2], v2[2], v3[2]]
            if debug:
                return z, debug_info
            return z
    
    if debug:
        return None, debug_info
    return None


def compare_elevations(tin_data: List[Triangle], gcp_data: pd.DataFrame, debug: bool = False) -> List[Dict]:
    """
    Compare TIN elevations with GCP elevations.
    Returns a list of dictionaries with comparison results.
    
    Args:
        tin_data: List of Triangle objects
        gcp_data: DataFrame with GCP points
        debug: If True, include diagnostic information in results
    """
    results = []
    
    # Calculate TIN bounding box for diagnostics and coordinate validation
    if tin_data:
        tin_min_x = min(t.min_x for t in tin_data)
        tin_max_x = max(t.max_x for t in tin_data)
        tin_min_y = min(t.min_y for t in tin_data)
        tin_max_y = max(t.max_y for t in tin_data)
        
        if debug:
            print(f"TIN Bounding Box: X=[{tin_min_x:.2f}, {tin_max_x:.2f}], Y=[{tin_min_y:.2f}, {tin_max_y:.2f}]")
            print(f"TIN has {len(tin_data)} triangles")
        
        # Check if coordinates might be swapped by testing first few GCP points
        # Count how many GCP points are within TIN bounds with normal vs swapped coordinates
        normal_in_bounds = 0
        swapped_in_bounds = 0
        
        for _, row in gcp_data.head(min(10, len(gcp_data))).iterrows():
            easting = float(row['easting'])
            northing = float(row['northing'])
            
            # Check normal order
            if tin_min_x <= easting <= tin_max_x and tin_min_y <= northing <= tin_max_y:
                normal_in_bounds += 1
            
            # Check swapped order
            if tin_min_x <= northing <= tin_max_x and tin_min_y <= easting <= tin_max_y:
                swapped_in_bounds += 1
        
        # If swapped coordinates match better, swap them
        coordinates_swapped = swapped_in_bounds > normal_in_bounds
        
        if debug:
            print(f"Coordinate check: {normal_in_bounds} points in bounds (normal), {swapped_in_bounds} points in bounds (swapped)")
            if coordinates_swapped:
                print("WARNING: Detected coordinate swap - GCP Easting/Northing appear to be swapped!")
    else:
        tin_min_x = tin_max_x = tin_min_y = tin_max_y = 0
        coordinates_swapped = False
    
    for idx, (_, row) in enumerate(gcp_data.iterrows()):
        gcp_id = str(row['id'])
        easting = float(row['easting'])
        northing = float(row['northing'])
        gcp_elevation = float(row['elevation'])
        gcp_type = str(row.get('type', ''))
        
        # Swap coordinates if detected
        if coordinates_swapped:
            x, y = northing, easting  # Swap: use northing as X, easting as Y
            if debug and idx == 0:
                print(f"Using swapped coordinates: GCP Easting->Y, Northing->X")
        else:
            x, y = easting, northing  # Normal: easting is X, northing is Y
        
        # Interpolate elevation from TIN - make sure we're calling it fresh for each point
        if debug:
            tin_elevation, debug_info = interpolate_elevation(tin_data, x, y, debug=True)
            if idx < 3:  # Print first 3 for debugging
                print(f"\nGCP {gcp_id}: Original=({easting:.2f}, {northing:.2f}), Using=({x:.2f}, {y:.2f})")
                print(f"  TIN Elevation: {tin_elevation}")
                print(f"  Found in triangle: {debug_info['found_in_triangle']}")
                print(f"  Used fallback: {debug_info['used_fallback']}")
                print(f"  Method: {debug_info['method']}")
                print(f"  Triangle index: {debug_info.get('triangle_index')}")
                if 'triangle_z_values' in debug_info:
                    print(f"  Triangle Z values: {debug_info['triangle_z_values']}")
        else:
            tin_elevation = interpolate_elevation(tin_data, x, y, debug=False)
        
        if tin_elevation is None:
            # Point is outside TIN coverage
            results.append({
                'gcp_id': gcp_id,
                'easting': easting,
                'northing': northing,
                'gcp_elevation': gcp_elevation,
                'tin_elevation': None,
                'discrepancy': None,
                'absolute_error': None,
                'gcp_type': gcp_type,
                'status': 'outside_coverage'
            })
        else:
            discrepancy = tin_elevation - gcp_elevation
            absolute_error = abs(discrepancy)
            
            results.append({
                'gcp_id': gcp_id,
                'easting': easting,
                'northing': northing,
                'gcp_elevation': gcp_elevation,
                'tin_elevation': tin_elevation,
                'discrepancy': discrepancy,
                'absolute_error': absolute_error,
                'gcp_type': gcp_type,
                'status': 'success'
            })
    
    return results


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate summary statistics from comparison results.
    """
    # Filter out points outside coverage
    valid_results = [r for r in results if r['status'] == 'success' and r['discrepancy'] is not None]
    
    if len(valid_results) == 0:
        return {
            'total_points': len(results),
            'valid_points': 0,
            'outside_coverage': len([r for r in results if r['status'] == 'outside_coverage']),
            'error': 'No valid points for statistics calculation'
        }
    
    discrepancies = [r['discrepancy'] for r in valid_results]
    absolute_errors = [r['absolute_error'] for r in valid_results]
    
    stats = {
        'total_points': len(results),
        'valid_points': len(valid_results),
        'outside_coverage': len([r for r in results if r['status'] == 'outside_coverage']),
        'mean_error': float(np.mean(discrepancies)),
        'std_error': float(np.std(discrepancies)),
        'rmse': float(np.sqrt(np.mean([e**2 for e in discrepancies]))),
        'max_error': float(np.max(absolute_errors)),
        'min_error': float(np.min(absolute_errors)),
        'max_positive_error': float(np.max(discrepancies)),
        'max_negative_error': float(np.min(discrepancies)),
    }
    
    # Error distribution by GCP type
    type_stats = {}
    for result in valid_results:
        gcp_type = result.get('gcp_type', 'Unknown')
        if gcp_type not in type_stats:
            type_stats[gcp_type] = {
                'count': 0,
                'errors': []
            }
        type_stats[gcp_type]['count'] += 1
        type_stats[gcp_type]['errors'].append(result['absolute_error'])
    
    # Calculate stats per type
    for gcp_type, data in type_stats.items():
        type_stats[gcp_type]['mean_error'] = float(np.mean(data['errors']))
        type_stats[gcp_type]['max_error'] = float(np.max(data['errors']))
        type_stats[gcp_type]['min_error'] = float(np.min(data['errors']))
        type_stats[gcp_type]['rmse'] = float(np.sqrt(np.mean([e**2 for e in data['errors']])))
        del type_stats[gcp_type]['errors']  # Remove raw errors list
    
    stats['by_type'] = type_stats
    
    return stats

