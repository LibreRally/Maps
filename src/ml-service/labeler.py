"""
OSM-Guided Point Cloud Labeler

Generates labeled training data by spatially joining 
point cloud clusters with OpenStreetMap features.
"""
import numpy as np
import os
import psycopg2
from pyproj import Transformer
from collections import defaultdict

# ASPRS ↔ OSM tag mapping
OSM_TO_ASPRS = {
    'building': 6,          # Building
    'road': 11,             # Road surface
    'highway': 11,          # Road surface
    'water': 9,             # Water
    'park': 3,              # Low vegetation (parks)
    'forest': 5,            # High vegetation
    'wood': 5,              # High vegetation
    'grass': 3,             # Low vegetation
    'farmland': 3,          # Low vegetation
    'railway': 10,          # Rail
    'power': 15,            # Transmission tower
    'man_made': 6,          # Building-like structures
}


def connect_db():
    """Connect to PostgreSQL."""
    return psycopg2.connect(
        host='localhost', port=5432,
        dbname='mapsdb', user='mapsuser', password='mapspass'
    )


def get_osm_features_in_bounds(conn, bounds_wgs84):
    """
    Query OSM buildings and features within a bounding box.
    
    Args:
        bounds_wgs84: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        List of dicts with osm_id, feature_type, geometry_wkt
    """
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                osm_id,
                COALESCE(building, highway, amenity, 'unknown') as feature_type,
                COALESCE(name, '') as name,
                COALESCE(building, '') as building_tag,
                COALESCE(highway, '') as highway_tag,
                ST_AsText(ST_Transform(way, 4326)) as geom_wkt,
                ST_AsText(ST_Transform(ST_Centroid(way), 4326)) as centroid_wkt,
                ST_Area(ST_Transform(way, 4326)::geography) as area_sqm
            FROM osm.planet_osm_polygon
            WHERE (building IS NOT NULL OR highway IS NOT NULL OR amenity IS NOT NULL)
              AND way && ST_Transform(
                  ST_MakeEnvelope(%s, %s, %s, %s, 4326), 3857)
            ORDER BY area_sqm DESC
            LIMIT 200
        """, (min_lon, min_lat, max_lon, max_lat))
        
        features = []
        for row in cur.fetchall():
            features.append({
                'osm_id': row[0],
                'feature_type': row[1],
                'name': row[2],
                'building_tag': row[3],
                'highway_tag': row[4],
                'geom_wkt': row[5],
                'centroid_wkt': row[6],
                'area_sqm': row[7],
            })
    
    return features


def label_clusters_with_osm(cluster_centroids_wgs84, osm_features, 
                             transformer_mga_to_wgs):
    """
    Label point cloud clusters by finding the closest OSM feature.
    
    Args:
        cluster_centroids_wgs84: List of (lon, lat) for each cluster
        osm_features: List of OSM feature dicts
        transformer_mga_to_wgs: pyproj Transformer
    
    Returns:
        Dict mapping cluster_id → (asprs_class, osm_feature_type, confidence)
    """
    labels = {}
    
    for cluster_id, (clon, clat) in enumerate(cluster_centroids_wgs84):
        best_dist = float('inf')
        best_feature = None
        
        for feat in osm_features:
            # Extract centroid from WKT
            centroid_wkt = feat['centroid_wkt']
            # Parse "POINT(lon lat)" 
            try:
                coords = centroid_wkt.replace('POINT(', '').replace(')', '').split()
                flon, flat = float(coords[0]), float(coords[1])
                dist = np.sqrt((clon - flon)**2 + (clat - flat)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_feature = feat
            except (ValueError, IndexError):
                continue
        
        if best_feature and best_dist < 0.001:  # ~100m at equator
            feature_type = best_feature['feature_type']
            # Map to ASPRS class
            asprs_class = OSM_TO_ASPRS.get(feature_type, 0)
            if best_feature['building_tag']:
                asprs_class = OSM_TO_ASPRS['building']
            elif best_feature['highway_tag']:
                asprs_class = OSM_TO_ASPRS['highway']
            
            if asprs_class > 0:
                labels[cluster_id] = (asprs_class, feature_type, 0.9)
    
    return labels


def generate_training_sample(las_path, output_dir, max_points=100000):
    """
    Generate a labeled training sample from a LAS tile.
    
    1. Read LAS file
    2. Run ground separation + clustering
    3. Query OSM features for the tile bounds
    4. Label clusters with OSM labels
    5. Save as numpy arrays for PointNet2 training
    
    Args:
        las_path: Path to LAS file
        output_dir: Directory to save training data
        max_points: Maximum points to process
    
    Returns:
        Dict with statistics
    """
    import laspy
    import os
    from classifier import separate_ground, cluster_points
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read LAS
    las = laspy.read(las_path)
    n_total = len(las.x)
    
    # Sample if too large
    if n_total > max_points:
        step = n_total // max_points
        idx = slice(0, n_total, step)
        xyz = np.vstack((
            np.array(las.x[idx]),
            np.array(las.y[idx]),
            np.array(las.z[idx])
        )).T
        rgb = np.vstack((
            np.array(las.red[idx]),
            np.array(las.green[idx]),
            np.array(las.blue[idx])
        )).T
    else:
        xyz = np.vstack((
            np.array(las.x), np.array(las.y), np.array(las.z)
        )).T
        rgb = np.vstack((
            np.array(las.red), np.array(las.green), np.array(las.blue)
        )).T
    
    n = len(xyz)
    print(f"Processing {n:,} points from {os.path.basename(las_path)}")
    
    # Ground separation
    ground_mask = separate_ground(xyz)
    print(f"  Ground: {ground_mask.sum():,} ({ground_mask.sum()/n*100:.1f}%)")
    
    # Compute WGS84 bounds
    transformer = Transformer.from_crs('EPSG:28355', 'EPSG:4326', always_xy=True)
    min_lon, min_lat = transformer.transform(xyz[:, 0].min(), xyz[:, 1].min())
    max_lon, max_lat = transformer.transform(xyz[:, 0].max(), xyz[:, 1].max())
    
    # Query OSM features
    conn = connect_db()
    try:
        osm_features = get_osm_features_in_bounds(
            conn, (min_lon, min_lat, max_lon, max_lat))
        print(f"  OSM features in bounds: {len(osm_features)}")
    finally:
        conn.close()
    
    # Cluster non-ground points
    non_ground = xyz[~ground_mask]
    if len(non_ground) > 100_000:
        ng_idx = np.random.choice(len(non_ground), 100_000, replace=False)
        cluster_labels_sample = cluster_points(non_ground[ng_idx], eps=3.0, min_samples=20)
        cluster_labels = np.full(len(non_ground), -1)
        for cid in np.unique(cluster_labels_sample):
            if cid >= 0:
                cluster_labels[ng_idx[cluster_labels_sample == cid]] = cid
    else:
        cluster_labels = cluster_points(non_ground, eps=3.0, min_samples=20)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  Clusters: {n_clusters}")
    
    # Compute cluster centroids in WGS84
    cluster_centroids = []
    cluster_ids = []
    for cid in np.unique(cluster_labels):
        if cid < 0:
            continue
        mask = cluster_labels == cid
        c_xyz = non_ground[mask]
        cx, cy = c_xyz[:, 0].mean(), c_xyz[:, 1].mean()
        clon, clat = transformer.transform(cx, cy)
        cluster_centroids.append((clon, clat))
        cluster_ids.append(cid)
    
    # Label clusters with OSM
    osm_labels = label_clusters_with_osm(
        cluster_centroids, osm_features, transformer)
    
    print(f"  OSM-labeled clusters: {len(osm_labels)}")
    
    # Build final labels array
    class_labels = np.ones(n, dtype=np.int32)  # Default: unclassified
    class_labels[ground_mask] = 2  # Ground
    
    for i, cl in enumerate(cluster_labels):
        if cl >= 0 and cl in osm_labels:
            asprs_class, feature_type, confidence = osm_labels[cl]
            ng_idx = np.where(~ground_mask)[0]
            class_labels[ng_idx[i]] = asprs_class
    
    # Statistics
    from collections import Counter
    stats = Counter(class_labels.tolist())
    print(f"  Label distribution: {dict(stats)}")
    
    # Save training data
    tile_name = os.path.splitext(os.path.basename(las_path))[0]
    np.savez_compressed(
        os.path.join(output_dir, f"{tile_name}_training.npz"),
        points=xyz.astype(np.float32),
        colors=rgb.astype(np.uint16),
        labels=class_labels.astype(np.int32),
        ground_mask=ground_mask,
        cluster_labels=cluster_labels,
    )
    
    # Save metadata
    metadata = {
        'tile': tile_name,
        'n_points': int(n),
        'n_ground': int(ground_mask.sum()),
        'n_clusters': n_clusters,
        'n_osm_labeled': len(osm_labels),
        'osm_features': len(osm_features),
        'label_distribution': {str(k): v for k, v in stats.items()},
        'bounds_wgs84': [min_lon, min_lat, max_lon, max_lat],
    }
    with open(os.path.join(output_dir, f"{tile_name}_meta.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


if __name__ == '__main__':
    import sys
    from pathlib import Path
    las_path = os.getenv("LAS_DATA_DIR", "")
    if not las_path:
        las_path = os.path.join(os.getenv("LAS_DATA_DIR", ""), "Tile_+014_+011.las")
    if not las_path or not os.path.exists(las_path):
        print("Set LAS_DATA_DIR to your LAS data directory, or provide a path.", file=sys.stderr)
        sys.exit(1)
    output_dir = str(Path(__file__).resolve().parent / "data")
    
    meta = generate_training_sample(las_path, output_dir, max_points=50000)
    print(f"\nMetadata saved: {meta}")
