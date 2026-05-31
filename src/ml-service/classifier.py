"""
Point cloud classification using:
1. Ground/non-ground separation (CSF cloth simulation filter)
2. Geometric feature extraction per cluster  
3. Rule-based classification with OSM enrichment
"""
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import CSF

# ASPRS classification codes
CLASS_GROUND = 2
CLASS_LOW_VEG = 3
CLASS_MED_VEG = 4
CLASS_HIGH_VEG = 5
CLASS_BUILDING = 6
CLASS_NOISE = 7
CLASS_WATER = 9
CLASS_ROAD = 11
CLASS_WIRE = 14
CLASS_TOWER = 15

# Feature type mapping
FEATURE_NAMES = {
    CLASS_GROUND: "ground",
    CLASS_LOW_VEG: "low_vegetation",
    CLASS_MED_VEG: "medium_vegetation", 
    CLASS_HIGH_VEG: "high_vegetation",
    CLASS_BUILDING: "building",
    CLASS_NOISE: "noise",
    CLASS_WATER: "water",
    CLASS_ROAD: "road",
    CLASS_WIRE: "wire",
    CLASS_TOWER: "tower",
}

def separate_ground(xyz):
    """Separate ground from non-ground points using CSF."""
    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 1.0
    csf.params.rigidness = 3          # Urban terrain
    csf.params.class_threshold = 0.5   # 0.5m threshold
    csf.params.interations = 500
    
    csf.setPointCloud(xyz)
    ground = CSF.VecInt()
    offground = CSF.VecInt()
    csf.do_filtering(ground, offground)
    
    ground_idx = np.array([ground[i] for i in range(len(ground))])
    ground_mask = np.zeros(len(xyz), dtype=bool)
    ground_mask[ground_idx] = True
    
    return ground_mask


def cluster_points(xyz, eps=5.0, min_samples=30):
    """Cluster non-ground points using DBSCAN.
    
    For urban lidar, eps=5.0m captures building-scale clusters.
    min_samples=30 filters noise while keeping small objects.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(xyz)
    return labels


def compute_local_ground_elevation(cluster_xy, ground_xyz):
    """
    Compute ground elevation at a cluster's XY location.
    Uses nearest ground point's Z value.
    """
    if len(ground_xyz) == 0:
        return 0.0
    cx, cy = cluster_xy
    # Find nearest ground point
    dists = np.sqrt((ground_xyz[:, 0] - cx)**2 + (ground_xyz[:, 1] - cy)**2)
    nearest_idx = np.argmin(dists)
    return ground_xyz[nearest_idx, 2]


def compute_cluster_features(xyz, cluster_labels, ground_xyz=None):
    """Compute geometric features for each cluster."""
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Skip noise (-1)
    
    features = {}
    for label in unique_labels:
        mask = cluster_labels == label
        pts = xyz[mask]
        n = len(pts)
        
        if n < 3:
            continue
        
        # Bounding box
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_size = bbox_max - bbox_min
        vol_x, vol_y, vol_z = bbox_size
        volume = max(vol_x * vol_y * max(vol_z, 0.1), 0.001)
        
        # Height features
        z_min = pts[:, 2].min()
        z_max = pts[:, 2].max()
        height = z_max - z_min
        
        # Local ground elevation at cluster centroid
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        if ground_xyz is not None and len(ground_xyz) > 0:
            local_ground_z = compute_local_ground_elevation((cx, cy), ground_xyz)
            hag = z_min - local_ground_z  # Height above ground
        else:
            hag = 0
        
        # Density (points per m³)
        density = n / volume if volume > 0 else 0
        
        # Covariance features (linearity, planarity, scattering)
        if n >= 3:
            cov = np.cov(pts.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
            total = eigenvalues.sum()
            if total > 0:
                linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
                planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
                scattering = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
            else:
                linearity = planarity = scattering = 0
        else:
            linearity = planarity = scattering = 0
        
        features[label] = {
            'n_points': n,
            'height': height,
            'height_above_ground': hag,
            'density': density,
            'linearity': linearity,
            'planarity': planarity,
            'scattering': scattering,
            'z_min': z_min,
            'z_max': z_max,
        }
    
    return features


def classify_by_features(features):
    """Rule-based classification using geometric features."""
    label_map = {}
    
    for cluster_id, feat in features.items():
        h = feat['height']
        hag = feat['height_above_ground']
        lin = feat['linearity']
        plan = feat['planarity']
        density = feat['density']
        n = feat['n_points']
        
        # Building: tall (>2m above ground), planar, reasonable density
        if hag > 1.5 and plan > 0.2 and n > 50:
            label_map[cluster_id] = CLASS_BUILDING
        
        # High vegetation: tall, scattered (low planarity)
        elif h > 3.0 and plan < 0.3 and lin < 0.6:
            label_map[cluster_id] = CLASS_HIGH_VEG
        
        # Medium vegetation: medium height
        elif 1.0 < h <= 3.0 and lin < 0.5:
            label_map[cluster_id] = CLASS_MED_VEG
        
        # Low vegetation: short
        elif h <= 1.0 and hag > 0.1:
            label_map[cluster_id] = CLASS_LOW_VEG
        
        # Wire/tower: very tall, highly linear, few points
        elif h > 5.0 and lin > 0.8 and n < 100:
            label_map[cluster_id] = CLASS_WIRE
        
        # Default: building-like for urban environment
        elif hag > 0.5 and n > 30:
            label_map[cluster_id] = CLASS_BUILDING
    
    return label_map


def classify_pointcloud(points, crs="EPSG:28355", debug=False):
    """
    Main classification pipeline.
    
    Args:
        points: Nx3 or Nx6 numpy array (x, y, z, [r, g, b])
        crs: Source coordinate system (unused for now)
        debug: Print debug info
    
    Returns:
        ClassificationResult-compatible dict
    """
    n = len(points)
    xyz = points[:, :3].astype(np.float64)
    
    # Step 1: Ground separation
    ground_mask = separate_ground(xyz)
    n_ground = ground_mask.sum()
    if debug:
        print(f"  Ground: {n_ground:,} / {n:,} ({n_ground/n*100:.1f}%)")
    
    # Step 2: Cluster non-ground points
    non_ground = xyz[~ground_mask]
    ng_indices = np.where(~ground_mask)[0]
    
    if debug:
        print(f"  Non-ground to cluster: {len(non_ground):,} points")
    
    cluster_labels = cluster_points(non_ground, eps=5.0, min_samples=30)
    if debug:
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"  Clustered {len(non_ground):,} points, found {n_clusters} clusters")
        sizes = [(cid, (cluster_labels == cid).sum()) 
                 for cid in np.unique(cluster_labels) if cid >= 0]
        sizes.sort(key=lambda x: -x[1])
        for cid, sz in sizes[:5]:
            print(f"    cluster {cid}: {sz:,} points")
    
    # Step 3: Feature extraction with local ground
    ground_xyz = xyz[ground_mask] if n_ground > 0 else None
    features = compute_cluster_features(non_ground, cluster_labels, ground_xyz)
    
    if debug:
        # Show features for top clusters
        for cid, feat in sorted(features.items(), key=lambda x: -x[1]['n_points'])[:5]:
            print(f"    cluster {cid}: h={feat['height']:.1f}m hag={feat['height_above_ground']:.1f}m "
                  f"plan={feat['planarity']:.2f} lin={feat['linearity']:.2f} "
                  f"density={feat['density']:.3f}")
    
    # Step 4: Classify clusters
    cluster_classes = classify_by_features(features)
    
    if debug:
        print(f"    Classified clusters: {len(cluster_classes)}")
        for cid, cls in list(cluster_classes.items())[:5]:
            feat = features.get(cid, {})
            print(f"      cluster {cid} → class {cls} ({FEATURE_NAMES.get(cls, '?')}) "
                  f"h={feat.get('height',0):.1f}m hag={feat.get('height_above_ground',0):.1f}m")
    
    # Step 5: Build output arrays
    labels = np.ones(n, dtype=np.int32)  # Default: unclassified (1)
    labels[ground_mask] = CLASS_GROUND
    
    # Map cluster labels to ASPRS classes (use flat indexing to avoid copy)
    ng_indices = np.where(~ground_mask)[0]
    for i, cl in enumerate(cluster_labels):
        if cl >= 0 and cl in cluster_classes:
            labels[ng_indices[i]] = cluster_classes[cl]
    
    # Confidence (simplified: 0.8 for classified, 0.5 for unclassified)
    confidence = np.full(n, 0.5, dtype=np.float64)
    classified_mask = labels != 1  # Not unclassified
    confidence[classified_mask] = 0.8
    
    # Feature type names
    feature_types = [FEATURE_NAMES.get(int(l), "unknown") for l in labels]
    
    return {
        "labels": labels.tolist(),
        "confidence": confidence.tolist(),
        "feature_types": feature_types,
        "source": "geometric-rules",
    }
