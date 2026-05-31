"""
LibreRally.Maps — LAS Point Cloud Processing Pipeline
Phase 0: Ground filtering + clustering on sample tile
"""
import sys, os, time
import numpy as np
import laspy
import CSF
from sklearn.cluster import DBSCAN
from collections import Counter

def read_las(path):
    """Read LAS file and extract XYZ, RGB, classification."""
    print(f"Reading {os.path.basename(path)}...")
    las = laspy.read(path)
    xyz = np.vstack((np.array(las.x), np.array(las.y), np.array(las.z))).T
    
    # Extract RGB if available
    has_rgb = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
    if has_rgb:
        rgb = np.vstack((np.array(las.red), np.array(las.green), np.array(las.blue))).T
    else:
        rgb = None
    
    # Original classification
    classification = np.array(las.classification, dtype=np.uint8)
    
    return las, xyz, rgb, classification

def csf_ground_filter(xyz):
    """
    Cloth Simulation Filter for ground/non-ground separation.
    
    Args:
        xyz: Nx3 numpy array of point coordinates
    
    Returns:
        ground_mask: boolean array, True = ground
    """
    print(f"Running CSF ground filter on {len(xyz):,} points...")
    
    csf = CSF.CSF()
    # Parameters tuned for urban aerial lidar
    csf.params.bSloopSmooth = True       # Enable slope smoothing
    csf.params.cloth_resolution = 1.0    # Cloth grid resolution (meters)
    csf.params.rigidness = 3             # 1=steep terrain, 3=flat terrain (urban)
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.5     # Distance threshold for ground (meters)
    csf.params.interations = 500
    
    csf.setPointCloud(xyz)
    
    ground = CSF.VecInt()
    offground = CSF.VecInt()
    csf.do_filtering(ground, offground)
    
    # Convert CSF VecInt to numpy array of indices
    ground_indices = np.array([ground[i] for i in range(len(ground))])
    ground_mask = np.zeros(len(xyz), dtype=bool)
    ground_mask[ground_indices] = True
    
    # Get the filtered points
    # ground_pts = csf.getPointCloud()[ground_mask]  # nope, API doesn't work like that
    
    pct = ground_mask.sum() / len(ground_mask) * 100
    print(f"  Ground: {ground_mask.sum():,} points ({pct:.1f}%)")
    print(f"  Non-ground: {(~ground_mask).sum():,} points ({100-pct:.1f}%)")
    
    return ground_mask

def cluster_points(xyz, eps=2.0, min_samples=10):
    """
    Cluster points using DBSCAN.
    
    Args:
        xyz: Nx3 numpy array
        eps: Maximum distance between points in a cluster (meters)
        min_samples: Minimum points to form a cluster
    
    Returns:
        labels: cluster labels (-1 = noise)
    """
    print(f"Clustering {len(xyz):,} points (eps={eps}m, min_samples={min_samples})...")
    t0 = time.time()
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(xyz)
    elapsed = time.time() - t0
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    print(f"  Found {n_clusters} clusters in {elapsed:.1f}s")
    print(f"  Noise points: {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")
    
    # Cluster size distribution
    cluster_sizes = Counter(labels[labels >= 0])
    if cluster_sizes:
        sizes = sorted(cluster_sizes.values())
        print(f"  Cluster sizes: min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}")
    
    return labels

def compute_height_above_ground(xyz, ground_mask, labels):
    """
    Compute height above ground for each cluster.
    For ground points, height = 0.
    For clustered non-ground points, height = min distance to ground.
    """
    # Placeholder - would need KD-tree for efficiency
    pass

def main():
    input_dir = os.getenv("LAS_DATA_DIR", "")
    if not input_dir:
        print("LAS_DATA_DIR environment variable not set. Point it to your LAS data directory.", file=sys.stderr)
        return 1
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a sample tile
    tile = 'Tile_+014_+011.las'  # Dense city center, ~3.6M pts
    input_path = os.path.join(input_dir, tile)
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return 1
    
    # Step 1: Read LAS
    las, xyz, rgb, classification = read_las(input_path)
    
    original_class_dist = Counter(classification)
    print(f"  Format: LAS {las.header.version}")
    print(f"  Points: {len(xyz):,}")
    print(f"  Bounds: X=[{xyz[:,0].min():.0f}, {xyz[:,0].max():.0f}]")
    print(f"          Y=[{xyz[:,1].min():.0f}, {xyz[:,1].max():.0f}]")
    print(f"          Z=[{xyz[:,2].min():.1f}, {xyz[:,2].max():.1f}]")
    print(f"  Original classification: {dict(original_class_dist)}")
    print(f"  Point format: {las.header.point_format.id}")
    print(f"  Has RGB: {rgb is not None}")
    if rgb is not None:
        print(f"  RGB range: R=[{rgb[:,0].min()}, {rgb[:,0].max()}], "
              f"G=[{rgb[:,1].min()}, {rgb[:,1].max()}], "
              f"B=[{rgb[:,2].min()}, {rgb[:,2].max()}]")
    
    # Step 2: Ground filtering
    ground_mask = csf_ground_filter(xyz)
    
    # Step 3: Cluster non-ground points
    non_ground_xyz = xyz[~ground_mask]
    indices = np.arange(len(non_ground_xyz))  # default: all indices
    if len(non_ground_xyz) > 1_000_000:
        # Sample for clustering to avoid memory/time issues
        sample_size = min(500_000, len(non_ground_xyz))
        print(f"\nSampling {sample_size:,} non-ground points for clustering...")
        indices = np.random.choice(len(non_ground_xyz), sample_size, replace=False)
        cluster_labels = cluster_points(non_ground_xyz[indices], eps=3.0, min_samples=20)
    else:
        cluster_labels = cluster_points(non_ground_xyz, eps=3.0, min_samples=20)
    
    # Step 4: Write output LAS with new classification
    # Reclassify: 2 = ground, 1 = unclassified (non-ground), 6 = building (clustered)
    new_class = np.ones(len(xyz), dtype=np.uint8)  # default: unclassified
    new_class[ground_mask] = 2  # ground
    
    if len(non_ground_xyz) > 1_000_000:
        # Clustered non-ground → class 6 (building)
        clustered_mask = cluster_labels >= 0
        # Map back to original non-ground indices
        ng_indices = np.where(~ground_mask)[0]
        sampled_indices = ng_indices[indices]
        new_class[sampled_indices[clustered_mask]] = 6  # building
    
    # Write
    output_path = os.path.join(output_dir, tile.replace('.las', '_classified.las'))
    print(f"\nWriting classified output to {output_path}...")
    
    out_las = laspy.create(point_format=las.header.point_format.id, file_version=str(las.header.version))
    out_las.x = las.x
    out_las.y = las.y
    out_las.z = las.z
    out_las.classification = new_class
    if rgb is not None:
        out_las.red = las.red
        out_las.green = las.green
        out_las.blue = las.blue
    
    # Copy header info
    out_las.header.offsets = las.header.offsets
    out_las.header.scales = las.header.scales
    
    out_las.write(output_path)
    
    # Summary
    final_dist = Counter(new_class)
    print(f"\n=== FINAL CLASSIFICATION ===")
    labels = {1: 'Unclassified (non-ground)', 2: 'Ground', 6: 'Building (clustered)'}
    for cls, count in sorted(final_dist.items()):
        pct = count / len(new_class) * 100
        cls_int = int(cls)
        label = labels.get(cls_int, f'Class {cls_int}')
        print(f"  Class {cls_int:2d}: {count:>10,} ({pct:5.1f}%)  {label}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
