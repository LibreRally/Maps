"""
Test the ML service pipeline end-to-end on a real LAS tile.
"""
import sys, os, time, json
import numpy as np
import laspy
from classifier import classify_pointcloud
from reconstruction import reconstruct_mesh

def main():
    las_dir = os.getenv("LAS_DATA_DIR", "")
    if not las_dir:
        print("LAS_DATA_DIR environment variable not set. Point it to your LAS data directory.", file=sys.stderr)
        return 1
    tile_file = 'Tile_+014_+011.las'
    las_path = os.path.join(las_dir, tile_file)
    
    print(f"Reading {tile_file}...")
    las = laspy.read(las_path)
    n_total = len(las.x)
    
    # Use a manageable sample — faster test
    n_sample = min(100_000, n_total)
    step = max(1, n_total // n_sample)
    idx = slice(0, n_total, step)
    
    xyz = np.vstack((
        np.array(las.x[idx]),
        np.array(las.y[idx]),
        np.array(las.z[idx])
    )).T
    
    print(f"Processing {len(xyz):,} points...")
    
    # Test classification
    t0 = time.time()
    result = classify_pointcloud(xyz, debug=True)
    classify_time = time.time() - t0
    print(f"\nClassification complete in {classify_time:.1f}s")
    
    # Label distribution
    from collections import Counter
    dist = Counter(result['labels'])
    print(f"\nLabel distribution:")
    names = {1: 'unclassified', 2: 'ground', 3: 'low_veg', 4: 'med_veg', 
             5: 'high_veg', 6: 'building', 7: 'noise', 9: 'water', 
             11: 'road', 14: 'wire', 15: 'tower'}
    for label, count in sorted(dist.items()):
        pct = count / len(result['labels']) * 100
        name = names.get(label, f'class_{label}')
        print(f"  {name:15s}: {count:>8,} ({pct:5.1f}%)")
    
    # Test reconstruction on a cluster
    labels = np.array(result['labels'], dtype=np.int32)
    building_mask = labels == 6  # Building class
    if building_mask.sum() > 4:
        print(f"\nReconstructing building cluster ({building_mask.sum()} points)...")
        t0 = time.time()
        mesh = reconstruct_mesh(xyz[building_mask], labels[building_mask], method="alpha_shape")
        recon_time = time.time() - t0
        print(f"Reconstruction complete in {recon_time:.1f}s")
        print(f"  Vertices: {mesh['vertex_count']:,}")
        print(f"  Faces: {mesh['face_count']:,}")
    else:
        print(f"\nNo building points found for reconstruction test")
    
    # Save results
    from pathlib import Path
    output_dir = str(Path(__file__).resolve().parent / "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save point cloud with labels
    classified = np.hstack([xyz, labels.reshape(-1, 1).astype(np.float64)])
    np.save(os.path.join(output_dir, 'classified_sample.npy'), classified)
    
    summary = {
        'tile': tile_file,
        'n_points': len(xyz),
        'classify_time_s': classify_time,
        'label_distribution': {names.get(k, str(k)): v for k, v in dist.items()},
    }
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to data/")
    return 0

if __name__ == '__main__':
    sys.exit(main())
