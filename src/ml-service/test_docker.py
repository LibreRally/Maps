"""Test the Docker ML service with real LAS data."""
import requests, numpy as np, laspy, os, sys, time

las_dir = os.getenv("LAS_DATA_DIR", "")
if not las_dir:
    print("LAS_DATA_DIR environment variable not set. Point it to your LAS data directory.", file=sys.stderr)
    raise SystemExit(1)
las_path = os.path.join(las_dir, "Tile_+014_+011.las")
las = laspy.read(las_path)

# Sample for speed
n = min(5000, len(las.x))
step = max(1, len(las.x) // n)
idx = slice(0, len(las.x), step)

xyz = np.vstack((np.array(las.x[idx]), np.array(las.y[idx]), np.array(las.z[idx]))).T[:n]
rgb = np.vstack((np.array(las.red[idx]), np.array(las.green[idx]), np.array(las.blue[idx]))).T[:n]

# Send to /classify
print(f"POST /classify with {n} points (XYZ+RGB)...")
t0 = time.time()
r = requests.post('http://localhost:8000/classify', json={
    'points': np.hstack([xyz, rgb]).tolist(),
    'crs': 'EPSG:28355'
}, timeout=300)
elapsed = time.time() - t0
result = r.json()
print(f"  {r.status_code} in {elapsed:.0f}s")
print(f"  Source: {result.get('source', '?')}")
from collections import Counter
dist = Counter(result['labels'])
names = {1:'unclassified',2:'ground',3:'low_veg',4:'med_veg',5:'high_veg',
         6:'building',7:'noise',9:'water',11:'road',14:'wire'}
for k,v in sorted(dist.items()):
    pct = v/len(result['labels'])*100
    print(f"    {names.get(k,str(k)):15s}: {v:>6,} ({pct:.1f}%)")

# Test /reconstruct on building points
labels = result['labels']
bldg_idx = [i for i,l in enumerate(labels) if l == 6]
if bldg_idx:
    bldg_pts = xyz[bldg_idx].tolist()
    bldg_labels = [labels[i] for i in bldg_idx]
    print(f"\nPOST /reconstruct with {len(bldg_pts)} building points (poisson)...")
    t0 = time.time()
    r2 = requests.post('http://localhost:8000/reconstruct', json={
        'points': bldg_pts, 'labels': bldg_labels, 'method': 'poisson'
    }, timeout=120)
    elapsed = time.time() - t0
    mesh = r2.json()
    print(f"  {r2.status_code} in {elapsed:.0f}s")
    print(f"  Vertices: {mesh.get('vertex_count',0):,}, Faces: {mesh.get('face_count',0):,}")
else:
    print("\nNo building points for reconstruction test")

print("\nOpen3D Docker E2E test complete")
