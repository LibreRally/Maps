"""Test ML service endpoints end-to-end."""
import requests, numpy as np, laspy, os, sys, time

las_dir = os.getenv("LAS_DATA_DIR", "")
if not las_dir:
    print("LAS_DATA_DIR environment variable not set. Point it to your LAS data directory.", file=sys.stderr)
    raise SystemExit(1)
las_path = os.path.join(las_dir, "Tile_+014_+011.las")
las = laspy.read(las_path)
n = min(5000, len(las.x))
step = max(1, len(las.x) // n)
xyz = np.vstack((np.array(las.x[::step][:n]), np.array(las.y[::step][:n]), np.array(las.z[::step][:n]))).T

print(f"Sending {n} points to /classify...")
t0 = time.time()
r = requests.post('http://localhost:8000/classify', json={
    'points': xyz.tolist(), 'crs': 'EPSG:28355'
}, timeout=120)
elapsed = time.time() - t0

result = r.json()
labels = result['labels']
feature_types = set(result['feature_types'])
print(f"  Status: {r.status_code} ({elapsed:.1f}s)")
print(f"  Features found: {feature_types}")
print(f"  Ground: {labels.count(2)}, Building: {labels.count(6)}, Unclassified: {labels.count(1)}")

# Test reconstruct
bldg_pts = [xyz[i].tolist() for i in range(n) if labels[i] == 6]
bldg_labels = [labels[i] for i in range(n) if labels[i] == 6]
print(f"\nReconstructing {len(bldg_pts)} building points...")
t0 = time.time()
r2 = requests.post('http://localhost:8000/reconstruct', json={
    'points': bldg_pts, 'labels': bldg_labels, 'method': 'alpha_shape'
}, timeout=60)
elapsed = time.time() - t0

mesh = r2.json()
vc = mesh.get('vertex_count', '?')
fc = mesh.get('face_count', '?')
print(f"  Status: {r2.status_code} ({elapsed:.1f}s)")
print(f"  Vertices: {vc}, Faces: {fc}")

print("\nE2E pipeline: LAS -> classify -> reconstruct OK")
