import os

import numpy as np

from classifier import (
    CLASS_BUILDING,
    CLASS_GROUND,
    CLASS_HIGH_VEG,
    CLASS_LOW_VEG,
    CLASS_MED_VEG,
    CLASS_NOISE,
    CLASS_ROAD,
    CLASS_TOWER,
    CLASS_WATER,
    CLASS_WIRE,
    FEATURE_NAMES,
)
from open3d_classifier import (
    classify_with_randlanet,
    get_inference_device,
    get_model_path,
    get_model_url,
    has_open3d_ml,
    reconstruct_with_open3d,
)

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    o3d = None
    _HAS_OPEN3D = False

try:
    import torch
except ImportError:
    torch = None


DEFAULT_MODEL_PATH = get_model_path()
MESHABLE_FEATURES = {
    feature.strip().lower()
    for feature in os.environ.get(
        "OPEN3D_MESHABLE_FEATURES",
        "ground,road,building,high_vegetation,medium_vegetation,low_vegetation,water,wire,tower",
    ).split(",")
    if feature.strip()
}
SEGMENT_CLUSTER_POINT_CAP = int(os.environ.get("OPEN3D_SEGMENT_CLUSTER_POINT_CAP", "250000"))
GROUND_DISTANCE_THRESHOLD = float(os.environ.get("OPEN3D_GROUND_DISTANCE_THRESHOLD", "0.2"))
GROUND_MIN_FRACTION = float(os.environ.get("OPEN3D_GROUND_MIN_FRACTION", "0.05"))
GROUND_RANSAC_ITERATIONS = int(os.environ.get("OPEN3D_GROUND_RANSAC_ITERATIONS", "800"))
GROUND_FALLBACK_PERCENTILE = float(os.environ.get("OPEN3D_GROUND_FALLBACK_PERCENTILE", "0.08"))
GROUND_FALLBACK_HEIGHT = float(os.environ.get("OPEN3D_GROUND_FALLBACK_HEIGHT", "0.45"))
RANDLANET_MIN_POINTS = int(os.environ.get("OPEN3D_RANDLANET_MIN_POINTS", "2048"))
DEGENERATE_DOMINANT_LABEL_THRESHOLD = float(
    os.environ.get("OPEN3D_DEGENERATE_DOMINANT_LABEL_THRESHOLD", "0.97")
)
GROUND_RECOVERY_FRACTION = float(os.environ.get("OPEN3D_GROUND_RECOVERY_FRACTION", "0.05"))

# ── Colour / PBR texture generation ──────────────────────────────
TEXTURE_SIZE = int(os.environ.get("OPEN3D_TEXTURE_SIZE", "1024"))
TEXTURE_BAKE_MAX_FACES = int(os.environ.get("OPEN3D_TEXTURE_BAKE_MAX_FACES", "50000"))
ENABLE_PBR_TEXTURES = os.environ.get("OPEN3D_ENABLE_PBR_TEXTURES", "1").lower() in {
    "1", "true", "yes", "on",
}
TEXTURE_OVERRIDE_DIR = os.environ.get("OPEN3D_TEXTURE_OVERRIDE_DIR", "")

CLASS_CLUSTER_SETTINGS = {
    CLASS_GROUND: (2.0, 120),
    CLASS_ROAD: (1.8, 80),
    CLASS_BUILDING: (1.4, 60),
    CLASS_HIGH_VEG: (1.2, 40),
    CLASS_MED_VEG: (1.0, 30),
    CLASS_LOW_VEG: (0.9, 20),
    CLASS_WIRE: (0.5, 12),
    CLASS_TOWER: (0.9, 18),
    CLASS_WATER: (2.0, 60),
    CLASS_NOISE: (0.6, 12),
}

CLASS_CONFIDENCE = {
    CLASS_GROUND: 0.88,
    CLASS_ROAD: 0.78,
    CLASS_BUILDING: 0.76,
    CLASS_HIGH_VEG: 0.7,
    CLASS_MED_VEG: 0.67,
    CLASS_LOW_VEG: 0.64,
    CLASS_WIRE: 0.72,
    CLASS_TOWER: 0.68,
    CLASS_WATER: 0.7,
    CLASS_NOISE: 0.45,
}

FEATURE_METHODS = {
    "ground": ("projected_surface", "alpha_shape", "ball_pivoting"),
    "road": ("projected_surface", "alpha_shape", "ball_pivoting"),
    "building": ("poisson", "alpha_shape", "ball_pivoting"),
    "high_vegetation": ("ball_pivoting", "alpha_shape"),
    "medium_vegetation": ("ball_pivoting", "alpha_shape"),
    "low_vegetation": ("ball_pivoting", "alpha_shape"),
    "wire": ("ball_pivoting", "alpha_shape"),
    "tower": ("poisson", "ball_pivoting", "alpha_shape"),
    "water": ("alpha_shape", "ball_pivoting"),
    "noise": ("ball_pivoting", "alpha_shape"),
    "unclassified": ("alpha_shape", "ball_pivoting"),
}


def get_runtime_info():
    model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0
    if not _HAS_OPEN3D:
        classifier = "legacy-geometric"
        reconstructor = "legacy-segmented"
    else:
        classifier = "randlanet" if has_open3d_ml() and model_exists else "open3d-geometric"
        reconstructor = "open3d-segmented"

    return {
        "open3d_available": _HAS_OPEN3D,
        "open3d_ml_available": has_open3d_ml(),
        "model_path": DEFAULT_MODEL_PATH,
        "model_url": get_model_url(),
        "model_present": model_exists,
        "classifier": classifier,
        "reconstructor": reconstructor,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "inference_device": get_inference_device(),
    }


def classify_semantic_labels(xyz, rgb=None):
    labels, confidence, _ = classify_semantic_labels_with_source(xyz, rgb)
    return labels, confidence


def classify_semantic_labels_with_source(xyz, rgb=None):
    if len(xyz) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), "empty"

    if not _HAS_OPEN3D:
        raise RuntimeError("Open3D is required but not available in the mlservice runtime.")

    if has_open3d_ml() and os.path.exists(DEFAULT_MODEL_PATH) and len(xyz) >= RANDLANET_MIN_POINTS:
        try:
            normalized_xyz = _normalize_for_inference(xyz)
            labels, confidence = classify_with_randlanet(normalized_xyz, rgb)
            labels, confidence, source = _recover_degenerate_ml_output(xyz, labels, confidence)
            return np.asarray(labels, dtype=np.int32), np.asarray(confidence, dtype=np.float32), source
        except Exception as exc:
            raise RuntimeError(f"Open3D-ML inference failed: {exc}") from exc

    labels, confidence = _classify_geometrically(xyz)
    return labels, confidence, "open3d-geometric"


def build_segment_ids(xyz, labels):
    if len(xyz) == 0:
        return np.empty(0, dtype=np.int32)

    if not _HAS_OPEN3D:
        raise RuntimeError("Open3D is required for segment generation.")

    segment_ids = np.full(len(xyz), -1, dtype=np.int32)
    next_segment_id = 1

    for class_id in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) == 0:
            continue

        if _should_short_circuit_segmentation(class_id, len(class_indices)):
            segment_ids[class_indices] = next_segment_id
            next_segment_id += 1
            continue

        points = xyz[class_indices]
        clusters = _segment_class_points(points, class_id)
        for cluster in clusters:
            member_indices = class_indices[np.asarray(cluster, dtype=np.int64)]
            if len(member_indices) < _minimum_segment_points(class_id):
                continue

            segment_ids[member_indices] = next_segment_id
            next_segment_id += 1

    return segment_ids


def estimate_ground_plane(xyz):
    if len(xyz) == 0:
        return None

    normals = _estimate_normals(xyz)
    _, plane_model = _detect_ground(xyz, normals=normals)
    return plane_model


def classify_segment_geometry(points, plane_model=None):
    if len(points) == 0:
        return CLASS_NOISE, CLASS_CONFIDENCE[CLASS_NOISE]

    normals = _estimate_normals(points)
    return _classify_cluster(points, normals, plane_model)


def reconstruct_segment_meshes_open3d(xyz, rgb, segment_ids, segments, max_reconstruction_points=0):
    if not _HAS_OPEN3D:
        raise RuntimeError("Open3D is required for mesh reconstruction.")

    meshes = []

    for segment in segments:
        if not _should_reconstruct_segment_mesh(segment):
            continue

        member_indices = np.where(segment_ids == segment["localId"])[0]
        if len(member_indices) < 10:
            continue

        points = xyz[member_indices]
        source_rgb = None
        if rgb is not None and len(rgb) == len(xyz):
            source_rgb = rgb[member_indices]
        if max_reconstruction_points > 0 and len(points) > max_reconstruction_points:
            step = max(len(points) // max_reconstruction_points, 1)
            points = points[::step]

        vertices = None
        faces = None
        for method in _mesh_methods_for_segment(segment):
            try:
                vertices, faces = reconstruct_with_open3d(points, method=method)
            except Exception:
                vertices = None
                faces = None

            if _is_valid_mesh(vertices, faces):
                break

        if not _is_valid_mesh(vertices, faces):
            continue

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        feature_type = segment["featureType"]

        segment["vertexCount"] = int(len(vertices))
        segment["faceCount"] = int(len(faces))

        # Display colour (flat median, always computed as fallback)
        display_color = _compute_segment_display_color(source_rgb)
        if display_color is not None:
            segment["displayColor"] = display_color.tolist()

        # Per-vertex colours from point cloud
        vertex_colors = _project_vertex_colors(points, source_rgb, vertices)

        # PBR textures
        base_texture_data = None
        normal_texture_data = None
        mesh_uvs = None
        if ENABLE_PBR_TEXTURES and vertex_colors is not None and len(faces) <= TEXTURE_BAKE_MAX_FACES:
            # Check for external (AI-upscaled) texture override first
            override_texture = _check_texture_override(segment["localId"], feature_type)
            if override_texture is not None:
                base_texture_data = override_texture
            else:
                mesh_uvs = _generate_uvs(vertices, faces, feature_type)
                if mesh_uvs is not None and len(mesh_uvs) == len(vertices):
                    base_texture_data = _bake_texture(
                        vertices, faces, vertex_colors, mesh_uvs,
                        TEXTURE_SIZE, feature_type,
                    )
                    mesh_normals = _compute_mesh_normals(vertices, faces)
                    if mesh_normals is not None:
                        normal_texture_data = _bake_normal_map(
                            vertices, faces, mesh_normals, mesh_uvs, TEXTURE_SIZE,
                        )

        mesh_dict = {
            "localId": segment["localId"],
            "asprsClass": segment["asprsClass"],
            "name": feature_type,
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "displayColor": None if display_color is None else display_color.tolist(),
            "vertexColors": None if vertex_colors is None else vertex_colors.tolist(),
            "uvs": None if mesh_uvs is None else mesh_uvs.tolist(),
            "baseColorTexture": base_texture_data,
            "normalTexture": normal_texture_data,
        }
        meshes.append(mesh_dict)

    return meshes


# ── Colour / PBR Texture Functions ───────────────────────────────

def _project_vertex_colors(source_xyz, source_rgb, mesh_vertices):
    """Project per-point RGB onto reconstructed mesh vertices via nearest-neighbor.

    Uses scipy.spatial.KDTree for efficient O(N log M) lookup.
    Returns (N, 3) uint8 array of per-vertex colours, or None if no source colour data.
    """
    if source_rgb is None:
        return None

    from scipy.spatial import KDTree

    source_rgb = np.asarray(source_rgb, dtype=np.float32)
    if source_rgb.ndim != 2 or source_rgb.shape[0] < 1 or source_rgb.shape[1] < 3:
        return None

    source_xyz = np.asarray(source_xyz[:, :3], dtype=np.float64)
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)

    if len(source_xyz) < 1 or len(mesh_vertices) < 1:
        return None

    tree = KDTree(source_xyz)
    _, indices = tree.query(mesh_vertices)
    return np.asarray(source_rgb[indices, :3], dtype=np.uint8)


def _compute_segment_display_color(segment_rgb):
    if segment_rgb is None:
        return None

    colors = np.asarray(segment_rgb, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[0] == 0 or colors.shape[1] < 3:
        return None

    median_color = np.median(colors[:, :3], axis=0)
    return np.clip(np.round(median_color), 0, 255).astype(np.uint8)


def _generate_uvs(vertices, faces, feature_type):
    """Generate UV coordinates for a mesh segment.

    Strategy by feature type:
      - ground / road: planar XY projection (fast, preserves proportions)
      - building / tower: trimesh LSCM unwrap → planar XY fallback
      - vegetation / wire: cylindrical projection → planar XY fallback
      - everything else: planar XY

    Returns (N, 2) float32 UV array in [0, 1] range, or None on failure.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)

    if feature_type in ("ground", "road"):
        return _planar_uv(v, axis=(0, 1))

    if feature_type in ("building", "tower"):
        uvs = _unwrap_uv(v, f)
        if uvs is not None:
            return uvs
        return _planar_uv(v, axis=(0, 1))

    if feature_type in ("high_vegetation", "medium_vegetation", "low_vegetation", "wire"):
        uvs = _cylindrical_uv(v)
        if uvs is not None:
            return uvs
        return _planar_uv(v, axis=(0, 1))

    return _planar_uv(v, axis=(0, 1))


def _planar_uv(vertices, axis=(0, 1)):
    """Planar UV projection using the two axes with largest spatial extent.

    Automatically selects the best 2 axes from XYZ to avoid degenerate
    (collinear) UVs which break Delaunay triangulation in griddata.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) < 3:
        # Too few vertices — return trivial UVs
        return np.zeros((len(v), 2), dtype=np.float32)

    # Pick the two axes with the largest range (avoid degenerate axis)
    ranges = np.ptp(v, axis=0)  # (3,) array
    if np.all(ranges < 1e-6):
        return np.zeros((len(v), 2), dtype=np.float32)
    # Descending order of range
    axis_order = np.argsort(-ranges)
    ax_u = int(axis_order[0])
    ax_v = int(axis_order[1])

    uv = v[:, [ax_u, ax_v]].copy()
    u_range = float(np.ptp(uv[:, 0]))
    v_range = float(np.ptp(uv[:, 1]))
    uv[:, 0] = (uv[:, 0] - float(np.min(uv[:, 0]))) / max(u_range, 1e-6)
    uv[:, 1] = (uv[:, 1] - float(np.min(uv[:, 1]))) / max(v_range, 1e-6)
    return np.asarray(uv, dtype=np.float32)


def _cylindrical_uv(vertices):
    """Cylindrical UV projection around the Z axis (useful for vegetation/wire clusters)."""
    v = vertices.copy()
    centroid_x = float(np.mean(v[:, 0], dtype=np.float64))
    centroid_y = float(np.mean(v[:, 1], dtype=np.float64))
    dx = v[:, 0] - centroid_x
    dy = v[:, 1] - centroid_y
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx * dx + dy * dy)

    angle_range = float(np.ptp(angles))
    if angle_range < 1e-6:
        return None

    u = (angles - float(np.min(angles))) / angle_range
    v_val = (v[:, 2] - float(np.min(v[:, 2]))) / max(float(np.ptp(v[:, 2])), 1e-6)

    return np.column_stack([u, v_val]).astype(np.float32)


def _unwrap_uv(vertices, faces):
    """LSCM (Least Squares Conformal Maps) UV unwrap via trimesh."""
    try:
        import trimesh

        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        uv = tm.unwrap()
        if uv is None or len(uv) != len(vertices):
            return None

        uv = np.asarray(uv, dtype=np.float64)
        has_nan = not np.all(np.isfinite(uv))
        if has_nan:
            return None

        uv[:, 0] = (uv[:, 0] - float(np.min(uv[:, 0]))) / max(float(np.ptp(uv[:, 0])), 1e-6)
        uv[:, 1] = (uv[:, 1] - float(np.min(uv[:, 1]))) / max(float(np.ptp(uv[:, 1])), 1e-6)
        return np.asarray(uv, dtype=np.float32)
    except Exception:
        return None


def _bake_texture(vertices, faces, vertex_colors, uvs, texture_size, feature_type="unclassified"):
    """Bake per-vertex colours to a PNG texture.

    Uses linear interpolation on the Delaunay triangulation of UV sample
    points, then fills remaining holes with the median colour.
    """
    if vertex_colors is None or uvs is None:
        return None

    faces_arr = np.asarray(faces, dtype=np.int32)
    if len(faces_arr) == 0 or len(faces_arr) > TEXTURE_BAKE_MAX_FACES:
        return None

    import io
    from PIL import Image
    from scipy.interpolate import griddata

    colors = np.asarray(vertex_colors, dtype=np.float32)
    uvs_arr = np.asarray(uvs, dtype=np.float32)

    # Downsample if too many vertices (griddata Delaunay is O(N²) worst-case)
    MAX_BAKE_SAMPLES = int(os.environ.get("OPEN3D_TEXTURE_BAKE_MAX_SAMPLES", "50000"))
    if len(colors) > MAX_BAKE_SAMPLES:
        idx = np.random.RandomState(42).choice(
            len(colors), MAX_BAKE_SAMPLES, replace=False,
        )
        colors = colors[idx]
        uvs_arr = uvs_arr[idx]

    # Sample points at vertex UV positions
    sample_x = uvs_arr[:, 0] * float(texture_size - 1)
    sample_y = uvs_arr[:, 1] * float(texture_size - 1)

    # Build target grid
    grid_x, grid_y = np.meshgrid(
        np.arange(texture_size, dtype=np.float32),
        np.arange(texture_size, dtype=np.float32),
        indexing="ij",
    )

    # Interpolate via linear method — falls back to NaN outside the convex hull
    # Interpolate via linear — produces NaN outside the convex hull.
    # Track coverage; if too sparse we fall back to nearest + blur.
    try:
        r = griddata(
            (sample_x, sample_y), colors[:, 0].astype(np.float64),
            (grid_x, grid_y), method="linear",
        )
        g = griddata(
            (sample_x, sample_y), colors[:, 1].astype(np.float64),
            (grid_x, grid_y), method="linear",
        )
        b = griddata(
            (sample_x, sample_y), colors[:, 2].astype(np.float64),
            (grid_x, grid_y), method="linear",
        )
    except (MemoryError, ValueError, RuntimeError):
        r = g = b = None

    nan_frac = (
        float(np.mean(~np.isfinite(r))) if r is not None else 1.0
    )
    if r is None or nan_frac > 0.3:
        # More than 30% outside convex hull — use nearest + smooth
        r = griddata(
            (sample_x, sample_y), colors[:, 0].astype(np.float64),
            (grid_x, grid_y), method="nearest",
        )
        g = griddata(
            (sample_x, sample_y), colors[:, 1].astype(np.float64),
            (grid_x, grid_y), method="nearest",
        )
        b = griddata(
            (sample_x, sample_y), colors[:, 2].astype(np.float64),
            (grid_x, grid_y), method="nearest",
        )
        # Apply small Gaussian blur to smooth Voronoi cell boundaries
        from scipy.ndimage import gaussian_filter
        r = gaussian_filter(r, sigma=1.5)
        g = gaussian_filter(g, sigma=1.5)
        b = gaussian_filter(b, sigma=1.5)

    # Fill NaN (outside convex hull) with median colour
    texture = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    for ch, arr in enumerate([r, g, b]):
        nan_mask = ~np.isfinite(arr)
        if nan_mask.any():
            median_val = float(np.median(colors[:, ch]))
            arr[nan_mask] = median_val
        texture[:, :, ch] = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    texture[:, :, 3] = 255

    buf = io.BytesIO()
    Image.fromarray(texture, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _dilate_texture(texture, coverage, max_iterations=8):
    """Dilate covered pixels outward to fill gaps in sparse rasterization.

    Uses repeated 3x3 morphological dilation: each uncovered pixel takes
    the median colour of its covered neighbours. Stops early when coverage
    reaches >95% or after max_iterations passes.
    """
    h, w = coverage.shape
    target = 0.95

    for iteration in range(max_iterations):
        covered_frac = float(np.mean(coverage >= 0.01))
        if covered_frac >= target:
            break

        # Build indices of uncovered pixels with covered neighbours
        new_texture = texture.copy()
        new_coverage = coverage.copy()
        changed = 0

        # Process in blocks for performance
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if coverage[y, x] >= 0.01:
                    continue

                # Gather covered neighbours in 3x3 window
                neighbours = []
                for dy in (-1, 0, 1):
                    ny = y + dy
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nx = x + dx
                        if coverage[ny, nx] >= 0.01:
                            neighbours.append(texture[ny, nx, :3])

                if neighbours:
                    arr = np.array(neighbours, dtype=np.float32)
                    new_texture[y, x, :3] = np.median(arr, axis=0).astype(np.uint8)
                    new_coverage[y, x] = 0.5
                    changed += 1

        if changed == 0:
            # No more expandable pixels — fully saturated
            break

        texture[:] = new_texture
        coverage[:] = new_coverage


def _edge_function(a, b, c):
    """Signed edge function for barycentric coordinate computation."""
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])


# (Unused helper kept for reference — _bake_normal_map delegates to _bake_texture)


def _bake_normal_map(vertices, faces, vertex_normals, uvs, texture_size):
    """Bake a tangent-space normal map from mesh geometry and UVs.

    Returns PNG bytes, or None if inputs are insufficient.
    """
    if vertex_normals is None or uvs is None or len(faces) == 0:
        return None

    faces_arr = np.asarray(faces, dtype=np.int32)
    verts = np.asarray(vertices, dtype=np.float64)
    norms = np.asarray(vertex_normals, dtype=np.float64)
    uvs_arr = np.asarray(uvs, dtype=np.float32)

    tangent_space_normals = _compute_tangent_normals(verts, faces_arr, uvs_arr, norms)
    return _bake_texture(
        vertices, faces,
        tangent_space_normals.astype(np.uint8),
        uvs, texture_size, feature_type="normal",
    )


def _compute_tangent_normals(vertices, faces, uvs, normals):
    """Compute per-vertex normals in tangent space from world-space mesh normals and UVs."""
    n_verts = len(vertices)
    tangents = np.zeros((n_verts, 3), dtype=np.float64)
    bitangents = np.zeros((n_verts, 3), dtype=np.float64)

    for tri in faces:
        i0, i1, i2 = tri
        p0, p1, p2 = vertices[[i0, i1, i2]]
        uv0, uv1, uv2 = uvs[[i0, i1, i2]]

        e1 = p1 - p0
        e2 = p2 - p0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0

        det = duv1[0] * duv2[1] - duv2[0] * duv1[1]
        if abs(det) < 1e-12:
            continue
        inv_det = 1.0 / det

        tangent = inv_det * (duv2[1] * e1 - duv1[1] * e2)
        bitangent = inv_det * (-duv2[0] * e1 + duv1[0] * e2)

        for idx in (i0, i1, i2):
            tangents[idx] += tangent
            bitangents[idx] += bitangent

    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.where(tangent_norms > 1e-8, tangents / tangent_norms, 0.0)
    bitangent_norms = np.linalg.norm(bitangents, axis=1, keepdims=True)
    bitangents = np.where(bitangent_norms > 1e-8, bitangents / bitangent_norms, 0.0)

    tan = tangents - normals * np.sum(tangents * normals, axis=1, keepdims=True)
    tan_norms = np.linalg.norm(tan, axis=1, keepdims=True)
    tan = np.where(tan_norms > 1e-8, tan / tan_norms, 0.0)
    bitan = np.cross(normals, tan)

    tn_x = np.sum(tan * normals, axis=1)
    tn_y = np.sum(bitan * normals, axis=1)
    tn_z = np.sum(normals * normals, axis=1)

    tangent_normals = np.stack([tn_x, tn_y, tn_z], axis=1)
    tangent_normals = np.clip(
        np.where(np.isfinite(tangent_normals), tangent_normals, 0.0),
        -1.0, 1.0,
    )
    return (tangent_normals * 0.5 + 0.5) * 255.0


def _compute_mesh_normals(vertices, faces):
    """Compute per-vertex normals from mesh geometry (area-weighted face normals)."""
    if len(faces) == 0:
        return None

    verts = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)
    normals = np.zeros((len(verts), 3), dtype=np.float64)

    for tri in faces_arr:
        i0, i1, i2 = tri
        p0, p1, p2 = verts[[i0, i1, i2]]
        fn = np.cross(p1 - p0, p2 - p0)
        fn_norm = np.linalg.norm(fn)
        if fn_norm < 1e-12:
            continue
        fn /= fn_norm
        area = fn_norm * 0.5
        for idx in (i0, i1, i2):
            normals[idx] += fn * area

    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(normal_norms > 1e-8, normals / normal_norms, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return normals


def _check_texture_override(segment_local_id, feature_type):
    """Check for an externally-supplied texture override for this segment.

    Looks in TEXTURE_OVERRIDE_DIR for files named:
      {local_id}.png, {local_id}_{feature_type}.png
    Returns PNG bytes if found, None otherwise.
    """
    if not TEXTURE_OVERRIDE_DIR or not os.path.isdir(TEXTURE_OVERRIDE_DIR):
        return None

    candidates = [
        os.path.join(TEXTURE_OVERRIDE_DIR, f"{segment_local_id}.png"),
        os.path.join(TEXTURE_OVERRIDE_DIR, f"{segment_local_id}_{feature_type}.png"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "rb") as handle:
                return handle.read()
    return None


def _build_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return point_cloud


def _normalize_for_inference(xyz):
    points = np.asarray(xyz, dtype=np.float64)
    origin = np.array(
        [
            float(np.mean(points[:, 0], dtype=np.float64)),
            float(np.mean(points[:, 1], dtype=np.float64)),
            float(np.min(points[:, 2])),
        ],
        dtype=np.float64,
    )
    return points - origin


def _recover_degenerate_ml_output(xyz, labels, confidence):
    labels = np.asarray(labels, dtype=np.int32)
    confidence = np.asarray(confidence, dtype=np.float32)

    geometric_labels = None
    geometric_confidence = None
    source = "randlanet"

    if _is_degenerate_label_distribution(labels):
        geometric_labels, geometric_confidence = _classify_geometrically(xyz)
        if _is_meaningfully_diverse(geometric_labels):
            return geometric_labels, geometric_confidence, "open3d-hybrid"

    ml_terrain_fraction = float(np.mean(np.isin(labels, (CLASS_GROUND, CLASS_ROAD)), dtype=np.float64))
    if ml_terrain_fraction < GROUND_RECOVERY_FRACTION:
        if geometric_labels is None or geometric_confidence is None:
            geometric_labels, geometric_confidence = _classify_geometrically(xyz)
        terrain_mask = np.isin(geometric_labels, (CLASS_GROUND, CLASS_ROAD))
        if np.mean(terrain_mask, dtype=np.float64) >= GROUND_RECOVERY_FRACTION:
            recovered = labels.copy()
            recovered_confidence = confidence.copy()
            recovered[terrain_mask] = geometric_labels[terrain_mask]
            recovered_confidence[terrain_mask] = np.maximum(
                recovered_confidence[terrain_mask],
                geometric_confidence[terrain_mask],
            )
            return recovered, recovered_confidence, "randlanet+ground-recovery"

    return labels, confidence, source


def _is_degenerate_label_distribution(labels):
    if len(labels) == 0:
        return True

    values, counts = np.unique(np.asarray(labels, dtype=np.int32), return_counts=True)
    dominant_ratio = float(np.max(counts) / len(labels))
    meaningful = values[np.isin(values, (CLASS_GROUND, CLASS_ROAD, CLASS_BUILDING, CLASS_HIGH_VEG, CLASS_MED_VEG, CLASS_LOW_VEG, CLASS_WIRE, CLASS_TOWER, CLASS_WATER))]
    return dominant_ratio >= DEGENERATE_DOMINANT_LABEL_THRESHOLD or len(meaningful) <= 1


def _is_meaningfully_diverse(labels):
    if len(labels) == 0:
        return False

    values = np.unique(np.asarray(labels, dtype=np.int32))
    meaningful = values[np.isin(values, (CLASS_GROUND, CLASS_ROAD, CLASS_BUILDING, CLASS_HIGH_VEG, CLASS_MED_VEG, CLASS_LOW_VEG, CLASS_WIRE, CLASS_TOWER, CLASS_WATER))]
    if len(meaningful) >= 2:
        return True

    terrain_fraction = np.mean(np.isin(labels, (CLASS_GROUND, CLASS_ROAD)), dtype=np.float64)
    return terrain_fraction >= GROUND_RECOVERY_FRACTION


def _classify_geometrically(xyz):
    labels = np.full(len(xyz), 1, dtype=np.int32)
    confidence = np.full(len(xyz), 0.45, dtype=np.float32)

    if len(xyz) < 10:
        return labels, confidence

    normals = _estimate_normals(xyz)
    ground_mask, plane_model = _detect_ground(xyz, normals=normals)

    if ground_mask.any():
        labels[ground_mask] = CLASS_GROUND
        confidence[ground_mask] = CLASS_CONFIDENCE[CLASS_GROUND]

    non_ground_indices = np.where(~ground_mask)[0]
    if len(non_ground_indices) == 0:
        return labels, confidence

    non_ground = xyz[non_ground_indices]
    cluster_labels = _cluster_points(non_ground, eps=max(_average_spacing(non_ground) * 8.0, 0.8), min_points=25)

    handled = np.zeros(len(non_ground_indices), dtype=bool)
    for cluster_id in sorted(int(value) for value in np.unique(cluster_labels) if int(value) >= 0):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = non_ground_indices[cluster_mask]
        if len(cluster_indices) < 12:
            continue

        cluster_points = xyz[cluster_indices]
        cluster_normals = normals[cluster_indices]
        class_id, class_confidence = _classify_cluster(cluster_points, cluster_normals, plane_model)
        labels[cluster_indices] = class_id
        confidence[cluster_indices] = class_confidence
        handled[np.where(cluster_mask)[0]] = True

    noise_indices = non_ground_indices[~handled]
    if len(noise_indices):
        noise_points = xyz[noise_indices]
        noise_height = float(np.ptp(noise_points[:, 2])) if len(noise_points) else 0.0
        if noise_height > 1.0:
            labels[noise_indices] = CLASS_LOW_VEG
            confidence[noise_indices] = CLASS_CONFIDENCE[CLASS_LOW_VEG] * 0.8
        else:
            labels[noise_indices] = CLASS_NOISE
            confidence[noise_indices] = CLASS_CONFIDENCE[CLASS_NOISE]

    return labels, confidence


def _detect_ground(xyz, normals=None):
    if len(xyz) < 50:
        return np.zeros(len(xyz), dtype=bool), None

    point_cloud = _build_point_cloud(xyz)
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=GROUND_DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=GROUND_RANSAC_ITERATIONS,
    )
    if len(inliers) < max(int(len(xyz) * GROUND_MIN_FRACTION), 25):
        return _detect_ground_by_height(xyz, normals=normals)

    normal = np.asarray(plane_model[:3], dtype=np.float64)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0.0:
        return _detect_ground_by_height(xyz, normals=normals)

    normal /= normal_norm
    if abs(normal[2]) < 0.7:
        return _detect_ground_by_height(xyz, normals=normals)

    distances = np.abs(xyz @ normal + float(plane_model[3]))
    return distances <= (GROUND_DISTANCE_THRESHOLD * 1.5), tuple(float(value) for value in plane_model)


def _detect_ground_by_height(xyz, normals=None):
    if len(xyz) < 50:
        return np.zeros(len(xyz), dtype=bool), None

    reference_z = float(np.quantile(xyz[:, 2], GROUND_FALLBACK_PERCENTILE))
    mask = xyz[:, 2] <= (reference_z + GROUND_FALLBACK_HEIGHT)

    if normals is not None and len(normals) == len(xyz):
        horizontal_mask = np.abs(normals[:, 2]) >= 0.65
        candidate_mask = mask & horizontal_mask
        if np.count_nonzero(candidate_mask) >= max(int(len(xyz) * GROUND_MIN_FRACTION), 25):
            mask = candidate_mask

    if np.count_nonzero(mask) < max(int(len(xyz) * GROUND_MIN_FRACTION), 25):
        return np.zeros(len(xyz), dtype=bool), None

    ground_z = float(np.median(xyz[mask, 2], dtype=np.float64))
    return mask, (0.0, 0.0, 1.0, -ground_z)


def _estimate_normals(xyz):
    point_cloud = _build_point_cloud(xyz)
    radius = max(_average_spacing(xyz) * 6.0, 0.5)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=48)
    )
    return np.asarray(point_cloud.normals, dtype=np.float64)


def _cluster_points(points, eps, min_points):
    """Cluster points using GPU grid-hashing when available, falling back to Open3D CPU DBSCAN.

    On GPU: PyTorch voxel-hashing + union-find connected components.
    On CPU: Open3D DBSCAN (identical semantics).
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)

    device = get_inference_device()
    if device == "cuda" and len(points) >= 512:
        try:
            return _cluster_points_gpu(points, eps, min_points, device)
        except Exception:
            pass

    return _cluster_points_o3d(points, eps, min_points)


def _cluster_points_o3d(points, eps, min_points):
    """Open3D CPU DBSCAN clustering (legacy fallback)."""
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)

    point_cloud = _build_point_cloud(points)
    return np.asarray(
        point_cloud.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False),
        dtype=np.int32,
    )


def _cluster_points_gpu(points, eps, min_points, device="cuda"):
    """GPU-accelerated grid-based clustering using PyTorch.

    Algorithm:
      1. Voxelize points into a regular grid with cell size = eps.
      2. Build an adjacency graph between occupied voxels (26-connectivity).
      3. Run union-find connected components on the voxel graph.
      4. Map voxel components back to per-point cluster labels.
      5. Drop clusters smaller than min_points.

    This is functionally equivalent to DBSCAN with the given eps but
    runs the heavy voxelization and neighbor search on GPU tensors.
    """
    import torch

    pts = torch.as_tensor(np.asarray(points, dtype=np.float32), device=device)
    n_points = len(pts)

    # ── 1. Voxelize ──────────────────────────────────────────────
    origin = pts.min(dim=0).values
    voxel_indices = ((pts - origin) / float(eps)).to(torch.int64)

    unique_voxels, inverse = torch.unique(voxel_indices, dim=0, return_inverse=True, sorted=False)
    n_voxels = len(unique_voxels)

    if n_voxels <= 1:
        labels = torch.full((n_points,), -1, dtype=torch.int32, device=device)
        if n_voxels == 1 and n_points >= min_points:
            labels[:] = 0
        return labels.cpu().numpy()

    # ── 2. Build voxel adjacency via hash map (CPU union-find) ───
    voxel_to_idx: dict[tuple[int, ...], int] = {}
    unique_voxels_cpu = unique_voxels.cpu().numpy()
    for i in range(n_voxels):
        voxel_to_idx[tuple(unique_voxels_cpu[i].tolist())] = i

    parent = list(range(n_voxels))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    # 13-directional half-neighborhood (avoids double-processing)
    _half_offsets = [
        (-1, 0, 0), (0, -1, 0), (0, 0, -1),
        (-1, -1, 0), (-1, 0, -1), (0, -1, -1),
        (-1, -1, -1), (-1, 1, 0), (-1, 0, 1),
        (0, -1, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1),
    ]

    for i in range(n_voxels):
        vx, vy, vz = unique_voxels_cpu[i]
        for dx, dy, dz in _half_offsets:
            neighbor = (vx + dx, vy + dy, vz + dz)
            j = voxel_to_idx.get(neighbor)
            if j is not None:
                _union(i, j)

    # ── 3. Assign component IDs to voxels ────────────────────────
    component_ids: dict[int, int] = {}
    component_counter = 0
    voxel_components = np.full(n_voxels, -1, dtype=np.int32)
    for i in range(n_voxels):
        root = _find(i)
        if root not in component_ids:
            component_ids[root] = component_counter
            component_counter += 1
        voxel_components[i] = component_ids[root]

    # ── 4. Map back to per-point labels ──────────────────────────
    inverse_cpu = inverse.cpu().numpy()
    labels = np.full(n_points, -1, dtype=np.int32)

    # Filter by min_points
    voxel_sizes = np.bincount(inverse_cpu, minlength=n_voxels)
    for i in range(n_voxels):
        if voxel_sizes[i] >= min_points:
            labels[inverse_cpu == i] = int(voxel_components[i])

    return labels


def _segment_class_points(points, class_id):
    if len(points) == 0:
        return []

    eps, min_points = _cluster_settings(points, class_id)
    cluster_labels = _cluster_points(points, eps=eps, min_points=min_points)

    clusters = [
        np.where(cluster_labels == cluster_id_value)[0]
        for cluster_id_value in sorted(int(value) for value in np.unique(cluster_labels) if int(value) >= 0)
    ]
    noise_indices = np.where(cluster_labels < 0)[0]
    clusters.extend(_recover_residual_clusters(points, noise_indices, class_id, eps, min_points))

    if not clusters and len(points) >= _minimum_segment_points(class_id):
        return [np.arange(len(points), dtype=np.int64)]

    return clusters


def _recover_residual_clusters(points, noise_indices, class_id, eps, min_points):
    if len(noise_indices) < _minimum_segment_points(class_id):
        return []

    if class_id in (CLASS_GROUND, CLASS_ROAD):
        return [noise_indices.astype(np.int64)]

    relaxed_min_points = max(_minimum_segment_points(class_id), max(min_points // 2, 8))
    relaxed_labels = _cluster_points(points[noise_indices], eps=eps * 1.35, min_points=relaxed_min_points)
    recovered = [
        noise_indices[np.where(relaxed_labels == cluster_id_value)[0]].astype(np.int64)
        for cluster_id_value in sorted(int(value) for value in np.unique(relaxed_labels) if int(value) >= 0)
    ]

    remaining = noise_indices[np.where(relaxed_labels < 0)[0]]
    if len(remaining) < max(_minimum_segment_points(class_id) * 3, int(len(points) * 0.2)):
        return recovered

    remaining_points = points[remaining]
    extent = np.ptp(remaining_points, axis=0)
    if max(float(extent[0]), float(extent[1])) <= max(eps * 20.0, 25.0):
        recovered.append(remaining.astype(np.int64))

    return recovered


def _cluster_settings(points, class_id):
    base_eps, min_points = CLASS_CLUSTER_SETTINGS.get(class_id, (1.0, 20))
    spacing = _average_spacing(points)
    return max(base_eps, spacing * 6.0), min_points


def _minimum_segment_points(class_id):
    if class_id in (CLASS_GROUND, CLASS_ROAD):
        return 30
    if class_id in (CLASS_WIRE, CLASS_TOWER):
        return 10
    return 15


def _average_spacing(points):
    """Estimate average point spacing using GPU-accelerated distance computation.

    On GPU: samples up to 10k points, computes pairwise distances via torch.cdist,
    then takes median of nearest-neighbor distances.
    On CPU: falls back to Open3D's compute_nearest_neighbor_distance.
    """
    if len(points) < 2:
        return 0.1

    device = get_inference_device()
    if device == "cuda" and len(points) >= 64:
        try:
            return _average_spacing_gpu(points)
        except Exception:
            pass

    point_cloud = _build_point_cloud(points)
    distances = np.asarray(point_cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return 0.1

    return max(float(np.median(finite)), 0.05)


def _average_spacing_gpu(points):
    """GPU-accelerated nearest-neighbor distance median via torch.cdist.

    Samples up to 10000 points to keep the N×N distance matrix tractable.
    """
    import torch

    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)

    # Downsample if needed to keep the distance matrix ≤ 100M entries
    if n > 10000:
        step = max(n // 10000, 1)
        pts = pts[::step]
        n = len(pts)

    t = torch.as_tensor(pts, device="cuda")
    # N×N pairwise distances
    dists = torch.cdist(t, t, p=2.0)

    # Mask self-distances (diagonal) so they don't become the nearest neighbor
    diag = torch.arange(n, device="cuda")
    dists[diag, diag] = float("inf")

    nearest = dists.min(dim=1).values
    finite = nearest[torch.isfinite(nearest)]
    if len(finite) == 0:
        return 0.1

    median_val = float(torch.median(finite).item())
    return max(median_val, 0.05)


def _classify_cluster(points, normals, plane_model):
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    size = bbox_max - bbox_min
    width, length = sorted(size[:2])
    height = float(size[2])
    footprint = float(size[0] * size[1])
    aspect_ratio = float(width / max(length, 1e-6))

    covariance = np.cov((points - points.mean(axis=0)).T)
    eigenvalues = np.sort(np.clip(np.linalg.eigvalsh(covariance), 1e-9, None))[::-1]
    linearity = float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])
    planarity = float((eigenvalues[1] - eigenvalues[2]) / eigenvalues[0])
    scattering = float(eigenvalues[2] / eigenvalues[0])
    horizontal_normals = float(np.mean(np.abs(normals[:, 2]))) if len(normals) else 0.0

    local_ground_z = bbox_min[2]
    if plane_model is not None and abs(plane_model[2]) > 1e-6:
        cx = float(np.mean(points[:, 0]))
        cy = float(np.mean(points[:, 1]))
        local_ground_z = float(-(plane_model[0] * cx + plane_model[1] * cy + plane_model[3]) / plane_model[2])

    height_above_ground = float(bbox_min[2] - local_ground_z)
    upper_height_above_ground = float(np.quantile(points[:, 2], 0.9) - local_ground_z)
    flat_surface = planarity >= 0.25 and footprint >= 6.0
    very_flat_surface = height <= 0.45 and horizontal_normals >= 0.85 and scattering <= 0.05
    elongated_surface = length >= 6.0 and aspect_ratio <= 0.45

    if height_above_ground <= 0.35 and flat_surface:
        if very_flat_surface and elongated_surface:
            return CLASS_ROAD, CLASS_CONFIDENCE[CLASS_ROAD]
        return CLASS_GROUND, CLASS_CONFIDENCE[CLASS_GROUND]

    if (
        upper_height_above_ground >= 1.5
        and planarity >= 0.35
        and footprint >= 1.5
        and (footprint <= 180.0 or height >= 3.5)
    ):
        return CLASS_BUILDING, CLASS_CONFIDENCE[CLASS_BUILDING]

    if height >= 5.0 and linearity >= 0.82 and max(width, length) <= 1.25:
        return CLASS_WIRE, CLASS_CONFIDENCE[CLASS_WIRE]

    if height >= 4.0 and max(width, length) <= 2.0 and planarity < 0.25:
        return CLASS_TOWER, CLASS_CONFIDENCE[CLASS_TOWER]

    if height >= 3.5 or upper_height_above_ground >= 3.0:
        return CLASS_HIGH_VEG, CLASS_CONFIDENCE[CLASS_HIGH_VEG]

    if height >= 1.5 or upper_height_above_ground >= 1.2:
        return CLASS_MED_VEG, CLASS_CONFIDENCE[CLASS_MED_VEG]

    if height > 0.25 or horizontal_normals < 0.8 or scattering > 0.08:
        return CLASS_LOW_VEG, CLASS_CONFIDENCE[CLASS_LOW_VEG]

    return CLASS_NOISE, CLASS_CONFIDENCE[CLASS_NOISE]


def _mesh_methods_for_feature(feature_type):
    return FEATURE_METHODS.get(feature_type, FEATURE_METHODS["unclassified"])


def _should_reconstruct_segment_mesh(segment):
    feature_type = segment.get("featureType")
    return isinstance(feature_type, str) and feature_type.lower() in MESHABLE_FEATURES


def _is_meshable_class_id(class_id):
    feature_type = FEATURE_NAMES.get(int(class_id))
    return isinstance(feature_type, str) and feature_type.lower() in MESHABLE_FEATURES


def _should_short_circuit_segmentation(class_id, point_count):
    if SEGMENT_CLUSTER_POINT_CAP <= 0 or point_count <= SEGMENT_CLUSTER_POINT_CAP:
        return False

    return int(class_id) not in (CLASS_BUILDING, CLASS_WIRE, CLASS_TOWER)


def _mesh_methods_for_segment(segment):
    feature_type = segment.get("featureType")
    methods = _mesh_methods_for_feature(feature_type)
    if not string_equals(feature_type, "building"):
        return methods

    bounds_min = np.asarray(segment.get("boundsMin", []), dtype=np.float64)
    bounds_max = np.asarray(segment.get("boundsMax", []), dtype=np.float64)
    if len(bounds_min) < 3 or len(bounds_max) < 3:
        return methods

    extent = bounds_max - bounds_min
    width, length = sorted(extent[:2])
    height = float(extent[2])
    aspect_ratio = float(width / max(length, 1e-6))
    if height <= 2.5 or aspect_ratio <= 0.2:
        return ("alpha_shape", "ball_pivoting", "poisson")

    return methods


def string_equals(left, right):
    return isinstance(left, str) and left.lower() == right.lower()


def _is_valid_mesh(vertices, faces):
    if vertices is None or faces is None:
        return False

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices.ndim == 2 and faces.ndim == 2 and len(vertices) >= 3 and len(faces) >= 1
