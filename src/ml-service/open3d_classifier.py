"""Open3D/Open3D-ML helpers for semantic inference and reconstruction."""
import os
import shutil
import threading
import urllib.request
import warnings

import numpy as np

# Suppress Open3D C++ warnings (invalid tetra, etc.) — these are normal
# for sparse/degenerate point sets and don't affect the reconstruction quality.
os.environ.setdefault("OPEN3D_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="open3d")

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    o3d = None
    _HAS_OPEN3D = False

_OPEN3D_ML_AVAILABLE = False
try:
    import open3d.ml.torch as _ml3d  # noqa: F401
    from open3d.ml.torch.models import RandLANet
    from open3d.ml.torch.pipelines import SemanticSegmentation

    _OPEN3D_ML_AVAILABLE = True
except ImportError:
    RandLANet = None
    SemanticSegmentation = None

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy.spatial import Delaunay
except ImportError:
    Delaunay = None


DEFAULT_MODEL_PATH = os.environ.get(
    "OPEN3D_ML_MODEL_PATH",
    "/app/models/randlanet_semantickitti_202201071330utc.pth",
)
DEFAULT_MODEL_URL = os.environ.get(
    "OPEN3D_ML_MODEL_URL",
    "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth",
)
_AUTO_DOWNLOAD_MODEL = os.environ.get("OPEN3D_ML_MODEL_AUTO_DOWNLOAD", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_INCLUDE_COLOR_FEATURES = os.environ.get("OPEN3D_ML_INCLUDE_COLOR_FEATURES", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_RANDLANET_LOCK = threading.Lock()
_RANDLANET_PIPELINE = None
_RANDLANET_PIPELINE_PATH = None
_RANDLANET_PIPELINE_DEVICE = None
MESH_BOUNDS_PADDING_XY = float(os.environ.get("OPEN3D_MESH_BOUNDS_PADDING_XY", "0.6"))
MESH_BOUNDS_PADDING_Z = float(os.environ.get("OPEN3D_MESH_BOUNDS_PADDING_Z", "0.5"))


# ── ASPRS ↔ Open3D-ML label mapping ──────────────────────────────
# RandLANet was trained on SemanticKITTI (19 classes)
# We map SemanticKITTI labels → ASPRS labels
SEMKITTI_TO_ASPRS = {
    0:  7,   # car/moving → noise (for now)
    1:  7,   # bicycle → noise
    2:  7,   # motorcycle → noise
    3:  7,   # truck → noise
    4:  7,   # other-vehicle → noise
    5:  7,   # person → noise
    6:  7,   # bicyclist → noise
    7:  7,   # motorcyclist → noise
    8:  11,  # road → road
    9:  11,  # parking → road
    10: 11,  # sidewalk → road
    11: 2,   # other-ground → ground
    12: 6,   # building → building
    13: 7,   # fence → noise
    14: 3,   # vegetation → low vegetation
    15: 5,   # trunk → high vegetation
    16: 2,   # terrain → ground
    17: 7,   # pole → noise
    18: 7,   # traffic-sign → noise
    19: 14,  # wire → wire
}


def has_open3d_ml():
    """Check if Open3D-ML is available."""
    return _OPEN3D_ML_AVAILABLE


def get_model_path():
    return DEFAULT_MODEL_PATH


def get_model_url():
    return DEFAULT_MODEL_URL


def get_inference_device():
    """Auto-detect the best available inference device.

    Returns 'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'.
    No environment variable gating — GPU is used whenever present.
    """
    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def ensure_model_checkpoint():
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH

    if not _AUTO_DOWNLOAD_MODEL:
        raise FileNotFoundError(f"Model not found at {DEFAULT_MODEL_PATH}")

    model_dir = os.path.dirname(DEFAULT_MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    with _RANDLANET_LOCK:
        if os.path.exists(DEFAULT_MODEL_PATH):
            return DEFAULT_MODEL_PATH

        download_path = f"{DEFAULT_MODEL_PATH}.download"
        try:
            with urllib.request.urlopen(DEFAULT_MODEL_URL) as response, open(download_path, "wb") as handle:
                shutil.copyfileobj(response, handle)

            os.replace(download_path, DEFAULT_MODEL_PATH)
        finally:
            if os.path.exists(download_path):
                os.remove(download_path)

    return DEFAULT_MODEL_PATH


def get_randlanet_pipeline():
    if not _OPEN3D_ML_AVAILABLE:
        raise RuntimeError("Open3D-ML not available")

    ckpt_path = ensure_model_checkpoint()
    device = get_inference_device()

    global _RANDLANET_PIPELINE
    global _RANDLANET_PIPELINE_DEVICE
    global _RANDLANET_PIPELINE_PATH

    with _RANDLANET_LOCK:
        if (
            _RANDLANET_PIPELINE is not None
            and _RANDLANET_PIPELINE_PATH == ckpt_path
            and _RANDLANET_PIPELINE_DEVICE == device
        ):
            return _RANDLANET_PIPELINE

        model = RandLANet(ckpt_path=ckpt_path)
        pipeline = SemanticSegmentation(model, dataset=None, device=device)
        pipeline.load_ckpt(ckpt_path=ckpt_path)

        _RANDLANET_PIPELINE = pipeline
        _RANDLANET_PIPELINE_PATH = ckpt_path
        _RANDLANET_PIPELINE_DEVICE = device
        return _RANDLANET_PIPELINE


def classify_with_randlanet(points, colors=None):
    """
    Classify a point cloud using RandLANet (Open3D-ML).
    
    Args:
        points: Nx3 numpy array (xyz)
        colors: Nx3 numpy array (rgb) optional
    
    Returns:
        labels: N array of ASPRS classification labels
        confidence: N array of confidence scores
    """
    if not _OPEN3D_ML_AVAILABLE:
        raise RuntimeError("Open3D-ML not available")

    pipeline = get_randlanet_pipeline()
    data = {
        "name": "input",
        "point": points.astype(np.float32),
        "label": np.zeros(len(points), dtype=np.int32),
    }
    if _INCLUDE_COLOR_FEATURES and colors is not None:
        max_color = 65535.0 if np.max(colors) > 255 else 255.0
        data["feat"] = colors.astype(np.float32) / max_color

    result = pipeline.run_inference(data)

    pred_labels = result.get("predict_labels", np.zeros(len(points), dtype=np.int32))
    asprs_labels = np.array([SEMKITTI_TO_ASPRS.get(int(l), 1) for l in pred_labels], dtype=np.int32)

    confidence = result.get("predict_scores")
    if confidence is None:
        confidence = np.full(len(points), 0.75, dtype=np.float32)
    else:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.ndim > 1:
            confidence = np.max(confidence, axis=1)

    return asprs_labels, np.asarray(confidence, dtype=np.float32)


def _reconstruct_projected_surface(points):
    if Delaunay is None or len(points) < 3:
        return None, None

    points = np.asarray(points, dtype=np.float64)
    xy = points[:, :2]
    xy_extent = np.ptp(xy, axis=0)
    if np.any(xy_extent <= 1e-6):
        return None, None

    normalized_xy = (xy - np.min(xy, axis=0)) / np.maximum(xy_extent, 1e-6)

    try:
        triangulation = Delaunay(normalized_xy)
    except Exception:
        return None, None

    faces = np.asarray(triangulation.simplices, dtype=np.int32)
    if len(faces) == 0:
        return None, None

    triangle_points = points[faces]
    edge_lengths_xy = np.stack(
        [
            np.linalg.norm(triangle_points[:, 0, :2] - triangle_points[:, 1, :2], axis=1),
            np.linalg.norm(triangle_points[:, 1, :2] - triangle_points[:, 2, :2], axis=1),
            np.linalg.norm(triangle_points[:, 2, :2] - triangle_points[:, 0, :2], axis=1),
        ],
        axis=1,
    )
    max_xy_edge = np.max(edge_lengths_xy, axis=1)
    triangle_height = np.ptp(triangle_points[:, :, 2], axis=1)

    allowed_edge = max(float(np.quantile(max_xy_edge, 0.8)) * 1.75, 1.5)
    allowed_height = max(float(np.quantile(triangle_height, 0.9)) * 2.0, 0.75)
    keep_mask = (max_xy_edge <= allowed_edge) & (triangle_height <= allowed_height)
    faces = faces[keep_mask]
    if len(faces) == 0:
        return None, None

    used_vertices = np.unique(faces)
    remap = np.full(len(points), -1, dtype=np.int32)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)
    vertices = points[used_vertices]
    remapped_faces = remap[faces]
    return vertices, remapped_faces


def _localize_points(points):
    points = np.asarray(points, dtype=np.float64)
    origin = np.mean(points, axis=0, dtype=np.float64)
    return points - origin, origin


def _trim_mesh_to_source_bounds(vertices, faces, source_points):
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        return None, None

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    source_points = np.asarray(source_points, dtype=np.float64)

    xy_padding = max(MESH_BOUNDS_PADDING_XY, 0.05)
    z_padding = max(MESH_BOUNDS_PADDING_Z, 0.05)

    source_min = np.min(source_points, axis=0)
    source_max = np.max(source_points, axis=0)
    source_min = np.array([source_min[0] - xy_padding, source_min[1] - xy_padding, source_min[2] - z_padding], dtype=np.float64)
    source_max = np.array([source_max[0] + xy_padding, source_max[1] + xy_padding, source_max[2] + z_padding], dtype=np.float64)

    valid_vertices = np.all((vertices >= source_min) & (vertices <= source_max), axis=1)
    if valid_vertices.all():
        return vertices, faces

    valid_faces = np.all(valid_vertices[faces], axis=1)
    faces = faces[valid_faces]
    if len(faces) == 0:
        return None, None

    used_vertices = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)
    vertices = vertices[used_vertices]
    faces = remap[faces]

    if len(vertices) < 3 or len(faces) == 0:
        return None, None

    return vertices, faces


def reconstruct_with_open3d(points, method="poisson"):
    """
    Surface reconstruction using Open3D.
    
    Methods:
        - "poisson": Poisson surface reconstruction (best quality)
        - "ball_pivoting": Ball pivoting algorithm
        - "alpha_shape": Alpha shapes
        - "projected_surface": 2.5D triangulation in XY for flat terrain/hardscape
    
    Returns:
        vertices: Nx3 array
        faces: Mx3 array (triangle indices)
    """
    if not _HAS_OPEN3D:
        raise RuntimeError("Open3D not available")

    if len(points) < 3:
        return None, None

    source_points = np.asarray(points, dtype=np.float64)
    local_points, origin = _localize_points(source_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(local_points)

    neighbor_distances = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
    finite_distances = neighbor_distances[np.isfinite(neighbor_distances)]
    mean_distance = max(float(np.median(finite_distances)) if len(finite_distances) else 0.1, 0.05)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(mean_distance * 6.0, 0.5), max_nn=48)
    )
    pcd.orient_normals_consistent_tangent_plane(24)

    if method == "projected_surface":
        vertices, faces = _reconstruct_projected_surface(local_points)
        if vertices is None or faces is None or len(faces) == 0:
            return None, None
        vertices = np.asarray(vertices, dtype=np.float64) + origin
        return _trim_mesh_to_source_bounds(vertices, np.asarray(faces, dtype=np.int32), source_points)

    if method == "poisson":
        # Depth 11-12 for finer detail (was 9-10)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=12 if len(points) < 50000 else 11,
            width=0,
            scale=1.1,
            linear_fit=False,
        )
        density_array = np.asarray(densities, dtype=np.float64)
        mesh.remove_vertices_by_mask(density_array < np.quantile(density_array, 0.02))
    elif method == "ball_pivoting":
        radii = [mean_distance * factor for factor in (1.5, 3.0, 6.0)]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii),
        )
    else:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            alpha=max(mean_distance * 8.0, 0.5),
        )

    if len(mesh.triangles) == 0:
        return None, None

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Only decimate very large meshes (>50k tris); skip for buildings to preserve detail
    TRI_DECIMATE_CAP = int(os.environ.get("OPEN3D_TRI_DECIMATE_CAP", "50000"))
    if len(mesh.triangles) > TRI_DECIMATE_CAP:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=TRI_DECIMATE_CAP,
        )

    mesh.remove_unreferenced_vertices()
    try:
        mesh.orient_triangles()
    except RuntimeError:
        pass

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vertices = vertices + origin

    # Only trim projected-surface meshes to bounds; skip Poisson/BPA/alpha
    # meshes so legitimate extrapolation around edges is preserved.
    if method == "projected_surface":
        return _trim_mesh_to_source_bounds(vertices, faces, source_points)
    return vertices, faces
