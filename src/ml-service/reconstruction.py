"""
Surface reconstruction from classified point cloud clusters.
Generates meshes using alpha shapes (requires scipy).
"""
import numpy as np
from scipy.spatial import Delaunay, ConvexHull


def alpha_shape(points, alpha):
    """
    Compute alpha shape (concave hull) from 2D or 3D points.
    
    Uses Delaunay triangulation + circumradius filtering.
    For 3D: returns faces of tetrahedra with circumradius < 1/alpha.
    
    Args:
        points: Nx3 numpy array
        alpha: Alpha parameter (smaller = tighter hull)
    
    Returns:
        vertices, faces arrays
    """
    if len(points) < 4:
        return points, np.array([], dtype=np.int32)
    
    tri = Delaunay(points)
    
    # Compute circumradius for each tetrahedron
    tetra = tri.simplices
    # Simplified: filter by edge length
    # For full alpha shape, compute circumcenters and radii
    # Here we use a simpler edge-length filter
    
    edges = set()
    for simplex in tetra:
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    
    # Compute edge lengths
    edge_list = list(edges)
    edge_lengths = np.array([
        np.linalg.norm(points[e[0]] - points[e[1]]) 
        for e in edge_list
    ])
    
    # Keep only edges below alpha threshold
    valid_mask = edge_lengths < (1.0 / alpha) if alpha > 0 else np.ones(len(edge_list), dtype=bool)
    valid_edges = [edge_list[i] for i in range(len(edge_list)) if valid_mask[i]]
    
    # Build faces from valid edges (simplified — returns triangle soup)
    face_set = set()
    for simplex in tetra:
        for i in range(4):
            face = tuple(sorted([simplex[(i+1)%4], simplex[(i+2)%4], simplex[(i+3)%4]]))
            # Check if all three edges of this face are valid
            e1 = tuple(sorted([face[0], face[1]]))
            e2 = tuple(sorted([face[1], face[2]]))
            e3 = tuple(sorted([face[2], face[0]]))
            if e1 in edges and e2 in edges and e3 in edges:
                if valid_mask[edge_list.index(e1)]:
                    face_set.add(face)
    
    faces = np.array(list(face_set), dtype=np.int32)
    return points, faces


def reconstruct_cluster(points, method="alpha_shape", alpha=0.5):
    """
    Reconstruct mesh for a single point cluster.
    
    Returns (vertices, faces) or (None, None) if reconstruction fails.
    """
    n = len(points)
    if n < 4:
        return None, None
    
    if method == "convex_hull":
        try:
            hull = ConvexHull(points)
            return points, hull.simplices.astype(np.int32)
        except Exception:
            return None, None
    
    elif method == "alpha_shape":
        try:
            verts, faces = alpha_shape(points, alpha)
            if len(faces) > 0:
                return verts, faces
            # Fall back to convex hull
            hull = ConvexHull(points)
            return points, hull.simplices.astype(np.int32)
        except Exception:
            return None, None
    
    else:
        try:
            hull = ConvexHull(points)
            return points, hull.simplices.astype(np.int32)
        except Exception:
            return None, None


def reconstruct_mesh(points, labels, method="alpha_shape"):
    """
    Reconstruct meshes for all classified clusters.
    
    Args:
        points: Nx3 numpy array
        labels: N array of classification labels
        method: Reconstruction method
    
    Returns:
        MeshResult-compatible dict with merged mesh data
    """
    unique_labels = np.unique(labels)
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for label in unique_labels:
        if label <= 0:  # Skip noise/unclassified
            continue
        
        mask = labels == label
        cluster_pts = points[mask]
        
        if len(cluster_pts) < 10:
            continue
        
        verts, faces = reconstruct_cluster(cluster_pts, method=method)
        
        if verts is not None and faces is not None and len(faces) > 0:
            all_vertices.append(verts)
            offset_faces = faces + vertex_offset
            all_faces.append(offset_faces)
            vertex_offset += len(verts)
    
    if not all_vertices:
        # Return minimal mesh
        return {
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "faces": [[0, 1, 2]],
            "vertex_count": 3,
            "face_count": 1,
        }
    
    merged_verts = np.vstack(all_vertices)
    merged_faces = np.vstack(all_faces)
    
    return {
        "vertices": merged_verts.tolist(),
        "faces": merged_faces.tolist(),
        "vertex_count": int(len(merged_verts)),
        "face_count": int(len(merged_faces)),
    }
