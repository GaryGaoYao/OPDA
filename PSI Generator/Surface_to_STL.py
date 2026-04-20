import argparse
from pathlib import Path

import numpy as np
import trimesh


EPS = 1e-12


def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        return v * 0.0
    return v / n


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Basic cleanup compatible with newer trimesh versions.
    """
    mesh.remove_unreferenced_vertices()

    nondeg = mesh.nondegenerate_faces()
    if nondeg is not None and len(nondeg) == len(mesh.faces):
        mesh.update_faces(nondeg)

    unique = mesh.unique_faces()
    if unique is not None and len(unique) == len(mesh.faces):
        mesh.update_faces(unique)

    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def load_single_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a single mesh. If the file contains a Scene, concatenate geometries.
    """
    mesh = trimesh.load_mesh(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("The input scene contains no geometry.")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded object is not a valid mesh.")

    return mesh


def get_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Boundary edges are edges used by only one face.
    """
    unique_edges = mesh.edges_unique
    inverse = mesh.edges_unique_inverse
    counts = np.bincount(inverse, minlength=len(unique_edges))
    boundary_edges = unique_edges[counts == 1]
    return boundary_edges


def ordered_boundary_loop(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Order vertices of one simple boundary loop.
    Assumes there is one main boundary loop.
    """
    boundary_edges = get_boundary_edges(mesh)

    if len(boundary_edges) == 0:
        raise ValueError("No open boundary found. The mesh may already be closed.")

    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    bad_vertices = [k for k, v in adj.items() if len(v) != 2]
    if bad_vertices:
        raise RuntimeError(
            "Boundary is not a simple single loop. "
            "This mesh may contain multiple holes, branches, or cracks."
        )

    start = boundary_edges[0][0]
    loop = [start]
    prev = None
    current = start

    while True:
        neighbors = adj[current]

        if prev is None:
            nxt = neighbors[0]
        else:
            candidates = [n for n in neighbors if n != prev]
            if not candidates:
                break
            nxt = candidates[0]

        if nxt == start:
            break

        loop.append(nxt)
        prev, current = current, nxt

        if len(loop) > len(adj) + 5:
            raise RuntimeError("Boundary loop ordering failed.")

    return np.array(loop, dtype=np.int64)


def bridge_loops(loop_a: np.ndarray, loop_b: np.ndarray) -> np.ndarray:
    """
    Create triangles bridging two ordered loops of equal length.
    """
    if len(loop_a) != len(loop_b):
        raise ValueError("Loop sizes do not match.")

    faces = []
    m = len(loop_a)
    for i in range(m):
        a0 = loop_a[i]
        a1 = loop_a[(i + 1) % m]
        b0 = loop_b[i]
        b1 = loop_b[(i + 1) % m]

        faces.append([a0, a1, b1])
        faces.append([a0, b1, b0])

    return np.asarray(faces, dtype=np.int64)


def compute_round_bins(
    v0: np.ndarray,
    v1: np.ndarray,
    loop: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each boundary vertex, compute:
    - center of arc
    - radius
    - start direction u
    - side direction b
    The rounded transition is a semicircle from original boundary to offset boundary.
    """
    centers = []
    radii = []
    u_dirs = []
    b_dirs = []

    for i, idx in enumerate(loop):
        prev_idx = loop[(i - 1) % len(loop)]
        next_idx = loop[(i + 1) % len(loop)]

        p0 = v0[idx]
        p1 = v1[idx]

        # Boundary tangent from neighboring boundary vertices
        tangent = unit_vector(v0[next_idx] - v0[prev_idx])
        if np.linalg.norm(tangent) < EPS:
            tangent = unit_vector(v0[next_idx] - v0[idx])

        # Offset direction
        n_dir = unit_vector(p1 - p0)
        if np.linalg.norm(n_dir) < EPS:
            raise RuntimeError("Offset direction is degenerate at boundary vertex.")

        # Side direction of the arc profile
        b_dir = np.cross(tangent, n_dir)
        b_dir = unit_vector(b_dir)
        if np.linalg.norm(b_dir) < EPS:
            raise RuntimeError("Failed to compute arc side direction.")

        center = 0.5 * (p0 + p1)
        radius = 0.5 * np.linalg.norm(p1 - p0)
        u_dir = unit_vector(p0 - center)

        centers.append(center)
        radii.append(radius)
        u_dirs.append(u_dir)
        b_dirs.append(b_dir)

    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        np.asarray(u_dirs, dtype=np.float64),
        np.asarray(b_dirs, dtype=np.float64),
    )


def choose_global_round_sign(
    mesh: trimesh.Trimesh,
    centers: np.ndarray,
    radii: np.ndarray,
    b_dirs: np.ndarray,
    flip_round: bool = False
) -> float:
    """
    Choose one global sign for arc bulging direction.
    By default, choose the side farther from mesh centroid on average.
    If the result is visually opposite, use --flip-round.
    """
    centroid = mesh.centroid
    score = 0.0

    for c, r, b in zip(centers, radii, b_dirs):
        p_plus = c + r * b
        p_minus = c - r * b
        score += np.linalg.norm(p_plus - centroid) - np.linalg.norm(p_minus - centroid)

    sign = 1.0 if score >= 0 else -1.0
    if flip_round:
        sign *= -1.0
    return sign


def build_round_rings(
    mesh: trimesh.Trimesh,
    v0: np.ndarray,
    v1: np.ndarray,
    loop: np.ndarray,
    round_steps: int = 8,
    flip_round: bool = False
) -> list[np.ndarray]:
    """
    Build intermediate arc rings between original boundary and offset boundary.
    round_steps:
        total segments of the semicircle.
        Example: 8 means 7 intermediate rings.
    """
    if round_steps < 2:
        return []

    centers, radii, u_dirs, b_dirs = compute_round_bins(v0, v1, loop)
    sign = choose_global_round_sign(mesh, centers, radii, b_dirs, flip_round=flip_round)

    rings = []
    for k in range(1, round_steps):
        theta = np.pi * k / round_steps
        pts = []

        for c, r, u, b in zip(centers, radii, u_dirs, b_dirs):
            p = c + r * (np.cos(theta) * u + np.sin(theta) * sign * b)
            pts.append(p)

        rings.append(np.asarray(pts, dtype=np.float64))

    return rings


def thicken_open_surface_with_rounded_edge(
    mesh: trimesh.Trimesh,
    thickness_mm: float = 0.7,
    outward: bool = True,
    round_steps: int = 8,
    flip_round: bool = False
) -> trimesh.Trimesh:
    """
    Convert an open surface patch into a closed shell with rounded edge transition.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Input must be a trimesh.Trimesh object.")

    mesh = mesh.copy()
    clean_mesh(mesh)

    if len(mesh.faces) == 0:
        raise ValueError("Input mesh has no faces.")

    if mesh.faces.shape[1] != 3:
        raise ValueError("Input mesh must be triangular.")

    if mesh.is_watertight:
        raise ValueError("Input mesh is already watertight. This script expects an open surface.")

    loop = ordered_boundary_loop(mesh)

    normals = mesh.vertex_normals.copy()
    if not outward:
        normals *= -1.0

    v0 = mesh.vertices.copy()
    v1 = v0 + normals * float(thickness_mm)

    n0 = len(v0)

    # Original and offset faces
    f0 = mesh.faces.copy()
    f1 = mesh.faces[:, ::-1] + n0

    # Build rounded edge rings
    rings = build_round_rings(
        mesh=mesh,
        v0=v0,
        v1=v1,
        loop=loop,
        round_steps=round_steps,
        flip_round=flip_round
    )

    all_vertices = [v0, v1]
    ring_indices = []

    cursor = 2 * n0
    for ring in rings:
        all_vertices.append(ring)
        idx = np.arange(cursor, cursor + len(loop), dtype=np.int64)
        ring_indices.append(idx)
        cursor += len(loop)

    vertices = np.vstack(all_vertices)

    all_faces = [f0, f1]

    # Sequence of loops to bridge:
    # original boundary -> intermediate round rings -> offset boundary
    boundary_original = loop
    boundary_offset = loop + n0

    bridge_sequence = [boundary_original] + ring_indices + [boundary_offset]

    for a, b in zip(bridge_sequence[:-1], bridge_sequence[1:]):
        all_faces.append(bridge_loops(a, b))

    faces = np.vstack(all_faces)

    solid = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    clean_mesh(solid)
    solid.fix_normals()

    return solid


def main():
    parser = argparse.ArgumentParser(
        description="Convert an open PSI base surface into a printable solid with rounded edge transition."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input mesh path (.stl, .obj, .ply)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output solid mesh path (.stl, .obj, .ply)"
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.7,
        help="Shell thickness in mm (default: 0.7)"
    )
    parser.add_argument(
        "--inward",
        action="store_true",
        help="Offset inward instead of outward"
    )
    parser.add_argument(
        "--round-steps",
        type=int,
        default=8,
        help="Number of semicircle segments for rounded edge (default: 8; larger = smoother)"
    )
    parser.add_argument(
        "--flip-round",
        action="store_true",
        help="Flip rounded edge bulging direction if the arc goes to the wrong side"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Loading mesh: {input_path}")
    mesh = load_single_mesh(str(input_path))
    print(f"[INFO] trimesh version: {trimesh.__version__}")
    print(f"[INFO] Input vertices: {len(mesh.vertices)}")
    print(f"[INFO] Input faces: {len(mesh.faces)}")
    print(f"[INFO] Input watertight: {mesh.is_watertight}")

    solid = thicken_open_surface_with_rounded_edge(
        mesh=mesh,
        thickness_mm=args.thickness,
        outward=not args.inward,
        round_steps=args.round_steps,
        flip_round=args.flip_round
    )

    print(f"[INFO] Output vertices: {len(solid.vertices)}")
    print(f"[INFO] Output faces: {len(solid.faces)}")
    print(f"[INFO] Output watertight: {solid.is_watertight}")
    print(f"[INFO] Exporting: {output_path}")

    solid.export(str(output_path))
    print("[DONE] Rounded-edge solid shell saved successfully.")


if __name__ == "__main__":
    main()
