import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


EPS = 1e-9
HARD_MIN_BRIDGE_MM = 2.0


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_unreferenced_vertices()

    nondeg = mesh.nondegenerate_faces()
    if nondeg is not None and len(nondeg) == len(mesh.faces):
        mesh.update_faces(nondeg)

    unique = mesh.unique_faces()
    if unique is not None and len(unique) == len(mesh.faces):
        mesh.update_faces(unique)

    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def load_single_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("The input scene contains no geometry.")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a valid mesh: {path}")

    return clean_mesh(mesh.copy())


def get_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    unique_edges = mesh.edges_unique
    inverse = mesh.edges_unique_inverse
    counts = np.bincount(inverse, minlength=len(unique_edges))
    boundary_edges = unique_edges[counts == 1]
    return boundary_edges


def ordered_boundary_loop(mesh: trimesh.Trimesh) -> np.ndarray:
    boundary_edges = get_boundary_edges(mesh)

    if len(boundary_edges) == 0:
        raise ValueError("No open boundary found. The carrier surface may already be closed.")

    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    bad_vertices = [k for k, v in adj.items() if len(v) != 2]
    if bad_vertices:
        raise RuntimeError(
            "Boundary is not a simple single loop. "
            "This carrier surface may contain multiple holes, branches, or cracks."
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


def fit_best_plane(points: np.ndarray):
    origin = points.mean(axis=0)
    X = points - origin
    _, _, vt = np.linalg.svd(X, full_matrices=False)

    e1 = vt[0]
    e2 = vt[1]
    n = vt[2]

    if np.dot(np.cross(e1, e2), n) < 0:
        e2 = -e2

    e1 = e1 / max(np.linalg.norm(e1), EPS)
    e2 = e2 / max(np.linalg.norm(e2), EPS)
    n = np.cross(e1, e2)
    n = n / max(np.linalg.norm(n), EPS)

    return origin, e1, e2, n


def project_to_plane(points: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    rel = points - origin
    x = rel @ e1
    y = rel @ e2
    return np.column_stack([x, y])


def unproject_from_plane(points_2d: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return origin[None, :] + points_2d[:, 0:1] * e1[None, :] + points_2d[:, 1:2] * e2[None, :]


def polygon_largest(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def to_parts(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    return []


def build_boundary_polygon_2d(carrier: trimesh.Trimesh):
    loop = ordered_boundary_loop(carrier)

    origin, e1, e2, n = fit_best_plane(carrier.vertices)
    boundary_pts_3d = carrier.vertices[loop]
    boundary_pts_2d = project_to_plane(boundary_pts_3d, origin, e1, e2)

    poly = Polygon(boundary_pts_2d)
    if not poly.is_valid:
        poly = poly.buffer(0)

    poly = polygon_largest(poly)

    if poly.area <= 0:
        raise ValueError("Boundary polygon area is invalid.")

    return poly, origin, e1, e2, n


def negative_buffer_safe(poly, dist: float):
    if dist <= 0:
        return poly
    g = poly.buffer(-dist)
    if not g.is_valid:
        g = g.buffer(0)
    return g


def build_obround_polygon_local(total_length: float, total_width: float):
    """
    Build a centered obround (stadium/slot) polygon in local 2D coordinates.
    Long axis is along local +Y / -Y.
    """
    if total_length <= 0 or total_width <= 0:
        raise ValueError("total_length and total_width must be > 0")
    if total_length < total_width:
        raise ValueError("total_length must be >= total_width")

    radius = total_width / 2.0
    straight_len = total_length - total_width

    if straight_len <= EPS:
        poly = Point(0.0, 0.0).buffer(radius)
    else:
        line = LineString([(0.0, -straight_len / 2.0), (0.0, straight_len / 2.0)])
        poly = line.buffer(radius, cap_style=1, join_style=1)

    return poly


def build_obround_polygon_2d(center_xy: np.ndarray, total_length: float, total_width: float):
    poly = build_obround_polygon_local(total_length, total_width)
    poly = affinity.translate(poly, xoff=float(center_xy[0]), yoff=float(center_xy[1]))
    return poly


def obround_area_2d(total_length: float, total_width: float) -> float:
    radius = total_width / 2.0
    straight_len = max(0.0, total_length - total_width)
    return straight_len * total_width + np.pi * radius * radius


def generate_obround_grid(bounds, slot_length: float, slot_width: float, bridge: float, origin_xy=(0.0, 0.0)):
    """
    Regular grid of obround centers.
    Long axis is along 2D y direction.
    """
    minx, miny, maxx, maxy = bounds

    pitch_x = slot_width + bridge
    pitch_y = slot_length + bridge

    if pitch_x <= 0 or pitch_y <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    ix_min = int(np.floor((minx - origin_xy[0]) / pitch_x)) - 2
    ix_max = int(np.ceil((maxx - origin_xy[0]) / pitch_x)) + 2
    iy_min = int(np.floor((miny - origin_xy[1]) / pitch_y)) - 2
    iy_max = int(np.ceil((maxy - origin_xy[1]) / pitch_y)) + 2

    centers = []
    for ix in range(ix_min, ix_max + 1):
        x = origin_xy[0] + ix * pitch_x
        for iy in range(iy_min, iy_max + 1):
            y = origin_xy[1] + iy * pitch_y
            centers.append([x, y])

    if len(centers) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return np.asarray(centers, dtype=np.float64)


def can_place_polygon_no_overlap(
    candidate_poly: Polygon,
    allowed_part: Polygon,
    boundary_poly: Polygon,
    accepted_polys: list,
    min_bridge: float,
    tol: float = 1e-6,
) -> bool:
    """
    Accept candidate only if:
    1) the whole candidate stays inside the allowed zone
    2) the whole candidate stays inside the global boundary
    3) distance to every accepted candidate is >= min_bridge
    """
    if not allowed_part.contains(candidate_poly):
        return False

    if not boundary_poly.contains(candidate_poly):
        return False

    for old_poly in accepted_polys:
        if candidate_poly.distance(old_poly) < (min_bridge - tol):
            return False

    return True


def collect_uniform_obrounds(
    boundary_poly: Polygon,
    frame: float,
    slot_length: float,
    slot_width: float,
    bridge: float,
):
    """
    Uniform obround distribution:
    - outside frame: solid border
    - inside frame: regular obround array
    - explicit non-overlap check
    """
    if frame < 0:
        raise ValueError("frame must be >= 0")
    if slot_length <= 0 or slot_width <= 0:
        raise ValueError("slot_length and slot_width must be > 0")
    if slot_length < slot_width:
        raise ValueError("slot_length must be >= slot_width")
    if bridge < HARD_MIN_BRIDGE_MM:
        raise ValueError(f"bridge must be >= {HARD_MIN_BRIDGE_MM:.1f} mm")
    if bridge < 0:
        raise ValueError("bridge must be >= 0")

    allowed_zone = negative_buffer_safe(boundary_poly, frame)
    if allowed_zone.is_empty:
        return np.zeros((0, 2), dtype=np.float64)

    origin_xy = np.array(boundary_poly.centroid.coords[0], dtype=np.float64)

    accepted_centers = []
    accepted_polys = []

    for part in to_parts(allowed_zone):
        centers_2d = generate_obround_grid(
            bounds=part.bounds,
            slot_length=slot_length,
            slot_width=slot_width,
            bridge=bridge,
            origin_xy=origin_xy,
        )

        for c in centers_2d:
            slot_poly = build_obround_polygon_2d(
                center_xy=c,
                total_length=slot_length,
                total_width=slot_width,
            )

            if not can_place_polygon_no_overlap(
                candidate_poly=slot_poly,
                allowed_part=part,
                boundary_poly=boundary_poly,
                accepted_polys=accepted_polys,
                min_bridge=bridge,
            ):
                continue

            accepted_polys.append(slot_poly)
            accepted_centers.append(c)

    if len(accepted_centers) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return np.asarray(accepted_centers, dtype=np.float64)


def summarize_layout_metrics(
    boundary_poly: Polygon,
    frame: float,
    slot_length: float,
    slot_width: float,
    bridge: float,
    centers_2d: np.ndarray,
):
    allowed_zone = negative_buffer_safe(boundary_poly, frame)

    boundary_area = float(boundary_poly.area)
    allowed_area = float(allowed_zone.area) if not allowed_zone.is_empty else 0.0

    slot_count = int(len(centers_2d))
    slot_area_single = float(obround_area_2d(slot_length, slot_width))
    total_open_area = float(slot_count * slot_area_single)

    pitch_x = float(slot_width + bridge)
    pitch_y = float(slot_length + bridge)
    cell_area = float(max(pitch_x * pitch_y, EPS))
    theoretical_packing_ratio = float(slot_area_single / cell_area)

    open_ratio_boundary = float(total_open_area / max(boundary_area, EPS))
    open_ratio_allowed = float(total_open_area / max(allowed_area, EPS)) if allowed_area > EPS else 0.0
    slots_per_100mm2 = float(slot_count / max(boundary_area, EPS) * 100.0)

    return {
        "boundary_area": round(boundary_area, 4),
        "allowed_area": round(allowed_area, 4),
        "slot_count": slot_count,
        "slot_area_single": round(slot_area_single, 4),
        "total_open_area": round(total_open_area, 4),
        "open_ratio_boundary": round(open_ratio_boundary, 4),
        "open_ratio_allowed": round(open_ratio_allowed, 4),
        "slots_per_100mm2": round(slots_per_100mm2, 4),
        "pitch_x": round(pitch_x, 4),
        "pitch_y": round(pitch_y, 4),
        "cell_area": round(cell_area, 4),
        "theoretical_packing_ratio": round(theoretical_packing_ratio, 4),
        "hard_min_bridge_mm": HARD_MIN_BRIDGE_MM,
    }


def map_2d_centers_to_surface(
    carrier: trimesh.Trimesh,
    centers_2d: np.ndarray,
    origin: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
):
    queries = unproject_from_plane(centers_2d, origin, e1, e2)
    closest, dist, tri_id = trimesh.proximity.closest_point(carrier, queries)

    normals = carrier.face_normals[tri_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(min=EPS)

    return closest, normals, dist


def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        return np.zeros_like(v)
    return v / n


def make_local_frame_from_normal_and_ref(
    normal_3d: np.ndarray,
    ref_y_3d: np.ndarray,
    ref_x_3d: np.ndarray,
):
    """
    Build a local frame [x, y, n] where:
    - n is local normal
    - y is as aligned as possible with the global reference y direction projected onto tangent plane
    - x completes the right-handed frame
    """
    n = unit_vector(normal_3d)

    y = ref_y_3d - np.dot(ref_y_3d, n) * n
    if np.linalg.norm(y) < EPS:
        y = ref_x_3d - np.dot(ref_x_3d, n) * n

    y = unit_vector(y)
    x = unit_vector(np.cross(y, n))
    y = unit_vector(np.cross(n, x))

    return x, y, n


def build_obround_cutter(
    center_3d: np.ndarray,
    normal_3d: np.ndarray,
    ref_x_3d: np.ndarray,
    ref_y_3d: np.ndarray,
    slot_length: float,
    slot_width: float,
    total_height: float,
) -> trimesh.Trimesh:
    """
    Build one 3D obround cutter.
    Long axis follows local projected Y direction.
    """
    poly_local = build_obround_polygon_local(slot_length, slot_width)
    prism = trimesh.creation.extrude_polygon(
        polygon=poly_local,
        height=float(total_height),
        mid_plane=True,
    )

    x_dir, y_dir, n_dir = make_local_frame_from_normal_and_ref(
        normal_3d=normal_3d,
        ref_y_3d=ref_y_3d,
        ref_x_3d=ref_x_3d,
    )

    T = np.eye(4)
    T[:3, 0] = x_dir
    T[:3, 1] = y_dir
    T[:3, 2] = n_dir
    T[:3, 3] = center_3d

    prism.apply_transform(T)
    return prism


def build_obround_cutters(
    centers_3d: np.ndarray,
    normals_3d: np.ndarray,
    ref_x_3d: np.ndarray,
    ref_y_3d: np.ndarray,
    slot_length: float,
    slot_width: float,
    thickness_mm: float,
    overshoot_mm: float,
):
    cutters = []
    total_height = 2.0 * (thickness_mm + overshoot_mm)

    for c, n in zip(centers_3d, normals_3d):
        cutters.append(
            build_obround_cutter(
                center_3d=c,
                normal_3d=n,
                ref_x_3d=ref_x_3d,
                ref_y_3d=ref_y_3d,
                slot_length=slot_length,
                slot_width=slot_width,
                total_height=total_height,
            )
        )

    return cutters


def boolean_subtract(shell: trimesh.Trimesh, cutters: list[trimesh.Trimesh], engine: str = "manifold"):
    if len(cutters) == 0:
        raise RuntimeError("No obround cutters were generated.")

    result = trimesh.boolean.difference(
        meshes=[shell] + cutters,
        engine=engine,
        check_volume=True,
    )
    result = clean_mesh(result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create a regular Y-axis obround-slot perforation on a PSI shell with explicit non-overlap control."
    )
    parser.add_argument("--shell", required=True, help="Input thickened watertight shell mesh (.stl/.obj/.ply)")
    parser.add_argument("--carrier", required=True, help="Input original open carrier surface (.stl/.obj/.ply)")
    parser.add_argument("--output", required=True, help="Output obround-slot shell mesh (.stl/.obj/.ply)")

    parser.add_argument("--thickness", type=float, default=0.7, help="Shell thickness in mm (default: 0.7)")
    parser.add_argument("--slot-length", type=float, default=2.4, help="Obround total length in mm, along Y direction (default: 2.4)")
    parser.add_argument("--slot-width", type=float, default=1.0, help="Obround width in mm (default: 1.0)")
    parser.add_argument("--bridge", type=float, default=2.0, help="Minimum material width between slots in mm (hard minimum: 2.0)")
    parser.add_argument("--frame", type=float, default=2.2, help="Solid border width in mm (default: 2.2)")
    parser.add_argument("--overshoot", type=float, default=1.0, help="Extra cutter reach beyond shell on both sides in mm (default: 1.0)")
    parser.add_argument("--engine", default="manifold", choices=["manifold", "blender"], help="Boolean backend (default: manifold)")
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional path to save 2D layout metrics as JSON",
    )

    args = parser.parse_args()

    shell_path = Path(args.shell)
    carrier_path = Path(args.carrier)
    output_path = Path(args.output)

    if not shell_path.exists():
        raise FileNotFoundError(f"Shell file not found: {shell_path}")
    if not carrier_path.exists():
        raise FileNotFoundError(f"Carrier file not found: {carrier_path}")

    if args.slot_length < args.slot_width:
        raise ValueError("--slot-length must be >= --slot-width")

    if args.bridge < HARD_MIN_BRIDGE_MM:
        raise ValueError(
            f"--bridge must be >= {HARD_MIN_BRIDGE_MM:.1f} mm to avoid overly dense perforation."
        )

    print(f"[INFO] Loading shell   : {shell_path}")
    shell = load_single_mesh(str(shell_path))
    print(f"[INFO] Loading carrier : {carrier_path}")
    carrier = load_single_mesh(str(carrier_path))

    print(f"[INFO] trimesh version   : {trimesh.__version__}")
    print(f"[INFO] shell watertight  : {shell.is_watertight}")
    print(f"[INFO] carrier watertight: {carrier.is_watertight}")
    print(f"[INFO] hard min bridge   : {HARD_MIN_BRIDGE_MM:.3f} mm")

    if not shell.is_watertight:
        raise ValueError("The thickened shell must be watertight before slot subtraction.")

    boundary_poly, origin, e1, e2, plane_n = build_boundary_polygon_2d(carrier)
    print(f"[INFO] 2D boundary area   : {boundary_poly.area:.3f}")

    centers_2d = collect_uniform_obrounds(
        boundary_poly=boundary_poly,
        frame=args.frame,
        slot_length=args.slot_length,
        slot_width=args.slot_width,
        bridge=args.bridge,
    )

    print(f"[INFO] kept slots total  : {len(centers_2d)}")
    if len(centers_2d) == 0:
        raise RuntimeError(
            "No obround slots survived. Try reducing --frame or reducing --slot-length / --slot-width. "
            "Note: --bridge cannot be reduced below the 2.0 mm hard threshold."
        )

    layout_metrics = summarize_layout_metrics(
        boundary_poly=boundary_poly,
        frame=args.frame,
        slot_length=args.slot_length,
        slot_width=args.slot_width,
        bridge=args.bridge,
        centers_2d=centers_2d,
    )

    print(f"[INFO] theoretical packing ratio : {layout_metrics['theoretical_packing_ratio']:.4f}")
    print(f"[INFO] open ratio allowed zone   : {layout_metrics['open_ratio_allowed']:.4f}")
    print(f"[INFO] slot density /100 mm^2   : {layout_metrics['slots_per_100mm2']:.4f}")

    if args.metrics_json:
        Path(args.metrics_json).write_text(
            json.dumps(layout_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] layout metrics saved    : {args.metrics_json}")

    centers_3d, normals_3d, snap_dist = map_2d_centers_to_surface(
        carrier=carrier,
        centers_2d=centers_2d,
        origin=origin,
        e1=e1,
        e2=e2,
    )

    print(f"[INFO] snap dist range   : {snap_dist.min():.4f} ~ {snap_dist.max():.4f} mm")
    print(f"[INFO] slot size         : length={args.slot_length:.3f} mm, width={args.slot_width:.3f} mm")
    print(f"[INFO] bridge            : {args.bridge:.3f} mm")
    print(f"[INFO] frame             : {args.frame:.3f} mm")

    cutters = build_obround_cutters(
        centers_3d=centers_3d,
        normals_3d=normals_3d,
        ref_x_3d=e1,
        ref_y_3d=e2,
        slot_length=args.slot_length,
        slot_width=args.slot_width,
        thickness_mm=args.thickness,
        overshoot_mm=args.overshoot,
    )
    print(f"[INFO] cutter count      : {len(cutters)}")

    print(f"[INFO] running boolean difference with engine='{args.engine}' ...")
    result = boolean_subtract(shell, cutters, engine=args.engine)

    print(f"[INFO] output watertight : {result.is_watertight}")
    print(f"[INFO] exporting        : {output_path}")
    result.export(str(output_path))
    print("[DONE] Uniform Y-axis obround-slot shell saved successfully.")


if __name__ == "__main__":
    main()
