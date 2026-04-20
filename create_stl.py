#!/usr/bin/env python

"""
coords_to_stl.py
----------------
Reads X and Z coordinates from a CSV or TXT file and generates a
watertight STL file extruded along the Y axis.

New feature:
  - Reads blockMeshDict to extract the maximum X coordinate (x_bm_max)
  - Finds the last point in the CSV where X == max(X in CSV), gets its Z (z_xmax)
  - Appends an extra point (x_bm_max, z_xmax) before extrusion so the STL
    connects flush to the blockMesh right boundary.

Usage:
  python coords_to_stl.py --input coordinates.csv --output my_shape.stl \
                           --y_min 0.0 --y_max 1.0 \
                           --blockmesh system/blockMeshDict

Arguments:
  --input       : Path to your CSV or TXT file with X, Z coordinates
  --output      : Output STL filename (default: output.stl)
  --y_min       : Start of Y extrusion (default: 0.0)
  --y_max       : End of Y extrusion -- should match your blockMesh Y extent
  --no_header   : Add this flag if your file has NO header row
  --x_col       : Zero-based column index for X (default: 0)
  --z_col       : Zero-based column index for Z (default: 1)
  --blockmesh   : Path to blockMeshDict (default: system/blockMeshDict)
                  Set to "none" to skip the extra-point insertion
"""

import argparse
import numpy as np
import re
import os
import sys


# ─────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────

def read_coordinates(filepath, has_header=True, x_col=0, z_col=1):
    """
    Read X, Z coordinates from CSV or space/tab-delimited TXT.
    x_col, z_col: zero-based column indices for X and Z values.
    Auto-detects column positions from header names (x, z case-insensitive).
    """
    points = []

    with open(filepath, "r") as f:
        content = f.read().strip()

    # Auto-detect delimiter
    delimiter = "," if "," in content else None  # None = whitespace

    lines = content.splitlines()

    if has_header and lines:
        header_line = lines[0]
        headers = [h.strip().lower() for h in
                   (header_line.split(delimiter) if delimiter else header_line.split())]
        print(f"  Detected header columns: {headers}")

        # Auto-detect only if user kept defaults
        if x_col == 0 and z_col == 1:
            if "x" in headers:
                x_col = headers.index("x")
            if "z" in headers:
                z_col = headers.index("z")

    print(f"  Using column {x_col} for X, column {z_col} for Z")
    start = 1 if has_header else 0

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(delimiter) if delimiter else line.split()
        try:
            x = float(parts[x_col].strip())
            z = float(parts[z_col].strip())
            points.append([x, z])
        except (ValueError, IndexError) as e:
            print(f"  Skipping unreadable line: '{line}' ({e})")

    return np.array(points)


def write_stl(filename, facets, solid_name="shape"):
    """Write list of (normal, [v0,v1,v2]) to ASCII STL."""
    with open(filename, "w") as f:
        f.write(f"solid {solid_name}\n")
        for normal, verts in facets:
            f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("    outer loop\n")
            for v in verts:
                f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


# ─────────────────────────────────────────────
#  blockMeshDict parser
# ─────────────────────────────────────────────

def parse_blockmesh_xmax(blockmesh_path):
    """
    Parse blockMeshDict and return the maximum X coordinate found
    in the vertices section.

    Handles both formats:
      vertices
      (
          (x y z)   // OpenFOAM style
          ...
      );

    Returns float or None if not found.
    """
    if not os.path.isfile(blockmesh_path):
        print(f"  WARNING: blockMeshDict not found at '{blockmesh_path}'")
        return None

    with open(blockmesh_path, "r") as f:
        content = f.read()

    # Strip C++ style comments
    content = re.sub(r"//.*", "", content)          # single-line
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)  # multi-line

    # Find the vertices block: vertices ( ... );
    match = re.search(r"vertices\s*\((.*?)\)\s*;", content, re.DOTALL)
    if not match:
        print("  WARNING: Could not find vertices block in blockMeshDict")
        return None

    vertices_block = match.group(1)

    # Extract all (x y z) vertex tuples
    vertex_pattern = re.compile(
        r"\(\s*([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s*\)"
    )
    x_values = []
    for m in vertex_pattern.finditer(vertices_block):
        x_values.append(float(m.group(1)))

    if not x_values:
        print("  WARNING: No vertex coordinates found in blockMeshDict")
        return None

    x_max = max(x_values)
    print(f"  blockMeshDict vertices X values found: {sorted(set(x_values))}")
    print(f"  blockMeshDict X_max = {x_max}")
    return x_max


# ─────────────────────────────────────────────
#  Peak raising with smooth blending
# ─────────────────────────────────────────────

def raise_peak_smooth(pts, dz=0.1, n_blend=50):
    """
    Raise the peak (max-Z point) by dz and smoothly *scale* the triangle
    sides so the shape is preserved — just taller.

    For each side (left / right of peak):
      - The base_z is the Z value at the n_blend-th neighbour (unchanged).
      - Every point between base and peak is vertically scaled so that
        the peak reaches (peak_z + dz) while the base stays fixed.
    """
    pts = pts.copy()
    peak_idx = int(np.argmax(pts[:, 1]))
    peak_x = pts[peak_idx, 0]
    peak_z = pts[peak_idx, 1]

    # Split into left-of-peak and right-of-peak by X coordinate
    left_mask = pts[:, 0] < peak_x - 1e-12
    right_mask = pts[:, 0] > peak_x + 1e-12

    # Left side: sort by X descending (closest to peak first)
    left_indices = np.where(left_mask)[0]
    left_sorted = left_indices[np.argsort(-pts[left_indices, 0])]
    left_blend = left_sorted[:n_blend]

    # Right side: sort by X ascending (closest to peak first)
    right_indices = np.where(right_mask)[0]
    right_sorted = right_indices[np.argsort(pts[right_indices, 0])]
    right_blend = right_sorted[:n_blend]

    def _scale_side(indices):
        if len(indices) == 0:
            return
        # base_z = Z at the outermost blend point (stays unchanged)
        base_z = pts[indices[-1], 1]
        h_peak = peak_z - base_z
        if h_peak < 1e-12:
            return
        # scale factor to stretch from base_z to (peak_z + dz)
        scale = (peak_z + dz - base_z) / h_peak
        for idx in indices:
            h = pts[idx, 1] - base_z
            pts[idx, 1] = base_z + h * scale

    _scale_side(left_blend)
    _scale_side(right_blend)

    # Raise the peak itself
    pts[peak_idx, 1] = peak_z + dz

    print(f"  Raised peak at index {peak_idx} (X={peak_x:.3f}) by {dz}, "
          f"scaled {len(left_blend)} left + {len(right_blend)} right neighbours")
    return pts


# ─────────────────────────────────────────────
#  Extra-point insertion
# ─────────────────────────────────────────────

def append_blockmesh_boundary_point(pts, x_bm_max):
    """
    1. Find the point(s) in pts where X == max(X in pts).
       Use the LAST such point (preserving the order of traversal).
    2. Get its Z value (z_xmax).
    3. Append a new point (x_bm_max, z_xmax) at the end of pts.

    This extends the profile to reach the blockMesh right boundary.
    """
    x_csv_max = np.max(pts[:, 0])
    x_csv_min = np.min(pts[:, 0])
    z_csv_max = np.max(pts[:, 1])
    pts = raise_peak_smooth(pts, dz=0.1, n_blend=100)
    z_csv_max = np.max(pts[:, 1])
    print(f" CSV Z range: {np.min(pts[:, 1])} to {z_csv_max}")
    # All points at the CSV x_max (tolerance for floats)
    tol = 1e-10 * (abs(x_csv_max) + 1.0)
    mask = np.abs(pts[:, 0] - x_csv_max) < tol
    candidates = pts[mask]

    # Take the last one in original ordering
    original_indices = np.where(mask)[0]
    last_idx = original_indices[-1]
    z_xmax = pts[last_idx, 1]

    print(f"  CSV  X_max = {x_csv_max}  (found at row index {last_idx}, Z = {z_xmax})")
    print(f"  Appending extra point: ({x_bm_max}, {z_xmax})")

    extra = np.array([
        [x_bm_max, z_xmax],
        [x_bm_max, 0],
        [x_csv_min, 0],
    ])
    return np.vstack([pts, extra])


# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def compute_normal(v0, v1, v2):
    e1 = np.array(v1) - np.array(v0)
    e2 = np.array(v2) - np.array(v0)
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n)
    return (n / norm).tolist() if norm > 1e-12 else [0.0, 0.0, 0.0]


def polygon_area_signed(pts):
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return area / 2.0


def ensure_ccw(pts):
    if polygon_area_signed(pts) < 0:
        return pts[::-1]
    return pts


def is_ear(pts, i, indices):
    n = len(indices)
    prev_i = indices[(i - 1) % n]
    curr_i = indices[i]
    next_i = indices[(i + 1) % n]
    a, b, c = pts[prev_i], pts[curr_i], pts[next_i]
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross <= 0:
        return False
    for j, idx in enumerate(indices):
        if idx in (prev_i, curr_i, next_i):
            continue
        if point_in_triangle(pts[idx], a, b, c):
            return False
    return True


def point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def triangulate_polygon(pts):
    pts = list(pts)
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points.")
    if n == 3:
        return [(0, 1, 2)]
    indices = list(range(n))
    triangles = []
    max_iter = n * n
    itr = 0
    while len(indices) > 3 and itr < max_iter:
        ear_found = False
        for i in range(len(indices)):
            if is_ear(pts, i, indices):
                prev_i = indices[(i - 1) % len(indices)]
                curr_i = indices[i]
                next_i = indices[(i + 1) % len(indices)]
                triangles.append((prev_i, curr_i, next_i))
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            print("  Warning: Could not fully triangulate. Partial result.")
            break
        itr += 1
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


# ─────────────────────────────────────────────
#  STL builder
# ─────────────────────────────────────────────

def build_prism_stl(xz_points, y_min, y_max):
    pts2d = ensure_ccw(xz_points.tolist())
    n = len(pts2d)
    front = [[p[0], y_min, p[1]] for p in pts2d]
    back  = [[p[0], y_max, p[1]] for p in pts2d]
    facets = []
    tri_indices = triangulate_polygon(pts2d)

    for (i0, i1, i2) in tri_indices:
        v0, v1, v2 = front[i0], front[i1], front[i2]
        n_vec = compute_normal(v0, v2, v1)
        facets.append((n_vec, [v0, v2, v1]))

    for (i0, i1, i2) in tri_indices:
        v0, v1, v2 = back[i0], back[i1], back[i2]
        n_vec = compute_normal(v0, v1, v2)
        facets.append((n_vec, [v0, v1, v2]))

    for i in range(n):
        j = (i + 1) % n
        f0, f1 = front[i], front[j]
        b0, b1 = back[i],  back[j]
        n_vec = compute_normal(f0, f1, b1)
        facets.append((n_vec, [f0, f1, b1]))
        n_vec = compute_normal(f0, b1, b0)
        facets.append((n_vec, [f0, b1, b0]))

    return facets


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert X-Z coordinates to STL (Y-extruded prism), "
                    "optionally extending to blockMesh X boundary.")
    parser.add_argument("--input",      required=True,                   help="Input CSV or TXT file")
    parser.add_argument("--output",     default="output.stl",            help="Output STL filename")
    parser.add_argument("--y_min",      type=float, default=0.0,         help="Y extrusion start")
    parser.add_argument("--y_max",      type=float, default=1.0,         help="Y extrusion end")
    parser.add_argument("--no_header",  action="store_true",             help="File has no header row")
    parser.add_argument("--x_col",      type=int,   default=0,           help="Zero-based column index for X (default: 0)")
    parser.add_argument("--z_col",      type=int,   default=1,           help="Zero-based column index for Z (default: 1)")
    parser.add_argument("--xorigin",    type=float, default=0.0,         help="Offset added to X coordinates from CSV (default: 0.0)")
    parser.add_argument("--blockmesh",  default="system/blockMeshDict",  help="Path to blockMeshDict (use \'none\' to skip)")
    args = parser.parse_args()

    # ── Read CSV / TXT ──
    print(f"\n  Reading coordinates from: {args.input}")
    pts = read_coordinates(args.input,
                           has_header=not args.no_header,
                           x_col=args.x_col,
                           z_col=args.z_col)
    print(f"  Loaded {len(pts)} points")

    # ── Apply X origin offset ──
    if args.xorigin != 0.0:
        print(f"  Applying X origin offset: {args.xorigin}")
        pts[:, 0] += args.xorigin

    if len(pts) < 3:
        print("  ERROR: Need at least 3 points to form a closed polygon.")
        sys.exit(1)

    # ── Optionally extend to blockMesh X boundary ──
    if args.blockmesh.lower() != "none":
        x_bm_max = parse_blockmesh_xmax(args.blockmesh)
        if x_bm_max is not None:
            pts = append_blockmesh_boundary_point(pts, x_bm_max)
            print(f"  Points after extension: {len(pts)}")
        else:
            print("  Skipping extra-point insertion (blockMeshDict parse failed).")
    else:
        print("  Skipping blockMeshDict (--blockmesh none)")

    print(f"\n  Final polygon points (X, Z):")
    

    # ── Build and write STL ──
    print(f"\n  Building STL prism (Y from {args.y_min} to {args.y_max})...")
    facets = build_prism_stl(pts, args.y_min, args.y_max)
    write_stl(args.output, facets)

    print(f"  STL written to  : {args.output}")
    print(f"  Total facets    : {len(facets)}")
    print(f"\n  Verify with: surfaceCheck {args.output}\n")


if __name__ == "__main__":
    main()
