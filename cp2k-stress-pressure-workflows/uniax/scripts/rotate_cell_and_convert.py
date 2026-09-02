#!/usr/bin/env python3
"""
Rotate hydro cell so the stress axis aligns with X/Y/Z, then compute fractional
coords via direct conversion f = inv(rotated_cell.T) @ r (no additional rotation).
Usage: rotate_cell_and_convert.py <axis> <src_cell> <src_xyz> <out_cell> <out_xyz>
  axis: aaxis | baxis | cbaxis
  aaxis: A along X (stress along X)
  baxis: B along Y (stress along Y)
  cbaxis: (b+c) along Z (stress along Z)
"""
import math
import sys
from pathlib import Path


def read_cell(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s[0] in ("#", "!"):
            continue
        parts = line.split()
        nums = [float(x) for x in parts if _num(x)]
        if len(nums) >= 3:
            rows.append(nums[-3:])
        if len(rows) == 3:
            break
    if len(rows) != 3:
        raise SystemExit(f"Could not parse 3 cell vectors from {path}")
    return rows


def _num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_xyz(path):
    lines = [ln.rstrip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    i = 0
    try:
        int(lines[0].split()[0])
        i = 2
    except (ValueError, IndexError):
        i = 0
    atoms = []
    for ln in lines[i:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        atoms.append((parts[0], [float(parts[1]), float(parts[2]), float(parts[3])]))
    return atoms


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def mat_mul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat_vec(m, v):
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def rotate_cell_aaxis(cell):
    """Rotate so A aligns with X."""
    a_vec = cell[0]
    ax, ay, az = a_vec[0], a_vec[1], a_vec[2]
    # Rotate in XY to put A in XZ (null ay)
    theta_z = math.atan2(-ay, ax) if (ax * ax + ay * ay) > 1e-20 else 0
    Rz = rot_z(theta_z)
    a1 = mat_vec(Rz, a_vec)
    # Rotate in XZ to put A along X (null a1[2])
    theta_y = math.atan2(-a1[2], a1[0]) if (a1[0] * a1[0] + a1[2] * a1[2]) > 1e-20 else 0
    Ry = rot_y(theta_y)
    R = mat_mul(Ry, Rz)
    return [[mat_vec(R, cell[i])[j] for j in range(3)] for i in range(3)]


def rotate_cell_baxis(cell):
    """Rotate so B aligns with Y."""
    b_vec = cell[1]
    v = [b_vec[0], b_vec[1], b_vec[2]]
    alpha = math.atan2(-v[2], v[1]) if (abs(v[1]) + abs(v[2])) > 1e-12 else 0
    Rx = rot_x(alpha)
    v1 = mat_vec(Rx, v)
    beta = math.atan2(v1[0], v1[1]) if (abs(v1[0]) + abs(v1[1])) > 1e-12 else 0
    Rz = rot_z(beta)
    R = mat_mul(Rz, Rx)
    return [[mat_vec(R, cell[i])[j] for j in range(3)] for i in range(3)]


def rotate_cell_cbaxis(cell):
    """Rotate so (b+c) aligns with Z."""
    b_vec, c_vec = cell[1], cell[2]
    v = [b_vec[0] + c_vec[0], b_vec[1] + c_vec[1], b_vec[2] + c_vec[2]]
    alpha = math.atan2(v[1], v[2]) if (abs(v[1]) + abs(v[2])) > 1e-12 else 0
    Rx = rot_x(alpha)
    v1 = mat_vec(Rx, v)
    beta = math.atan2(-v1[0], v1[2]) if (abs(v1[0]) + abs(v1[2])) > 1e-12 else 0
    Ry = rot_y(beta)
    R = mat_mul(Ry, Rx)
    return [[mat_vec(R, cell[i])[j] for j in range(3)] for i in range(3)]


def inv3(m):
    a, b, c = m[0][0], m[0][1], m[0][2]
    d, e, f = m[1][0], m[1][1], m[1][2]
    g, h, i = m[2][0], m[2][1], m[2][2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-18:
        raise SystemExit("Cell matrix singular")
    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def main():
    if len(sys.argv) != 6:
        print("Usage: rotate_cell_and_convert.py <axis> <src_cell> <src_xyz> <out_cell> <out_xyz>", file=sys.stderr)
        sys.exit(2)
    axis = sys.argv[1].lower()
    src_cell = Path(sys.argv[2])
    src_xyz = Path(sys.argv[3])
    out_cell = Path(sys.argv[4])
    out_xyz = Path(sys.argv[5])

    cell = read_cell(src_cell)
    if axis == "aaxis":
        cell_rot = rotate_cell_aaxis(cell)
    elif axis == "baxis":
        cell_rot = rotate_cell_baxis(cell)
    elif axis == "cbaxis":
        cell_rot = rotate_cell_cbaxis(cell)
    else:
        raise SystemExit(f"Unknown axis: {axis}")

    # Fractional coords from hydro cell+coords direct conversion (no rotation applied)
    cell_T = [[cell[j][i] for j in range(3)] for i in range(3)]
    inv_cell_T = inv3(cell_T)

    atoms = read_xyz(src_xyz)
    if not atoms:
        raise SystemExit(f"No atoms in {src_xyz}")

    out_cell.write_text(
        "\n".join(
            f"{'A' if i == 0 else 'B' if i == 1 else 'C'}   {cell_rot[i][0]:18.15f} {cell_rot[i][1]:18.15f} {cell_rot[i][2]:18.15f}"
            for i in range(3)
        )
        + "\n"
    )
    lines = ["SCALED T"]
    for el, r in atoms:
        f = [sum(inv_cell_T[i][j] * r[j] for j in range(3)) for i in range(3)]
        lines.append(f"{el:2s} {f[0]: .12f} {f[1]: .12f} {f[2]: .12f}")
    out_xyz.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
