#!/usr/bin/env python3
"""
Cell utilities: read/write CP2K cell files, convert between
cell vectors and (a,b,c,α,β,γ), perturb parameters for finite differences.
"""
from pathlib import Path

import numpy as np


def read_cell(path):
    """Read CP2K cell file, return 3x3 numpy array (A,B,C as rows)."""
    path = Path(path)
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "!")):
            continue
        parts = line.split()
        nums = []
        for x in parts:
            try:
                nums.append(float(x))
            except ValueError:
                pass
        if len(nums) >= 3:
            rows.append(nums[-3:])
        if len(rows) == 3:
            break
    if len(rows) != 3:
        raise ValueError(f"Could not parse 3 cell vectors from {path}")
    return np.array(rows, dtype=float)


def write_cell(path, cell):
    """Write 3x3 cell matrix to CP2K format (A, B, C rows)."""
    path = Path(path)
    with open(path, "w") as f:
        for name, row in zip(("A", "B", "C"), cell):
            f.write(f"{name:<3} {row[0]:18.15f} {row[1]:18.15f} {row[2]:18.15f}\n")


def cell_to_abc_angles(cell):
    """
    Convert cell matrix to (a, b, c, α, β, γ).
    a,b,c in Angstrom; α,β,γ in degrees.
    α = angle(B,C), β = angle(A,C), γ = angle(A,B).
    """
    a_vec = cell[0]
    b_vec = cell[1]
    c_vec = cell[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        raise ValueError("Zero cell vector length")
    alpha = np.degrees(np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1, 1)))
    beta = np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1, 1)))
    return (a, b, c, alpha, beta, gamma)


def abc_angles_to_cell(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    Convert (a,b,c,α,β,γ) to 3x3 cell matrix.
    Standard crystallographic convention:
    A = (a, 0, 0)
    B = (b*cos(γ), b*sin(γ), 0)
    C = (c*cos(β), c*(cos(α)-cos(β)cos(γ))/sin(γ), c_z)
    """
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    sg = np.sin(gamma)
    if abs(sg) < 1e-10:
        raise ValueError("sin(gamma) near zero - degenerate cell")
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    c_x = c * np.cos(beta)
    c_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sg
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    if c_z_sq < -1e-10:
        raise ValueError("Invalid cell angles - c_z^2 < 0")
    c_z = np.sqrt(max(0, c_z_sq))
    c_vec = np.array([c_x, c_y, c_z])
    return np.array([a_vec, b_vec, c_vec])


def perturb_param(cell, param_idx, signed_delta):
    """
    Perturb one cell parameter by signed_delta.
    param_idx: 0=a, 1=b, 2=c, 3=α, 4=β, 5=γ
    signed_delta: positive or negative change (lengths in Å, angles in degrees).
    Returns new 3x3 cell matrix with same orientation as input cell.
    """
    a, b, c, alpha, beta, gamma = cell_to_abc_angles(cell)
    params = [a, b, c, alpha, beta, gamma]
    params[param_idx] += signed_delta
    # Sanity bounds
    if param_idx < 3:
        params[param_idx] = max(params[param_idx], 0.1)
    else:
        params[param_idx] = np.clip(params[param_idx], 1.0, 179.0)
    cell_standard_new = abc_angles_to_cell(*params)
    return apply_cell_orientation(cell, cell_standard_new)


def get_perturbation_deltas(cell, delta_length=0.001, delta_angle=0.05, delta_length_ang=None):
    """
    Return list of (param_idx, delta) for central differences.
    delta_length: relative perturbation for a,b,c when delta_length_ang is None (e.g. 0.001 = 0.1%)
    delta_angle: absolute perturbation for α,β,γ in degrees.
    delta_length_ang: if set, use this absolute value [Å] for all lengths (overrides delta_length).
    """
    a, b, c, alpha, beta, gamma = cell_to_abc_angles(cell)
    if delta_length_ang is not None:
        dl = float(delta_length_ang)
        length_deltas = [(0, dl), (1, dl), (2, dl)]
    else:
        length_deltas = [
            (0, a * delta_length),
            (1, b * delta_length),
            (2, c * delta_length),
        ]
    return length_deltas + [
        (3, delta_angle),
        (4, delta_angle),
        (5, delta_angle),
    ]


def rotate_cell_to_axis_frame(cell, axis):
    """
    Rotate cell to axis frame: aaxis=A along X, baxis=B along Y, cbaxis=(B+C) along Z.
    Returns 3x3 cell matrix (A,B,C as rows).
    """
    cell = np.asarray(cell, dtype=float)
    axis = axis.lower()

    def rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    if axis == "aaxis":
        a_vec = cell[0]
        theta_z = np.arctan2(-a_vec[1], a_vec[0]) if (a_vec[0] ** 2 + a_vec[1] ** 2) > 1e-20 else 0
        Rz = rot_z(theta_z)
        a1 = Rz @ a_vec
        theta_y = np.arctan2(-a1[2], a1[0]) if (a1[0] ** 2 + a1[2] ** 2) > 1e-20 else 0
        Ry = rot_y(theta_y)
        R = Ry @ Rz
    elif axis == "baxis":
        b_vec = cell[1]
        alpha = np.arctan2(-b_vec[2], b_vec[1]) if (abs(b_vec[1]) + abs(b_vec[2])) > 1e-12 else 0
        Rx = rot_x(alpha)
        v1 = Rx @ b_vec
        beta = np.arctan2(v1[0], v1[1]) if (abs(v1[0]) + abs(v1[1])) > 1e-12 else 0
        Rz = rot_z(beta)
        R = Rz @ Rx
    elif axis == "cbaxis":
        v = cell[1] + cell[2]
        alpha = np.arctan2(v[1], v[2]) if (abs(v[1]) + abs(v[2])) > 1e-12 else 0
        Rx = rot_x(alpha)
        v1 = Rx @ v
        beta = np.arctan2(-v1[0], v1[2]) if (abs(v1[0]) + abs(v1[2])) > 1e-12 else 0
        Ry = rot_y(beta)
        R = Ry @ Rx
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return (R @ cell.T).T


def infer_strong_pairs(cell, orth_tol_deg=2.0):
    """
    Infer strong pair couplings (i<j) for q=[a,b,c,alpha,beta,gamma].

    Assumptions:
    - Length-length couplings are treated as strong.
    - Length-angle couplings are treated as strong.
    - Angle-angle couplings are treated as strong.
    """
    _ = (cell, orth_tol_deg)
    return [(i, j) for i in range(6) for j in range(i + 1, 6)]


def apply_cell_orientation(cell_orig, cell_standard_new):
    """
    Return cell_standard_new rotated to match the orientation of cell_orig.

    abc_angles_to_cell always produces A-along-X. The input cell_orig may have
    any orientation (e.g. B along Y, or (b+c) along Z, or an intermediate
    from optimization). This restores that orientation to the new cell so
    the uniaxial stress axis is preserved.

    cell_orig, cell_standard_new: 3x3 arrays (A,B,C as rows)
    Returns: 3x3 cell with same shape as cell_standard_new, same orientation as cell_orig.
    """
    # cell_standard_orig has same (a,b,c,α,β,γ) as cell_orig but A-along-X
    abc = cell_to_abc_angles(cell_orig)
    cell_standard_orig = abc_angles_to_cell(*abc)
    # R such that cell_orig = R @ cell_standard_orig (rows as vectors)
    R = np.linalg.solve(cell_standard_orig.T, cell_orig.T).T
    return R @ cell_standard_new
