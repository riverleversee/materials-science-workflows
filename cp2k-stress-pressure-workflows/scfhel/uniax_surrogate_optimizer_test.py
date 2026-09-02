#!/usr/bin/env python3
"""
Test optimizer for uniaxial stress matching on scfhel surrogate surface.

Workflow:
1. Coarse grid at 40% tolerance; filter to 20% and 10% matches
2. Local minima in enthalpy at 20%/40%; local minima in second cost (H + 1 meV per 10% eig deviation) at 20%/40%; force fine grid at any 10% match not already a candidate
3. Fine grid 5^6 (20% tol) at each merged center; boundary/30° rules for local minima
4. Gradient descent with combined eigenvalue + enthalpy forces (second cost not used after fine-grid selection)
7. Uniqueness by eigenvector (20 deg, H-ordered)
First step goal: 0.3 GPa (delta P) away from hydrostatic.
"""
from __future__ import annotations

import itertools
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed  # ProcessPoolExecutor may need full permissions
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Convert pressure×volume to Hartree: 1 bar·Å³ = 1e5 Pa × 1e-30 m³ = 1e-25 J; 1 Ha ≈ 4.3597e-18 J
BAR_ANG3_TO_HA = 2.2937122783963248e-8  # (bar · Å³) → Ha
GPA_TO_BAR = 10000.0
# 1 meV in Ha (1 eV = 1/27.211386245988 Ha)
MEV_TO_HA = 1e-3 / 27.211386245988
PARAM_LABELS = ("a", "b", "c", "alpha", "beta", "gamma")
STRESS_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")

# CP2K enthalpy polynomial and stress response (parameter_trend_matrices.txt)
_H_data: Optional[Tuple[float, np.ndarray, Dict[Tuple[int, int], float]]] = None  # (H0, dH, cH)
_trend_q0: Optional[np.ndarray] = None
_trend_deltas: Optional[np.ndarray] = None
_E0: Optional[float] = None
_P_iso: Optional[float] = None

# Stress surrogate terms (bar): sigma(x) = sigma0 + d_single@x + sum_{i<=j} c_ij x_i x_j
_SIGMA0: Optional[np.ndarray] = None  # (6,)
_DSINGLE: Optional[np.ndarray] = None  # (6,6)
_CPAIRS: Optional[Dict[Tuple[int, int], np.ndarray]] = None  # (i,j)->(6,)


def _workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_cp2k_param_path() -> Path:
    env = os.environ.get("CP2K_PARAM_TREND_PATH")
    if env:
        return Path(env)
    for p in [
        _workflow_root() / "uniax/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt",
    ]:
        if p.exists():
            return p
    return _workflow_root() / "uniax/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt"


def vec6_to_mat3(v6: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = v6
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def mat3_to_vec6(mat: np.ndarray) -> np.ndarray:
    return np.array([
        mat[0, 0], mat[1, 1], mat[2, 2],
        mat[0, 1], mat[0, 2], mat[1, 2],
    ], dtype=float)


def voigt_strain_to_mat3(eps_voigt: np.ndarray) -> np.ndarray:
    """Voigt [ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy] (engineering γ = 2ε) -> 3x3 symmetric strain."""
    exx, eyy, ezz = eps_voigt[0], eps_voigt[1], eps_voigt[2]
    eyz, exz, exy = eps_voigt[3] / 2.0, eps_voigt[4] / 2.0, eps_voigt[5] / 2.0
    return np.array([[exx, exy, exz], [exy, eyy, eyz], [exz, eyz, ezz]], dtype=float)


def mat3_strain_to_voigt(E_mat: np.ndarray) -> np.ndarray:
    """Symmetric 3x3 strain E -> Voigt [ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy] (γ = 2ε)."""
    return np.array([
        E_mat[0, 0], E_mat[1, 1], E_mat[2, 2],
        2 * E_mat[1, 2], 2 * E_mat[0, 2], 2 * E_mat[0, 1],
    ], dtype=float)


def stress_strain_double_contraction(sigma_voigt: np.ndarray, eps_voigt: np.ndarray) -> float:
    """
    Scalar σ : ε = σ_ij ε_ij (frame-invariant Frobenius product).
    Same value in any orthonormal basis; use this so the work term does not depend on
    the choice of coordinate frame.
    """
    sigma_mat = vec6_to_mat3(sigma_voigt)
    eps_mat = voigt_strain_to_mat3(eps_voigt)
    return float(np.sum(sigma_mat * eps_mat))


def principal_sorted(mat3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(mat3)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def abc_to_cell(a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """Convert (a,b,c,α,β,γ) to 3x3 cell matrix in 'standard' orientation: A along X, B in XY plane.
    Rows = A, B, C (cell = h^T with CP2K columns = A,B,C). The 15gpa job uses init_cell.cell with
    a different orientation (A in XZ, B 3D); when trend is loaded we apply that orientation in cell_at_x."""
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    sg = np.sin(gamma)
    if abs(sg) < 1e-10:
        sg = 1e-10
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    c_x = c * np.cos(beta)
    c_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sg
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    c_z = np.sqrt(max(0.0, c_z_sq))
    c_vec = np.array([c_x, c_y, c_z])
    return np.array([a_vec, b_vec, c_vec])


def cell_to_abc(cell: np.ndarray) -> np.ndarray:
    """Return (a, b, c, alpha_deg, beta_deg, gamma_deg)."""
    a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    alpha = np.degrees(np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c + 1e-15), -1, 1)))
    beta = np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c + 1e-15), -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b + 1e-15), -1, 1)))
    return np.array([a, b, c, alpha, beta, gamma], dtype=float)


def strain_from_cell_deformation(cell0: np.ndarray, cell1: np.ndarray) -> np.ndarray:
    """
    Green-Lagrange strain from cell0 to cell1.
    Uses F = cell1 @ inv(cell0) so that strain transforms correctly under rotation:
    if both cells are rotated by R (cell' = R @ cell), then E' = R E R^T and the
    work σ : ε is frame-invariant (energy conserved under rotation).
    With rows = A,B,C we have r_cart = cell @ r_frac (column), so F = cell1 @ inv(cell0).
    E = 0.5 * (F^T F - I).
    Returns 6-vector Voigt: [ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy] (engineering shear = 2*ε_ij).
    Components are in the Cartesian frame in which cell0, cell1 are given.
    """
    F = cell1 @ np.linalg.inv(cell0)
    E = 0.5 * (F.T @ F - np.eye(3))
    # Voigt: xx, yy, zz, yz, xz, xy; engineering shear γ = 2ε
    return np.array([
        E[0, 0], E[1, 1], E[2, 2],
        2 * E[1, 2], 2 * E[0, 2], 2 * E[0, 1],
    ], dtype=float)


def compute_grid_bounds(
    eig_current: np.ndarray,
    eig_goal: np.ndarray,
    d_sigma: np.ndarray,
    deltas: np.ndarray,
    q0: np.ndarray,
    bounds_scale: float = 1.0,
) -> np.ndarray:
    """
    Determine max |x_i| for each parameter (grid extent = 3*dq in each direction).

    For each param i: max stress change from linear response = max_j |d_sigma[j,i]|.
    Eigenvalue change required = max_k |eig_goal[k] - eig_current[k]|.
    Param change to achieve that stress change: dq_i such that d_sigma * dq_i ~ delta_eig.
    Use largest |d_sigma| for that param. Triple the result for grid extent.
    bounds_scale: multiply bounds by this (e.g. 1.3 for 30% increase).
    """
    delta_eig_max = float(np.max(np.abs(eig_goal - eig_current)))
    if delta_eig_max < 1e-6:
        delta_eig_max = 100.0  # fallback bar

    bounds = np.zeros(6, dtype=float)
    for i in range(6):
        # Max absolute stress change per unit x from param i (linear part)
        max_dsigma = float(np.max(np.abs(d_sigma[:, i])))
        if max_dsigma < 1e-10:
            bounds[i] = 80.0  # default
        else:
            # x_i change to alter stress by delta_eig_max
            x_for_eig = delta_eig_max / max_dsigma
            bounds[i] = 3.0 * bounds_scale * x_for_eig  # 3x in x-space, scaled by bounds_scale
        bounds[i] = min(bounds[i], 100.0)  # cap
    return bounds


def predict_sigma(x: np.ndarray) -> np.ndarray:
    sigma_base, _target, d_single, c_pairs, _deltas, _q0 = get_surrogate_data()
    s = sigma_base + d_single @ x
    for (i, j), c in c_pairs.items():
        s = s + c * x[i] * x[j]
    return s


def predict_eigs(x: np.ndarray) -> np.ndarray:
    mat = vec6_to_mat3(predict_sigma(x))
    vals, _ = principal_sorted(mat)
    return vals


def x_to_q(x: np.ndarray) -> np.ndarray:
    """Convert x (displacement) to abc/angles q = q0 + x * deltas."""
    _, _, _, _, deltas, q0 = get_surrogate_data()
    return q0 + x * deltas


def cell_at_x(x: np.ndarray) -> np.ndarray:
    """Cell at state x (standard lower-triangular convention)."""
    q = x_to_q(x)
    cell_standard = abc_to_cell(q[0], q[1], q[2], q[3], q[4], q[5])
    return cell_standard


def cell_volume(cell: np.ndarray) -> float:
    return float(np.abs(np.linalg.det(cell)))


def strain_work(x_prev: np.ndarray, x_curr: np.ndarray) -> float:
    """
    Strain work (energy density) from state x_prev to x_curr.
    Literature: W = ∫ σ : dε. For linear path (trapezoidal rule):
    W = (1/2)(σ_0 + σ_1) : (ε_1 - ε_0)  [correct for linear elasticity]
    NOT (1/2) Δσ : Δε which is incorrect.
    Voigt: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy] · [ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy]
    Units: bar × dimensionless = bar (energy per unit volume).
    Returns strain work density in bar.

    Convention: σ : ε is computed as the Frobenius product of the 3x3 tensors
    (stress_strain_double_contraction), so the scalar is frame-invariant—same
    value in any orthonormal basis. Thus the work (and the enthalpy-like state
    variable) does not depend on the choice of coordinate frame. Stress and
    strain are taken in the same frame (job/cell frame) so the pair is
    work-conjugate; the resulting scalar is the physical work.
    """
    cell_prev = cell_at_x(x_prev)
    cell_curr = cell_at_x(x_curr)
    delta_eps = strain_from_cell_deformation(cell_prev, cell_curr)
    sigma_prev = predict_sigma(x_prev)
    sigma_curr = predict_sigma(x_curr)
    sigma_avg = 0.5 * (sigma_prev + sigma_curr)
    return stress_strain_double_contraction(sigma_avg, delta_eps)


def strain_work_total_ha(x_prev: np.ndarray, x_curr: np.ndarray) -> float:
    """
    Total strain work in Ha (same units as internal energy E).
    W_density from strain_work is in bar (energy/volume); multiply by V [Å³] and
    BAR_ANG3_TO_HA to get energy in Ha. Uses current-state volume V = det(cell_curr).
    """
    cell_curr = cell_at_x(x_curr)
    V = cell_volume(cell_curr)
    sw_bar = strain_work(x_prev, x_curr)
    return sw_bar * V * BAR_ANG3_TO_HA


def enthalpy_like(x_curr: np.ndarray, x_prev: np.ndarray) -> float:
    """
    Enthalpy-like relative to x_prev: ΔE + strain_work [Ha].
    ΔE = E(x_curr) - E(x_prev), with E = H - P*V from parameterization (Ha).
    Strain work = (1/2)(σ_0 + σ_1) : Δε × V × BAR_ANG3_TO_HA (density in bar, × volume in Å³, → Ha).
    All terms in Ha so the sum is dimensionally consistent with internal energy.

    Crystal rotation: The enthalpy-like free energy does NOT include crystal orientation
    as a degree of freedom. The state is only x → (a,b,c,α,β,γ). The strain-work term
    σ : Δε is computed as a frame-invariant scalar (Frobenius product), so the
    enthalpy-like value does not depend on the choice of coordinate frame—it is
    the same physical quantity regardless of which orthonormal basis is used.
    """
    dE = predict_E(x_curr) - predict_E(x_prev)
    sw_ha = strain_work_total_ha(x_prev, x_curr)
    return dE + sw_ha


def _load_cp2k_parameter_trend(path: Path, cycle: int = 1) -> None:
    """Load CP2K enthalpy polynomial, q0/deltas, and stress surrogate from parameter_trend_matrices.txt."""
    global _H_data, _trend_q0, _trend_deltas, _E0, _P_iso
    global _SIGMA0, _DSINGLE, _CPAIRS

    from parameter_trend_io import load_cp2k_parameter_trend

    q0, deltas, sigma0, _target, d_single, c_pairs, e0, h0, p_iso, dH, cH = load_cp2k_parameter_trend(
        path, cycle=cycle
    )
    _trend_q0 = q0
    _trend_deltas = deltas
    _H_data = (h0, dH, cH)
    _E0 = e0
    _P_iso = p_iso
    _SIGMA0 = sigma0
    _DSINGLE = d_single
    _CPAIRS = c_pairs


def predict_E(x: np.ndarray) -> float:
    """Internal energy E [Ha] from CP2K enthalpy polynomial: E = H(x) - P(x)*V(x)."""
    if _H_data is None:
        get_surrogate_data()
    H0, dH, cH = _H_data
    h = H0 + float(np.dot(dH, x))
    for (i, j), c in cH.items():
        h += c * x[i] * x[j]
    p = float(np.trace(vec6_to_mat3(predict_sigma(x)))) / 3.0
    v = cell_volume(cell_at_x(x))
    return h - p * v * BAR_ANG3_TO_HA


def get_surrogate_data():
    global _SIGMA0, _DSINGLE, _CPAIRS
    if _trend_q0 is None or _trend_deltas is None or _H_data is None or _SIGMA0 is None:
        path = _default_cp2k_param_path()
        if not path.exists():
            raise FileNotFoundError(
                f"CP2K parameter file not found at {path}. Set CP2K_PARAM_TREND_PATH to your "
                "uniax_manual/.../parameter_trend_matrices.txt"
            )
        _load_cp2k_parameter_trend(path)

    sigma_base = _SIGMA0.copy()
    target = sigma_base.copy()
    return sigma_base, target, _DSINGLE, _CPAIRS, _trend_deltas, _trend_q0


@dataclass
class OptimizerConfig:
    delta_p_gpa: float = 0.3  # first step: 0.3 GPa away from hydrostatic
    coarse_tol_frac: float = 0.40  # 40% of goal change
    fine_tol_frac: float = 0.20  # 20% for dense grid
    trace_filter_factor: float = 3.0  # reject if |trace - trace_target| > 3 * eig_tol
    eig_force_q1_frac: float = 0.10  # q1 = min(10% of eig change, 0.05 GPa)
    eig_force_q2_factor: float = 2.0  # q2 = 2*q1
    eig_force_max_factor: float = 1.3  # max eig force = 1.3 * enthalpy gradient
    grad_fd_scale: float = 1e-5  # 0.001% of length for 1 GPa
    uniq_eigvec_deg: float = 15.0
    max_grid_points: int = 500_000  # safety limit
    n_fine_per_axis: int = 6  # 6^6 dense grid at each fine-grid center
    max_coarse_for_dense: Optional[int] = None  # None = all coarse matches get dense grid
    bounds_scale: float = 1.0  # grid bounds multiplier
    conv_dH_meV: float = 0.1  # convergence: stop when |ΔH| < 0.1 meV (enthalpy-like, in Ha)
    # Optional: for dual-grid runs, exclude coarse/fine points inside this bounds box (|x_i| <= exclude_bounds_i)
    exclude_bounds: Optional[np.ndarray] = None


def make_target_uniaxial(p_iso_bar: float, delta_p_gpa: float) -> np.ndarray:
    """
    Uniaxial: one axis +2x, others -x, trace unchanged.
    deltaP = lam3 - lam1 = 3x => x = deltaP/3.
    lam1 = lam2 = P_iso - deltaP/3, lam3 = P_iso + 2*deltaP/3
    """
    delta_p_bar = delta_p_gpa * GPA_TO_BAR
    x = delta_p_bar / 3.0
    lam1 = lam2 = p_iso_bar - x
    lam3 = p_iso_bar + 2 * x
    # Return stress tensor (diagonal in principal frame - we use eigenvalues)
    return np.array([lam1, lam2, lam3], dtype=float)


def run_coarse_grid(
    eig_initial: np.ndarray,
    eig_goal: np.ndarray,
    bounds: np.ndarray,
    n_per_axis: int,
    config: OptimizerConfig,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Coarse grid search with trace pre-filter (regular grid).
    Returns list of (x, eig_pred, eig_resid) for points within coarse_tol_frac.
    """
    eig_tol = config.coarse_tol_frac * float(np.max(np.abs(eig_goal - eig_initial)))
    trace_target = np.sum(eig_goal)
    trace_tol = config.trace_filter_factor * eig_tol  # trace within 3x eig error

    axes = []
    for i in range(6):
        ax = np.linspace(-bounds[i], bounds[i], n_per_axis)
        axes.append(ax)

    mesh = np.array(np.meshgrid(*axes, indexing="ij")).reshape(6, -1).T
    return _coarse_grid_loop(mesh, eig_goal, eig_tol, trace_target, trace_tol)


def run_coarse_grid_random(
    eig_initial: np.ndarray,
    eig_goal: np.ndarray,
    bounds: np.ndarray,
    n_sample: int,
    config: OptimizerConfig,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Coarse random grid with trace pre-filter."""
    eig_tol = config.coarse_tol_frac * float(np.max(np.abs(eig_goal - eig_initial)))
    trace_target = np.sum(eig_goal)
    trace_tol = config.trace_filter_factor * eig_tol

    rng = np.random.default_rng(42)
    mesh = rng.uniform(-bounds, bounds, (n_sample, 6))
    return _coarse_grid_loop(mesh, eig_goal, eig_tol, trace_target, trace_tol)


def _coarse_grid_loop(
    mesh: np.ndarray,
    eig_goal: np.ndarray,
    eig_tol: float,
    trace_target: float,
    trace_tol: float,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for x in mesh:
        s = predict_sigma(x)
        trace_pred = s[0] + s[1] + s[2]
        if np.abs(trace_pred - trace_target) > trace_tol:
            continue
        eig_pred = predict_eigs(x)
        resid = eig_goal - eig_pred
        if np.max(np.abs(resid)) <= eig_tol:
            matches.append((x.copy(), eig_pred.copy(), resid.copy()))
    return matches


def coarse_boundary_adjacent_stats(
    coarse_matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    bounds: np.ndarray,
    n_per: int,
) -> Tuple[int, int, int]:
    """
    Returns (n_at_boundary, n_with_adjacent_agreeing_nonboundary, n_with_better_at_neighbor).
    - At boundary: any coordinate at grid edge (index 0 or n_per-1).
    - Adjacent agreeing non-boundary: has a matching neighbor (diff ±1 in one dim) that is not at boundary.
    - Better at neighbor: has a matching neighbor with lower resid_max.
    """
    if not coarse_matches:
        return 0, 0, 0
    step = 2.0 * bounds / max(n_per - 1, 1)
    idx_to_match: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for x, eig, resid in coarse_matches:
        idx = tuple(int(round((x[i] + bounds[i]) / (step[i] + 1e-15))) for i in range(6))
        idx_to_match[idx] = (x, eig, resid)

    n_at_boundary = 0
    n_adjacent_agreeing_nonboundary = 0
    n_better_at_neighbor = 0

    for idx, (x, eig, resid) in idx_to_match.items():
        resid_max = float(np.max(np.abs(resid)))
        at_boundary = any(idx[i] == 0 or idx[i] == n_per - 1 for i in range(6))
        if at_boundary:
            n_at_boundary += 1

        has_adjacent_nonboundary = False
        has_better_neighbor = False
        for d in range(6):
            for delta in (-1, 1):
                ni = list(idx)
                ni[d] = idx[d] + delta
                if ni[d] < 0 or ni[d] >= n_per:
                    continue
                nidx = tuple(ni)
                if nidx in idx_to_match:
                    neighbor_at_boundary = any(ni[i] == 0 or ni[i] == n_per - 1 for i in range(6))
                    if not neighbor_at_boundary:
                        has_adjacent_nonboundary = True
                    n_resid = np.max(np.abs(idx_to_match[nidx][2]))
                    if n_resid < resid_max:
                        has_better_neighbor = True

        if has_adjacent_nonboundary:
            n_adjacent_agreeing_nonboundary += 1
        if has_better_neighbor:
            n_better_at_neighbor += 1

    return n_at_boundary, n_adjacent_agreeing_nonboundary, n_better_at_neighbor


def count_coarse_no_bordering_match(
    coarse_matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    bounds: np.ndarray,
    n_per: int,
    eigvec_deg: float = 30.0,
) -> int:
    """
    Count matches with no bordering stress-matched site. Uses Moore neighborhood.
    Neighbors with principal eigenvector > eigvec_deg apart are NOT considered
    bordering (different stress tensor character).
    """
    if not coarse_matches:
        return 0
    step = 2.0 * bounds / max(n_per - 1, 1)
    idx_to_match: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for x, eig, resid in coarse_matches:
        idx = tuple(int(round((x[i] + bounds[i]) / (step[i] + 1e-15))) for i in range(6))
        mat = vec6_to_mat3(predict_sigma(x))
        _, eigvecs = principal_sorted(mat)
        vec = eigvecs[:, 2]
        idx_to_match[idx] = (x, eig, resid, vec)

    n_isolated = 0
    for idx, (x, eig, resid, vec) in idx_to_match.items():
        has_neighbor = False
        for d0 in (-1, 0, 1):
            for d1 in (-1, 0, 1):
                for d2 in (-1, 0, 1):
                    for d3 in (-1, 0, 1):
                        for d4 in (-1, 0, 1):
                            for d5 in (-1, 0, 1):
                                if d0 == d1 == d2 == d3 == d4 == d5 == 0:
                                    continue
                                nidx = (idx[0] + d0, idx[1] + d1, idx[2] + d2,
                                        idx[3] + d3, idx[4] + d4, idx[5] + d5)
                                if all(0 <= nidx[i] < n_per for i in range(6)) and nidx in idx_to_match:
                                    _, _, _, vecj = idx_to_match[nidx]
                                    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(vec, vecj)), 0, 1)))
                                    if angle <= eigvec_deg:
                                        has_neighbor = True
                                        break
                            if has_neighbor:
                                break
                        if has_neighbor:
                            break
                    if has_neighbor:
                        break
                if has_neighbor:
                    break
            if has_neighbor:
                break
        if not has_neighbor:
            n_isolated += 1
    return n_isolated


def _boundary_indices(
    matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    bounds: np.ndarray,
    n_per: int,
) -> set:
    """Return set of match indices that are at grid boundary."""
    if not matches:
        return set()
    step = 2.0 * bounds / max(n_per - 1, 1)
    boundary = set()
    for i, (x, _, _) in enumerate(matches):
        idx = tuple(int(round((x[j] + bounds[j]) / (step[j] + 1e-15))) for j in range(6))
        if any(idx[k] == 0 or idx[k] == n_per - 1 for k in range(6)):
            boundary.add(i)
    return boundary


def count_principal_eigvec_isolated(
    matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    deg_threshold: float = 30.0,
    bounds: Optional[np.ndarray] = None,
    n_per: Optional[int] = None,
) -> int:
    """
    Count matches whose principal eigenvector is > deg_threshold from all others.
    If bounds and n_per provided, compare only to non-boundary matches (exclude boundary from comparison set).
    Uses |dot| so v and -v are equivalent.
    """
    if len(matches) <= 1:
        return len(matches)
    exclude = set()
    if bounds is not None and n_per is not None:
        exclude = _boundary_indices(matches, bounds, n_per)
    vecs = []
    for x, _, _ in matches:
        mat = vec6_to_mat3(predict_sigma(x))
        _, eigvecs = principal_sorted(mat)
        vecs.append(eigvecs[:, 2])  # principal (max eigenvalue) axis
    n_isolated = 0
    for i in range(len(vecs)):
        min_angle = 180.0
        others = [j for j in range(len(vecs)) if j != i and j not in exclude]
        if not others:
            continue  # no non-boundary others to compare to; skip
        for j in others:
            angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(vecs[i], vecs[j])), 0, 1)))
            min_angle = min(min_angle, angle)
        if min_angle > deg_threshold:
            n_isolated += 1
    return n_isolated


def run_dense_grid(
    coarse_matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    eig_initial: np.ndarray,
    eig_goal: np.ndarray,
    config: OptimizerConfig,
    n_fine_per_axis: int = 6,
    coarse_bounds: Optional[np.ndarray] = None,
    coarse_n_per: int = 13,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Dense 6^6 grid at EACH coarse match. Extent = out to nearest coarse grid points.
    Tolerance 20%. Returns (x, eig_pred, eig_resid, enthalpy_like) for matches.
    """
    if not coarse_matches:
        return []

    eig_tol = config.fine_tol_frac * float(np.max(np.abs(eig_goal - eig_initial)))
    trace_target = np.sum(eig_goal)
    trace_tol = config.trace_filter_factor * eig_tol
    x_prev = np.zeros(6)  # initial state

    # Coarse grid step = extent to nearest coarse grid point
    if coarse_bounds is not None:
        step = 2.0 * coarse_bounds / max(coarse_n_per - 1, 1)
    else:
        xs = np.array([m[0] for m in coarse_matches])
        span = np.maximum(np.max(xs, axis=0) - np.min(xs, axis=0), 0.5)
        step = span / 4  # fallback

    matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for x_c, _, _ in coarse_matches:
        # 6^6 grid centered at x_c, extent ±step/2 in each dimension
        axes = [
            np.linspace(x_c[i] - step[i] / 2, x_c[i] + step[i] / 2, n_fine_per_axis)
            for i in range(6)
        ]
        mesh = np.array(np.meshgrid(*axes, indexing="ij")).reshape(6, -1).T
        for x in mesh:
            s = predict_sigma(x)
            if np.abs(s[0] + s[1] + s[2] - trace_target) > trace_tol:
                continue
            eig_pred = predict_eigs(x)
            resid = eig_goal - eig_pred
            if np.max(np.abs(resid)) <= eig_tol:
                H = enthalpy_like(x, x_prev)
                matches.append((x.copy(), eig_pred.copy(), resid.copy(), H))

    return matches


def find_local_minima_simple(
    dense_matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    radius: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Local minima: no other matching point within radius has lower enthalpy-like.
    """
    if len(dense_matches) <= 1:
        return list(dense_matches)

    minima = []
    for i, (x, eig, resid, H) in enumerate(dense_matches):
        is_min = True
        for j, (xj, _, _, Hj) in enumerate(dense_matches):
            if j == i:
                continue
            if np.linalg.norm(x - xj) <= radius and Hj < H:
                is_min = False
                break
        if is_min:
            minima.append((x, eig, resid, H))

    return minima


def find_local_minima_grid(
    dense_matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    x_center: np.ndarray,
    step: np.ndarray,
    n_fine: int = 6,
    eigvec_deg: float = 30.0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Local minima using grid connectivity: neighbors differ by at most 1 step
    in each axis. Neighbors with principal eigenvector > eigvec_deg apart are
    NOT considered (different stress tensor character).
    """
    if len(dense_matches) <= 1:
        return list(dense_matches)

    fine_step = step / max(n_fine - 1, 1)
    x_min = x_center - step / 2

    def x_to_idx(x: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(round((x[i] - x_min[i]) / (fine_step[i] + 1e-15))) for i in range(6))

    idx_to_match: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = {}
    for x, eig, resid, H in dense_matches:
        idx = x_to_idx(x)
        mat = vec6_to_mat3(predict_sigma(x))
        _, eigvecs = principal_sorted(mat)
        vec = eigvecs[:, 2]
        idx_to_match[idx] = (x, eig, resid, H, vec)

    minima = []
    for idx, (x, eig, resid, H, vec) in idx_to_match.items():
        is_min = True
        for d0 in (-1, 0, 1):
            for d1 in (-1, 0, 1):
                for d2 in (-1, 0, 1):
                    for d3 in (-1, 0, 1):
                        for d4 in (-1, 0, 1):
                            for d5 in (-1, 0, 1):
                                if d0 == d1 == d2 == d3 == d4 == d5 == 0:
                                    continue
                                nidx = (idx[0] + d0, idx[1] + d1, idx[2] + d2,
                                        idx[3] + d3, idx[4] + d4, idx[5] + d5)
                                if nidx in idx_to_match:
                                    _, _, _, Hj, vecj = idx_to_match[nidx]
                                    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(vec, vecj)), 0, 1)))
                                    if angle <= eigvec_deg and Hj < H:
                                        is_min = False
                                        break
                            if not is_min:
                                break
                        if not is_min:
                            break
                    if not is_min:
                        break
                if not is_min:
                    break
            if not is_min:
                break
        if is_min:
            minima.append((x, eig, resid, H))

    return minima


def coarse_grid_index(x: np.ndarray, bounds: np.ndarray, n_per: int) -> Tuple[int, ...]:
    """Grid index of x for the regular coarse grid (same convention as find_local_minima_coarse_grid)."""
    step = 2.0 * bounds / max(n_per - 1, 1)
    x_min = -bounds
    return tuple(int(round((x[i] - x_min[i]) / (step[i] + 1e-15))) for i in range(6))


def find_local_minima_coarse_grid(
    coarse_matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    bounds: np.ndarray,
    n_per: int,
    x_prev: np.ndarray,
    eigvec_deg: float = 30.0,
    cost_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float], float]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Local minima on coarse grid by cost (default: enthalpy-like H). Neighbors = grid neighbors (max 1 step/axis).
    Points not meeting the match threshold are not in the set, so treated as higher cost (boundary).
    Neighbors with principal eigenvector > eigvec_deg apart are NOT considered (different stress tensor character).
    cost_fn(x, eig, resid, H) -> float; if None, use H.
    Returns (x, eig, resid, H) for coarse matches that are local minima in the chosen cost.
    """
    if not coarse_matches:
        return []
    step = 2.0 * bounds / max(n_per - 1, 1)
    x_min = -bounds

    def x_to_idx(x: np.ndarray) -> Tuple[int, ...]:
        return coarse_grid_index(x, bounds, n_per)

    # Build idx -> (x, eig, resid, H, vec, cost)
    idx_to_match: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]] = {}
    for x, eig, resid in coarse_matches:
        H = enthalpy_like(x, x_prev)
        cost = cost_fn(x, eig, resid, H) if cost_fn is not None else H
        idx = x_to_idx(x)
        mat = vec6_to_mat3(predict_sigma(x))
        _, eigvecs = principal_sorted(mat)
        vec = eigvecs[:, 2]
        idx_to_match[idx] = (x, eig, resid, H, vec, cost)

    minima = []
    for idx, (x, eig, resid, H, vec, cost) in idx_to_match.items():
        is_min = True
        for d0 in (-1, 0, 1):
            for d1 in (-1, 0, 1):
                for d2 in (-1, 0, 1):
                    for d3 in (-1, 0, 1):
                        for d4 in (-1, 0, 1):
                            for d5 in (-1, 0, 1):
                                if d0 == d1 == d2 == d3 == d4 == d5 == 0:
                                    continue
                                nidx = (idx[0] + d0, idx[1] + d1, idx[2] + d2,
                                        idx[3] + d3, idx[4] + d4, idx[5] + d5)
                                if nidx in idx_to_match:
                                    _, _, _, Hj, vecj, costj = idx_to_match[nidx]
                                    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(vec, vecj)), 0, 1)))
                                    if angle <= eigvec_deg and costj < cost:
                                        is_min = False
                                        break
                            if not is_min:
                                break
                        if not is_min:
                            break
                    if not is_min:
                        break
                if not is_min:
                    break
            if not is_min:
                break
        if is_min:
            minima.append((x, eig, resid, H))

    return minima


def hydroopt_max_step_per_01_gpa(p_iso_gpa: float, q0: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """
    Max parameter change (in x-space) per 0.1 GPa from hydroopt.
    Uses 14/15 GPa cells as proxy; scale to current pressure if needed.
    """
    # Use hydrostatic cells at 14 and 15 GPa (surrogate is ~15 GPa)
    cell_14 = np.array([
        [4.8309359310, -0.0000000016, 0.3347874435],
        [0.0000000170, 4.8525084690, 0.0000000001],
        [-2.0780415830, -0.0000000129, 6.2030249320],
    ])
    cell_15 = np.array([
        [4.8150567070, -0.0000000026, 0.3404446083],
        [0.0000000153, 4.8513009610, 0.0000000021],
        [-2.0606332550, -0.0000000112, 6.1667503390],
    ])
    q14 = cell_to_abc(cell_14)
    q15 = cell_to_abc(cell_15)
    dq_per_gpa = np.abs(q15 - q14)
    dq_per_01_gpa = 0.1 * dq_per_gpa
    # Convert to x-space: dx = dq / deltas
    dx_max = dq_per_01_gpa / deltas
    return np.maximum(dx_max, 1e-6)


def grad_enthalpy_like(x: np.ndarray, x_prev: np.ndarray, fd_eps: float = 1e-8) -> np.ndarray:
    """Gradient of enthalpy-like w.r.t. x via finite difference."""
    g = np.zeros(6)
    for i in range(6):
        xp, xm = x.copy(), x.copy()
        xp[i] += fd_eps
        xm[i] -= fd_eps
        g[i] = (enthalpy_like(xp, x_prev) - enthalpy_like(xm, x_prev)) / (2 * fd_eps)
    return g


def grad_eig_mismatch(x: np.ndarray, eig_goal: np.ndarray, fd_step: float) -> np.ndarray:
    """
    Gradient of cost = sum_k (eig_goal[k]-eig_pred[k])^2.
    grad_i = -2 * sum_k (eig_goal[k]-eig_pred[k]) * d(eig_pred[k])/dx_i
    """
    fd_step = max(fd_step, 1e-8)
    eig0 = predict_eigs(x)
    d_eig_dx = np.zeros((3, 6))
    for i in range(6):
        xp, xm = x.copy(), x.copy()
        xp[i] += fd_step
        xm[i] -= fd_step
        ep, em = predict_eigs(xp), predict_eigs(xm)
        d_eig_dx[:, i] = (ep - em) / (2 * fd_step)
    resid = eig_goal - eig0
    grad = -2.0 * resid @ d_eig_dx
    return grad


def eigenvalue_forces(
    x: np.ndarray,
    eig_goal: np.ndarray,
    q1_bar: float,
    q2_bar: float,
    max_force_per_param: np.ndarray,
    fd_step: float,
) -> np.ndarray:
    """
    Eigenvalue forces (descent direction): zero for |resid| < q1; linear q1->q2; max at q2.
    Max force per param = 1.3 * |enthalpy gradient|.
    grad_eig = gradient of cost (points up); force = -grad for descent.
    """
    eig_pred = predict_eigs(x)
    resid = eig_goal - eig_pred
    g_raw = grad_eig_mismatch(x, eig_goal, fd_step)

    # Scale: when all |resid| < q1, force=0. Otherwise scale by max residual
    max_resid = float(np.max(np.abs(resid)))
    if max_resid < q1_bar:
        scale = 0.0
    elif max_resid < q2_bar:
        scale = (max_resid - q1_bar) / (q2_bar - q1_bar)
    else:
        scale = 1.0

    forces = -scale * g_raw  # descent = -grad
    for i in range(6):
        forces[i] = np.clip(forces[i], -max_force_per_param[i], max_force_per_param[i])
    return forces


def gradient_descent_step(
    x: np.ndarray,
    x_prev: np.ndarray,
    eig_goal: np.ndarray,
    config: OptimizerConfig,
    dx_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """One gradient descent step. Returns (x_new, total_force, step_was_limited).
    step_was_limited=True if the step was clipped by dx_max (do not treat as converged).
    """
    delta_eig = np.max(np.abs(eig_goal - predict_eigs(np.zeros(6))))
    q1_bar = min(0.10 * delta_eig, 0.05 * GPA_TO_BAR)
    q2_bar = config.eig_force_q2_factor * q1_bar

    g_H = grad_enthalpy_like(x, x_prev)
    max_force = config.eig_force_max_factor * np.abs(g_H)
    max_force = np.maximum(max_force, 1e-10)

    # fd_step: 0.001% of length for 1 GPa. Length ~5 Ang, 1 GPa. So 5e-5 Ang. deltas=0.005 => dx~0.01
    fd_step = max(config.grad_fd_scale * delta_eig / 1000.0, 1e-8)
    f_eig = eigenvalue_forces(x, eig_goal, q1_bar, q2_bar, max_force, fd_step)

    # Total descent direction: -grad_H + f_eig (f_eig already descent)
    total_force = -g_H + f_eig

    # Step size: scale so no param exceeds dx_max
    step = total_force.copy()
    step_norm = np.linalg.norm(step)
    if step_norm < 1e-15:
        return x.copy(), total_force, False
    scale = 1.0
    for i in range(6):
        if abs(step[i]) > dx_max[i]:
            scale = min(scale, dx_max[i] / (abs(step[i]) + 1e-15))
    step_was_limited = scale < 0.9999  # step was clipped by dx_max
    step *= scale
    return x + step, total_force, step_was_limited


def run_gradient_descent_until_converged(
    x_start: np.ndarray,
    x_prev: np.ndarray,
    eig_goal: np.ndarray,
    config: OptimizerConfig,
    q0: np.ndarray,
    deltas: np.ndarray,
    max_steps: int = 500,
) -> Tuple[np.ndarray, int, float]:
    """
    Run gradient descent from x_start until |ΔH| < config.conv_dH_meV (in meV, converted to Ha).
    Does not count as converged if the step was limited by dx_max (must converge by small ΔH
    with an unclipped step). Returns (x_converged, n_steps, H_final).
    """
    conv_ha = config.conv_dH_meV * MEV_TO_HA  # 0.1 meV in Ha
    p_iso_gpa = float(np.mean(predict_eigs(np.zeros(6)))) / GPA_TO_BAR
    dx_max = hydroopt_max_step_per_01_gpa(p_iso_gpa, q0, deltas)

    x = x_start.copy()
    H_old = enthalpy_like(x, x_prev)
    for step in range(max_steps):
        x_new, _, step_was_limited = gradient_descent_step(x, x_prev, eig_goal, config, dx_max)
        H_new = enthalpy_like(x_new, x_prev)
        dH_small = np.abs(H_new - H_old) < conv_ha
        if dH_small and not step_was_limited:
            return x_new, step + 1, float(H_new)
        x = x_new
        H_old = H_new
    return x, max_steps, float(H_old)


def principal_vec_at_x(x: np.ndarray) -> np.ndarray:
    """Principal (max eigenvalue) eigenvector of stress at x."""
    mat = vec6_to_mat3(predict_sigma(x))
    _, eigvecs = principal_sorted(mat)
    return eigvecs[:, 2].copy()


def angle_between_principal_vecs_deg(x1: np.ndarray, x2: np.ndarray) -> float:
    """Angle in degrees between principal stress eigenvectors at x1 and x2 (0-90, |dot|)."""
    v1 = principal_vec_at_x(x1)
    v2 = principal_vec_at_x(x2)
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0, 1))))


def unique_minima_by_eigenvector(
    minima: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    deg_threshold: float = 15.0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    Group by eigenvector (within 15 deg). Label by min enthalpy in group.
    Returns (x, eig, resid, H, eigvec_principal) per unique group.
    """
    if not minima:
        return []

    sorted_m = sorted(minima, key=lambda m: m[3])
    result = []
    used = [False] * len(sorted_m)

    for i, (x, eig, resid, H) in enumerate(sorted_m):
        if used[i]:
            continue
        mat = vec6_to_mat3(predict_sigma(x))
        _, vecs = principal_sorted(mat)
        vec_principal = vecs[:, 2]  # principal (max eigenvalue) axis

        group = [(x, eig, resid, H, vec_principal)]
        used[i] = True

        for j, (xj, eigj, residj, Hj) in enumerate(sorted_m):
            if j <= i or used[j]:
                continue
            matj = vec6_to_mat3(predict_sigma(xj))
            _, vecsj = principal_sorted(matj)
            vj = vecsj[:, 2]
            angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(vec_principal, vj)), 0, 1)))
            if angle <= deg_threshold:
                group.append((xj, eigj, residj, Hj, vj))
                used[j] = True

        # Representative: min H in group
        rep = min(group, key=lambda g: g[3])
        result.append(rep)

    return result


def group_matches_by_principal_eigvec(
    matches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    deg_threshold: float = 30.0,
) -> List[List[Tuple[int, np.ndarray, np.ndarray]]]:
    """
    Group matches by principal eigenvector (within deg_threshold).
    Returns list of groups; each group is [(idx, x, eigvec), ...].
    Uses |dot| so v and -v are equivalent.
    """
    if not matches:
        return []
    vecs = []
    for x, _, _ in matches:
        mat = vec6_to_mat3(predict_sigma(x))
        _, eigvecs = principal_sorted(mat)
        vecs.append(eigvecs[:, 2])
    used = [False] * len(matches)
    groups = []
    for i in range(len(matches)):
        if used[i]:
            continue
        group = [(i, matches[i][0], vecs[i])]
        used[i] = True
        for j in range(i + 1, len(matches)):
            if used[j]:
                continue
            angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(vecs[i], vecs[j])), 0, 1)))
            if angle <= deg_threshold:
                group.append((j, matches[j][0], vecs[j]))
                used[j] = True
        groups.append(group)
    return groups


def _run_scale_task(
    scale: float,
    eig_initial: np.ndarray,
    eig_goal: np.ndarray,
) -> Tuple[float, List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]], List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]]:
    """Run pipeline for one scale (for parallel execution). Loads trend in process. Returns (scale, converged, groups)."""
    config = replace(OptimizerConfig(), bounds_scale=scale, conv_dH_meV=0.2)
    converged, groups = _run_pipeline_for_scale(scale, config, eig_initial, eig_goal, lambda _: None)
    return (scale, converged, groups)


def _run_fine_grid_for_center(
    args: Tuple[
        np.ndarray,          # x_c
        np.ndarray,          # eig_initial
        np.ndarray,          # eig_goal
        np.ndarray,          # bounds
        int,                 # n_per
        OptimizerConfig,     # config (with bounds_scale, tolerances, etc.)
    ]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Helper for parallel fine-grid evaluation at a single center.
    Returns list of (x, eig, resid, H) local minima from the dense grid around x_c.
    """
    x_c, eig_initial, eig_goal, bounds, n_per, config = args
    c = config
    step = 2.0 * bounds / max(n_per - 1, 1)
    dense = run_dense_grid(
        [(x_c, np.zeros(3), np.zeros(3))], eig_initial, eig_goal, c,
        n_fine_per_axis=c.n_fine_per_axis, coarse_bounds=bounds, coarse_n_per=n_per,
    )
    if not dense:
        return []
    mins = find_local_minima_grid(dense, x_center=x_c, step=step, n_fine=c.n_fine_per_axis)
    return mins


def _run_pipeline_for_scale(
    scale: float,
    config: OptimizerConfig,
    eig_initial: np.ndarray,
    eig_goal: np.ndarray,
    log: Callable[[str], None],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]], List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]]:
    """Run full pipeline (coarse -> fine -> GD) for given bounds_scale. Returns (converged, groups)."""
    import time
    c = replace(config, bounds_scale=scale, conv_dH_meV=config.conv_dH_meV)
    sigma_base, target, d_single, c_pairs, deltas, q0 = get_surrogate_data()
    bounds = compute_grid_bounds(eig_initial, eig_goal, d_single, deltas, q0, c.bounds_scale)
    n_per = 13
    delta_eig = float(np.max(np.abs(eig_goal - eig_initial)))
    eig_tol_20 = c.fine_tol_frac * delta_eig
    eig_tol_40 = c.coarse_tol_frac * delta_eig
    eig_tol_10 = 0.10 * delta_eig  # 10% mismatch: force fine grid if not already a candidate

    t0 = time.perf_counter()
    coarse_matches_40 = run_coarse_grid(eig_initial, eig_goal, bounds, n_per, c)
    # Dual-grid support: optionally exclude points inside a smaller-scale bounds box
    if c.exclude_bounds is not None:
        coarse_matches_40 = [m for m in coarse_matches_40 if x_outside_bounds(m[0], c.exclude_bounds)]
    matches_20 = [m for m in coarse_matches_40 if np.max(np.abs(m[2])) <= eig_tol_20]
    matches_10 = [m for m in coarse_matches_40 if np.max(np.abs(m[2])) <= eig_tol_10]
    x_prev = np.zeros(6)

    # Local minima by enthalpy (H)
    mins_20 = find_local_minima_coarse_grid(matches_20, bounds, n_per, x_prev, eigvec_deg=30.0)
    mins_40 = find_local_minima_coarse_grid(coarse_matches_40, bounds, n_per, x_prev, eigvec_deg=30.0)

    # Second cost: H + 1 meV per 10% deviation per eigenvalue (only for identifying fine-grid candidates)
    # 10% deviation = |resid_i|/eig_tol_10; cost2 = H + MEV_TO_HA * sum_i(|resid_i|/eig_tol_10)
    def second_cost_fn(x: np.ndarray, eig: np.ndarray, resid: np.ndarray, H: float) -> float:
        return H + MEV_TO_HA * float(np.sum(np.abs(resid) / eig_tol_10))

    mins_20_C2 = find_local_minima_coarse_grid(
        matches_20, bounds, n_per, x_prev, eigvec_deg=30.0, cost_fn=second_cost_fn
    )
    mins_40_C2 = find_local_minima_coarse_grid(
        coarse_matches_40, bounds, n_per, x_prev, eigvec_deg=30.0, cost_fn=second_cost_fn
    )

    # Merge fine-grid centers: H-minima (20% then 40%), then C2-minima, then 10% points not already in
    idx_fine: set = set()
    fine_centers: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    def add_center(m: Tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> None:
        idx = coarse_grid_index(m[0], bounds, n_per)
        if idx not in idx_fine:
            idx_fine.add(idx)
            fine_centers.append(m)

    for m in mins_20:
        add_center(m)
    for m in mins_40:
        add_center(m)
    for m in mins_20_C2:
        add_center(m)
    for m in mins_40_C2:
        add_center(m)
    for m in matches_10:
        add_center((m[0], m[1], m[2], enthalpy_like(m[0], x_prev)))

    # Fine-grid evaluation: parallelize over centers to speed up the dense 6^6 grids
    step = 2.0 * bounds / max(n_per - 1, 1)
    all_fine_mins: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    if fine_centers:
        # Package arguments per center; config c is picklable (dataclass of simple types/arrays)
        fine_tasks = [(x_c, eig_initial, eig_goal, bounds, n_per, c) for x_c, _, _, _ in fine_centers]
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as ex:
            for mins in ex.map(_run_fine_grid_for_center, fine_tasks):
                if mins:
                    all_fine_mins.extend(mins)

    converged: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]] = []
    if all_fine_mins:
        for x0, eig0, resid0, H0 in all_fine_mins:
            x_fin, n_steps, H_fin = run_gradient_descent_until_converged(
                x0, x_prev, eig_goal, c, q0, deltas, max_steps=200
            )
            eig_fin = predict_eigs(x_fin)
            resid_fin = eig_goal - eig_fin
            converged.append((x_fin, eig_fin, resid_fin, H_fin, n_steps))

    minima_4 = [(x, eig, resid, H) for x, eig, resid, H, _ in converged]
    groups = unique_minima_by_eigenvector(minima_4, deg_threshold=15.0)
    return converged, groups


def _vec_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0, 1))))


def group_all_minima_by_eigenvector(
    converged_by_scale: Dict[float, List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]]],
    deg_threshold: float = 20.0,
) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]]:
    """
    Group all converged minima by principal eigvec (within deg_threshold). Processing order
    is by ascending H: the lowest-H minimum starts the first group; all minima within
    deg_threshold of its principal eigenvector are assigned to that group and removed from
    consideration; then the next lowest-H remaining starts the next group, and so on.
    Each minimum is (x, eig, resid, H, n_steps); we tag with scale.
    Returns list of groups; each group is list of (x, eig, resid, H, scale).
    """
    all_tagged: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    for scale, converged in converged_by_scale.items():
        for x, eig, resid, H, _ in converged:
            all_tagged.append((x, eig, resid, H, scale))
    if not all_tagged:
        return []

    # Sort by H ascending; cluster by principal eigenvector (lowest H first)
    all_tagged = sorted(all_tagged, key=lambda m: m[3])
    used = [False] * len(all_tagged)
    groups: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]] = []

    for i, (x, eig, resid, H, scale) in enumerate(all_tagged):
        if used[i]:
            continue
        vec = principal_vec_at_x(x)
        group = [(x, eig, resid, H, scale)]
        used[i] = True
        for j, (xj, eigj, residj, Hj, scalej) in enumerate(all_tagged):
            if j <= i or used[j]:
                continue
            vj = principal_vec_at_x(xj)
            if _vec_angle_deg(vec, vj) <= deg_threshold:
                group.append((xj, eigj, residj, Hj, scalej))
                used[j] = True
        groups.append(group)
    return groups


def subgroup_by_parameters(
    group: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]],
    param_thresholds: np.ndarray,
) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]]:
    """
    Within an eigenvector-based group, form subgroups based on parameter proximity.

    Each subgroup requires that ALL parameters for any two members differ by no more
    than param_thresholds[i] in component i. We enforce this pairwise within each
    subgroup: a new member joins a subgroup only if it is within the thresholds of
    every existing member in that subgroup.
    """
    if len(group) <= 1:
        return [group]

    # Sort by H so that each subgroup tends to be anchored at low enthalpy
    remaining = sorted(group, key=lambda m: m[3])
    subgroups: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]] = []

    for x, eig, resid, H, scale in remaining:
        placed = False
        for sg in subgroups:
            ok = True
            for xs, _, _, _, _ in sg:
                if np.any(np.abs(x - xs) > param_thresholds):
                    ok = False
                    break
            if ok:
                sg.append((x, eig, resid, H, scale))
                placed = True
                break
        if not placed:
            subgroups.append([(x, eig, resid, H, scale)])
    return subgroups


def x_outside_bounds(x: np.ndarray, bounds: np.ndarray) -> bool:
    """True if any |x[i]| > bounds[i] (point outside the grid box)."""
    return bool(np.any(np.abs(x) > bounds))


def main():
    import time
    config = OptimizerConfig(conv_dH_meV=0.2)
    sigma_base, target, d_single, c_pairs, deltas, q0 = get_surrogate_data()
    eig_initial = principal_sorted(vec6_to_mat3(sigma_base))[0]
    p_iso_bar = float(np.mean(eig_initial))
    eig_goal = make_target_uniaxial(p_iso_bar, config.delta_p_gpa)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("Initial eigenvalues [bar]: " + str(eig_initial))
    log("Target eigenvalues [bar]: " + str(eig_goal))
    log(f"Delta P [GPa]: {config.delta_p_gpa}")
    log(f"Convergence: |ΔH| < {config.conv_dH_meV} meV")

    t_main_start = time.perf_counter()
    scales_list = [1.0]
    results: Dict[float, Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]], List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]]] = {}

    log("\nRunning single-grid scale: " + str(scales_list))
    for scale in scales_list:
        log(f"\nBOUNDS_SCALE = {scale} ...")
        t0 = time.perf_counter()
        c = replace(config, bounds_scale=scale, conv_dH_meV=0.2, exclude_bounds=None)
        converged, groups = _run_pipeline_for_scale(scale, c, eig_initial, eig_goal, log)
        results[scale] = (converged, groups)
        log(f"  Scale {scale}: {len(converged)} converged, {len(groups)} groups in {time.perf_counter()-t0:.1f} s")

    for scale in scales_list:
        converged, groups = results[scale]
        log(f"\n--- Scale {scale} --- Groups ({len(groups)}):")
        for i, (x, eig, resid, H, vec) in enumerate(sorted(groups, key=lambda g: g[3])):
            log(f"  group {i+1}: H={H:.6e}, resid_max={np.max(np.abs(resid)):.1f} bar")

    # Write all minima to JSON (one file per scale) for later testing/characterization.
    # For dual-grid runs (0.7 and 1.3 only), add a suffix so we don't overwrite earlier results.
    out_dir = Path(__file__).resolve().parent
    is_dual_grid = False
    suffix = ""
    for scale in scales_list:
        converged = results[scale][0]
        minima_data = []
        for x, eig, resid, H, n_steps in converged:
            vec = principal_vec_at_x(x).tolist()
            minima_data.append({
                "x": x.tolist(),
                "eig": eig.tolist(),
                "resid": resid.tolist(),
                "H": float(H),
                "n_steps": int(n_steps),
                "resid_max": float(np.max(np.abs(resid))),
                "principal_vec": vec,
            })
        out_path = out_dir / f"minima_scale_{scale}{suffix}.json"
        with open(out_path, "w") as f:
            json.dump({"scale": scale, "minima": minima_data}, f, indent=1)
        log(f"  Wrote {len(minima_data)} minima to {out_path}")

    # Unified grouping across all scales (15°)
    converged_by_scale = {s: results[s][0] for s in scales_list}
    # Group by principal eigenvector (acceptance angle 20°); order is by H ascending so
    # each group is anchored by the lowest-H member among those not yet assigned.
    unified_groups = group_all_minima_by_eigenvector(converged_by_scale, deg_threshold=20.0)
    log("\n" + "=" * 60)
    log("UNIFIED GROUPS (all scales merged by 15° eigvec)")
    log("=" * 60)
    log(f"Total unified groups: {len(unified_groups)}")

    # Subgrouping within each eigenvector group based on parameter changes.
    # Use the same param scaling as grid-bounds computation: for each param i,
    # x_for_eig[i] = delta_eig_max / max|d_sigma[:,i]|, where delta_eig_max
    # is the largest eigenvalue change over the three principal stresses.
    sigma_base, target, d_single, c_pairs, deltas, q0 = get_surrogate_data()
    eig_current = eig_initial
    delta_eig_max = float(np.max(np.abs(eig_goal - eig_current)))
    if delta_eig_max < 1e-6:
        delta_eig_max = 100.0
    x_for_eig = np.zeros(6, dtype=float)
    for i in range(6):
        max_dsigma = float(np.max(np.abs(d_single[:, i])))
        if max_dsigma < 1e-10:
            x_for_eig[i] = 80.0
        else:
            x_for_eig[i] = delta_eig_max / max_dsigma

    # For enthalpy comparison we need the global best H over all groups
    global_min_H_local = min(m[3] for g in unified_groups for m in g)

    log("\nParameter-based subgroups within eigenvector groups (different thresholds as fractions of x_for_eig):")
    for frac in (1.0 / 3.0, 2.0 / 3.0, 1.0):
        param_thresholds = frac * x_for_eig
        groups_with_subgroups = 0
        total_subgroups = 0
        enthalpy_deltas_meV: List[float] = []
        subgroups_within_10_meV: int = 0
        for gidx, group in enumerate(unified_groups):
            subgs = subgroup_by_parameters(group, param_thresholds)
            if len(subgs) > 1:
                groups_with_subgroups += 1
                total_subgroups += len(subgs)
                # Record this group's best H relative to global minimum
                best_H_group = min(m[3] for m in group)
                d_meV = (best_H_group - global_min_H_local) / MEV_TO_HA
                enthalpy_deltas_meV.append(d_meV)
                # For each subgroup, measure how far its best H is above the group's best H
                for sg in subgs:
                    best_H_sub = min(m[3] for m in sg)
                    d_group_meV = (best_H_sub - best_H_group) / MEV_TO_HA
                    if d_group_meV <= 10.0 + 1e-9:  # include 10 meV
                        subgroups_within_10_meV += 1
        if enthalpy_deltas_meV:
            log(
                f"  Threshold = {frac:.3f} * x_for_eig: "
                f"groups_with_subgroups={groups_with_subgroups}, "
                f"total_subgroups={total_subgroups}, "
                f"ΔH(group best vs global) meV range = "
                f"[{min(enthalpy_deltas_meV):.3f}, {max(enthalpy_deltas_meV):.3f}], "
                f"subgroups with best_H within 10 meV of group best: {subgroups_within_10_meV}"
            )
        else:
            log(
                f"  Threshold = {frac:.3f} * x_for_eig: "
                "no groups produced more than one subgroup."
            )

    global_min_H = min(m[3] for g in unified_groups for m in g)
    log(f"Global best H (all scales): {global_min_H:.6e} Ha")

    # Best H missed: how far off in meV? Count > 0.5 meV and > 0.1 meV
    log("\nGroups where each scale missed the best H — how far off (meV)?")
    thresh_01_meV_ha = 0.1 * MEV_TO_HA
    thresh_05_meV_ha = 0.5 * MEV_TO_HA
    for s in scales_list:
        n_gt_05 = 0
        n_gt_01 = 0
        deltas_meV = []
        for group in unified_groups:
            group_best = min(m[3] for m in group)
            scale_members = [m for m in group if m[4] == s]
            if not scale_members:
                continue
            scale_best = min(m[3] for m in scale_members)
            delta_ha = scale_best - group_best
            if delta_ha <= 0:
                continue
            delta_meV = delta_ha / MEV_TO_HA
            deltas_meV.append(delta_meV)
            if delta_ha >= thresh_05_meV_ha:
                n_gt_05 += 1
            if delta_ha >= thresh_01_meV_ha:
                n_gt_01 += 1
        log(f"  Scale {s}: missed by > 0.5 meV: {n_gt_05} groups; by > 0.1 meV: {n_gt_01} groups")
        if deltas_meV:
            log(f"    (delta meV: min={min(deltas_meV):.3f}, max={max(deltas_meV):.3f})")

    groups_missing_best: Dict[float, List[int]] = {s: [] for s in scales_list}
    for gidx, group in enumerate(unified_groups):
        best_H = min(m[3] for m in group)
        scales_achieving_best = [m[4] for m in group if m[3] == best_H]
        for s in scales_list:
            if s not in scales_achieving_best:
                groups_missing_best[s].append(gidx)
    for s in scales_list:
        if groups_missing_best[s]:
            best_H_per_group = [(gidx, min(m[3] for m in unified_groups[gidx])) for gidx in groups_missing_best[s]]
            for gidx, best_H in sorted(best_H_per_group, key=lambda t: -t[1])[:15]:
                log(f"    scale {s} missed group #{gidx+1} (best H={best_H:.6e})")

    # For each scale and each group where that scale missed the group's best H:
    # (1) ΔH of missed best vs global min (meV), (2) ΔH of scale's identified min vs missed best (meV), or "no member"
    log("\n" + "=" * 60)
    log("GROUPS WHERE SCALE MISSED THE GROUP'S BEST H: ΔH vs global, and scale's min vs missed best (meV)")
    log("=" * 60)
    for s in scales_list:
        if not groups_missing_best[s]:
            continue
        log(f"\n  Scale {s}:")
        rows = []
        for gidx in groups_missing_best[s]:
            group = unified_groups[gidx]
            group_best_H = min(m[3] for m in group)
            delta_missed_vs_global_meV = (group_best_H - global_min_H) / MEV_TO_HA
            scale_members = [m for m in group if m[4] == s]
            if scale_members:
                scale_best_H = min(m[3] for m in scale_members)
                delta_identified_vs_missed_meV = (scale_best_H - group_best_H) / MEV_TO_HA
                rows.append((gidx, group_best_H, delta_missed_vs_global_meV, delta_identified_vs_missed_meV, True))
            else:
                rows.append((gidx, group_best_H, delta_missed_vs_global_meV, None, False))
        for gidx, group_best_H, d_global_meV, d_id_vs_missed_meV, has_member in sorted(rows, key=lambda r: (-r[2], r[0])):
            if has_member:
                log(f"    group #{gidx+1}: missed best H={group_best_H:.6e} Ha; ΔH(missed vs global)={d_global_meV:.3f} meV; ΔH(scale's min vs missed best)={d_id_vs_missed_meV:.3f} meV")
            else:
                log(f"    group #{gidx+1}: missed best H={group_best_H:.6e} Ha; ΔH(missed vs global)={d_global_meV:.3f} meV; scale has no member in this group")

    # Cross-scale: groups only at larger scale; bounding box
    sigma_base, target, d_single, c_pairs, deltas, q0 = get_surrogate_data()
    bounds_by_scale = {s: compute_grid_bounds(eig_initial, eig_goal, d_single, deltas, q0, s) for s in scales_list}
    log("\n" + "=" * 60)
    log("GROUPS ONLY AT LARGER SCALE: how many missing due to bounding box?")
    log("=" * 60)
    pairs = [(0.7, 1.0), (0.7, 1.3), (0.7, 1.5), (1.0, 1.3), (1.0, 1.5), (1.3, 1.5)]
    missed_by_bounds: Dict[Tuple[float, float], List[Tuple[int, Tuple]]] = {}
    for small, large in pairs:
        # Skip pairs that involve scales we did not run in this configuration
        if small not in scales_list or large not in scales_list:
            continue
        groups_only_at_large = []
        for gidx, group in enumerate(unified_groups):
            scales_in_group = set(m[4] for m in group)
            if large in scales_in_group and small not in scales_in_group:
                best_from_large = min((m for m in group if m[4] == large), key=lambda m: m[3])
                groups_only_at_large.append((gidx, best_from_large))
        bounds_small = bounds_by_scale[small]
        missed_due_to_box = [(gidx, best) for gidx, best in groups_only_at_large if x_outside_bounds(best[0], bounds_small)]
        missed_by_bounds[(small, large)] = missed_due_to_box
        n_due_to_box = len(missed_due_to_box)
        log(f"  Groups at {large} but NOT at {small}: {len(groups_only_at_large)}; outside {small} bounds: {n_due_to_box}")
        if missed_due_to_box:
            # Enthalpy difference (meV) between global min and each missed group's best H
            deltas_meV = [(best[3] - global_min_H) / MEV_TO_HA for _gidx, best in missed_due_to_box]
            lowest_delta = min(deltas_meV)
            highest_delta = max(deltas_meV)
            log(f"    ΔH vs global min (meV) for these missed-by-bounds groups: min={lowest_delta:.3f}, max={highest_delta:.3f}")
            for (gidx, best), d_meV in sorted(zip(missed_due_to_box, deltas_meV), key=lambda t: t[1]):
                log(f"      group #{gidx+1}: best H={best[3]:.6e} Ha, ΔH={d_meV:.3f} meV")

    log("\n" + "=" * 60)
    log("SUMMARY: lowest-enthalpy missed-by-bounds group (ΔH vs global min, meV)")
    log("=" * 60)
    for (small, large), missed_due_to_box in missed_by_bounds.items():
        if missed_due_to_box:
            deltas_meV = [(best[3] - global_min_H) / MEV_TO_HA for _gidx, best in missed_due_to_box]
            lowest_delta = min(deltas_meV)
            gidx_low, best_low = min(missed_due_to_box, key=lambda t: (t[1][3] - global_min_H))
            log(f"  {small} vs {large}: {len(missed_due_to_box)} groups missed (outside {small} box). Lowest enthalpy missed group: ΔH = {lowest_delta:.3f} meV (group #{gidx_low+1}, H={best_low[3]:.6e} Ha)")
        else:
            log(f"  {small} vs {large}: no groups missed due to bounds")

    # Minima with no 15°-match at other scale: this analysis is only meaningful when multiple
    # distinct scales are present (beyond the dual-grid 0.7/1.3 configuration). Since we have
    # settled on the dual-grid approach as best, we skip this extra cross-scale reporting here.

    t_total = time.perf_counter() - t_main_start
    log(f"\nTotal time: {t_total:.1f} s ({t_total/60:.1f} min)")


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a 3x3 orthogonal matrix with det = +1 (rotation)."""
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_strain_work_frame_invariance() -> None:
    """
    Test that enthalpy-like strain energy (1/2)(σ_0+σ_1):Δε is frame-invariant.
    Setup: orthorhombic cell, distorted cell; hydrostatic stress on orig, orthorhombic
    stress on distorted. Apply two random rotations to both cells and both stress tensors;
    compute strain in each frame (rotated E = R E R^T), then strain work in each frame.
    The two strain-work values must agree (same physical scalar).
    """
    rng = np.random.default_rng(2025)
    # 1. Orthorhombic cell (rows = A, B, C)
    a, b, c = 5.0, 6.0, 7.0
    cell_orig = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=float)
    # 2. Distorted: stretch and slight shear
    cell_dist = np.array([
        [a * 1.02, 0.01, 0.02],
        [0.01, b * 0.98, 0.015],
        [0.02, 0.015, c * 1.01],
    ], dtype=float)
    # 3. Hydrostatic stress on orig (bar), orthorhombic on distorted
    p = 150000.0  # bar
    sigma_orig = p * np.eye(3)
    sigma_dist = np.diag([p * 1.1, p * 0.95, p * 1.05])  # orthorhombic principal stress
    # 4. Two random rotations
    R1 = _random_rotation(rng)
    R2 = _random_rotation(rng)
    # 5. Base strain (orig -> dist) in unrotated frame; E is 3x3
    eps_voigt_base = strain_from_cell_deformation(cell_orig, cell_dist)
    E_base = voigt_strain_to_mat3(eps_voigt_base)
    # 5b. Recompute strain from rotated cells (now gives strain in rotated frame: E' = R E R^T)
    eps_recomputed_R1 = strain_from_cell_deformation(R1 @ cell_orig, R1 @ cell_dist)
    eps_recomputed_R2 = strain_from_cell_deformation(R2 @ cell_orig, R2 @ cell_dist)
    print("Strain from cell deformation (F = cell1 @ inv(cell0), conserves energy under rotation):")
    print("  base (unrotated cells):     ", np.array2string(eps_voigt_base, precision=6))
    print("  recomputed from R1 cells:  ", np.array2string(eps_recomputed_R1, precision=6))
    print("  recomputed from R2 cells:  ", np.array2string(eps_recomputed_R2, precision=6))
    print("  recomputed R1 == base?     ", np.allclose(eps_recomputed_R1, eps_voigt_base))
    print("  recomputed R2 == base?     ", np.allclose(eps_recomputed_R2, eps_voigt_base))
    # Strain in rotated frame should equal recomputed (same physical convention)
    eps_rotated_R1 = mat3_strain_to_voigt(R1 @ E_base @ R1.T)
    eps_rotated_R2 = mat3_strain_to_voigt(R2 @ E_base @ R2.T)
    print("  recomputed R1 == R@E@R^T? ", np.allclose(eps_recomputed_R1, eps_rotated_R1))
    print("  recomputed R2 == R@E@R^T? ", np.allclose(eps_recomputed_R2, eps_rotated_R2))
    # 6. For each rotation: rotated cells, rotated stress, strain from recomputed (in that frame)
    def strain_work_in_frame(R: np.ndarray) -> float:
        cell_orig_R = R @ cell_orig
        cell_dist_R = R @ cell_dist
        # Strain in this frame: recompute from rotated cells (gives E' = R E R^T)
        eps_R = strain_from_cell_deformation(cell_orig_R, cell_dist_R)
        sigma_orig_R = R @ sigma_orig @ R.T
        sigma_dist_R = R @ sigma_dist @ R.T
        sigma_orig_R_v = mat3_to_vec6(sigma_orig_R)
        sigma_dist_R_v = mat3_to_vec6(sigma_dist_R)
        sigma_avg = 0.5 * (sigma_orig_R_v + sigma_dist_R_v)
        return stress_strain_double_contraction(sigma_avg, eps_R)
    W1 = strain_work_in_frame(R1)
    W2 = strain_work_in_frame(R2)
    W0 = strain_work_in_frame(np.eye(3))  # unrotated
    # 7. All three must agree (frame-invariant scalar)
    rtol = 1e-10
    assert np.isclose(W0, W1, rtol=rtol), f"W0={W0} W1={W1}"
    assert np.isclose(W0, W2, rtol=rtol), f"W0={W0} W2={W2}"
    assert np.isclose(W1, W2, rtol=rtol), f"W1={W1} W2={W2}"
    print("test_strain_work_frame_invariance: OK (W same in unrotated, R1, R2)")
    print(f"  W (bar) = {W0:.6e}")


if __name__ == "__main__":
    import sys
    if "--test-frame-invariance" in sys.argv:
        test_strain_work_frame_invariance()
        sys.exit(0)
    main()
