#!/usr/bin/env python3
"""
Manual cell optimization via finite-difference stress gradients.
Two-phase: --prepare writes inputs + run_fd_sp.sh; bash runs srun directly;
--postprocess reads outputs and computes new cell.
Avoids running srun from Python subprocess (which caused 100x+ SCF slowdown).
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Allow imports from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cell_utils import (
    abc_angles_to_cell,
    cell_to_abc_angles,
    get_perturbation_deltas,
    infer_strong_pairs,
    perturb_param,
    read_cell,
    rotate_cell_to_axis_frame,
    write_cell,
)
from stress_parser import parse_stress_from_output

STRESS_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")
PARAM_LABELS = ("a", "b", "c", "alpha", "beta", "gamma")
BAR_ANG3_TO_HARTREE = 2.2937122783963248e-8


def append_fd_monotonicity_notes(log_path, cycle):
    """Keep progress-log section in p-only mode (no p/m monotonic bracket test)."""
    out = Path(log_path)
    with open(out, "a") as f:
        f.write(f"FD monotonicity check (cycle {cycle})\n")
        f.write("  p-only scheme active: p/m monotonic bracket test skipped.\n")
        f.write("===========\n")


def append_parameter_trend_matrices(
    log_path,
    cycle,
    q,
    deltas,
    d_single,
    c_pairs,
    strong_pairs,
    sigma_base,
    target,
    solver_iters,
    x_final,
    sigma_pred,
    e_base,
    v_base,
    h_base,
    target_p_iso,
    d_h_single,
    d_h_dq,
    h_pairs_all,
    h_pair_curvature,
):
    """
    Append p-only trend matrices and nonlinear solver diagnostics for one cycle.
    d_single: 6x6 matrix with columns d_i = sigma_i - sigma0.
    c_pairs: dict[(i,j)] -> 6-vector c_ij interaction term.
    """
    out = Path(log_path)
    with open(out, "a") as f:
        f.write(f"Cycle {cycle}\n")
        f.write("Parameter state q = [a,b,c,alpha,beta,gamma]:\n")
        f.write(
            f"  a={q[0]:.9f}  b={q[1]:.9f}  c={q[2]:.9f}  "
            f"alpha={q[3]:.9f}  beta={q[4]:.9f}  gamma={q[5]:.9f}\n"
        )
        f.write("Deltas used (per parameter):\n")
        for j, (_param_idx, delta) in enumerate(deltas):
            unit = "A" if j < 3 else "deg"
            f.write(f"  {PARAM_LABELS[j]}: {delta:.9f} [{unit}]\n")
        f.write("Base stress [xx yy zz xy xz yz] bar:\n")
        f.write("  " + " ".join(f"{x: .6f}" for x in sigma_base) + "\n")
        f.write("Target stress [xx yy zz xy xz yz] bar:\n")
        f.write("  " + " ".join(f"{x: .6f}" for x in target) + "\n")

        f.write("Trend matrix dSingle = sigma_p - sigma_base (rows=stress comps, cols=parameters):\n")
        f.write("  cols: a b c alpha beta gamma\n")
        for i, sname in enumerate(STRESS_LABELS):
            f.write(f"  {sname:>3} " + " ".join(f"{d_single[i, j]: .6f}" for j in range(6)) + "\n")

        f.write("Strong pair interactions c_ij = sigma_pp - sigma_pi - sigma_pj + sigma_base:\n")
        for i, j in strong_pairs:
            v = c_pairs[(i, j)]
            f.write(
                f"  ({PARAM_LABELS[i]},{PARAM_LABELS[j]}): "
                + " ".join(f"{x: .6f}" for x in v)
                + "\n"
            )
        f.write(f"Self-consistent iterations: {solver_iters}\n")
        f.write("Final x (dimensionless multipliers of deltas):\n")
        f.write("  " + " ".join(f"{x: .6f}" for x in x_final) + "\n")
        f.write("Predicted stress at applied step [xx yy zz xy xz yz] bar:\n")
        f.write("  " + " ".join(f"{x: .6f}" for x in sigma_pred) + "\n")
        f.write("Predicted residual (target - predicted) [xx yy zz xy xz yz] bar:\n")
        f.write("  " + " ".join(f"{x: .6f}" for x in (target - sigma_pred)) + "\n")

        f.write("Base energetics:\n")
        f.write(f"  E_base [Ha]: {e_base:.12f}\n")
        f.write(f"  V_base [A^3]: {v_base:.9f}\n")
        f.write(f"  P_target_iso [bar]: {target_p_iso:.6f}\n")
        f.write(f"  H_base = E + P_iso*V [Ha]: {h_base:.12f}\n")
        f.write("Enthalpy singles dH = H_p - H_base:\n")
        for j in range(6):
            unit = "A" if j < 3 else "deg"
            f.write(
                f"  {PARAM_LABELS[j]}: dH={d_h_single[j]: .12e} [Ha], "
                f"dH/d{PARAM_LABELS[j]}={d_h_dq[j]: .12e} [Ha/{unit}]\n"
            )
        f.write("Pairwise enthalpy interactions k_ij [Ha] (includes diagonal i=j):\n")
        for i, j in all_pairs_with_diag():
            v = h_pairs_all[(i, j)]
            f.write(f"  ({PARAM_LABELS[i]},{PARAM_LABELS[j]}): {v: .12e}\n")
        f.write("Normalized enthalpy pair curvature K_ij = k_ij/(delta_i*delta_j):\n")
        f.write("  rows/cols: a b c alpha beta gamma\n")
        for i, pname in enumerate(PARAM_LABELS):
            f.write(f"  {pname:>6} " + " ".join(f"{h_pair_curvature[i, j]: .6e}" for j in range(6)) + "\n")
        f.write("===========\n")


def get_or_create_strong_pairs(work_dir, cell, orth_tol_deg):
    """Persist strong/weak coupling assumptions per run folder across cycles."""
    path = Path(work_dir) / "strong_pairs.json"
    policy = "all_offdiag_strong"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("policy") == policy:
                return [tuple(p) for p in data["strong_pairs"]]
        except Exception:
            pass
    strong = infer_strong_pairs(cell, orth_tol_deg=orth_tol_deg)
    path.write_text(
        json.dumps({"policy": policy, "strong_pairs": strong, "orth_tol_deg": orth_tol_deg}, indent=2) + "\n"
    )
    return strong


def predict_sigma(sigma_base, d_single, c_pairs, x):
    """Predict stress from p-only nonlinear local model for dimensionless multipliers x."""
    sigma = sigma_base + d_single @ x
    for (i, j), cij in c_pairs.items():
        sigma = sigma + cij * x[i] * x[j]
    return sigma


def all_offdiag_pairs():
    """Return all i<j off-diagonal parameter pairs for q of length 6."""
    return [(i, j) for i in range(6) for j in range(i + 1, 6)]


def all_pairs_with_diag():
    """Return all i<=j parameter pairs for q of length 6."""
    return [(i, j) for i in range(6) for j in range(i, 6)]


def parse_total_energy_from_output(out_path):
    """
    Parse last total CP2K FORCE_EVAL energy in Hartree from output.
    Returns None on parse failure.
    """
    path = Path(out_path)
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    matches = re.findall(
        r"ENERGY\|\s+Total FORCE_EVAL[^\n]*?(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    # Fallback for some print formats
    matches = re.findall(
        r"Total energy:\s*(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def cell_volume_ang3(cell):
    return float(abs(np.linalg.det(np.array(cell, dtype=float))))


def write_full_coupling_report(path, cycle, c_pairs_all):
    """
    Write full coupling report for one cycle.
    Uses max absolute component of c_ij as scalar strength metric.
    """
    out = Path(path)
    M = np.zeros((6, 6), dtype=float)
    for (i, j), v in c_pairs_all.items():
        s = float(np.max(np.abs(v)))
        M[i, j] = s
        M[j, i] = s
    with open(out, "w") as f:
        f.write(f"Full coupling report (cycle {cycle})\n")
        f.write("Scalar strength matrix S_ij = max(|c_ij components|) [bar]\n")
        f.write("Rows/cols: a b c alpha beta gamma\n")
        for i, name in enumerate(PARAM_LABELS):
            f.write(f"{name:>6} " + " ".join(f"{M[i, j]: .6f}" for j in range(6)) + "\n")
        f.write("\nPair interaction vectors c_ij [xx yy zz xy xz yz] (bar, includes diagonal i=j):\n")
        for i, j in all_pairs_with_diag():
            if (i, j) not in c_pairs_all:
                continue
            v = c_pairs_all[(i, j)]
            f.write(
                f"({PARAM_LABELS[i]},{PARAM_LABELS[j]}): "
                + " ".join(f"{x: .6f}" for x in v)
                + "\n"
            )


def do_prepare(args):
    """Write base + p-only + strong-pair(++) cell/input files and run_fd_sp.sh."""
    work_dir = Path(args.work_dir)
    cell_path = Path(args.cell)
    coords_path = Path(args.coords)
    target = np.array(args.target_stress, dtype=float)
    delta_len = getattr(args, "delta_length", None)
    delta_ang = args.delta_angle
    delta_length_ang = getattr(args, "delta_length_ang", None)

    if not cell_path.exists():
        print(f"Error: cell file not found: {cell_path}", file=sys.stderr)
        sys.exit(1)
    if not coords_path.exists():
        print(f"Error: coords file not found: {coords_path}", file=sys.stderr)
        sys.exit(1)

    cell = read_cell(cell_path)
    deltas = get_perturbation_deltas(cell, delta_len or 0.001, delta_ang, delta_length_ang=delta_length_ang)

    if args.submit_dir:
        submit_dir = Path(args.submit_dir).resolve()
    else:
        submit_dir = work_dir.parent.parent / "submitfiles"
    inp_template = (submit_dir / args.inp_template).resolve()
    if not inp_template.exists():
        inp_template = (work_dir / args.inp_template).resolve()
    if not inp_template.exists():
        print(f"Error: input template not found: {inp_template}", file=sys.stderr)
        sys.exit(1)

    work_cell = work_dir / "init_cell.cell"
    work_coords = work_dir / "coordinates_init.xyz"
    if coords_path.resolve() != work_coords.resolve():
        import shutil
        shutil.copy(coords_path, work_coords)

    template = inp_template.read_text()
    runs = []

    def write_inp(suffix, tmpl):
        # Unique PROJECT per run to avoid overwriting restart/trajectory
        import re
        proj = f"QM_fd_{suffix}".replace("-", "m")
        out = re.sub(r"PROJECT\s+\S+", f"PROJECT {proj}", tmpl, count=1)
        return out

    axis = getattr(args, "axis", None)
    def to_axis_frame(c):
        return rotate_cell_to_axis_frame(c, axis) if axis else c

    # Base (ensure axis frame)
    cell_base = to_axis_frame(cell)
    write_cell(work_dir / "cell_base.cell", cell_base)
    (work_dir / "sp_base.inp").write_text(write_inp("base", template))
    runs.append(("base", "cell_base.cell", "sp_base.inp", "sp_base.out"))

    strong_pairs = get_or_create_strong_pairs(work_dir, cell, args.orth_tol_deg)
    use_all_pairs = bool(args.full_coupling_first_step) and int(getattr(args, "cycle", 0)) == 1
    pair_list = all_offdiag_pairs() if use_all_pairs else strong_pairs

    # p-only singles
    for j, (param_idx, delta) in enumerate(deltas):
        cell_plus = to_axis_frame(perturb_param(cell, param_idx, +delta))
        write_cell(work_dir / f"cell_p{j}.cell", cell_plus)
        (work_dir / f"sp_p{j}.inp").write_text(write_inp(f"p{j}", template))
        runs.append((f"p{j}", f"cell_p{j}.cell", f"sp_p{j}.inp", f"sp_p{j}.out"))

    # pair ++ displacements for stress couplings (off-diagonal strong/all)
    for i, j in pair_list:
        di = deltas[i][1]
        dj = deltas[j][1]
        cell_pair = perturb_param(perturb_param(cell, i, +di), j, +dj)
        cell_pair = to_axis_frame(cell_pair)
        suffix = f"pp{i}_{j}"
        write_cell(work_dir / f"cell_{suffix}.cell", cell_pair)
        (work_dir / f"sp_{suffix}.inp").write_text(write_inp(suffix, template))
        runs.append((suffix, f"cell_{suffix}.cell", f"sp_{suffix}.inp", f"sp_{suffix}.out"))

    # diagonal ++ displacements (i=i) for enthalpy curvature terms
    for i in range(6):
        di = deltas[i][1]
        cell_pair = perturb_param(perturb_param(cell, i, +di), i, +di)
        cell_pair = to_axis_frame(cell_pair)
        suffix = f"pp{i}_{i}"
        write_cell(work_dir / f"cell_{suffix}.cell", cell_pair)
        (work_dir / f"sp_{suffix}.inp").write_text(write_inp(suffix, template))
        runs.append((suffix, f"cell_{suffix}.cell", f"sp_{suffix}.inp", f"sp_{suffix}.out"))

    # Write run_fd_sp.sh - caller creates run_fd_cmd.txt with srun template
    run_cmd_file = (work_dir / (args.run_cmd_file or "run_fd_cmd.txt")).resolve()
    run_fd_sh = work_dir / "run_fd_sp.sh"
    work_dir_abs = work_dir.resolve()
    if run_cmd_file.exists():
        run_cmd_tpl = run_cmd_file.read_text().strip()
        lines = ["#!/bin/bash", "set -e", f"cd {work_dir_abs} || exit 1", ""]
        for _suffix, cell_f, inp_f, out_f in runs:
            lines.append(f'cp "{cell_f}" init_cell.cell')
            lines.append(f'{run_cmd_tpl} -o "{out_f}" -i "{inp_f}"')
            lines.append("")
        lines.append("# Restore base cell for postprocess (run_fd overwrites init_cell.cell each run)")
        lines.append('cp "cell_base.cell" init_cell.cell')
        lines.append("# Cleanup GeoOpt restart/trajectory (keep .out for stress)")
        lines.append("rm -f QM_fd_*-* 2>/dev/null || true")
        run_fd_sh.write_text("\n".join(lines) + "\n")
        run_fd_sh.chmod(0o755)
    else:
        print(f"Warning: {run_cmd_file} not found; create it with the srun command template", file=sys.stderr)
        sys.exit(1)


def do_postprocess(args):
    """Read p-only + pair outputs, solve self-consistent step, write new cell."""
    work_dir = Path(args.work_dir)
    cell_path = Path(args.cell)
    target = np.array(args.target_stress, dtype=float)
    step_frac = args.step_fraction
    delta_len = getattr(args, "delta_length", None)
    delta_ang = args.delta_angle
    delta_length_ang = getattr(args, "delta_length_ang", None)
    reg = args.regularization

    cell = read_cell(cell_path)
    deltas = get_perturbation_deltas(cell, delta_len or 0.001, delta_ang, delta_length_ang=delta_length_ang)

    def get_stress(suffix):
        out_file = work_dir / f"sp_{suffix}.out"
        stress = parse_stress_from_output(out_file)
        if stress is None:
            print(f"Error: Could not parse stress from {out_file}", file=sys.stderr)
            sys.exit(1)
        return np.array(stress)

    def get_energy(suffix):
        out_file = work_dir / f"sp_{suffix}.out"
        energy = parse_total_energy_from_output(out_file)
        if energy is None:
            print(f"Error: Could not parse total energy from {out_file}", file=sys.stderr)
            sys.exit(1)
        return float(energy)

    def get_volume_from_cell_file(cell_file):
        return cell_volume_ang3(read_cell(work_dir / cell_file))

    target_p_iso = float(np.mean(target[:3]))

    sigma_base = get_stress("base")
    e_base = get_energy("base")
    v_base = get_volume_from_cell_file("cell_base.cell")
    h_base = e_base + target_p_iso * v_base * BAR_ANG3_TO_HARTREE
    q = np.array(cell_to_abc_angles(cell))
    r0 = target - sigma_base

    # p-only trends: d_i = sigma_p(i) - sigma_base
    d_single = np.zeros((6, 6))
    sigma_p = {}
    h_p = {}
    for j in range(6):
        sigma_p[j] = get_stress(f"p{j}")
        d_single[:, j] = sigma_p[j] - sigma_base
        e_pj = get_energy(f"p{j}")
        v_pj = get_volume_from_cell_file(f"cell_p{j}.cell")
        h_p[j] = e_pj + target_p_iso * v_pj * BAR_ANG3_TO_HARTREE

    delta_vec = np.array([d for _p, d in deltas], dtype=float)
    d_h_single = np.array([h_p[j] - h_base for j in range(6)], dtype=float)
    d_h_dq = np.divide(d_h_single, delta_vec, out=np.zeros_like(d_h_single), where=np.abs(delta_vec) > 0)

    # Fixed strong couplings (persisted per run)
    strong_pairs = get_or_create_strong_pairs(work_dir, cell, args.orth_tol_deg)
    # Evaluate all pair couplings if pp outputs exist (including diagonal i=j)
    c_pairs_all = {}
    h_pairs_all = {}
    for i, j in all_pairs_with_diag():
        out_pp = work_dir / f"sp_pp{i}_{j}.out"
        if not out_pp.exists():
            continue
        sigma_pp = get_stress(f"pp{i}_{j}")
        if i == j:
            c_pairs_all[(i, j)] = sigma_pp - 2.0 * sigma_p[i] + sigma_base
        else:
            c_pairs_all[(i, j)] = sigma_pp - sigma_p[i] - sigma_p[j] + sigma_base

        e_pp = get_energy(f"pp{i}_{j}")
        v_pp = get_volume_from_cell_file(f"cell_pp{i}_{j}.cell")
        h_pp = e_pp + target_p_iso * v_pp * BAR_ANG3_TO_HARTREE
        if i == j:
            h_pairs_all[(i, j)] = h_pp - 2.0 * h_p[i] + h_base
        else:
            h_pairs_all[(i, j)] = h_pp - h_p[i] - h_p[j] + h_base

    h_pair_curvature = np.zeros((6, 6), dtype=float)
    for (i, j), hij in h_pairs_all.items():
        denom = delta_vec[i] * delta_vec[j]
        val = hij / denom if abs(denom) > 0 else 0.0
        h_pair_curvature[i, j] = val
        h_pair_curvature[j, i] = val

    # Solver uses fixed strong pairs only
    c_pairs = {}
    for i, j in strong_pairs:
        if (i, j) in c_pairs_all:
            c_pairs[(i, j)] = c_pairs_all[(i, j)]
        else:
            # Should not happen unless prepare/postprocess mismatch; fallback no interaction
            c_pairs[(i, j)] = np.zeros(6)

    if getattr(args, "progress_log", None):
        append_fd_monotonicity_notes(args.progress_log, getattr(args, "cycle", 0))

    def solve_linear(A, b):
        cond_A = np.linalg.cond(A)
        reg_use = reg if (cond_A > 1e10 or not np.isfinite(cond_A)) else 0.0
        A_reg = A + reg_use * np.eye(6)
        try:
            x = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        return x

    # Initial guess from singles-only linear solve
    x = solve_linear(d_single, r0)

    def build_eff_matrix(x_vec):
        M_eff = d_single.copy()
        for i in range(6):
            add = np.zeros(6)
            for j in range(6):
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                if key in c_pairs:
                    add += c_pairs[key] * x_vec[j]
            M_eff[:, i] = d_single[:, i] + add
        return M_eff

    solver_iters = 0
    for k in range(args.sc_max_iter):
        solver_iters = k + 1
        M_eff = build_eff_matrix(x)
        x_tilde = solve_linear(M_eff, r0)
        x_next = (1.0 - args.sc_eta) * x + args.sc_eta * x_tilde
        if np.linalg.norm(x_next - x) < args.sc_tol:
            x = x_next
            break
        x = x_next

    dq_full = x * delta_vec

    dq = step_frac * dq_full
    # Trust radius: scale entire step so largest overshoot defines step size (uniform scaling)
    max_dl = getattr(args, "max_delta_length_ang", 0.1)
    max_da = getattr(args, "max_delta_angle_deg", 1.0)
    scale = 1.0
    max_len = np.max(np.abs(dq[0:3]))
    if max_len > max_dl:
        scale = min(scale, max_dl / max_len)
    max_ang = np.max(np.abs(dq[3:6]))
    if max_ang > max_da:
        scale = min(scale, max_da / max_ang)
    dq = dq * scale
    x_applied = np.divide(dq, delta_vec, out=np.zeros_like(dq), where=np.abs(delta_vec) > 0)
    sigma_pred = predict_sigma(sigma_base, d_single, c_pairs, x_applied)

    if getattr(args, "trend_log", None):
        append_parameter_trend_matrices(
            args.trend_log,
            getattr(args, "cycle", 0),
            q,
            deltas,
            d_single,
            c_pairs,
            strong_pairs,
            sigma_base,
            target,
            solver_iters,
            x_applied,
            sigma_pred,
            e_base,
            v_base,
            h_base,
            target_p_iso,
            d_h_single,
            d_h_dq,
            h_pairs_all,
            h_pair_curvature,
        )
    if int(getattr(args, "cycle", 0)) == 1 and bool(args.full_coupling_first_step):
        report_path = Path(work_dir) / args.full_coupling_report
        write_full_coupling_report(report_path, getattr(args, "cycle", 0), c_pairs_all)

    q_new = q + dq
    q_new[0:3] = np.maximum(q_new[0:3], 0.1)
    q_new[3:6] = np.clip(q_new[3:6], 1.0, 179.0)

    cell_standard_new = abc_angles_to_cell(*q_new)
    # Sanity check: reject overflow/corruption only (valid cells can have zero components, e.g. A=(a,0,0))
    if np.any(np.abs(cell_standard_new) > 1e6):
        print(f"  Manual step: REJECTED - cell overflow (max={np.max(np.abs(cell_standard_new)):.1e})", file=sys.stderr)
        sys.exit(1)
    write_cell(cell_path, cell_standard_new)
    print(f"  Manual step: residual norm {np.linalg.norm(r0):.1f} bar")


def main():
    parser = argparse.ArgumentParser(description="Manual cell optimization step (prepare/postprocess)")
    parser.add_argument("mode", choices=["prepare", "postprocess"], help="prepare: write inputs; postprocess: compute new cell")
    parser.add_argument("--work-dir", required=True, help="Working directory")
    parser.add_argument("--cell", required=True, help="Path to init_cell.cell")
    parser.add_argument("--coords", required=True, help="Path to coordinates_init.xyz")
    parser.add_argument("--target-stress", required=True, nargs=6, type=float, metavar=("SXX", "SYY", "SZZ", "SXY", "SXZ", "SYZ"))
    parser.add_argument("--step-fraction", type=float, default=0.3)
    parser.add_argument("--delta-length", type=float, default=None, help="Relative FD for a,b,c (ignored if --delta-length-ang set)")
    parser.add_argument("--delta-length-ang", type=float, default=0.005, help="Absolute FD for a,b,c [Å] (default 0.005)")
    parser.add_argument("--delta-angle", type=float, default=0.05, help="Absolute FD for α,β,γ [deg] (~0.005 Å length-scale for L~6 Å)")
    parser.add_argument("--inp-template", default="GeoOpt_fd.inp", help="GeoOpt_fd.inp for relaxed stress (recommended)")
    parser.add_argument("--axis", choices=["aaxis", "baxis", "cbaxis"], help="Rotate perturbed cells to axis frame (A along X, B along Y, or B+C along Z)")
    parser.add_argument("--submit-dir", default=None)
    parser.add_argument("--run-cmd-file", default="run_fd_cmd.txt", help="File with srun command template (one line)")
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--orth-tol-deg", type=float, default=2.0, help="Near-orthogonality tolerance for weak length-length coupling [deg]")
    parser.add_argument("--no-full-coupling-first-step", action="store_false", dest="full_coupling_first_step", default=True, help="Disable full off-diagonal pair couplings at cycle 1")
    parser.add_argument("--full-coupling-report", default="full_coupling_matrix_cycle1.txt", help="Filename for full coupling report at cycle 1")
    parser.add_argument("--sc-eta", type=float, default=0.2, help="Self-consistent solver damping eta")
    parser.add_argument("--sc-max-iter", type=int, default=20, help="Max self-consistent iterations")
    parser.add_argument("--sc-tol", type=float, default=1e-4, help="Self-consistent convergence tolerance on x")
    parser.add_argument("--max-delta-length-ang", type=float, default=0.1, help="Max change in a,b,c per step [Å] (trust radius)")
    parser.add_argument("--max-delta-angle-deg", type=float, default=1.0, help="Max change in α,β,γ per step [deg] (trust radius)")
    parser.add_argument("--cycle", type=int, default=0, help="Current cycle number (for progress logging)")
    parser.add_argument("--progress-log", default=None, help="Path to append FD monotonicity diagnostics")
    parser.add_argument("--trend-log", default="parameter_trend_matrices.txt", help="Path to append FD trend/pair interaction matrices")
    args = parser.parse_args()

    if args.mode == "prepare":
        do_prepare(args)
    else:
        do_postprocess(args)


if __name__ == "__main__":
    main()
