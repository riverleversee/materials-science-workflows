#!/usr/bin/env python3
"""
CP2K ΔP ramp driver for one uniaxial axis.

Phase A: self-consistent local minimum (<=2 FD re-fits + gradient descent).
Phase B: ramp ΔP from 1 to 5 GPa; full CP2K FD re-fit + GD at every integer GPa
checkpoint (1, 2, 3, 4, 5).

Env (required):
  AXIS_OUT_DIR, START_CELL, START_COORDS
Env (optional):
  PRESSURE_GPA (default 15), OPT_SCRIPT, CENTERED_FD_SCRIPT, WORKFLOW_ROOT
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

GPA_TO_BAR = 10000.0
PHASE_A_MAX_ITERS = 2
PHASE_A_CONV_LEN_ANG = 0.01
PHASE_A_CONV_ANG_DEG = 0.05
RAMP_MAX_DP_GPA = 5.0


def _parse_full_fd_gpas() -> set[float]:
    """GPa checkpoints that trigger a full CP2K FD re-fit (default: every integer 1..5)."""
    raw = os.environ.get("RAMP_FULL_FD_GPAS", "").strip()
    if not raw:
        return {float(t) for t in range(1, int(RAMP_MAX_DP_GPA) + 1)}
    out: set[float] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(float(part))
    return out


def load_optimizer(opt_script: Path):
    spec = importlib.util.spec_from_file_location("uniaxopt_mod", str(opt_script))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class EvalPoint:
    x: np.ndarray
    cell: np.ndarray
    sigma_bar: np.ndarray
    eigs_bar: np.ndarray
    principal_vec: np.ndarray
    E_ha: float


def format_deltaP_label(dp_gpa: float) -> str:
    return f"{dp_gpa:.2f}".replace(".", "p")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_state_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_cell_file(path: Path, cell: np.ndarray) -> None:
    with path.open("w") as f:
        for lab, vec in zip(["A", "B", "C"], cell):
            f.write(f"{lab} {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n")


def p_iso_bar_from_eigs(eigs_bar: np.ndarray) -> float:
    return float(np.mean(eigs_bar))


def ramp_config(mod) -> Any:
    return mod.OptimizerConfig(conv_dH_meV=0.2)


def main() -> None:
    axis_root = Path(os.environ["AXIS_OUT_DIR"]).resolve()
    start_cell = Path(os.environ["START_CELL"]).resolve()
    start_coords = Path(os.environ["START_COORDS"]).resolve()
    pressure_gpa = float(os.environ.get("PRESSURE_GPA", "15.0"))
    workflow_root = Path(
        os.environ.get("WORKFLOW_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()

    opt_script = Path(
        os.environ.get(
            "OPT_SCRIPT",
            workflow_root / "scfhel/uniax_surrogate_optimizer_test.py",
        )
    ).resolve()

    centered_fd = Path(
        os.environ.get(
            "CENTERED_FD_SCRIPT",
            workflow_root / "uniax/deltaPramp/uniax_param_centered_fd.sh",
        )
    ).resolve()
    if not centered_fd.is_file():
        raise FileNotFoundError(f"Centered FD script not found: {centered_fd}")

    ensure_dir(axis_root)
    min_refine_dir = axis_root / "min_refine"
    ensure_dir(min_refine_dir)

    def run_cp2k_fd(cell_path: Path, coords_path: Path, out_dir: Path) -> None:
        ensure_dir(out_dir)
        env = os.environ.copy()
        env.update(
            {
                "CENTER_CELL": str(cell_path),
                "CENTER_COORDS": str(coords_path),
                "OUT_DIR": str(out_dir),
                "PRESSURE_GPA": str(pressure_gpa),
                "WORKFLOW_ROOT": str(workflow_root),
            }
        )
        subprocess.check_call(["bash", str(centered_fd)], env=env)

    def eval_surrogate_at(cell_path: Path, coords_path: Path, out_dir: Path) -> Any:
        run_cp2k_fd(cell_path, coords_path, out_dir)
        os.environ["CP2K_PARAM_TREND_PATH"] = str(out_dir / "parameter_trend_matrices.txt")
        return load_optimizer(opt_script)

    # -------- Phase A: minimum refinement --------
    current_cell = start_cell
    current_coords = start_coords
    best_eval: EvalPoint | None = None
    mod_phase_a: Any = None
    q_prev = None

    for it in range(1, PHASE_A_MAX_ITERS + 1):
        it_dir = min_refine_dir / f"iter_{it}"
        ensure_dir(it_dir)
        mod_it = eval_surrogate_at(current_cell, current_coords, it_dir)
        mod_phase_a = mod_it

        sigma0, _t, _ds, _cp, deltas, q0 = mod_it.get_surrogate_data()
        eig_initial = mod_it.principal_sorted(mod_it.vec6_to_mat3(sigma0))[0]
        p_iso_bar = float(np.mean(eig_initial))
        eig_goal = mod_it.make_target_uniaxial(p_iso_bar, 0.0)

        x0 = np.zeros(6)
        x_prev = np.zeros(6)
        cfg = ramp_config(mod_it)
        x_min, _n_steps, _H_like = mod_it.run_gradient_descent_until_converged(
            x0, x_prev, eig_goal, cfg, q0, deltas, max_steps=200
        )

        best_eval = EvalPoint(
            x=x_min,
            cell=mod_it.cell_at_x(x_min),
            sigma_bar=mod_it.predict_sigma(x_min),
            eigs_bar=mod_it.predict_eigs(x_min),
            principal_vec=mod_it.principal_vec_at_x(x_min),
            E_ha=mod_it.predict_E(x_min),
        )

        q = mod_it.cell_to_abc(best_eval.cell)
        if q_prev is not None:
            dq = np.abs(q - q_prev)
            if np.max(dq[0:3]) < PHASE_A_CONV_LEN_ANG and np.max(dq[3:6]) < PHASE_A_CONV_ANG_DEG:
                break
        q_prev = q.copy()

        current_cell = it_dir / "final_cell.cell"
        write_cell_file(current_cell, best_eval.cell)
        coords_out = it_dir / "coordinates_final.xyz"
        shutil.copy2(current_coords, coords_out)
        current_coords = coords_out

    if best_eval is None or mod_phase_a is None:
        raise RuntimeError("Minimum refinement did not produce an eval point")

    phase_a_end = axis_root / "phase_a_end"
    ensure_dir(phase_a_end)
    write_cell_file(phase_a_end / "final_cell.cell", best_eval.cell)
    shutil.copy2(current_coords, phase_a_end / "coordinates_final.xyz")
    trend_src = min_refine_dir / f"iter_{PHASE_A_MAX_ITERS}" / "parameter_trend_matrices.txt"
    for it in range(PHASE_A_MAX_ITERS, 0, -1):
        candidate = min_refine_dir / f"iter_{it}" / "parameter_trend_matrices.txt"
        if candidate.is_file():
            trend_src = candidate
            break
    if trend_src.is_file():
        shutil.copy2(trend_src, phase_a_end / "parameter_trend_matrices.txt")
    write_state_json(
        phase_a_end / "state.json",
        {
            "pressure_gpa": pressure_gpa,
            "deltaP_GPa": 0.0,
            "p_iso_bar": p_iso_bar_from_eigs(best_eval.eigs_bar),
            "eigs_bar": best_eval.eigs_bar.tolist(),
            "principal_vec": best_eval.principal_vec.tolist(),
            "E_Ha": float(best_eval.E_ha),
            "Hlike_total_Ha": 0.0,
        },
    )

    # -------- Phase B: ΔP ramp 1..5 GPa --------
    Hlike_total = 0.0
    current_cell = phase_a_end / "final_cell.cell"
    current_coords = phase_a_end / "coordinates_final.xyz"

    targets = [float(t) for t in range(1, int(RAMP_MAX_DP_GPA) + 1)]
    full_fd_gpas = _parse_full_fd_gpas()

    def gd_on_mod(mod_eval: Any, target_dp: float) -> Tuple[EvalPoint, float]:
        sigma0, _t, _ds, _cp, deltas, q0 = mod_eval.get_surrogate_data()
        eig_initial = mod_eval.principal_sorted(mod_eval.vec6_to_mat3(sigma0))[0]
        p_iso_bar = float(np.mean(eig_initial))
        eig_goal = mod_eval.make_target_uniaxial(p_iso_bar, target_dp)
        x0 = np.zeros(6)
        x_ref = np.zeros(6)
        cfg = ramp_config(mod_eval)
        x_min, _n_steps, H_like = mod_eval.run_gradient_descent_until_converged(
            x0, x_ref, eig_goal, cfg, q0, deltas, max_steps=200
        )
        ep = EvalPoint(
            x=x_min,
            cell=mod_eval.cell_at_x(x_min),
            sigma_bar=mod_eval.predict_sigma(x_min),
            eigs_bar=mod_eval.predict_eigs(x_min),
            principal_vec=mod_eval.principal_vec_at_x(x_min),
            E_ha=mod_eval.predict_E(x_min),
        )
        return ep, float(H_like)

    for t in targets:
        full_fd = t in full_fd_gpas
        dp_label = format_deltaP_label(t)
        out_dir = axis_root / f"deltaP_{dp_label}"
        ensure_dir(out_dir)

        if full_fd:
            center_cell = out_dir / "center_cell.cell"
            center_coords = out_dir / "center_coords.xyz"
            shutil.copy2(current_cell, center_cell)
            shutil.copy2(current_coords, center_coords)
            mod_eval = eval_surrogate_at(center_cell, center_coords, out_dir)
            ep, _H_like = gd_on_mod(mod_eval, t)
            Hlike_total += float(mod_eval.enthalpy_like(ep.x, np.zeros(6)))
        else:
            raise RuntimeError(
                f"Checkpoint ΔP={t} GPa is not in RAMP_FULL_FD_GPAS={sorted(full_fd_gpas)}; "
                "surrogate-only steps are disabled (too inaccurate without re-centering FD)."
            )

        write_cell_file(out_dir / "surrogate_center.cell", mod_eval.cell_at_x(np.zeros(6)))
        write_cell_file(out_dir / "final_cell.cell", ep.cell)
        shutil.copy2(current_coords, out_dir / "coordinates_final.xyz")
        current_cell = out_dir / "final_cell.cell"

        write_state_json(
            out_dir / "state.json",
            {
                "pressure_gpa": pressure_gpa,
                "deltaP_GPa": float(t),
                "p_iso_bar": p_iso_bar_from_eigs(ep.eigs_bar),
                "eigs_bar": ep.eigs_bar.tolist(),
                "principal_vec": ep.principal_vec.tolist(),
                "E_Ha": float(ep.E_ha),
                "Hlike_total_Ha": float(Hlike_total),
                "full_fd_refit": full_fd,
            },
        )


if __name__ == "__main__":
    main()
