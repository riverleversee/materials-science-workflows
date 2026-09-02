#!/usr/bin/env python3
"""
Build eigenvector-group axis folders and axes_manifest.json for the CP2K ΔP ramp.

Reads optimizer minima JSON (default: scfhel/minima_scale_1.0.json), groups by
principal stress eigenvector (20°), picks the lowest-H representative per group,
and writes one axis folder per group under uniax/deltaPramp/axes/.

Inputs (env vars):
  CP2K_PARAM_TREND_PATH — parameter_trend_matrices.txt for q0/deltas/dSingle
  MINIMA_JSON — path to minima JSON (default: scfhel/minima_scale_1.0.json)
  HYDRO_COORDS — coordinates_final.xyz from hydrostatic opt (default: cp2k/hydroopt)
  PRESSURE_GPA — base mean pressure [GPa] (default: 15)
  EIGVEC_DEG — eigenvector grouping angle [deg] (default: 20)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DELTA_PRAMP = WORKFLOW_ROOT / "uniax" / "deltaPramp"


def _abc_to_cell(a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    sg = math.sin(gamma)
    if abs(sg) < 1e-10:
        sg = 1e-10
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0])
    c_x = c * math.cos(beta)
    c_y = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sg
    c_z = math.sqrt(max(0.0, c * c - c_x * c_x - c_y * c_y))
    c_vec = np.array([c_x, c_y, c_z])
    return np.array([a_vec, b_vec, c_vec], dtype=float)


def _write_cell(path: Path, cell: np.ndarray) -> None:
    with path.open("w") as f:
        for lab, vec in zip(["A", "B", "C"], cell):
            f.write(f"{lab} {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n")


def _vec_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / (np.linalg.norm(v1) + 1e-15)
    v2 = v2 / (np.linalg.norm(v2) + 1e-15)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(v1, v2))), 0.0, 1.0))))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parameter_trend_io import load_cp2k_parameter_trend

    pressure_gpa = float(os.environ.get("PRESSURE_GPA", "15.0"))
    p_int = int(round(pressure_gpa))
    eigvec_deg = float(os.environ.get("EIGVEC_DEG", "20.0"))

    param_path = Path(
        os.environ.get(
            "CP2K_PARAM_TREND_PATH",
            WORKFLOW_ROOT / "uniax/uniax_manual/cbaxis/1gpadelta/15gpa/parameter_trend_matrices.txt",
        )
    ).expanduser()
    if not param_path.is_file():
        raise FileNotFoundError(f"CP2K parameter trend not found: {param_path}")

    minima_path = Path(
        os.environ.get("MINIMA_JSON", Path(__file__).resolve().parent / "minima_scale_1.0.json")
    ).expanduser()
    if not minima_path.is_file():
        raise FileNotFoundError(f"Minima JSON not found: {minima_path}")

    hydro_root = WORKFLOW_ROOT / "cp2k" / "hydroopt" / f"{p_int}gpa"
    default_coords_candidates = [
        hydro_root / "reopt" / "coordinates_final.xyz",
        hydro_root / "coordinates_final.xyz",
    ]
    hydro_coords_env = os.environ.get("HYDRO_COORDS", "").strip()
    if hydro_coords_env:
        hydro_coords = Path(hydro_coords_env).expanduser()
    else:
        hydro_coords = next((p for p in default_coords_candidates if p.is_file()), default_coords_candidates[0])
    if not hydro_coords.is_file():
        raise FileNotFoundError(f"Hydrostatic coordinates not found: {hydro_coords}")

    q0, deltas, _sigma0, _target, d_single, _c_pairs, *_rest = load_cp2k_parameter_trend(param_path)

    all_min: List[Dict[str, Any]] = json.loads(minima_path.read_text())["minima"]
    all_min.sort(key=lambda m: float(m["H"]))

    used = [False] * len(all_min)
    groups: List[List[int]] = []
    for i, mi in enumerate(all_min):
        if used[i]:
            continue
        vi = np.array(mi["principal_vec"], dtype=float)
        grp = [i]
        used[i] = True
        for j in range(i + 1, len(all_min)):
            if used[j]:
                continue
            vj = np.array(all_min[j]["principal_vec"], dtype=float)
            if _vec_angle_deg(vi, vj) <= eigvec_deg:
                grp.append(j)
                used[j] = True
        groups.append(grp)

    axes_root = DELTA_PRAMP / "axes"
    axes_root.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []

    group_order = sorted(range(len(groups)), key=lambda gi: min(float(all_min[k]["H"]) for k in groups[gi]))
    for new_gid, gi in enumerate(group_order):
        best_k = min(groups[gi], key=lambda k: float(all_min[k]["H"]))
        m_best = all_min[best_k]
        x = np.array(m_best["x"], dtype=float)
        vec = np.array(m_best["principal_vec"], dtype=float)
        H = float(m_best["H"])

        axis_dir = axes_root / f"axis_{new_gid}"
        axis_dir.mkdir(parents=True, exist_ok=True)

        q = q0 + x * deltas
        cell = _abc_to_cell(q[0], q[1], q[2], q[3], q[4], q[5])
        cell_path = axis_dir / "final_cell.cell"
        _write_cell(cell_path, cell)

        coords_path = axis_dir / "coordinates_final.xyz"
        coords_path.write_text(hydro_coords.read_text())

        entries.append(
            {
                "group_id": new_gid,
                "subgroup_id": 0,
                "best_H_like_ha": H,
                "principal_vec": vec.tolist(),
                "path_cell": str(cell_path.resolve()),
                "path_coords": str(coords_path.resolve()),
            }
        )

    manifest_path = DELTA_PRAMP / "axes_manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(entries)} group axes to {manifest_path}")


if __name__ == "__main__":
    main()
