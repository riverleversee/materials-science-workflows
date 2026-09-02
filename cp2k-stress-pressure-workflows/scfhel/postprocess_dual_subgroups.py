#!/usr/bin/env python3
"""
Post-process dual-grid (0.7 + 1.3) minima: group by 20° eigenvector (H-ordered), then form
subgroups using 1.0 * x_for_eig for both lengths and angles. No re-run of optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Load optimizer module so we get surrogate + grouping helpers
import uniax_surrogate_optimizer_test as opt
from uniax_surrogate_optimizer_test import (
    principal_sorted,
    vec6_to_mat3,
    make_target_uniaxial,
    OptimizerConfig,
)


def load_dual_minima(dir_path: Path):
    """Load minima from minima_scale_0.7_dual07_13.json and minima_scale_1.3_dual07_13.json.
    Returns converged_by_scale: {0.7: [(x, eig, resid, H, n_steps), ...], 1.3: ...}.
    """
    converged_by_scale = {}
    for scale in (0.7, 1.3):
        path = dir_path / f"minima_scale_{scale}_dual07_13.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        minima = []
        for m in data["minima"]:
            x = np.array(m["x"], dtype=float)
            eig = np.array(m["eig"], dtype=float)
            resid = np.array(m["resid"], dtype=float)
            H = float(m["H"])
            n_steps = int(m["n_steps"])
            minima.append((x, eig, resid, H, n_steps))
        converged_by_scale[scale] = minima
    return converged_by_scale


def main():
    base_dir = Path(__file__).resolve().parent
    converged_by_scale = load_dual_minima(base_dir)

    # Surrogate and x_for_eig (same as main script)
    sigma_base, target, d_single, c_pairs, deltas, q0 = opt.get_surrogate_data()
    config = OptimizerConfig()
    eig_initial = principal_sorted(vec6_to_mat3(sigma_base))[0]
    p_iso_bar = float(np.mean(eig_initial))
    eig_goal = make_target_uniaxial(p_iso_bar, config.delta_p_gpa)
    delta_eig_max = float(np.max(np.abs(eig_goal - eig_initial)))
    if delta_eig_max < 1e-6:
        delta_eig_max = 100.0
    x_for_eig = np.zeros(6, dtype=float)
    for i in range(6):
        max_dsigma = float(np.max(np.abs(d_single[:, i])))
        if max_dsigma < 1e-10:
            x_for_eig[i] = 80.0
        else:
            x_for_eig[i] = delta_eig_max / max_dsigma

    # 1.0 * x_for_eig for both lengths and angles
    param_thresholds = 1.0 * x_for_eig
    thresh_physical = param_thresholds * deltas

    # Group by eigenvector (20° acceptance); main script uses H-ascending order (lowest first).
    unified_groups = opt.group_all_minima_by_eigenvector(converged_by_scale, deg_threshold=20.0)
    print(f"Unified groups (20° eigvec, H-ordered): {len(unified_groups)}")

    # Subgroups with 1.0 * x_for_eig
    total_subgroups = 0
    groups_with_subgroups = 0
    for group in unified_groups:
        subgs = opt.subgroup_by_parameters(group, param_thresholds)
        total_subgroups += len(subgs)
        if len(subgs) > 1:
            groups_with_subgroups += 1
    print(f"Thresholds: 1.0 * x_for_eig (lengths Å: [{thresh_physical[0]:.4f}, {thresh_physical[1]:.4f}, {thresh_physical[2]:.4f}], angles deg: [{thresh_physical[3]:.4f}, {thresh_physical[4]:.4f}, {thresh_physical[5]:.4f}])")
    print(f"Groups with >1 subgroup: {groups_with_subgroups}")
    print(f"Total subgroups: {total_subgroups}")

    # a,b,c split among groups (not aligned): H-best per group -> range and variance
    _, _, _, _, deltas, q0 = opt.get_surrogate_data()

    abc_list = []
    H_best_list = []
    for group in unified_groups:
        best = min(group, key=lambda m: m[3])  # (x, eig, resid, H, scale)
        x = best[0]
        H = best[3]
        q = q0 + x * deltas
        a, b, c = q[0], q[1], q[2]
        abc_list.append([a, b, c])
        H_best_list.append(H)

    abc = np.array(abc_list)  # (n_groups, 3)
    print("\n--- a, b, c across eigenvector groups (H-best per group) ---")
    for i, name in enumerate(["a (Å)", "b (Å)", "c (Å)"]):
        col = abc[:, i]
        print(f"  {name}: range [{col.min():.4f}, {col.max():.4f}], variance = {col.var():.6f}")
    print(f"  (n_groups = {len(unified_groups)})")


if __name__ == "__main__":
    main()
