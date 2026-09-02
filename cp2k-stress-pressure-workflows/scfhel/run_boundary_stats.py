#!/usr/bin/env python3
"""Run coarse grid boundary/adjacent stats for 20% and 30% tolerance (bounds_scale=1.3)."""
import sys
sys.path.insert(0, ".")
import numpy as np
from uniax_surrogate_optimizer_test import (
    get_surrogate_data,
    run_coarse_grid,
    compute_grid_bounds,
    make_target_uniaxial,
    principal_sorted,
    vec6_to_mat3,
    coarse_boundary_adjacent_stats,
    count_principal_eigvec_isolated,
    group_matches_by_principal_eigvec,
    _boundary_indices,
    OptimizerConfig,
)

def main():
    config = OptimizerConfig()
    config.bounds_scale = 1.3
    sigma_base, target, d_single, c_pairs, deltas, q0 = get_surrogate_data()
    eig_initial = principal_sorted(vec6_to_mat3(sigma_base))[0]
    p_iso_bar = float(sum(eig_initial) / 3)
    eig_goal = make_target_uniaxial(p_iso_bar, config.delta_p_gpa)
    bounds = compute_grid_bounds(eig_initial, eig_goal, d_single, deltas, q0, config.bounds_scale)
    n_per = 13

    print("Grid bounds (x):", bounds)
    print("\nCOARSE GRID BOUNDARY/ADJACENT STATS (bounds_scale=1.3)")
    print("=" * 60)
    for tol_pct in (20, 30):
        config.coarse_tol_frac = tol_pct / 100.0
        coarse_tol = run_coarse_grid(eig_initial, eig_goal, bounds, n_per, config)
        n_bnd, n_adj, n_better = coarse_boundary_adjacent_stats(coarse_tol, bounds, n_per)
        n_isolated = count_principal_eigvec_isolated(
            coarse_tol, deg_threshold=30.0, bounds=bounds, n_per=n_per
        )
        print(f"  {tol_pct}% coarse tol: {len(coarse_tol)} matches, {n_bnd} at boundary; "
              f"{n_adj} with adjacent agreeing non-boundary neighbor; "
              f"{n_better} with better agreement at neighbor; "
              f"{n_isolated} with principal eigvec >30° from all non-boundary others")
    print("=" * 60)

    # Principal stress axis grouping (30% tol, 30° threshold)
    config.coarse_tol_frac = 0.30
    coarse_30 = run_coarse_grid(eig_initial, eig_goal, bounds, n_per, config)
    groups = group_matches_by_principal_eigvec(coarse_30, deg_threshold=30.0)
    boundary_idx = _boundary_indices(coarse_30, bounds, n_per)

    print("\nPRINCIPAL STRESS AXIS GROUPING (30% tol, 30° threshold)")
    print("=" * 60)
    for g, group in enumerate(groups):
        idx_list = [gi[0] for gi in group]
        n_bnd = sum(1 for i in idx_list if i in boundary_idx)
        rep_vec = group[0][2].copy()
        if rep_vec[0] < 0:
            rep_vec *= -1  # normalize to +x hemisphere for readability
        print(f"  Group {g+1}: {len(group)} matches ({n_bnd} at boundary)")
        print(f"    Representative axis: [{rep_vec[0]:+.4f}, {rep_vec[1]:+.4f}, {rep_vec[2]:+.4f}]")
        print(f"    Match indices: {idx_list}")
    print("=" * 60)

if __name__ == "__main__":
    main()
