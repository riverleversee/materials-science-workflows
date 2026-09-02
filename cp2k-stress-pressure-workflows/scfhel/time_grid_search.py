#!/usr/bin/env python3
"""Time the grid search (coarse + dense) - stops after fine grid finishes."""
import time

import numpy as np

from uniax_surrogate_optimizer_test import (
    get_surrogate_data,
    run_coarse_grid,
    run_dense_grid,
    compute_grid_bounds,
    make_target_uniaxial,
    principal_sorted,
    vec6_to_mat3,
    OptimizerConfig,
    PARAM_LABELS,
)

def main():
    config = OptimizerConfig()
    config.max_coarse_for_dense = 50  # limit for timing

    t0 = time.perf_counter()
    sigma_base, target, d_single, c_pairs, deltas, q0 = get_surrogate_data()
    eig_initial = principal_sorted(vec6_to_mat3(sigma_base))[0]
    p_iso_bar = float(np.mean(eig_initial))
    eig_goal = make_target_uniaxial(p_iso_bar, config.delta_p_gpa)
    bounds = compute_grid_bounds(eig_initial, eig_goal, d_single, deltas, q0)
    t_setup = time.perf_counter() - t0
    print(f"Setup: {t_setup:.2f} s")

    t0 = time.perf_counter()
    n_per = 13
    coarse_matches = run_coarse_grid(eig_initial, eig_goal, bounds, n_per, config)
    t_coarse = time.perf_counter() - t0
    print(f"Coarse grid ({n_per}^6 points, {len(coarse_matches)} matches): {t_coarse:.2f} s")

    t0 = time.perf_counter()
    coarse_for_dense = coarse_matches[: config.max_coarse_for_dense] if config.max_coarse_for_dense else coarse_matches
    dense_matches = run_dense_grid(
        coarse_for_dense, eig_initial, eig_goal, config,
        n_fine_per_axis=6, coarse_bounds=bounds, coarse_n_per=n_per,
    )
    t_dense = time.perf_counter() - t0
    n_dense_points = len(coarse_for_dense) * (6 ** 6)
    print(f"Dense grid ({len(coarse_for_dense)} coarse x 5^6 = {n_dense_points} points, {len(dense_matches)} matches): {t_dense:.2f} s")

    total = t_setup + t_coarse + t_dense
    print(f"\nTotal (through fine grid): {total:.2f} s")
    print(f"  Coarse: {100*t_coarse/total:.1f}%")
    print(f"  Dense:  {100*t_dense/total:.1f}%")


if __name__ == "__main__":
    main()
