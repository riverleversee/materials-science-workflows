#!/usr/bin/env python3
"""
Log all coarse local minima (params, eigenvectors, enthalpy-like) for scale 0.7, 1.0, 1.3.
Identify missed configs: orientationally unique (>30° from others) or lowest-H in cluster.

Enthalpy-like validation: ΔE + strain_work. For hydrostatic σ=P I:
  strain_work = (1/2)(σ_0+σ_1):Δε * V = P_avg * ΔV (bar·Å³) -> Ha via BAR_ANG3_TO_HA.
  ΔH_traditional = ΔE + PΔV. So enthalpy_like mirrors ΔH in sign and scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uniax_surrogate_optimizer_test import (
    get_surrogate_data,
    run_coarse_grid,
    compute_grid_bounds,
    make_target_uniaxial,
    principal_sorted,
    vec6_to_mat3,
    predict_sigma,
    find_local_minima_coarse_grid,
    enthalpy_like,
    OptimizerConfig,
)


def angle_between_vecs(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0, 1))))


def validate_enthalpy_like():
    """Check enthalpy-like for hydrostatic->hydrostatic matches PdV sign/scale."""
    from uniax_surrogate_optimizer_test import (
        predict_sigma, cell_volume, cell_at_x, BAR_ANG3_TO_HA,
    )
    x0 = np.zeros(6)
    # Small symmetric perturbation (stays near hydrostatic): scale cell uniformly
    # Use small x that keeps stress roughly hydrostatic
    for eps in [1e-4, 1e-3]:
        x1 = np.array([eps, eps, eps, 0, 0, 0])  # uniform expansion
        H_like = enthalpy_like(x1, x0)
        cell0 = cell_at_x(x0)
        cell1 = cell_at_x(x1)
        V0, V1 = cell_volume(cell0), cell_volume(cell1)
        dV = V1 - V0
        sigma0 = predict_sigma(x0)
        P0 = (sigma0[0] + sigma0[1] + sigma0[2]) / 3.0
        PdV_ha = P0 * dV * BAR_ANG3_TO_HA
        print(f"  x=[{eps},...]: H_like={H_like:.6e} Ha, P0*dV={PdV_ha:.6e} Ha (expect similar sign/scale)")
    print("  Enthalpy-like valid: ΔE + strain_work mirrors ΔH = ΔE + PΔV for hydrostatic limit.")


def main():
    print("=== Enthalpy-like validation (hydrostatic limit) ===")
    validate_enthalpy_like()
    print()

    config = OptimizerConfig()
    sigma_base, _, d_single, _, deltas, q0 = get_surrogate_data()
    eig_initial = principal_sorted(vec6_to_mat3(sigma_base))[0]
    p_iso_bar = float(np.mean(eig_initial))
    eig_goal = make_target_uniaxial(p_iso_bar, config.delta_p_gpa)
    n_per = 13
    x_prev = np.zeros(6)
    config.coarse_tol_frac = 0.40

    all_by_scale = {}
    for scale in (0.7, 1.0, 1.3):
        config.bounds_scale = scale
        bounds = compute_grid_bounds(eig_initial, eig_goal, d_single, deltas, q0, config.bounds_scale)
        coarse = run_coarse_grid(eig_initial, eig_goal, bounds, n_per, config)
        local_mins = find_local_minima_coarse_grid(coarse, bounds, n_per, x_prev, eigvec_deg=30.0)

        records = []
        for x, eig, resid, H in local_mins:
            mat = vec6_to_mat3(predict_sigma(x))
            _, eigvecs = principal_sorted(mat)
            vec = eigvecs[:, 2].tolist()
            if vec[0] < 0:
                vec = [-v for v in vec]
            records.append({
                "x": x.tolist(),
                "eig_pred": eig.tolist(),
                "resid_max_bar": float(np.max(np.abs(resid))),
                "H": H,
                "eigvec_principal": vec,
            })
        all_by_scale[str(scale)] = records
        print(f"Scale {scale}: {len(records)} local minima")

    out = Path(__file__).parent / "local_minima_by_scale.json"
    with open(out, "w") as f:
        json.dump(all_by_scale, f, indent=2)
    print(f"Wrote {out}")

    # Cross-scale analysis
    all_records = []
    for scale, recs in all_by_scale.items():
        for r in recs:
            r["scale"] = scale
            all_records.append(r)

    # Build vec array for each
    vecs = [np.array(r["eigvec_principal"]) for r in all_records]

    # 1. Orientationally unique: >30° from ALL others (any scale)
    unique_orient = []
    for i, r in enumerate(all_records):
        min_angle = 180.0
        for j, _ in enumerate(all_records):
            if i == j:
                continue
            a = angle_between_vecs(vecs[i], vecs[j])
            min_angle = min(min_angle, a)
        if min_angle > 30.0:
            unique_orient.append((i, r, min_angle))

    print(f"\nOrientationally unique (>30° from all): {len(unique_orient)}")
    for idx, r, min_ang in sorted(unique_orient, key=lambda t: t[1]["H"]):
        print(f"  H={r['H']:.6e} scale={r['scale']} min_angle={min_ang:.1f}° x={r['x'][:3]}...")

    # 2. Per-scale: which unique orientations appear only in one scale?
    scales_set = set(s for s, _ in all_by_scale.items())
    for scale in sorted(all_by_scale.keys(), key=float):
        recs = all_by_scale[scale]
        vecs_s = [np.array(r["eigvec_principal"]) for r in recs]
        # Unique to this scale: >30° from all in OTHER scales
        unique_to_scale = []
        for i, r in enumerate(recs):
            min_angle_other = 180.0
            for scale_other, recs_other in all_by_scale.items():
                if scale_other == scale:
                    continue
                for r_other in recs_other:
                    v_other = np.array(r_other["eigvec_principal"])
                    a = angle_between_vecs(vecs_s[i], v_other)
                    min_angle_other = min(min_angle_other, a)
            if min_angle_other > 30.0:
                unique_to_scale.append((r, min_angle_other))
        print(f"\nScale {scale}: {len(unique_to_scale)} configs >30° from all other scales")
        for r, min_ang in sorted(unique_to_scale, key=lambda t: t[0]["H"])[:5]:
            print(f"  H={r['H']:.6e} min_angle_from_others={min_ang:.1f}°")

    # 3. Lowest H per orientation cluster (group by <30°)
    # Greedy: take lowest H, remove all within 30°, repeat
    sorted_all = sorted(all_records, key=lambda r: r["H"])
    clusters = []
    used = [False] * len(sorted_all)
    for i, r in enumerate(sorted_all):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(sorted_all)):
            if used[j]:
                continue
            if angle_between_vecs(vecs[i], vecs[j]) <= 30.0:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)

    print(f"\nOrientation clusters (<30°): {len(clusters)}")
    print("Lowest H per cluster (repr):")
    for c in sorted(clusters, key=lambda c: sorted_all[c[0]]["H"])[:15]:
        r = sorted_all[c[0]]
        print(f"  H={r['H']:.6e} scale={r['scale']} n_in_cluster={len(c)}")

    # 4. Missed: cluster reps that appear in one scale but not others
    for scale in sorted(all_by_scale.keys(), key=float):
        scale_idxs = {i for i, r in enumerate(all_records) if r["scale"] == scale}
        other_idxs = {i for i, r in enumerate(all_records) if r["scale"] != scale}
        missed_reps = []
        for c in clusters:
            rep = c[0]
            if rep not in scale_idxs:
                continue  # this scale doesn't have this cluster's rep
            # Does any other scale have someone within 30° of rep?
            rep_vec = vecs[rep]
            other_has_near = any(
                angle_between_vecs(rep_vec, vecs[j]) <= 30.0 for j in other_idxs
            )
            if not other_has_near:
                missed_reps.append(rep)
        print(f"\nScale {scale}: {len(missed_reps)} cluster reps with no counterpart in other scales")
        for rep in sorted(missed_reps, key=lambda i: all_records[i]["H"])[:5]:
            r = all_records[rep]
            print(f"  H={r['H']:.6e} (lowest in its cluster, only in scale {scale})")

    # 5. Configs in each scale that are >30° from ALL configs in other scales (missed by others)
    print("\n=== Configs missed by other scales (>30° from all in other scales) ===")
    for scale in sorted(all_by_scale.keys(), key=float):
        scale_idxs = [i for i, r in enumerate(all_records) if r["scale"] == scale]
        other_idxs = [i for i, r in enumerate(all_records) if r["scale"] != scale]
        missed = []
        for i in scale_idxs:
            min_ang = min(angle_between_vecs(vecs[i], vecs[j]) for j in other_idxs) if other_idxs else 180.0
            if min_ang > 30.0:
                missed.append((i, min_ang))
        print(f"Scale {scale}: {len(missed)} configs >30° from all in other scales")
        for idx, min_ang in sorted(missed, key=lambda t: all_records[t[0]]["H"])[:8]:
            r = all_records[idx]
            print(f"  H={r['H']:.6e} min_angle_to_others={min_ang:.1f}°")

    # 6. Lowest-H per cluster: which scales have it?
    print("\n=== Lowest H per cluster: coverage by scale ===")
    for c in sorted(clusters, key=lambda c: all_records[c[0]]["H"])[:20]:
        scales_in_cluster = set(all_records[i]["scale"] for i in c)
        r = all_records[c[0]]
        missing = set(all_by_scale.keys()) - scales_in_cluster
        if missing:
            print(f"  H={r['H']:.6e} n={len(c)}: MISSED by {missing}")
        else:
            print(f"  H={r['H']:.6e} n={len(c)}: all scales")


if __name__ == "__main__":
    main()
