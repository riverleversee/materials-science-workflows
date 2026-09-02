#!/usr/bin/env python3
"""
Compare low-enthalpy minima (principal stress axes) to compliance tensor at x=0.

Compliance tensor meaning:
  - S = inv(C), C = d_sigma/d_epsilon. Eigenvector v of S: strain mode ε = v for stress σ = v/λ (λ = eigenvalue).
  - Eigenvalue λ = compliance: larger λ = softer (more strain per unit stress), smaller λ = stiffer.
  - v is a 6-vector Voigt [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy].
  - The 3D direction of the strain mode is NOT (v0,v1,v2): that is the diagonal of the strain tensor.
    Correct: build 3x3 strain E from v, take principal eigenvector of E (= direction of max normal strain).

Minimum enthalpy vs compliance (H = E + P*V is minimized):
  - Under applied stress σ, strain ε = Sσ. Work done ON the cell = (1/2)σ:ε = (1/2)σ:Sσ → stored strain energy, adds to E.
  - So H = E + P*V increases by (1/2)σ:Sσ (plus P*ΔV). To minimize H we want strain energy (1/2)σ:Sσ as small as possible.
  - For fixed uniaxial |σ|, (1/2)σ:Sσ is smallest when σ is along the STIFFEST compliance direction (smallest λ).
  - So minimum-enthalpy expectation: principal stress aligns with STIFFEST mode (not softest).
"""
import json
import sys
from pathlib import Path

import numpy as np

from uniax_surrogate_optimizer_test import build_elastic_tensors

# Strain Voigt order in this code: [xx, yy, zz, yz, xz, xy] (engineering shear = 2*e_ij)
def strain_vec6_to_mat3(v6: np.ndarray) -> np.ndarray:
    """Convert 6-vector [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy] to 3x3 symmetric strain."""
    exx, eyy, ezz = v6[0], v6[1], v6[2]
    eyz, exz, exy = v6[3] / 2.0, v6[4] / 2.0, v6[5] / 2.0
    return np.array([[exx, exy, exz], [exy, eyy, eyz], [exz, eyz, ezz]], dtype=float)


def principal_strain_direction_from_voigt6(v6: np.ndarray) -> np.ndarray | None:
    """Principal eigenvector (direction of max normal strain) from compliance eigenvector 6-vec."""
    E = strain_vec6_to_mat3(v6)
    vals, vecs = np.linalg.eigh(E)
    # Largest eigenvalue = principal strain direction
    imax = int(np.argmax(np.abs(vals)))
    d = vecs[:, imax].ravel()
    n = np.linalg.norm(d)
    if n < 1e-12:
        return None
    d = d / n
    if d[0] < 0:
        d = -d
    return d


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0, 1))))


def main():
    C, S = build_elastic_tensors(np.zeros(6))
    S_vals, S_vecs = np.linalg.eigh(S)
    order = np.argsort(S_vals)[::-1]  # largest compliance (softest) first
    S_vals = S_vals[order]
    S_vecs = S_vecs[:, order]

    # For each S eigenvector: 3D direction = principal strain direction of the 3x3 strain from v
    comp_axes = []
    comp_normal_frac = []
    for j in range(6):
        v = S_vecs[:, j]
        norm_full = np.linalg.norm(v)
        norm_normal = np.linalg.norm(v[:3])
        frac = (norm_normal / norm_full) ** 2 if norm_full > 1e-10 else 0.0
        comp_normal_frac.append(frac)
        d = principal_strain_direction_from_voigt6(v)
        comp_axes.append(d)

    comp_physical = [comp_axes[j] is not None for j in range(6)]
    physical_axes = [j for j in range(6) if comp_physical[j]]
    physical_dirs = [comp_axes[j] for j in physical_axes]
    # Stiffest mode = smallest compliance λ => minimum strain energy (1/2)σ:Sσ for fixed uniaxial |σ|
    stiffest_idx = int(np.argmin(S_vals))
    softest_idx = int(np.argmax(S_vals))

    print("=" * 72)
    print("COMPLIANCE TENSOR at x=0")
    print("=" * 72)
    print("S eigenvector = strain mode (ε = v for σ = v/λ). Larger λ = softer.")
    print("3D direction = principal strain axis of 3x3 strain from v (not diagonal v0,v1,v2).")
    print("Eigenvalues (compliance, bar^-1):", S_vals)
    print(f"  Softest mode (largest λ): {softest_idx+1}; stiffest (smallest λ): {stiffest_idx+1}.")
    print("  H = E + PV minimized → strain energy (1/2)σ:Sσ minimum → stress along STIFFEST (min λ).")
    print("\nPer mode: compliance λ, normal_frac, principal strain direction [nx,ny,nz]:")
    for j in range(6):
        p = " [3D]" if comp_physical[j] else " (degenerate/mixed)"
        if comp_axes[j] is not None:
            d = comp_axes[j]
            print(f"  Mode {j+1}  λ={S_vals[j]:.2e}  normal_frac={comp_normal_frac[j]:.1%}{p}")
            print(f"         principal strain dir = [{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]")
        else:
            print(f"  Mode {j+1}  λ={S_vals[j]:.2e}  normal_frac={comp_normal_frac[j]:.1%}{p}")

    # Load cluster reps from log (or recompute)
    json_path = Path(__file__).parent / "local_minima_by_scale.json"
    if not json_path.exists():
        print("Run log_local_minima_by_scale.py first")
        return

    with open(json_path) as f:
        all_by_scale = json.load(f)

    all_records = []
    for scale, recs in all_by_scale.items():
        for r in recs:
            r["scale"] = scale
            all_records.append(r)

    sorted_all = sorted(all_records, key=lambda r: r["H"])

    # Build clusters from sorted_all (sorted by H), group within 30°
    used = [False] * len(sorted_all)
    clusters = []
    for i in range(len(sorted_all)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        vi = np.array(sorted_all[i]["eigvec_principal"])
        for j in range(i + 1, len(sorted_all)):
            if used[j]:
                continue
            vj = np.array(sorted_all[j]["eigvec_principal"])
            if angle_deg(vi, vj) <= 30.0:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)

    clusters_sorted = sorted(clusters, key=lambda c: sorted_all[c[0]]["H"])
    cluster_reps_sorted = [sorted_all[c[0]] for c in clusters_sorted]

    # ---- Low-enthalpy focus: top N clusters by H ----
    n_low = 15
    low_H_reps = cluster_reps_sorted[:n_low]
    H_low = low_H_reps[0]["H"] if low_H_reps else None
    H_high = low_H_reps[-1]["H"] if low_H_reps else None

    print("\n" + "=" * 72)
    print("LOW-ENTHALPY CLUSTERS vs COMPLIANCE (principal stress eigvec)")
    print("=" * 72)
    print(f"Using lowest-H rep per cluster (30° grouping). Showing top {n_low} clusters by H.")
    if low_H_reps:
        print(f"H range: {H_low:.6e} .. {H_high:.6e} [Ha]")
    print("Compliance 3D dir = principal strain axis of each mode (from 3x3 strain matrix).")
    if low_H_reps and comp_axes[stiffest_idx] is not None:
        v_best = np.array(low_H_reps[0]["eigvec_principal"])
        ang_stiffest = angle_deg(v_best, comp_axes[stiffest_idx])
        print(f"Lowest-H cluster angle to STIFFEST mode (mode {stiffest_idx+1}, min H expectation): {ang_stiffest:.1f}°.")
    print()

    print(f"{'Cluster':<8} {'H [Ha]':>14} {'n':>4} | Min angle (all 6) | Best mode | Angle to stiffest")
    print("-" * 72)
    for k, r in enumerate(low_H_reps):
        v = np.array(r["eigvec_principal"])
        min_ang_all = 180.0
        best_j = -1
        for j in range(6):
            if comp_axes[j] is None:
                continue
            a = angle_deg(v, comp_axes[j])
            if a < min_ang_all:
                min_ang_all = a
                best_j = j
        min_ang_stiffest = angle_deg(v, comp_axes[stiffest_idx]) if comp_axes[stiffest_idx] is not None else float("nan")
        n_in_cluster = len(clusters_sorted[k])
        best_label = f"Mode {best_j+1}" if best_j >= 0 else "—"
        print(f"  {k+1:<6} {r['H']:>14.6e} {n_in_cluster:>4} | {min_ang_all:>14.1f}° | {best_label:>8} | {min_ang_stiffest:>6.1f}°")

    # ---- REVIEW: Cases vs compliance predictions ----
    print("\n" + "=" * 72)
    print("REVIEW: Cases vs compliance predictions")
    print("=" * 72)
    print("Prediction (H = E + PV minimized): principal stress should align with STIFFEST")
    print(f"  mode (mode {stiffest_idx+1}, λ={S_vals[stiffest_idx]:.2e} bar^-1).")
    print("  Stiffest principal strain dir =", np.round(comp_axes[stiffest_idx], 4).tolist())
    print()
    agrees = []
    for k, r in enumerate(low_H_reps):
        ang_s = angle_deg(np.array(r["eigvec_principal"]), comp_axes[stiffest_idx]) if comp_axes[stiffest_idx] is not None else 180.0
        # Agree = within 30° of stiffest
        agree = ang_s <= 30.0
        if agree:
            agrees.append(k + 1)
    print(f"Clusters with principal stress within 30° of stiffest (mode {stiffest_idx+1}): {agrees if agrees else 'none'} ({len(agrees)}/{len(low_H_reps)}).")
    print()
    print("Case summary (low-H clusters, sorted by H):")
    print(f"  {'#':<4} {'H [Ha]':>14} {'Best mode':>10} {'Angle to stiffest':>18} {'Matches prediction?':<20}")
    print("  " + "-" * 68)
    for k, r in enumerate(low_H_reps):
        v = np.array(r["eigvec_principal"])
        best_j = -1
        min_a = 180.0
        for j in range(6):
            if comp_axes[j] is None:
                continue
            a = angle_deg(v, comp_axes[j])
            if a < min_a:
                min_a = a
                best_j = j
        ang_stiff = angle_deg(v, comp_axes[stiffest_idx]) if comp_axes[stiffest_idx] is not None else float("nan")
        match = "yes (≤30°)" if ang_stiff <= 30.0 else "no"
        print(f"  {k+1:<4} {r['H']:>14.6e} Mode {best_j+1:<4} {ang_stiff:>14.1f}°       {match:<20}")
    print()
    if agrees:
        print(f"Verdict: {len(agrees)} cluster(s) match the min-H prediction (stress near stiffest).")
    else:
        print("Verdict: No cluster has principal stress within 30° of stiffest; lowest-H aligns best with other modes.")
    print("  (Full H surface includes enthalpy curvature and P*V; linear compliance at x=0 is only part of the story.)")

    print("\n" + "=" * 72)
    print("DETAILED: Low-H cluster reps vs each compliance axis (angles in deg)")
    print("=" * 72)
    for k, r in enumerate(low_H_reps[:10]):
        v = np.array(r["eigvec_principal"])
        print(f"\nCluster {k+1}  H={r['H']:.6e}  stress principal = [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]")
        for j in range(6):
            if comp_axes[j] is None:
                continue
            a = angle_deg(v, comp_axes[j])
            mark = " (principal strain dir)" if comp_physical[j] else ""
            print(f"  vs compliance mode {j+1} (λ={S_vals[j]:.2e}): {a:>5.1f}°{mark}")


if __name__ == "__main__":
    out_path = Path(__file__).parent / "compliance_vs_minima_results.txt"
    with open(out_path, "w") as f:
        class Tee:
            def __init__(self, file_handle, real_stdout):
                self.file = file_handle
                self.real = real_stdout
            def write(self, data):
                self.file.write(data)
                self.real.write(data)
            def flush(self):
                self.file.flush()
                self.real.flush()
        old_stdout = sys.stdout
        sys.stdout = Tee(f, old_stdout)
        try:
            main()
        finally:
            sys.stdout = old_stdout
    print(f"\n[Results also written to {out_path}]")
