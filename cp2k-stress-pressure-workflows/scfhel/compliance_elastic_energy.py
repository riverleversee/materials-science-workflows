#!/usr/bin/env python3
"""
Apply a uniaxial stress tensor (desired eigenvalues) along specified axes and use
the compliance tensor at x=0 to predict the elastic strain energy (1/2)σ:ε.

Coordinates: stress and compliance are in the same Cartesian (cell) frame. We build
σ in that frame with principal axis along a given unit vector n, then ε = Sσ, and
strain energy density = (1/2) σ : ε with correct Voigt pairing.

Voigt: stress [xx, yy, zz, xy, xz, yz]; strain from S is [xx, yy, zz, yz, xz, xy].
"""
from pathlib import Path

import numpy as np

from uniax_surrogate_optimizer_test import (
    build_elastic_tensors,
    mat3_to_vec6,
)
from compare_minima_to_compliance import (
    strain_vec6_to_mat3,
    principal_strain_direction_from_voigt6,
)

# Target uniaxial eigenvalues [bar]: two equal, one different (from build_data target)
# Principal axis for the "different" value (156667) is the uniaxial direction.
P_ISO_BAR = 146667.0
P_UNIQ_BAR = 156667.0  # the axis we orient along n


def uniaxial_stress_3x3(n: np.ndarray, p_iso: float = P_ISO_BAR, p_uniq: float = P_UNIQ_BAR) -> np.ndarray:
    """
    Build 3x3 stress tensor with eigenvalues (p_iso, p_iso, p_uniq) and the p_uniq
    principal axis along unit vector n (in cell frame).
    σ = p_iso*I + (p_uniq - p_iso)*n⊗n.
    """
    n = np.asarray(n, dtype=float).ravel()
    n = n / (np.linalg.norm(n) + 1e-15)
    return p_iso * np.eye(3) + (p_uniq - p_iso) * np.outer(n, n)


def stress_voigt_from_principal_axis(n: np.ndarray) -> np.ndarray:
    """Uniaxial stress with axis n; return Voigt [xx, yy, zz, xy, xz, yz]."""
    sigma_3x3 = uniaxial_stress_3x3(n)
    return mat3_to_vec6(sigma_3x3)


def strain_energy_density(sigma_vec: np.ndarray, eps_vec: np.ndarray) -> float:
    """
    (1/2) σ : ε [bar] with correct Voigt pairing.
    sigma_vec order: [xx, yy, zz, xy, xz, yz]
    eps_vec order:   [xx, yy, zz, yz, xz, xy] (from S @ sigma)
    σ:ε = σ_xx*ε_xx + σ_yy*ε_yy + σ_zz*ε_zz + σ_xy*γ_xy + σ_xz*γ_xz + σ_yz*γ_yz
    """
    return 0.5 * (
        sigma_vec[0] * eps_vec[0]
        + sigma_vec[1] * eps_vec[1]
        + sigma_vec[2] * eps_vec[2]
        + sigma_vec[3] * eps_vec[5]
        + sigma_vec[4] * eps_vec[4]
        + sigma_vec[5] * eps_vec[3]
    )


def main():
    # Compliance at x=0; S maps stress [xx,yy,zz,xy,xz,yz] -> strain [xx,yy,zz,yz,xz,xy]
    C, S = build_elastic_tensors(np.zeros(6))
    S_vals, S_vecs = np.linalg.eigh(S)
    order = np.argsort(S_vals)[::-1]
    S_vals = S_vals[order]
    S_vecs = S_vecs[:, order]

    stiffest_idx = int(np.argmin(S_vals))
    softest_idx = int(np.argmax(S_vals))

    # Principal strain directions (3D) for each compliance mode
    comp_dirs = []
    for j in range(6):
        d = principal_strain_direction_from_voigt6(S_vecs[:, j])
        comp_dirs.append(d)

    # Directions to test: stiffest, softest, and a couple of low-H cluster axes
    labels = []
    directions = []
    labels.append("Stiffest (mode 6, min λ)")
    directions.append(comp_dirs[stiffest_idx])
    labels.append("Softest (mode 1, max λ)")
    directions.append(comp_dirs[softest_idx])
    # Add mode 2 and 4 (high normal-frac) for comparison
    labels.append("Mode 2")
    directions.append(comp_dirs[1])
    labels.append("Mode 4")
    directions.append(comp_dirs[3])

    # Load lowest-H cluster principal stress from log if available
    json_path = Path(__file__).parent / "local_minima_by_scale.json"
    if json_path.exists():
        import json
        with open(json_path) as f:
            data = json.load(f)
        all_recs = []
        for scale, recs in data.items():
            for r in recs:
                r["scale"] = scale
                all_recs.append(r)
        all_recs.sort(key=lambda r: r["H"])
        for i, r in enumerate(all_recs[:3]):
            v = np.array(r["eigvec_principal"], dtype=float)
            v = v / (np.linalg.norm(v) + 1e-15)
            if v[0] < 0:
                v = -v
            labels.append(f"Low-H cluster rep {i+1} (H={r['H']:.2e})")
            directions.append(v)

    print("=" * 72)
    print("ELASTIC STRAIN ENERGY FROM COMPLIANCE FOR UNIAXIAL STRESS ALONG AXES")
    print("=" * 72)
    print("Uniaxial stress: eigenvalues (146667, 146667, 156667) bar.")
    print("Principal axis for 156667 bar is the direction n. σ = p_iso*I + (p_uniq-p_iso)*n⊗n.")
    print("Strain ε = S σ; strain energy density = (1/2) σ : ε [bar].")
    print("All in same Cartesian (cell) frame.")
    print()
    print(f"{'Direction':<45} {'(1/2)σ:Sσ [bar]':>18} {'n (first 3)':<25}")
    print("-" * 72)

    collected = []
    for lab, n in zip(labels, directions):
        if n is None:
            continue
        sigma_vec = stress_voigt_from_principal_axis(n)
        eps_vec = S @ sigma_vec  # strain in order [xx,yy,zz,yz,xz,xy]
        w = strain_energy_density(sigma_vec, eps_vec)
        collected.append((lab, w))
        print(f"  {lab:<43} {w:>18.6f}   [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]")

    by_energy = sorted(collected, key=lambda x: x[1])
    print()
    print("Ranked by (1/2)σ:Sσ (algebraically smallest first; S has negative eigenvalues so values can be < 0):")
    for i, (lab, w) in enumerate(by_energy):
        print(f"  {i+1}. {lab}: {w:.2f} bar")
    print()
    print("Min-H prediction (H = E+PV): strain energy (1/2)σ:Sσ minimum → stiffest direction.")
    print("(When S is not positive definite, the stiffest axis need not give the algebraically smallest (1/2)σ:Sσ.)")


if __name__ == "__main__":
    main()
