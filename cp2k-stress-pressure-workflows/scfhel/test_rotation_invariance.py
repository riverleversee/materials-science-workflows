#!/usr/bin/env python3
"""
Fast test: what changes under a common rotation of cell0 and cell1?

We compute strain directly in each frame via the optimizer's
strain_from_cell_deformation(). We do NOT "rotate E manually".

Facts:
  - E components depend on the chosen Cartesian frame, so they generally change.
  - Rotation-invariant scalars (eigenvalues of E, tr(E), ||E||_F, det(E)) are unchanged.
  - The scalar work-like contraction sigma:E is invariant if sigma is rotated the same way.
"""

from __future__ import annotations

import numpy as np

import uniax_surrogate_optimizer_test as opt


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random proper rotation via QR, det=+1."""
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def invariants(E: np.ndarray) -> dict:
    vals = np.linalg.eigvalsh(E)
    return {
        "eig_sorted": np.sort(vals),
        "tr": float(np.trace(E)),
        "fro": float(np.linalg.norm(E)),
        "det": float(np.linalg.det(E)),
    }


def main() -> None:
    rng = np.random.default_rng(0)

    # Non-degenerate baseline cells (rows are A,B,C).
    q0 = np.array([4.7, 4.8, 6.0, 93.0, 101.0, 88.0], dtype=float)
    cell0 = opt.abc_to_cell(*q0)

    dq = np.array([0.02, -0.01, 0.03, 0.2, -0.1, 0.15], dtype=float)
    cell1 = opt.abc_to_cell(*(q0 + dq))

    # Baseline strain in baseline frame
    eps0 = opt.strain_from_cell_deformation(cell0, cell1)
    E0 = opt.voigt_strain_to_mat3(eps0)
    inv0 = invariants(E0)
    V0 = float(abs(np.linalg.det(cell1)))  # volume at "current" cell (matches optimizer's convention)

    print("Baseline eps(voigt):", eps0)
    print("Baseline invariants:", {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in inv0.items()})
    print("Baseline V(det(cell1)):", V0)

    for k in range(3):
        R = random_rotation(rng)
        cell0p = R @ cell0
        cell1p = R @ cell1
        V1 = float(abs(np.linalg.det(cell1p)))

        # Strain computed directly in the rotated Cartesian frame
        eps1 = opt.strain_from_cell_deformation(cell0p, cell1p)
        E1 = opt.voigt_strain_to_mat3(eps1)
        inv1 = invariants(E1)

        # Components change (generally)
        max_comp_delta = float(np.max(np.abs(eps1 - eps0)))

        # Invariants should match
        eig_delta = float(np.max(np.abs(inv1["eig_sorted"] - inv0["eig_sorted"])))
        tr_delta = abs(inv1["tr"] - inv0["tr"])
        fro_delta = abs(inv1["fro"] - inv0["fro"])
        det_delta = abs(inv1["det"] - inv0["det"])

        # Work-like contraction invariance if sigma is rotated consistently.
        sigma = rng.normal(size=(3, 3))
        sigma = 0.5 * (sigma + sigma.T)
        sigma_p = R @ sigma @ R.T
        w0 = float(np.sum(sigma * E0))
        w1 = float(np.sum(sigma_p * E1))
        w_delta = abs(w1 - w0)

        # Include volume factor: (sigma:E) * V should also be invariant (since V is rotation-invariant).
        sv0 = w0 * V0
        sv1 = w1 * V1
        sv_delta = abs(sv1 - sv0)

        print(f"\nRotation {k}:")
        print("  max|eps' - eps|     =", max_comp_delta, "(expected nonzero)")
        print("  max|eig(E')-eig(E)| =", eig_delta)
        print("  |tr' - tr|          =", tr_delta)
        print("  |fro' - fro|        =", fro_delta)
        print("  |det' - det|        =", det_delta)
        print("  |(sigma':E')-(sigma:E)| =", w_delta)
        print("  |(sigma':E')V'-(sigma:E)V| =", sv_delta)

    # Sanity: if cell1==cell0, strain should be ~0 even after rotation
    R = random_rotation(rng)
    eps_same = opt.strain_from_cell_deformation(R @ cell0, R @ cell0)
    print("\nSame-cell rotated strain max abs:", float(np.max(np.abs(eps_same))))


if __name__ == "__main__":
    main()

