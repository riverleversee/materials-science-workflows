#!/usr/bin/env python3
"""
Build and search an eigenvalue-based FD surrogate from CP2K outputs.

Model (dimensionless x where dq = x * delta):
  lambda_k(x) = lambda0_k + sum_i dlam[k,i]*x_i + sum_{i<=j} clam[k,i,j]*x_i*x_j

The stress-tensor surrogate is also reconstructed to report principal axes.
Enthalpy/internal-energy are computed from FD energy+volume surrogate:
  H(x) = H0 + sum_i dH_i*x_i + sum_{i<=j} kH_ij*x_i*x_j
  E(x) = H(x) - P_iso * V(q0 + dq) * conv(bar*A^3->Ha)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BAR_ANG3_TO_HARTREE = 2.2937122783963248e-8
PARAM_LABELS = ("a", "b", "c", "alpha", "beta", "gamma")


def _import_uniax_scripts(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    from stress_parser import parse_stress_from_output  # type: ignore
    from cell_utils import read_cell, cell_to_abc_angles, abc_angles_to_cell  # type: ignore

    return parse_stress_from_output, read_cell, cell_to_abc_angles, abc_angles_to_cell


def parse_total_energy_from_output(out_path: Path) -> float:
    text = out_path.read_text(errors="replace")
    matches = re.findall(
        r"ENERGY\|\s+Total FORCE_EVAL[^\n]*?(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return float(matches[-1])
    matches = re.findall(
        r"Total energy:\s*(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return float(matches[-1])
    raise ValueError(f"Could not parse energy from {out_path}")


def vec6_to_mat3(v6: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = v6
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def principal_sorted(mat3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(mat3)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def cell_volume_from_abc(q: np.ndarray, abc_angles_to_cell) -> float:
    cell = abc_angles_to_cell(*q)
    return float(abs(np.linalg.det(cell)))


@dataclass
class FDData:
    q0: np.ndarray
    deltas: np.ndarray
    sigma0: np.ndarray
    sigma_p: Dict[int, np.ndarray]
    sigma_pp: Dict[Tuple[int, int], np.ndarray]
    eig0: np.ndarray
    eig_p: Dict[int, np.ndarray]
    eig_pp: Dict[Tuple[int, int], np.ndarray]
    H0: float
    E0: float
    P_iso: float
    H_p: Dict[int, float]
    H_pp: Dict[Tuple[int, int], float]
    dlam: np.ndarray
    clam: Dict[Tuple[int, int], np.ndarray]
    d_sigma: np.ndarray
    c_sigma: Dict[Tuple[int, int], np.ndarray]
    dH: np.ndarray
    cH: Dict[Tuple[int, int], float]


def build_fd_data(
    work_dir: Path,
    scripts_dir: Path,
    target_stress6: np.ndarray,
    deltas: np.ndarray,
) -> FDData:
    parse_stress, read_cell, cell_to_abc, abc_to_cell = _import_uniax_scripts(scripts_dir)

    sigma0 = np.array(parse_stress(work_dir / "sp_base.out"), dtype=float)
    mat0 = vec6_to_mat3(sigma0)
    eig0, _ = principal_sorted(mat0)
    E0 = parse_total_energy_from_output(work_dir / "sp_base.out")
    q0 = np.array(cell_to_abc(read_cell(work_dir / "cell_base.cell")), dtype=float)
    V0 = cell_volume_from_abc(q0, abc_to_cell)
    P_iso = float(np.mean(target_stress6[:3]))
    H0 = E0 + P_iso * V0 * BAR_ANG3_TO_HARTREE

    sigma_p: Dict[int, np.ndarray] = {}
    eig_p: Dict[int, np.ndarray] = {}
    H_p: Dict[int, float] = {}
    for i in range(6):
        s = np.array(parse_stress(work_dir / f"sp_p{i}.out"), dtype=float)
        sigma_p[i] = s
        eig_p[i], _ = principal_sorted(vec6_to_mat3(s))
        Ei = parse_total_energy_from_output(work_dir / f"sp_p{i}.out")
        qi = q0.copy()
        qi[i] += deltas[i]
        Vi = cell_volume_from_abc(qi, abc_to_cell)
        H_p[i] = Ei + P_iso * Vi * BAR_ANG3_TO_HARTREE

    sigma_pp: Dict[Tuple[int, int], np.ndarray] = {}
    eig_pp: Dict[Tuple[int, int], np.ndarray] = {}
    H_pp: Dict[Tuple[int, int], float] = {}
    for i in range(6):
        for j in range(i, 6):
            out = work_dir / f"sp_pp{i}_{j}.out"
            if not out.exists():
                continue
            s = np.array(parse_stress(out), dtype=float)
            sigma_pp[(i, j)] = s
            eig_pp[(i, j)], _ = principal_sorted(vec6_to_mat3(s))
            Eij = parse_total_energy_from_output(out)
            qij = q0.copy()
            qij[i] += deltas[i]
            qij[j] += deltas[j]
            Vij = cell_volume_from_abc(qij, abc_to_cell)
            H_pp[(i, j)] = Eij + P_iso * Vij * BAR_ANG3_TO_HARTREE

    dlam = np.zeros((3, 6), dtype=float)
    d_sigma = np.zeros((6, 6), dtype=float)
    dH = np.zeros(6, dtype=float)
    for i in range(6):
        dlam[:, i] = eig_p[i] - eig0
        d_sigma[:, i] = sigma_p[i] - sigma0
        dH[i] = H_p[i] - H0

    clam: Dict[Tuple[int, int], np.ndarray] = {}
    c_sigma: Dict[Tuple[int, int], np.ndarray] = {}
    cH: Dict[Tuple[int, int], float] = {}
    for i in range(6):
        for j in range(i, 6):
            if (i, j) not in eig_pp:
                continue
            if i == j:
                clam[(i, j)] = eig_pp[(i, j)] - 2.0 * eig_p[i] + eig0
                c_sigma[(i, j)] = sigma_pp[(i, j)] - 2.0 * sigma_p[i] + sigma0
                cH[(i, j)] = H_pp[(i, j)] - 2.0 * H_p[i] + H0
            else:
                clam[(i, j)] = eig_pp[(i, j)] - eig_p[i] - eig_p[j] + eig0
                c_sigma[(i, j)] = sigma_pp[(i, j)] - sigma_p[i] - sigma_p[j] + sigma0
                cH[(i, j)] = H_pp[(i, j)] - H_p[i] - H_p[j] + H0

    return FDData(
        q0=q0,
        deltas=deltas,
        sigma0=sigma0,
        sigma_p=sigma_p,
        sigma_pp=sigma_pp,
        eig0=eig0,
        eig_p=eig_p,
        eig_pp=eig_pp,
        H0=H0,
        E0=E0,
        P_iso=P_iso,
        H_p=H_p,
        H_pp=H_pp,
        dlam=dlam,
        clam=clam,
        d_sigma=d_sigma,
        c_sigma=c_sigma,
        dH=dH,
        cH=cH,
    )


def predict_eigs(fd: FDData, x: np.ndarray) -> np.ndarray:
    lam = fd.eig0 + fd.dlam @ x
    for (i, j), c in fd.clam.items():
        lam = lam + c * x[i] * x[j]
    return lam


def predict_sigma(fd: FDData, x: np.ndarray) -> np.ndarray:
    s = fd.sigma0 + fd.d_sigma @ x
    for (i, j), c in fd.c_sigma.items():
        s = s + c * x[i] * x[j]
    return s


def predict_H(fd: FDData, x: np.ndarray) -> float:
    h = fd.H0 + float(np.dot(fd.dH, x))
    for (i, j), c in fd.cH.items():
        h += c * x[i] * x[j]
    return float(h)


def predict_E(fd: FDData, x: np.ndarray, abc_angles_to_cell) -> float:
    q = fd.q0 + x * fd.deltas
    V = cell_volume_from_abc(q, abc_angles_to_cell)
    return float(predict_H(fd, x) - fd.P_iso * V * BAR_ANG3_TO_HARTREE)


def eig_obj(fd: FDData, x: np.ndarray, target_eigs: np.ndarray, w: np.ndarray) -> float:
    r = target_eigs - predict_eigs(fd, x)
    rw = w * r
    return float(np.dot(rw, rw))


def local_refine(
    fd: FDData,
    x0: np.ndarray,
    step: np.ndarray,
    x_bounds: np.ndarray,
    target_eigs: np.ndarray,
    w: np.ndarray,
    max_iter: int,
) -> Tuple[np.ndarray, float]:
    x = x0.copy()
    f = eig_obj(fd, x, target_eigs, w)
    for _ in range(max_iter):
        improved = False
        for k in range(6):
            for sgn in (-1.0, 1.0):
                xt = x.copy()
                xt[k] = np.clip(xt[k] + sgn * step[k], -x_bounds[k], x_bounds[k])
                ft = eig_obj(fd, xt, target_eigs, w)
                if ft < f:
                    x, f = xt, ft
                    improved = True
        for i in range(6):
            for j in range(i + 1, 6):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        xt = x.copy()
                        xt[i] = np.clip(xt[i] + si * step[i], -x_bounds[i], x_bounds[i])
                        xt[j] = np.clip(xt[j] + sj * step[j], -x_bounds[j], x_bounds[j])
                        ft = eig_obj(fd, xt, target_eigs, w)
                        if ft < f:
                            x, f = xt, ft
                            improved = True
        if not improved:
            break
    return x, f


def build_axis(bounds: float, step: float) -> np.ndarray:
    n = int(np.floor((2.0 * bounds) / step)) + 1
    return np.linspace(-bounds, bounds, n)


def rms6_aeq(q0: np.ndarray, dq: np.ndarray) -> float:
    l_alpha = 0.5 * (q0[1] + q0[2])
    l_beta = 0.5 * (q0[0] + q0[2])
    l_gamma = 0.5 * (q0[0] + q0[1])
    vec = np.array(
        [
            abs(dq[0]),
            abs(dq[1]),
            abs(dq[2]),
            abs(np.deg2rad(dq[3])) * l_alpha,
            abs(np.deg2rad(dq[4])) * l_beta,
            abs(np.deg2rad(dq[5])) * l_gamma,
        ],
        dtype=float,
    )
    return float(np.sqrt(np.mean(vec * vec)))


def main():
    ap = argparse.ArgumentParser(description="Eigenvalue FD surrogate search and ranking.")
    ap.add_argument("--work-dir", required=True, help="Directory containing sp_*.out and cell_*.cell")
    ap.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parents[1] / "uniax/uniax_manual/scripts"))
    ap.add_argument("--target-stress", nargs=6, type=float, required=True, metavar=("SXX", "SYY", "SZZ", "SXY", "SXZ", "SYZ"))
    ap.add_argument("--delta-length-ang", type=float, default=0.005)
    ap.add_argument("--delta-angle-deg", type=float, default=0.05)
    ap.add_argument("--x-bounds", nargs=6, type=float, default=[80, 80, 80, 100, 100, 100])
    ap.add_argument("--coarse-steps", nargs=6, type=float, default=[20, 20, 20, 30, 20, 30])
    ap.add_argument("--mid-steps", nargs=6, type=float, default=[0.32, 0.32, 0.32, 0.4, 0.4, 0.4])
    ap.add_argument("--fine-steps", nargs=6, type=float, default=[0.16, 0.16, 0.16, 0.2, 0.2, 0.2])
    ap.add_argument("--seed-count", type=int, default=180)
    ap.add_argument("--uniq-seed-dist", type=float, default=1.5)
    ap.add_argument("--uniq-final-dist", type=float, default=0.75)
    ap.add_argument("--mid-max-iter", type=int, default=120)
    ap.add_argument("--fine-max-iter", type=int, default=500)
    ap.add_argument("--group-threshold-aeq", type=float, default=0.05)
    ap.add_argument(
        "--eig-tol-bar",
        type=float,
        default=3000.0,
        help="Max allowed absolute principal-stress residual per eigenvalue [bar] for accepted minima",
    )
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--out-prefix", default="eigen_fd_search")
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    scripts_dir = Path(args.scripts_dir).resolve()
    target_stress6 = np.array(args.target_stress, dtype=float)
    target_eigs, _ = principal_sorted(vec6_to_mat3(target_stress6))
    deltas = np.array([args.delta_length_ang] * 3 + [args.delta_angle_deg] * 3, dtype=float)
    x_bounds = np.array(args.x_bounds, dtype=float)
    coarse_steps = np.array(args.coarse_steps, dtype=float)
    mid_steps = np.array(args.mid_steps, dtype=float)
    fine_steps = np.array(args.fine_steps, dtype=float)

    parse_stress, read_cell, cell_to_abc, abc_to_cell = _import_uniax_scripts(scripts_dir)
    _ = (parse_stress, read_cell, cell_to_abc)  # keep import for parity and lint silence

    fd = build_fd_data(work_dir, scripts_dir, target_stress6, deltas)

    # Build search mesh and refine
    axes = [build_axis(x_bounds[i], coarse_steps[i]) for i in range(6)]
    mesh = np.array(np.meshgrid(*axes, indexing="ij")).reshape(6, -1).T
    w_eigs = np.array([1.0, 1.0, 1.0], dtype=float)
    vals = np.array([eig_obj(fd, x, target_eigs, w_eigs) for x in mesh], dtype=float)
    order = np.argsort(vals)

    refined: List[Tuple[np.ndarray, float]] = []
    for ii in order[: args.seed_count]:
        xr, fr = local_refine(fd, mesh[ii], mid_steps, x_bounds, target_eigs, w_eigs, args.mid_max_iter)
        refined.append((xr, fr))
    refined.sort(key=lambda t: t[1])

    uniq_seeds: List[np.ndarray] = []
    for x, _f in refined:
        if not any(np.linalg.norm(x - u) < args.uniq_seed_dist for u in uniq_seeds):
            uniq_seeds.append(x)

    finals: List[dict] = []
    for x0 in uniq_seeds:
        x, f = local_refine(fd, x0, fine_steps, x_bounds, target_eigs, w_eigs, args.fine_max_iter)
        lam = predict_eigs(fd, x)
        rlam = target_eigs - lam
        sig6 = predict_sigma(fd, x)
        sig3 = vec6_to_mat3(sig6)
        evals_sig, evecs_sig = principal_sorted(sig3)
        # Uniaxial/principal axis: eigenvector of largest principal stress
        uniax_vec = evecs_sig[:, 2]
        H = predict_H(fd, x)
        E = predict_E(fd, x, abc_to_cell)
        dq = x * fd.deltas
        finals.append(
            {
                "x": x,
                "eig_obj": float(f),
                "lam_pred": lam,
                "eig_resid": rlam,
                "sigma_eval_pred": evals_sig,
                "uniax_vec": uniax_vec,
                "H_pred": H,
                "E_pred": E,
                "dH": H - fd.H0,
                "dE": E - fd.E0,
                "dq": dq,
                "rms6_aeq": rms6_aeq(fd.q0, dq),
            }
        )

    # Apply principal-stress residual tolerance filter
    finals_tol = [row for row in finals if float(np.max(np.abs(row["eig_resid"]))) <= args.eig_tol_bar]
    if finals_tol:
        finals = finals_tol
    else:
        print(
            f"Warning: no minima satisfy eig tolerance {args.eig_tol_bar:.3f} bar; "
            "reporting unfiltered minima.",
            file=sys.stderr,
        )

    # De-duplicate local minima in x-space
    finals_sorted = sorted(finals, key=lambda d: d["eig_obj"])
    uniq_finals: List[dict] = []
    for row in finals_sorted:
        if not any(np.linalg.norm(row["x"] - u["x"]) < args.uniq_final_dist for u in uniq_finals):
            uniq_finals.append(row)

    # Group by 0.05 A-equivalent style on dq-equivalent vector, keep lowest enthalpy representative
    l_alpha = 0.5 * (fd.q0[1] + fd.q0[2])
    l_beta = 0.5 * (fd.q0[0] + fd.q0[2])
    l_gamma = 0.5 * (fd.q0[0] + fd.q0[1])

    def eq_vec(dq: np.ndarray) -> np.ndarray:
        return np.array(
            [
                abs(dq[0]),
                abs(dq[1]),
                abs(dq[2]),
                abs(np.deg2rad(dq[3])) * l_alpha,
                abs(np.deg2rad(dq[4])) * l_beta,
                abs(np.deg2rad(dq[5])) * l_gamma,
            ],
            dtype=float,
        )

    remaining = list(range(len(uniq_finals)))
    groups: List[dict] = []
    while remaining:
        i = remaining.pop(0)
        rep = uniq_finals[i]
        rep_eq = eq_vec(rep["dq"])
        members = [i]
        keep = []
        for j in remaining:
            if np.all(np.abs(eq_vec(uniq_finals[j]["dq"]) - rep_eq) <= args.group_threshold_aeq):
                members.append(j)
            else:
                keep.append(j)
        remaining = keep
        best = min(members, key=lambda k: uniq_finals[k]["H_pred"])
        groups.append({"best": best, "members": members})

    grouped = []
    for g in groups:
        row = dict(uniq_finals[g["best"]])
        row["n_equiv"] = len(g["members"])
        grouped.append(row)
    grouped.sort(key=lambda d: d["H_pred"])

    # Write coefficient report
    coeff_path = work_dir / f"{args.out_prefix}_coefficients.txt"
    with open(coeff_path, "w") as f:
        f.write("Eigenvalue FD surrogate coefficients\n")
        f.write("Model: lam_k(x)=lam0_k + sum_i dlam[k,i] x_i + sum_{i<=j} clam[k,i,j] x_i x_j\n")
        f.write("Parameters: a b c alpha beta gamma\n")
        f.write(f"Target eigenvalues [bar] (ascending): {' '.join(f'{v:.6f}' for v in target_eigs)}\n")
        f.write(f"Base eigenvalues [bar]   (ascending): {' '.join(f'{v:.6f}' for v in fd.eig0)}\n")
        f.write("Deltas [A, A, A, deg, deg, deg]: " + " ".join(f"{d:.8f}" for d in fd.deltas) + "\n\n")
        f.write("Linear eigenvalue trends dlam = eig_p - eig0 (rows=eig1,eig2,eig3; cols=params):\n")
        for k in range(3):
            f.write(f"eig{k+1}: " + " ".join(f"{fd.dlam[k, j]: .6f}" for j in range(6)) + "\n")
        f.write("\nPairwise eigenvalue interactions clam_ij (i<=j):\n")
        for i in range(6):
            for j in range(i, 6):
                if (i, j) in fd.clam:
                    c = fd.clam[(i, j)]
                    f.write(f"({PARAM_LABELS[i]},{PARAM_LABELS[j]}): " + " ".join(f"{x: .6f}" for x in c) + "\n")
        f.write("\nEnergy/enthalpy coefficients:\n")
        f.write(f"E0 [Ha]: {fd.E0:.12f}\n")
        f.write(f"H0 [Ha]: {fd.H0:.12f}\n")
        f.write("dH_i [Ha]: " + " ".join(f"{fd.dH[i]: .12e}" for i in range(6)) + "\n")
        f.write("cH_ij [Ha], i<=j:\n")
        for i in range(6):
            for j in range(i, 6):
                if (i, j) in fd.cH:
                    f.write(f"({PARAM_LABELS[i]},{PARAM_LABELS[j]}): {fd.cH[(i, j)]: .12e}\n")

    # Write grouped minima (enthalpy-ranked)
    csv_path = work_dir / f"{args.out_prefix}_grouped_ranked.csv"
    fields = [
        "rank_by_enthalpy",
        "n_equiv",
        "H_pred_Ha",
        "E_pred_Ha",
        "dH_Ha",
        "dE_Ha",
        "eig_obj",
        "eig_resid_norm",
        "eig_resid_max_abs_bar",
        "eig_pred_1",
        "eig_pred_2",
        "eig_pred_3",
        "uniax_vec_x",
        "uniax_vec_y",
        "uniax_vec_z",
        "rms6_aeq",
        "da",
        "db",
        "dc",
        "dalpha_deg",
        "dbeta_deg",
        "dgamma_deg",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, row in enumerate(grouped[: args.top_n], 1):
            eig_resid = row["eig_resid"]
            dq = row["dq"]
            uv = row["uniax_vec"]
            w.writerow(
                {
                    "rank_by_enthalpy": rank,
                    "n_equiv": row["n_equiv"],
                    "H_pred_Ha": f"{row['H_pred']:.12f}",
                    "E_pred_Ha": f"{row['E_pred']:.12f}",
                    "dH_Ha": f"{row['dH']:.12e}",
                    "dE_Ha": f"{row['dE']:.12e}",
                    "eig_obj": f"{row['eig_obj']:.6f}",
                    "eig_resid_norm": f"{np.linalg.norm(eig_resid):.6f}",
                    "eig_resid_max_abs_bar": f"{np.max(np.abs(eig_resid)):.6f}",
                    "eig_pred_1": f"{row['lam_pred'][0]:.6f}",
                    "eig_pred_2": f"{row['lam_pred'][1]:.6f}",
                    "eig_pred_3": f"{row['lam_pred'][2]:.6f}",
                    "uniax_vec_x": f"{uv[0]:.9f}",
                    "uniax_vec_y": f"{uv[1]:.9f}",
                    "uniax_vec_z": f"{uv[2]:.9f}",
                    "rms6_aeq": f"{row['rms6_aeq']:.9f}",
                    "da": f"{dq[0]:.9f}",
                    "db": f"{dq[1]:.9f}",
                    "dc": f"{dq[2]:.9f}",
                    "dalpha_deg": f"{dq[3]:.9f}",
                    "dbeta_deg": f"{dq[4]:.9f}",
                    "dgamma_deg": f"{dq[5]:.9f}",
                }
            )

    # Console summary for quick comparison
    print(f"Wrote coefficients: {coeff_path}")
    print(f"Wrote grouped minima: {csv_path}")
    print(f"Unique minima: {len(uniq_finals)}  Grouped representatives: {len(grouped)}")
    print("Top cases by enthalpy (rank, H_pred, E_pred, eig_resid_max_abs_bar, n_equiv, uniax_vec):")
    for rank, row in enumerate(grouped[: min(args.top_n, 10)], 1):
        uv = row["uniax_vec"]
        print(
            f"{rank:2d}  H={row['H_pred']:.12f}  E={row['E_pred']:.12f}  "
            f"max|dlam|={np.max(np.abs(row['eig_resid'])):.3f}  n={row['n_equiv']}  "
            f"u=({uv[0]:+.6f},{uv[1]:+.6f},{uv[2]:+.6f})"
        )


if __name__ == "__main__":
    main()

