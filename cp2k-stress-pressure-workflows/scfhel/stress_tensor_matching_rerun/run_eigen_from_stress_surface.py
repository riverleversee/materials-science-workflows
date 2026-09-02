#!/usr/bin/env python3
"""
Eigenvalue matching on top of the parameterized stress-tensor surrogate.

Uses stress surrogate from trend log:
  sigma(x) = sigma0 + dSingle*x + sum_{i<j} c_ij*x_i*x_j

Eigenvalues are computed directly from sigma(x) (no separate eigenvalue fit).
Enthalpy/internal-energy are evaluated from the enthalpy surrogate:
  H(x) = H0 + dH*x + sum_{i<=j} k_ij*x_i*x_j
  E(x) = H(x) - P_iso*V(q0 + x*delta)

Workflow:
  1) multistart local search minimizing eigenvalue mismatch objective
  2) unique-minima dedup in x-space
  3) filter by max principal-stress mismatch <= eig_tol_bar
  4) sort by enthalpy and write outputs
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BAR_ANG3_TO_HA = 2.2937122783963248e-8
PARAMS = ("a", "b", "c", "alpha", "beta", "gamma")
STRESS_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")


def parse_cycle_block(text: str, cycle: int) -> str:
    m = re.search(rf"Cycle {cycle}\n(.*?)\n===========", text, flags=re.S)
    if not m:
        raise ValueError(f"Cycle {cycle} block not found")
    return m.group(1)


def parse_row_after_label(block: str, label: str) -> np.ndarray:
    m = re.search(rf"{re.escape(label)}\n\s+([^\n]+)", block)
    if not m:
        raise ValueError(f"Missing label: {label}")
    return np.fromstring(m.group(1), sep=" ")


def parse_model(block: str):
    qm = re.search(
        r"a=([0-9Ee+\-.]+)\s+b=([0-9Ee+\-.]+)\s+c=([0-9Ee+\-.]+)\s+"
        r"alpha=([0-9Ee+\-.]+)\s+beta=([0-9Ee+\-.]+)\s+gamma=([0-9Ee+\-.]+)",
        block,
    )
    if not qm:
        raise ValueError("Failed to parse q0")
    q0 = np.array([float(qm.group(i)) for i in range(1, 7)], dtype=float)

    deltas = np.zeros(6, dtype=float)
    for i, p in enumerate(PARAMS):
        dm = re.search(rf"\b{p}:\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+\[(?:A|deg)\]", block)
        if not dm:
            raise ValueError(f"Missing delta for {p}")
        deltas[i] = float(dm.group(1))

    sigma0 = parse_row_after_label(block, "Base stress [xx yy zz xy xz yz] bar:")
    target = parse_row_after_label(block, "Target stress [xx yy zz xy xz yz] bar:")

    d_single = np.zeros((6, 6), dtype=float)
    for i, s in enumerate(STRESS_LABELS):
        rm = re.search(rf"\n\s*{s}\s+([^\n]+)", block)
        if not rm:
            raise ValueError(f"Missing dSingle row for {s}")
        d_single[i, :] = np.fromstring(rm.group(1), sep=" ")

    idx = {p: i for i, p in enumerate(PARAMS)}
    c_pairs: Dict[Tuple[int, int], np.ndarray] = {}
    for mm in re.finditer(
        r"\((a|b|c|alpha|beta|gamma),(a|b|c|alpha|beta|gamma)\):\s+([^\n]+)",
        block,
    ):
        vals = np.fromstring(mm.group(3), sep=" ")
        if vals.size == 6:
            i, j = idx[mm.group(1)], idx[mm.group(2)]
            c_pairs[(min(i, j), max(i, j))] = vals

    em = re.search(r"E_base \[Ha\]:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", block)
    hm = re.search(r"H_base = E \+ P_iso\*V \[Ha\]:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", block)
    if not em or not hm:
        raise ValueError("Missing base energetics")
    e0 = float(em.group(1))
    h0 = float(hm.group(1))
    p_iso = float(np.mean(target[:3]))

    dH = np.zeros(6, dtype=float)
    for i, p in enumerate(PARAMS):
        hm_i = re.search(rf"\b{p}:\s+dH=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+\[Ha\]", block)
        if not hm_i:
            raise ValueError(f"Missing dH for {p}")
        dH[i] = float(hm_i.group(1))

    k_section = re.search(
        r"Pairwise enthalpy interactions k_ij \[Ha\] \(includes diagonal i=j\):\n(.*?)\nNormalized enthalpy pair curvature",
        block,
        flags=re.S,
    )
    if not k_section:
        raise ValueError("Missing pairwise enthalpy section")
    cH: Dict[Tuple[int, int], float] = {}
    for mm in re.finditer(
        r"\((a|b|c|alpha|beta|gamma),(a|b|c|alpha|beta|gamma)\):\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
        k_section.group(1),
    ):
        i, j = idx[mm.group(1)], idx[mm.group(2)]
        cH[(min(i, j), max(i, j))] = float(mm.group(3))

    return q0, deltas, sigma0, target, d_single, c_pairs, e0, h0, p_iso, dH, cH


def abc_to_cell(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    sg = np.sin(gamma)
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    c_x = c * np.cos(beta)
    c_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sg
    c_z = np.sqrt(max(0.0, c * c - c_x * c_x - c_y * c_y))
    return np.array([a_vec, b_vec, np.array([c_x, c_y, c_z])], dtype=float)


def vec6_to_mat3(v):
    xx, yy, zz, xy, xz, yz = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def principal_sorted(v6):
    vals, vecs = np.linalg.eigh(vec6_to_mat3(v6))
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def main():
    ap = argparse.ArgumentParser(description="Eigenvalue matching from stress-tensor surrogate.")
    ap.add_argument("--trend-log", required=True)
    ap.add_argument("--cycle", type=int, default=1)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-prefix", default="eigen_from_stress")
    ap.add_argument("--eig-tol-bar", type=float, default=2000.0, help="Per-eigenvalue mismatch tolerance [bar]")
    ap.add_argument("--x-bounds", nargs=6, type=float, default=[80, 80, 80, 100, 100, 100])
    ap.add_argument("--coarse-steps", nargs=6, type=float, default=[20, 20, 20, 30, 20, 30])
    ap.add_argument("--mid-steps", nargs=6, type=float, default=[0.32, 0.32, 0.32, 0.4, 0.4, 0.4])
    ap.add_argument("--fine-steps", nargs=6, type=float, default=[0.16, 0.16, 0.16, 0.2, 0.2, 0.2])
    ap.add_argument("--seed-count", type=int, default=180)
    ap.add_argument("--uniq-seed-dist", type=float, default=1.5)
    ap.add_argument("--uniq-final-dist", type=float, default=0.75)
    ap.add_argument("--mid-max-iter", type=int, default=120)
    ap.add_argument("--fine-max-iter", type=int, default=500)
    ap.add_argument("--random-start-count", type=int, default=120, help="Extra random multistarts for robustness")
    ap.add_argument("--jump-iterations", type=int, default=12, help="Basin-hopping jumps per seed")
    ap.add_argument(
        "--jump-scales",
        nargs=6,
        type=float,
        default=[4, 4, 4, 7, 7, 7],
        help="Normal jump scales in x-space for [a,b,c,alpha,beta,gamma]",
    )
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument(
        "--group-threshold-aeq",
        type=float,
        default=0.05,
        help="Component-wise equivalence threshold in A-equivalent space for grouping",
    )
    args = ap.parse_args()

    text = Path(args.trend_log).read_text()
    block = parse_cycle_block(text, args.cycle)
    q0, delta, sigma0, target6, d_single, c_pairs, e0, h0, p_iso, dH, cH = parse_model(block)
    target_eig, _ = principal_sorted(target6)

    x_bounds = np.array(args.x_bounds, dtype=float)
    coarse_steps = np.array(args.coarse_steps, dtype=float)
    mid_steps = np.array(args.mid_steps, dtype=float)
    fine_steps = np.array(args.fine_steps, dtype=float)
    jump_scales = np.array(args.jump_scales, dtype=float)
    rng = np.random.default_rng(args.rng_seed)

    def pred_sigma(x):
        s = sigma0 + d_single @ x
        for (i, j), c in c_pairs.items():
            s = s + c * x[i] * x[j]
        return s

    def pred_eig(x):
        vals, vecs = principal_sorted(pred_sigma(x))
        return vals, vecs

    def eig_resid(x):
        vals, _ = pred_eig(x)
        return target_eig - vals

    def eig_obj(x):
        r = eig_resid(x)
        return float(np.dot(r, r))

    def pred_h(x):
        h = h0 + float(np.dot(dH, x))
        for (i, j), c in cH.items():
            h += c * x[i] * x[j]
        return h

    def pred_e(x):
        q = q0 + x * delta
        v = abs(np.linalg.det(abc_to_cell(*q)))
        return pred_h(x) - p_iso * v * BAR_ANG3_TO_HA

    def local_refine(x0, step, max_iter):
        x = x0.copy()
        f = eig_obj(x)
        for _ in range(max_iter):
            improved = False
            for k in range(6):
                for sgn in (-1.0, 1.0):
                    xt = x.copy()
                    xt[k] = np.clip(xt[k] + sgn * step[k], -x_bounds[k], x_bounds[k])
                    ft = eig_obj(xt)
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
                            ft = eig_obj(xt)
                            if ft < f:
                                x, f = xt, ft
                                improved = True
            if not improved:
                break
        return x, f

    axes = [np.linspace(-x_bounds[i], x_bounds[i], int(np.floor((2.0 * x_bounds[i]) / coarse_steps[i])) + 1) for i in range(6)]
    mesh = np.array(np.meshgrid(*axes, indexing="ij")).reshape(6, -1).T
    mesh_obj = np.array([eig_obj(x) for x in mesh], dtype=float)
    order = np.argsort(mesh_obj)

    refined = []
    for ii in order[: args.seed_count]:
        x, f = local_refine(mesh[ii], mid_steps, args.mid_max_iter)
        refined.append((x, f))
    refined.sort(key=lambda t: t[1])

    uniq_seed = []
    for x, _f in refined:
        if not any(np.linalg.norm(x - u) < args.uniq_seed_dist for u in uniq_seed):
            uniq_seed.append(x)

    robust_seeds = list(uniq_seed)
    for _ in range(args.random_start_count):
        robust_seeds.append(rng.uniform(-x_bounds, x_bounds))
    for x0 in uniq_seed[: min(30, len(uniq_seed))]:
        robust_seeds.append(np.clip(x0 + rng.normal(0.0, jump_scales, size=6), -x_bounds, x_bounds))

    finals = []
    for x0 in robust_seeds:
        x, f = local_refine(x0, fine_steps, args.fine_max_iter)
        for _ in range(args.jump_iterations):
            xj = np.clip(x + rng.normal(0.0, jump_scales, size=6), -x_bounds, x_bounds)
            xj, _ = local_refine(xj, mid_steps, args.mid_max_iter)
            xj, fj = local_refine(xj, fine_steps, args.fine_max_iter)
            if fj < f:
                x, f = xj, fj
        rr = eig_resid(x)
        vals, vecs = pred_eig(x)
        # principal axis = eigenvector for largest principal stress
        u = vecs[:, 2]
        finals.append(
            {
                "x": x,
                "obj": float(f),
                "eig_resid": rr,
                "eig_resid_norm": float(np.linalg.norm(rr)),
                "eig_resid_max": float(np.max(np.abs(rr))),
                "eig_pred": vals,
                "uniax_vec": u,
                "H": float(pred_h(x)),
                "E": float(pred_e(x)),
                "dH": float(pred_h(x) - h0),
                "dE": float(pred_e(x) - e0),
                "dq": x * delta,
            }
        )

    finals = sorted(finals, key=lambda d: d["obj"])
    uniq_finals = []
    for row in finals:
        if not any(np.linalg.norm(row["x"] - u["x"]) < args.uniq_final_dist for u in uniq_finals):
            uniq_finals.append(row)

    # Tolerance filter in eigenvalue space
    filt = [row for row in uniq_finals if row["eig_resid_max"] <= args.eig_tol_bar]
    if filt:
        uniq_finals = filt
    else:
        print(
            f"Warning: no minima satisfy eig tolerance {args.eig_tol_bar:.3f} bar; "
            "reporting unfiltered minima."
        )

    # Sort by enthalpy as requested
    uniq_finals = sorted(uniq_finals, key=lambda d: d["H"])

    # Group by component-wise A-equivalent dq similarity, representative = minimum enthalpy.
    l_alpha = 0.5 * (q0[1] + q0[2])
    l_beta = 0.5 * (q0[0] + q0[2])
    l_gamma = 0.5 * (q0[0] + q0[1])

    def dq_aeq_vec(dq):
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
    groups = []
    while remaining:
        i = remaining.pop(0)
        rep = uniq_finals[i]
        rep_vec = dq_aeq_vec(rep["dq"])
        members = [i]
        keep = []
        for j in remaining:
            if np.all(np.abs(dq_aeq_vec(uniq_finals[j]["dq"]) - rep_vec) <= args.group_threshold_aeq):
                members.append(j)
            else:
                keep.append(j)
        remaining = keep
        best = min(members, key=lambda k: uniq_finals[k]["H"])
        groups.append({"best": best, "members": members})

    grouped = []
    for g in groups:
        row = dict(uniq_finals[g["best"]])
        row["n_equiv"] = len(g["members"])
        grouped.append(row)
    grouped = sorted(grouped, key=lambda d: d["H"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{args.output_prefix}_ranked_by_enthalpy_cycle{args.cycle}.txt"
    csv_path = out_dir / f"{args.output_prefix}_ranked_by_enthalpy_cycle{args.cycle}.csv"
    gtxt_path = out_dir / f"{args.output_prefix}_grouped_ranked_by_enthalpy_cycle{args.cycle}.txt"
    gcsv_path = out_dir / f"{args.output_prefix}_grouped_ranked_by_enthalpy_cycle{args.cycle}.csv"

    with open(txt_path, "w") as f:
        f.write(f"Eigenvalue-from-stress local minima (cycle {args.cycle})\n")
        f.write(f"Input trend log: {args.trend_log}\n")
        f.write(f"Eigenvalue tolerance per entry [bar]: {args.eig_tol_bar:.6f}\n")
        f.write(f"Target principal stresses [bar] (asc): {' '.join(f'{v:.6f}' for v in target_eig)}\n")
        f.write(f"Accepted minima count: {len(uniq_finals)}\n")
        f.write("Columns: rank H E eig_resid_max eig_resid_norm eig_pred[3] uniax_vec[3] dq[6]\n\n")
        for i, row in enumerate(uniq_finals, 1):
            f.write(
                f"{i:4d}  H={row['H']:.12f}  E={row['E']:.12f}  "
                f"eig_resid_max={row['eig_resid_max']:.6f}  eig_resid_norm={row['eig_resid_norm']:.6f}\n"
            )
            f.write("      eig_pred=" + " ".join(f"{v:.6f}" for v in row["eig_pred"]) + "\n")
            f.write("      eig_resid=" + " ".join(f"{v:.6f}" for v in row["eig_resid"]) + "\n")
            f.write("      uniax_vec=" + " ".join(f"{v:.9f}" for v in row["uniax_vec"]) + "\n")
            f.write("      dq=" + " ".join(f"{v:.9f}" for v in row["dq"]) + "\n")

    with open(csv_path, "w", newline="") as f:
        fields = [
            "rank",
            "H",
            "E",
            "dH",
            "dE",
            "eig_obj",
            "eig_resid_norm",
            "eig_resid_max",
            "eig_pred_1",
            "eig_pred_2",
            "eig_pred_3",
            "eig_resid_1",
            "eig_resid_2",
            "eig_resid_3",
            "uniax_vec_x",
            "uniax_vec_y",
            "uniax_vec_z",
            "da",
            "db",
            "dc",
            "dalpha_deg",
            "dbeta_deg",
            "dgamma_deg",
        ]
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        for i, row in enumerate(uniq_finals, 1):
            dq = row["dq"]
            wcsv.writerow(
                {
                    "rank": i,
                    "H": f"{row['H']:.12f}",
                    "E": f"{row['E']:.12f}",
                    "dH": f"{row['dH']:.12e}",
                    "dE": f"{row['dE']:.12e}",
                    "eig_obj": f"{row['obj']:.12f}",
                    "eig_resid_norm": f"{row['eig_resid_norm']:.12f}",
                    "eig_resid_max": f"{row['eig_resid_max']:.12f}",
                    "eig_pred_1": f"{row['eig_pred'][0]:.12f}",
                    "eig_pred_2": f"{row['eig_pred'][1]:.12f}",
                    "eig_pred_3": f"{row['eig_pred'][2]:.12f}",
                    "eig_resid_1": f"{row['eig_resid'][0]:.12f}",
                    "eig_resid_2": f"{row['eig_resid'][1]:.12f}",
                    "eig_resid_3": f"{row['eig_resid'][2]:.12f}",
                    "uniax_vec_x": f"{row['uniax_vec'][0]:.12f}",
                    "uniax_vec_y": f"{row['uniax_vec'][1]:.12f}",
                    "uniax_vec_z": f"{row['uniax_vec'][2]:.12f}",
                    "da": f"{dq[0]:.12f}",
                    "db": f"{dq[1]:.12f}",
                    "dc": f"{dq[2]:.12f}",
                    "dalpha_deg": f"{dq[3]:.12f}",
                    "dbeta_deg": f"{dq[4]:.12f}",
                    "dgamma_deg": f"{dq[5]:.12f}",
                }
            )

    with open(gtxt_path, "w") as f:
        f.write(f"Grouped eigenvalue-from-stress minima (cycle {args.cycle})\n")
        f.write(f"Input trend log: {args.trend_log}\n")
        f.write(f"Eigenvalue tolerance per entry [bar]: {args.eig_tol_bar:.6f}\n")
        f.write(f"Grouping threshold (A-eq, component-wise): {args.group_threshold_aeq:.6f}\n")
        f.write(f"Target principal stresses [bar] (asc): {' '.join(f'{v:.6f}' for v in target_eig)}\n")
        f.write(f"Grouped representatives: {len(grouped)}\n")
        f.write("Columns: rank n_equiv H E eig_resid_max eig_resid_norm eig_pred[3] uniax_vec[3] dq[6]\n\n")
        for i, row in enumerate(grouped, 1):
            f.write(
                f"{i:4d}  n_equiv={row['n_equiv']:4d}  H={row['H']:.12f}  E={row['E']:.12f}  "
                f"eig_resid_max={row['eig_resid_max']:.6f}  eig_resid_norm={row['eig_resid_norm']:.6f}\n"
            )
            f.write("      eig_pred=" + " ".join(f"{v:.6f}" for v in row["eig_pred"]) + "\n")
            f.write("      eig_resid=" + " ".join(f"{v:.6f}" for v in row["eig_resid"]) + "\n")
            f.write("      uniax_vec=" + " ".join(f"{v:.9f}" for v in row["uniax_vec"]) + "\n")
            f.write("      dq=" + " ".join(f"{v:.9f}" for v in row["dq"]) + "\n")

    with open(gcsv_path, "w", newline="") as f:
        fields = [
            "rank",
            "n_equiv",
            "H",
            "E",
            "dH",
            "dE",
            "eig_obj",
            "eig_resid_norm",
            "eig_resid_max",
            "eig_pred_1",
            "eig_pred_2",
            "eig_pred_3",
            "eig_resid_1",
            "eig_resid_2",
            "eig_resid_3",
            "uniax_vec_x",
            "uniax_vec_y",
            "uniax_vec_z",
            "da",
            "db",
            "dc",
            "dalpha_deg",
            "dbeta_deg",
            "dgamma_deg",
        ]
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        for i, row in enumerate(grouped, 1):
            dq = row["dq"]
            wcsv.writerow(
                {
                    "rank": i,
                    "n_equiv": row["n_equiv"],
                    "H": f"{row['H']:.12f}",
                    "E": f"{row['E']:.12f}",
                    "dH": f"{row['dH']:.12e}",
                    "dE": f"{row['dE']:.12e}",
                    "eig_obj": f"{row['obj']:.12f}",
                    "eig_resid_norm": f"{row['eig_resid_norm']:.12f}",
                    "eig_resid_max": f"{row['eig_resid_max']:.12f}",
                    "eig_pred_1": f"{row['eig_pred'][0]:.12f}",
                    "eig_pred_2": f"{row['eig_pred'][1]:.12f}",
                    "eig_pred_3": f"{row['eig_pred'][2]:.12f}",
                    "eig_resid_1": f"{row['eig_resid'][0]:.12f}",
                    "eig_resid_2": f"{row['eig_resid'][1]:.12f}",
                    "eig_resid_3": f"{row['eig_resid'][2]:.12f}",
                    "uniax_vec_x": f"{row['uniax_vec'][0]:.12f}",
                    "uniax_vec_y": f"{row['uniax_vec'][1]:.12f}",
                    "uniax_vec_z": f"{row['uniax_vec'][2]:.12f}",
                    "da": f"{dq[0]:.12f}",
                    "db": f"{dq[1]:.12f}",
                    "dc": f"{dq[2]:.12f}",
                    "dalpha_deg": f"{dq[3]:.12f}",
                    "dbeta_deg": f"{dq[4]:.12f}",
                    "dgamma_deg": f"{dq[5]:.12f}",
                }
            )

    print(f"Wrote TXT: {txt_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote grouped TXT: {gtxt_path}")
    print(f"Wrote grouped CSV: {gcsv_path}")
    print(f"Accepted minima: {len(uniq_finals)}")
    print(f"Grouped representatives: {len(grouped)}")
    if uniq_finals:
        best = uniq_finals[0]
        print(
            "Best (enthalpy-ranked): "
            f"H={best['H']:.12f} E={best['E']:.12f} "
            f"eig_resid_max={best['eig_resid_max']:.6f}"
        )


if __name__ == "__main__":
    main()

