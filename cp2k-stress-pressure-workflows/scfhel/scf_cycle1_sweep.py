#!/usr/bin/env python3
"""
Stress-tensor surrogate coefficients from CP2K finite-difference runs.
Source job directory: CP2Kbenz/uniax_manual/cbaxis/1gpadelta/15gpa
  (or uniaxsubmits_apptainer/.../15gpa); parameter_trend_matrices.txt Cycle 1.
Conventions (match CP2K and trend file):
  - Stress Voigt order: [xx, yy, zz, xy, xz, yz] (bar).
  - Cell: CP2K uses h with columns = A,B,C; uniax uses cell with rows = A,B,C (cell = h^T).
"""
import numpy as np


def build_data():
    sigma_base = np.array([150009.643528, 150033.751761, 150028.777817, -4.162008, 11.300253, 3.169662], dtype=float)
    target = np.array([146667.0, 146667.0, 156667.0, 0.0, 0.0, 0.0], dtype=float)
    d_single = np.array(
        [
            [-1672.616780, -797.219500, -549.650818, -80.686740, 336.848760, -172.559064],
            [-728.412896, -1637.921884, -398.239443, -86.213299, 34.149246, -12.789727],
            [-771.067978, -523.836724, -1127.678167, 151.627860, 368.309499, 67.355665],
            [137.419065, 14.484761, -10.342839, 21.209998, 28.957101, 396.184359],
            [-295.616764, 129.618756, 45.681903, -45.988769, 305.262297, 9.565615],
            [83.363293, 92.942138, -144.007684, 137.875597, 0.740014, 49.761487],
        ],
        dtype=float,
    )
    deltas = np.array([0.005, 0.005, 0.005, 0.05, 0.05, 0.05], dtype=float)
    c_pairs = {
        (0, 1): np.array([18.606999, 23.820295, -5.830451, -6.499394, -8.603489, 1.106675]),
        (0, 2): np.array([26.626494, -4.346910, 10.857048, -5.937825, -1.879631, 5.602235]),
        (0, 3): np.array([22.215721, 8.407935, 5.369852, 0.038214, -4.079758, 2.881220]),
        (0, 4): np.array([-5.076568, -17.847875, 1.503802, 16.977483, -6.988035, -6.579788]),
        (0, 5): np.array([38.404595, -9.512366, 0.680317, -6.586773, 12.664108, 6.408039]),
        (1, 2): np.array([51.948276, -5.623323, -0.762095, -9.776723, 3.022823, -2.118237]),
        (1, 3): np.array([25.224823, 33.609753, 6.343600, 1.676962, -2.816658, 5.986543]),
        (1, 4): np.array([8.722500, 5.453366, 0.562774, 18.823611, -4.007095, -7.270613]),
        (1, 5): np.array([40.833247, 1.460385, 5.717800, 1.144342, 13.313346, 10.912959]),
        (2, 3): np.array([50.595000, -1.024904, 24.167107, -12.284288, 18.296322, 7.789615]),
        (2, 4): np.array([22.814185, -21.996135, 10.848253, 17.726486, 5.014092, -7.148198]),
        (2, 5): np.array([40.109516, -5.665407, 5.863833, -24.241037, 2.781123, 4.878708]),
        (3, 4): np.array([27.141675, 3.937907, -3.430901, -11.953685, -3.552724, -1.020849]),
        (3, 5): np.array([32.525861, 10.597099, 16.061181, 3.348845, 2.790560, 10.511908]),
        (4, 5): np.array([22.732270, -7.418891, 9.215664, 15.026681, 13.200605, 2.078318]),
    }
    return sigma_base, target, d_single, c_pairs, deltas


def solve_linear(A, b, reg=1e-6):
    cond_A = np.linalg.cond(A)
    reg_use = reg if (cond_A > 1e10 or not np.isfinite(cond_A)) else 0.0
    A_reg = A + reg_use * np.eye(6)
    try:
        return np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A_reg, b, rcond=None)[0]


def build_eff_matrix(d_single, c_pairs, x_vec):
    M_eff = d_single.copy()
    for i in range(6):
        add = np.zeros(6)
        for j in range(6):
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in c_pairs:
                add += c_pairs[key] * x_vec[j]
        M_eff[:, i] = d_single[:, i] + add
    return M_eff


def predict_sigma(sigma_base, d_single, c_pairs, x):
    sigma = sigma_base + d_single @ x
    for (i, j), cij in c_pairs.items():
        sigma += cij * x[i] * x[j]
    return sigma


def run_sc(eta, max_iter, tol=1e-4, step_fraction=0.3, max_dl=0.1, max_da=1.0):
    sigma_base, target, d_single, c_pairs, deltas = build_data()
    r0 = target - sigma_base
    x = solve_linear(d_single, r0)
    for k in range(max_iter):
        M_eff = build_eff_matrix(d_single, c_pairs, x)
        x_tilde = solve_linear(M_eff, r0)
        x_next = (1.0 - eta) * x + eta * x_tilde
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next

    dq_full = x * deltas
    dq = step_fraction * dq_full
    scale = 1.0
    max_len = np.max(np.abs(dq[0:3]))
    if max_len > max_dl:
        scale = min(scale, max_dl / max_len)
    max_ang = np.max(np.abs(dq[3:6]))
    if max_ang > max_da:
        scale = min(scale, max_da / max_ang)
    dq *= scale
    x_applied = np.divide(dq, deltas, out=np.zeros_like(dq), where=np.abs(deltas) > 0)
    sigma_pred = predict_sigma(sigma_base, d_single, c_pairs, x_applied)
    resid = target - sigma_pred
    return x, x_applied, sigma_pred, resid, scale


def main():
    cases = [(0.5, 12), (0.2, 20), (0.3, 20), (0.1, 30)]
    for eta, mi in cases:
        x_raw, x_app, sigma_pred, resid, scale = run_sc(eta, mi)
        print(f"\n=== eta={eta}, max_iter={mi} ===")
        print("x_raw:", " ".join(f"{v: .6f}" for v in x_raw))
        print("x_applied:", " ".join(f"{v: .6f}" for v in x_app))
        print(f"trust_scale: {scale:.6f}")
        print("pred_sigma:", " ".join(f"{v: .3f}" for v in sigma_pred))
        print("pred_resid:", " ".join(f"{v: .3f}" for v in resid))
        print(f"pred_resid_norm: {np.linalg.norm(resid):.3f}")


if __name__ == "__main__":
    main()
