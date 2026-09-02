#!/usr/bin/env python3
import numpy as np
import importlib.util
from pathlib import Path


def load_base_module():
    p = Path(__file__).parent / "scf_cycle1_sweep.py"
    spec = importlib.util.spec_from_file_location("scfbase", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def apply_limits(x, deltas, step_fraction=0.3, max_dl=0.1, max_da=1.0):
    dq_full = x * deltas
    dq = step_fraction * dq_full
    scale = 1.0
    ml = np.max(np.abs(dq[:3]))
    if ml > max_dl:
        scale = min(scale, max_dl / ml)
    ma = np.max(np.abs(dq[3:]))
    if ma > max_da:
        scale = min(scale, max_da / ma)
    dq *= scale
    x_applied = np.divide(dq, deltas, out=np.zeros_like(dq), where=np.abs(deltas) > 0)
    return x_applied, scale


def phi_norm2(m, x):
    sigma_base, target, d_single, c_pairs, _ = m.build_data()
    sigma = m.predict_sigma(sigma_base, d_single, c_pairs, x)
    r = target - sigma
    return float(np.dot(r, r)), r, sigma


def jac_resid_fd(m, x, eps=1e-3):
    # Jacobian of residual r(x)=target-sigma_pred(x) wrt x by FD
    _, r0, _ = phi_norm2(m, x)
    J = np.zeros((6, 6))
    for i in range(6):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        _, rp, _ = phi_norm2(m, xp)
        _, rm, _ = phi_norm2(m, xm)
        J[:, i] = (rp - rm) / (2 * eps)
    return J, r0


def sc_fixed_point(m, eta=0.2, max_iter=20, tol=1e-4):
    sigma_base, target, d_single, c_pairs, _ = m.build_data()
    r0 = target - sigma_base
    x = m.solve_linear(d_single, r0)
    for _ in range(max_iter):
        M_eff = m.build_eff_matrix(d_single, c_pairs, x)
        x_tilde = m.solve_linear(M_eff, r0)
        x_next = (1.0 - eta) * x + eta * x_tilde
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
    return x


def sc_best_prefix(m, eta=0.2, max_iter=50):
    sigma_base, target, d_single, c_pairs, _ = m.build_data()
    r0 = target - sigma_base
    x = m.solve_linear(d_single, r0)
    best_x = x.copy()
    best_f, _, _ = phi_norm2(m, x)
    for _ in range(max_iter):
        M_eff = m.build_eff_matrix(d_single, c_pairs, x)
        x_tilde = m.solve_linear(M_eff, r0)
        x = (1.0 - eta) * x + eta * x_tilde
        f, _, _ = phi_norm2(m, x)
        if f < best_f:
            best_f = f
            best_x = x.copy()
    return best_x


def gn_backtracking(m, max_iter=40, lam=1e-3):
    x = np.zeros(6)
    f, r, _ = phi_norm2(m, x)
    for _ in range(max_iter):
        J, r = jac_resid_fd(m, x)
        A = J.T @ J + lam * np.eye(6)
        b = -J.T @ r
        try:
            d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(A, b, rcond=None)[0]
        alpha = 1.0
        accepted = False
        for _ls in range(12):
            xt = x + alpha * d
            ft, _, _ = phi_norm2(m, xt)
            if ft < f:
                x = xt
                f = ft
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        if np.linalg.norm(alpha * d) < 1e-5:
            break
    return x


def newton_resid_backtracking(m, max_iter=30, lam=1e-3):
    x = np.zeros(6)
    f, r, _ = phi_norm2(m, x)
    for _ in range(max_iter):
        J, r = jac_resid_fd(m, x)
        A = J + lam * np.eye(6)
        try:
            d = np.linalg.solve(A, -r)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(A, -r, rcond=None)[0]
        alpha = 1.0
        accepted = False
        for _ls in range(12):
            xt = x + alpha * d
            ft, _, _ = phi_norm2(m, xt)
            if ft < f:
                x = xt
                f = ft
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        if np.linalg.norm(alpha * d) < 1e-5:
            break
    return x


def random_restart_local(m, n=200, scale=8.0):
    rng = np.random.default_rng(7)
    best_x = np.zeros(6)
    best_f, _, _ = phi_norm2(m, best_x)
    for _ in range(n):
        x0 = rng.normal(0.0, scale, size=6)
        f0, _, _ = phi_norm2(m, x0)
        if f0 < best_f:
            best_f = f0
            best_x = x0
    return best_x


def report(name, m, x):
    deltas = m.build_data()[4]
    x_app, scale = apply_limits(x, deltas)
    f, r, s = phi_norm2(m, x_app)
    print(f"{name:28s} resid_norm={np.sqrt(f):10.3f}  trust_scale={scale:6.3f}")
    return np.sqrt(f), x_app, r, s


def main():
    m = load_base_module()
    print("Benchmarking local solver variants on provided Cycle-1 surrogate")
    results = []

    x_sc = sc_fixed_point(m, eta=0.2, max_iter=20)
    results.append(("SC eta0.2 iter20",) + report("SC eta0.2 iter20", m, x_sc))

    x_scb = sc_best_prefix(m, eta=0.2, max_iter=50)
    results.append(("SC best-prefix eta0.2",) + report("SC best-prefix eta0.2", m, x_scb))

    x_sc1 = sc_best_prefix(m, eta=0.1, max_iter=50)
    results.append(("SC best-prefix eta0.1",) + report("SC best-prefix eta0.1", m, x_sc1))

    x_gn = gn_backtracking(m, max_iter=40, lam=1e-3)
    results.append(("Gauss-Newton+LS",) + report("Gauss-Newton+LS", m, x_gn))

    x_nr = newton_resid_backtracking(m, max_iter=30, lam=1e-3)
    results.append(("Newton-resid+LS",) + report("Newton-resid+LS", m, x_nr))

    x_rr = random_restart_local(m, n=600, scale=10.0)
    results.append(("RandomRestartPick",) + report("RandomRestartPick", m, x_rr))

    results_sorted = sorted(results, key=lambda t: t[1])
    print("\nRanked by predicted residual norm:")
    for i, (name, rn, *_rest) in enumerate(results_sorted, 1):
        print(f"{i:2d}. {name:28s}  {rn:.3f}")


if __name__ == "__main__":
    main()
