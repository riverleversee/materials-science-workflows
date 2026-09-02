#!/usr/bin/env python3
"""
Best-effort adaptive minima counting for the Cycle-1 surrogate model.

Workflow:
1) Coarse anisotropic grid over full bounds.
2) Keep promising regions (low objective) and refine locally.
3) Run local coordinate/pair descent from refined seeds.
4) Cluster converged points to count distinct minima.

Note: This is adaptive (not mathematically certified exact).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

import importlib.util
from pathlib import Path


def load_model():
    p = Path(__file__).parent / "scf_cycle1_sweep.py"
    spec = importlib.util.spec_from_file_location("scfbase", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@dataclass
class Result:
    x: np.ndarray
    obj: float
    resid: np.ndarray


def build_objective():
    m = load_model()
    sigma_base, target, d_single, c_pairs, _deltas = m.build_data()
    # User-selected weighted residual objective (diag > shear)
    w = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=float)

    def pred_sigma(x: np.ndarray) -> np.ndarray:
        s = sigma_base + d_single @ x
        for (i, j), cij in c_pairs.items():
            s = s + cij * x[i] * x[j]
        return s

    def residual(x: np.ndarray) -> np.ndarray:
        return target - pred_sigma(x)

    def obj(x: np.ndarray) -> float:
        r = residual(x)
        rw = w * r
        return float(np.dot(rw, rw))

    return obj, residual


def make_axis(bounds: float, step: float) -> np.ndarray:
    # Ensure symmetric axis including both ends.
    n = int(np.floor((2 * bounds) / step)) + 1
    vals = np.linspace(-bounds, bounds, n)
    return vals


def coarse_grid_search(obj, x_bounds):
    # Sparse grid; angle dimensions (3,5) intentionally sparser near 90 deg.
    coarse_steps = np.array([20.0, 20.0, 20.0, 30.0, 20.0, 30.0], dtype=float)
    axes = [make_axis(x_bounds[i], coarse_steps[i]) for i in range(6)]
    mesh = np.array(list(itertools.product(*axes)), dtype=float)
    vals = np.array([obj(x) for x in mesh], dtype=float)
    return mesh, vals, coarse_steps


def local_refine(obj, x0, x_bounds, step, max_iter=250):
    x = x0.copy()
    f = obj(x)
    for _ in range(max_iter):
        improved = False
        # Coordinate moves
        for k in range(6):
            for sgn in (-1.0, 1.0):
                xt = x.copy()
                xt[k] = np.clip(xt[k] + sgn * step[k], -x_bounds[k], x_bounds[k])
                ft = obj(xt)
                if ft < f:
                    x, f = xt, ft
                    improved = True
        # Pair moves for coupling
        for i in range(6):
            for j in range(i + 1, 6):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        xt = x.copy()
                        xt[i] = np.clip(xt[i] + si * step[i], -x_bounds[i], x_bounds[i])
                        xt[j] = np.clip(xt[j] + sj * step[j], -x_bounds[j], x_bounds[j])
                        ft = obj(xt)
                        if ft < f:
                            x, f = xt, ft
                            improved = True
        if not improved:
            break
    return x, f


def cluster_results(results, dist_thresh=0.8):
    # results sorted ascending obj
    clusters = []
    for r in results:
        placed = False
        for c in clusters:
            if np.linalg.norm(r.x - c["rep"].x) < dist_thresh:
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": r, "count": 1})
    return clusters


def main():
    obj, residual = build_objective()
    # Bounds from user request: lengths +/-0.4 A, angles +/-5 deg.
    # Convert to x-space: dq = x * delta, delta=[0.005,0.005,0.005,0.05,0.05,0.05]
    x_bounds = np.array([80.0, 80.0, 80.0, 100.0, 100.0, 100.0], dtype=float)
    # Fine step = 0.1% of full grid size: len=0.16, ang=0.2
    fine_step = np.array([0.16, 0.16, 0.16, 0.2, 0.2, 0.2], dtype=float)
    mid_step = fine_step * 2.0

    mesh, vals, coarse_steps = coarse_grid_search(obj, x_bounds)
    order = np.argsort(vals)
    top = order[:180]  # promising seeds from sparse grid

    # Mid refinement around top seeds
    refined = []
    for idx in top:
        x0 = mesh[idx]
        xr, fr = local_refine(obj, x0, x_bounds, mid_step, max_iter=120)
        refined.append((xr, fr))

    # Dense refinement from unique mid points
    refined.sort(key=lambda t: t[1])
    uniq = []
    for x, f in refined:
        if not any(np.linalg.norm(x - u) < 1.5 for u in uniq):
            uniq.append(x)
        if len(uniq) >= 70:
            break

    finals = []
    for x0 in uniq:
        xr, fr = local_refine(obj, x0, x_bounds, fine_step, max_iter=500)
        rr = residual(xr)
        finals.append(Result(x=xr, obj=fr, resid=rr))

    finals.sort(key=lambda r: r.obj)
    clusters = cluster_results(finals, dist_thresh=0.75)

    strict01 = [c for c in clusters if np.all(np.abs(c["rep"].resid) < 0.1)]
    strict1 = [c for c in clusters if np.all(np.abs(c["rep"].resid) < 1.0)]
    strict10 = [c for c in clusters if np.all(np.abs(c["rep"].resid) < 10.0)]

    print("Adaptive sparse->dense search (best-effort, not certified exact)")
    print(f"coarse_grid_points={len(mesh)}")
    print(f"coarse_steps={coarse_steps.tolist()}")
    print(f"mid_seeds={len(top)} unique_mid={len(uniq)} final_refined={len(finals)}")
    print(f"distinct_minima_found={len(clusters)}")
    print(f"minima with all |residual_i| < 0.1 bar: {len(strict01)}")
    print(f"minima with all |residual_i| < 1.0 bar: {len(strict1)}")
    print(f"minima with all |residual_i| < 10.0 bar: {len(strict10)}")

    if clusters:
        best = clusters[0]["rep"]
        print("best_weighted_obj=", f"{best.obj:.6f}")
        print("best_residual=", " ".join(f"{v:.6f}" for v in best.resid))
        print("best_x=", " ".join(f"{v:.6f}" for v in best.x))


if __name__ == "__main__":
    main()

