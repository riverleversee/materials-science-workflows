"""Parse CP2K parameter_trend_matrices.txt blocks for scfhel surrogates."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PARAMS = ("a", "b", "c", "alpha", "beta", "gamma")
STRESS_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")


def parse_cycle_block(text: str, cycle: int = 1) -> str:
    m = re.search(rf"Cycle {cycle}\n(.*?)\n===========", text, flags=re.S)
    if not m:
        raise ValueError(f"Cycle {cycle} block not found")
    return m.group(1)


def parse_row_after_label(block: str, label: str) -> np.ndarray:
    m = re.search(rf"{re.escape(label)}\n\s+([^\n]+)", block)
    if not m:
        raise ValueError(f"Missing label: {label}")
    return np.fromstring(m.group(1), sep=" ")


def parse_cp2k_model(block: str):
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
        raise ValueError("Missing base energetics in trend log block")
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


def load_cp2k_parameter_trend(path: Path, cycle: int = 1):
    block = parse_cycle_block(path.read_text(), cycle=cycle)
    return parse_cp2k_model(block)
