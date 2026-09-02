"""File I/O for unit cells and XYZ geometry / mode displacements."""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

AtomRecord = Tuple[str, np.ndarray]


def read_cell(cell_file: str) -> np.ndarray:
    """
    Read lattice vectors from a cell file.

    Supported formats:
      - Three lines: ``A x y z``, ``B x y z``, ``C x y z``
      - One line with 12 tokens: ``A x y z B x y z C x y z``
      - Extended XYZ header: ``Lattice="a1 a2 a3 b1 b2 b3 c1 c2 c3"``

    Returns a 3x3 matrix whose **columns** are lattice vectors A, B, C (Angstrom).
    """
    with open(cell_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    lattice_line = next((line for line in lines if "Lattice=" in line), None)
    if lattice_line:
        match = re.search(r'Lattice="([^"]+)"', lattice_line)
        if not match:
            raise ValueError(f"Malformed Lattice= line in {cell_file}")
        parts = match.group(1).split()
        if len(parts) != 9:
            raise ValueError(f"Expected 9 lattice components in {cell_file}")
        floats = [float(value) for value in parts]
        return np.column_stack(
            (
                np.array(floats[0:3]),
                np.array(floats[3:6]),
                np.array(floats[6:9]),
            )
        )

    tokens: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            tokens.extend(stripped.split())

    if len(tokens) == 12:
        a = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        b = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
        c = np.array([float(tokens[9]), float(tokens[10]), float(tokens[11])])
        return np.column_stack((a, b, c))

    vectors: List[np.ndarray] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        vectors.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
        if len(vectors) == 3:
            break

    if len(vectors) < 3:
        raise ValueError(f"{cell_file} does not contain three lattice vectors")
    return np.column_stack(vectors)


def read_xyz(xyz_file: str) -> Tuple[List[AtomRecord], str]:
    """Read a single-frame XYZ file of atomic positions."""
    with open(xyz_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if len(lines) < 3:
        raise ValueError(f"{xyz_file} is too short for XYZ format")

    natoms = int(lines[0].strip())
    comment = lines[1].strip() if len(lines) > 1 else ""
    atoms: List[AtomRecord] = []
    for line in lines[2 : 2 + natoms]:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        label = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2]) if len(parts) > 2 else 0.0
            z = float(parts[3]) if len(parts) > 3 else 0.0
            pos = np.array([x, y, z])
        except ValueError as exc:
            raise ValueError(f"Error parsing atom coordinates in line: {line}") from exc
        atoms.append((label, pos))
    return atoms, comment


def read_normal_mode_xyz(xyz_file: str) -> Tuple[List[AtomRecord], str]:
    """Read an XYZ file whose coordinates are normal-mode displacement vectors."""
    return read_xyz(xyz_file)
