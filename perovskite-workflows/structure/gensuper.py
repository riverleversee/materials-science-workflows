"""
gensuper.py — geometry-only mixed-halide perovskite supercell builder.

Builds Pb / MA / halide positions and optional LAMMPS topology (bonds/angles/dihedrals).
Force-field parameters (masses, charges, pair/bond/angle/dihedral coeffs) are NOT
embedded: supply them yourself before running LAMMPS (e.g. merge into the data file
or add coefficient commands in the input script).

Composition: if the cwd basename matches ``Nperbr`` (e.g. ``50perbr``), that Br
fraction is used; otherwise set ``BR_FRACTION`` (0–1) in the environment.
"""
from __future__ import annotations

import math
import os
import random
import re
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# --- User parameters
nx = 10
ny = 10
nz = 10
spacing = 6.2
box_origin = (0.0, 0.0, 0.0)
output_filename = "lammps_data_full_supercell.data"
random_seed = 42

# Atom type IDs (topology only; match your force field)
AT_TYPE_C_MA = 2
AT_TYPE_N_MA = 4
AT_TYPE_H5 = 5
AT_TYPE_H6 = 6
AT_TYPE_PB = 7
AT_TYPE_I = 8
AT_TYPE_BR = 9

# Placeholder charges — replace with your force-field charges before production MD
PLACEHOLDER_CHARGE = 0.0

# MA template Cartesian offsets (geometry only)
_example_positions = [
    (-0.752882, -0.000000, -0.000000),
    (0.752882, -0.000000, -0.000000),
    (-1.099122, 0.061931, 1.034173),
    (-1.098989, -0.926615, -0.463469),
    (-1.099000, 0.864662, -0.570756),
    (1.120151, -0.806980, 0.532445),
    (1.120129, 0.864608, 0.432656),
    (1.120113, -0.057596, -0.965106),
]

half = spacing / 2.0
_template_offsets = [(x, y, z) for (x, y, z) in _example_positions]
_local_ma_types = [AT_TYPE_C_MA, AT_TYPE_N_MA] + [AT_TYPE_H5] * 3 + [AT_TYPE_H6] * 3

_local_bonds = [
    (1, 2, 4),
    (1, 3, 5),
    (1, 4, 5),
    (1, 5, 5),
    (2, 6, 6),
    (2, 7, 6),
    (2, 8, 6),
]
_local_angles = [
    (3, 1, 2, 6),
    (4, 1, 2, 6),
    (5, 1, 2, 6),
    (1, 2, 6, 7),
    (1, 2, 7, 7),
    (1, 2, 8, 7),
    (3, 1, 4, 8),
    (3, 1, 5, 8),
    (4, 1, 5, 8),
    (6, 2, 7, 9),
    (6, 2, 8, 9),
    (7, 2, 8, 9),
]
_local_dihedrals = [
    (3, 1, 2, 6),
    (3, 1, 2, 7),
    (3, 1, 2, 8),
    (4, 1, 2, 6),
    (4, 1, 2, 7),
    (4, 1, 2, 8),
    (5, 1, 2, 6),
    (5, 1, 2, 7),
    (5, 1, 2, 8),
]

_halide_local_offsets = [
    (half, 0.0, 0.0),
    (0.0, half, 0.0),
    (0.0, 0.0, half),
]


def br_fraction_from_cwd() -> float:
    env = os.environ.get("BR_FRACTION")
    if env is not None:
        return float(env)
    folder_name = os.path.basename(os.getcwd())
    match = re.match(r"(\d+)perbr", folder_name)
    if match:
        return int(match.group(1)) / 100.0
    raise ValueError(
        "Set BR_FRACTION (0-1) or run from a directory named like '50perbr'"
    )


def iter_pb_positions(nx, ny, nz, spacing, origin=(0.0, 0.0, 0.0)):
    ox, oy, oz = origin
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                yield (ox + ix * spacing, oy + iy * spacing, oz + iz * spacing)


def write_data_file(filename, nx, ny, nz, n_br, seed=None):
    if seed is not None:
        random.seed(seed)

    pb_positions = list(iter_pb_positions(nx, ny, nz, spacing, box_origin))
    n_pb = len(pb_positions)
    halides_per_pb = len(_halide_local_offsets)
    total_halides = n_pb * halides_per_pb

    if n_br < 0 or n_br > total_halides:
        raise ValueError("n_br must be between 0 and total number of halide sites")

    atom_lines = []
    bonds = []
    angles = []
    dihedrals = []

    atom_id = 0
    bond_id = 0
    angle_id = 0
    dihedral_id = 0

    for pos in pb_positions:
        atom_id += 1
        atom_lines.append(
            (atom_id, 0, AT_TYPE_PB, PLACEHOLDER_CHARGE, pos[0], pos[1], pos[2])
        )

    for mi, pb_pos in enumerate(pb_positions, start=1):
        cx, cy, cz = (pb_pos[0] + half, pb_pos[1] + half, pb_pos[2] + half)
        local_atom_ids = []
        rotation = R.random()
        rotated_offsets = rotation.apply(_template_offsets)

        for offset, atype in zip(rotated_offsets, _local_ma_types):
            atom_id += 1
            local_atom_ids.append(atom_id)
            ox, oy, oz = offset
            atom_lines.append(
                (
                    atom_id,
                    mi,
                    atype,
                    PLACEHOLDER_CHARGE,
                    cx + ox,
                    cy + oy,
                    cz + oz,
                )
            )

        for a_local, b_local, btype in _local_bonds:
            bond_id += 1
            bonds.append(
                (bond_id, btype, local_atom_ids[a_local - 1], local_atom_ids[b_local - 1])
            )
        for i_local, j_local, k_local, atype in _local_angles:
            angle_id += 1
            angles.append(
                (
                    angle_id,
                    atype,
                    local_atom_ids[i_local - 1],
                    local_atom_ids[j_local - 1],
                    local_atom_ids[k_local - 1],
                )
            )
        for a_local, b_local, c_local, d_local in _local_dihedrals:
            dihedral_id += 1
            dihedrals.append(
                (
                    dihedral_id,
                    3,
                    local_atom_ids[a_local - 1],
                    local_atom_ids[b_local - 1],
                    local_atom_ids[c_local - 1],
                    local_atom_ids[d_local - 1],
                )
            )

    halide_atom_ids = []
    for pb_pos in pb_positions:
        for offset in _halide_local_offsets:
            atom_id += 1
            atom_lines.append(
                (
                    atom_id,
                    0,
                    AT_TYPE_I,
                    PLACEHOLDER_CHARGE,
                    pb_pos[0] + offset[0],
                    pb_pos[1] + offset[1],
                    pb_pos[2] + offset[2],
                )
            )
            halide_atom_ids.append(atom_id)

    br_choice = set(random.sample(halide_atom_ids, n_br))
    atom_lines = [
        (aid, mid, AT_TYPE_BR, PLACEHOLDER_CHARGE, x, y, z)
        if (aid in br_choice and atype == AT_TYPE_I)
        else (aid, mid, atype, q, x, y, z)
        for (aid, mid, atype, q, x, y, z) in atom_lines
    ]

    n_atoms = len(atom_lines)
    n_bonds = len(bonds)
    n_angles = len(angles)
    n_dihedrals = len(dihedrals)

    xlo, xhi = box_origin[0], box_origin[0] + nx * spacing
    ylo, yhi = box_origin[1], box_origin[1] + ny * spacing
    zlo, zhi = box_origin[2], box_origin[2] + nz * spacing

    with open(filename, "w") as f:
        f.write("# Geometry-only supercell (no force-field coeffs)\n\n")
        f.write(f"{n_atoms:12d} atoms\n")
        f.write(f"{n_bonds:12d} bonds\n")
        f.write(f"{n_angles:12d} angles\n")
        f.write(f"{n_dihedrals:12d} dihedrals\n")
        f.write(f"{0:12d} impropers\n\n")
        f.write(f"{9:12d} atom types\n")
        f.write(f"{6:12d} bond types\n")
        f.write(f"{9:12d} angle types\n")
        f.write(f"{3:12d} dihedral types\n")
        f.write(f"{0:12d} improper types\n\n")
        f.write(f"{xlo:12.6f} {xhi:12.6f} xlo xhi\n")
        f.write(f"{ylo:12.6f} {yhi:12.6f} ylo yhi\n")
        f.write(f"{zlo:12.6f} {zhi:12.6f} zlo zhi\n\n")

        f.write("# Insert Masses and force-field coefficient blocks before running LAMMPS.\n\n")
        f.write("Atoms # full\n\n")
        for aid, mid, atype, q, x, y, z in atom_lines:
            f.write(
                f"{aid:8d} {mid:6d} {atype:6d} {q:18.12f} {x:12.8f} {y:12.8f} {z:12.8f}\n"
            )
        f.write("\n")

        f.write("Bonds\n\n")
        for bid, btype, a1, a2 in bonds:
            f.write(f"{bid:8d} {btype:4d} {a1:8d} {a2:8d}\n")
        f.write("\n")

        f.write("Angles\n\n")
        for aidn, atype, a1, a2, a3 in angles:
            f.write(f"{aidn:8d} {atype:4d} {a1:8d} {a2:8d} {a3:8d}\n")
        f.write("\n")

        f.write("Dihedrals\n\n")
        for did, dtype, a1, a2, a3, a4 in dihedrals:
            f.write(f"{did:8d} {dtype:4d} {a1:8d} {a2:8d} {a3:8d} {a4:8d}\n")
        f.write("\n")

    print(
        f"Wrote {filename}: {n_atoms} atoms, {n_bonds} bonds, "
        f"{n_angles} angles, {n_dihedrals} dihedrals "
        f"(Br sites: {n_br}/{total_halides}; charges/coeffs are placeholders)"
    )


if __name__ == "__main__":
    f = br_fraction_from_cwd()
    n_br = round(nx * ny * nz * 3 * f)
    print(f"Br fraction={f}, n_br={n_br}")
    write_data_file(output_filename, nx, ny, nz, n_br, seed=random_seed)
