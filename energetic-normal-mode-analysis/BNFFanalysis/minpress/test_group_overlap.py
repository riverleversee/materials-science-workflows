import argparse
import os
import sys


def _as_set(flags_list):
    return {i for i, v in enumerate(flags_list) if int(v) == 1}


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mode_analysis.paths import optimized_cell_path, resolve_data_dir, trajectory_path

    ap = argparse.ArgumentParser(
        description=(
            "Test group-flag overlap for furazan/furoxano/rings/nitro on one structure.\n"
            "Imports the canonical driver and prints connectivity-based group assignments."
        )
    )
    ap.add_argument("--data-dir", default=None, help="Study root (see README). Defaults to NMA_DATA_DIR or BNFFanalysis/minpress/")
    ap.add_argument("--pressure", type=int, default=0, choices=[0, 4, 10], help="Pressure in GPa")
    ap.add_argument("--mode", type=int, default=120, help="Mode index (anime_XXX.xyz)")
    ap.add_argument("--bond-cutoff", type=float, default=1.6, help="Bond cutoff (Angstrom)")
    args = ap.parse_args()

    import importlib.util

    driver_path = os.path.join(
        repo_root,
        "BNFFanalysis",
        "minpress",
        "ModeAnalysis_River_Traj_r2scale_massweight_groups.py",
    )
    spec = importlib.util.spec_from_file_location("nma_groups_driver", driver_path)
    mag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mag)

    data_dir = resolve_data_dir(args.data_dir)
    cell = optimized_cell_path(data_dir, args.pressure)
    xyz = trajectory_path(data_dir, args.pressure, args.mode)

    if not os.path.exists(cell):
        raise FileNotFoundError(cell)
    if not os.path.exists(xyz):
        raise FileNotFoundError(xyz)

    original_atoms, supercell_atoms, _cell_matrix = mag.create_supercell(cell, xyz)
    bond_matrix, *_rest = mag.analyze_supercell_bonds(supercell_atoms, original_atoms, args.bond_cutoff)
    flags = mag.identify_groups(supercell_atoms, original_atoms, bond_matrix)

    furazan = _as_set(flags["furazan"])
    furoxano = _as_set(flags["furoxano"])
    rings = _as_set(flags["rings"])
    furox_oxo = _as_set(flags["furoxano_oxo"])
    nitro = _as_set(flags["nitro"])
    furoxano_ring_component = furoxano & rings

    print(f"data_dir={data_dir}")
    print(f"Structure: pressure={args.pressure} GPa, xyz={os.path.basename(xyz)}")
    print(f"N(original atoms)={len(original_atoms)}  N(supercell atoms)={len(supercell_atoms)}")
    print("")
    print("Group sizes (original-atom indices flagged):")
    print(f"  nitro:         {len(nitro)}")
    print(f"  furazan:       {len(furazan)}")
    print(f"  furoxano:      {len(furoxano)}   (ring + exocyclic O)")
    print(f"  furoxano_oxo:  {len(furox_oxo)}  (exocyclic O only)")
    print(f"  rings:         {len(rings)}      (furazan ring + furoxano ring)")
    print("")
    print("Key overlap checks:")
    print(f"  furazan ⊆ rings?                  {furazan.issubset(rings)}")
    print(f"  furoxano_oxo ∩ rings = ∅ ?         {len(furox_oxo & rings) == 0}")
    print(f"  furazan ∩ furoxano (any) = ∅ ?     {len(furazan & furoxano) == 0}")
    print(f"  furazan ∩ (furoxano ring) = ∅ ?    {len(furazan & furoxano_ring_component) == 0}")
    print(f"  rings == furazan ∪ (furoxano ring)? {rings == (furazan | furoxano_ring_component)}")

    if len(furazan) == 0 and len(furoxano) == 0:
        print("WARNING: No furazan or furoxano groups detected with this bond cutoff.")


if __name__ == "__main__":
    main()
