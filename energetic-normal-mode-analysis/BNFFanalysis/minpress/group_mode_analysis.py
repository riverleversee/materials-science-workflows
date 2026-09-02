"""
Canonical group-resolved normal-mode analysis for BNFF / DNTF energetic crystals.

Library API: run_full_analysis(), make_plots(), identify_groups(), ...
CLI entry point: ../../run_group_analysis.py

Data layout (under NMA_DATA_DIR or BNFFanalysis/minpress/):
  optimized_cell{P}GPa.cell
  {P}GPa/AnimationFiles/anime_{mode}.xyz
"""

import os
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import json

# Study directory for displacement intermediates and JSON output when outdir is omitted.
BASEDIR = os.path.dirname(os.path.abspath(__file__))

# Defaults used only by the legacy __main__ block below (prefer run_group_analysis.py).
DEFAULT_PRESSURES_GPA = [0, 4, 10]
DEFAULT_BOND_CUTOFF = 1.6
DEFAULT_DISTANCE_THRESHOLD = 3.2
DEFAULT_OUTPUT_PREFIX = "modes_groups"
TRAJ_PREFIX = "anime_"
TRAJ_SUFFIX = ".xyz"


def read_cell(cell_file):
    """
    Reads unit cell vectors from a file.

    Supported formats:
      Option 1 (3 lines):
         A x1 y1 z1
         B x2 y2 z2
         C x3 y3 z3
      Option 2 (1 line with 12 tokens):
         A x1 y1 z1 B x2 y2 z2 C x3 y3 z3
      Option 3 (line containing Lattice="x1 y1 z1 x2 y2 z2 x3 y3 z3")

    Returns:
      A 3x3 numpy array (cell_matrix) whose columns are the lattice vectors A, B, and C.
    """
    with open(cell_file, 'r') as f:
        lines = f.readlines()

    lattice_line = next((line for line in lines if 'Lattice=' in line), None)

    if lattice_line:
        match = re.search(r'Lattice="([^"]+)"', lattice_line)
        if not match:
            raise ValueError("Lattice data not properly formatted.")
        parts = match.group(1).split()
        if len(parts) != 9:
            raise ValueError("Expected 9 components for 3 lattice vectors.")
        try:
            floats = list(map(float, parts))
            A = np.array(floats[0:3])
            B = np.array(floats[3:6])
            C = np.array(floats[6:9])
        except ValueError:
            raise ValueError("Lattice values must be valid floats.")
    else:
        # fallback to original logic
        tokens = []
        for line in lines:
            if line.strip():
                tokens.extend(line.strip().split())

        if len(tokens) == 12:
            A = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            B = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
            C = np.array([float(tokens[9]), float(tokens[10]), float(tokens[11])])
        elif len(lines) >= 3:
            vectors = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    raise ValueError("Error parsing a cell vector in the line: " + line)
                vectors.append(vec)
            if len(vectors) < 3:
                raise ValueError("The cell file must contain three lattice vectors.")
            A, B, C = vectors[:3]
        else:
            raise ValueError("Cell file does not have enough data.")

    cell_matrix = np.column_stack((A, B, C))
    return cell_matrix


def read_xyz(xyz_file):
    """
    Reads an xyz file of atomic positions.
    
    Expected file format:
      number_of_atoms
      comment_line
      atom_label    x    y    z
      ...
    
    Returns:
      A tuple (atoms, comment) where atoms is a list of tuples:
         (atom_label, numpy array([x, y, z]))
    """
    atoms = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError("XYZ file does not have enough lines.")
    
    try:
        natoms = int(lines[0].strip())
    except Exception as e:
        raise ValueError("The first line must be an integer (number of atoms).") from e
    
    comment = lines[1].strip() if len(lines) > 1 else ""
    
    for line in lines[2:2+natoms]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        label = parts[0]
        try:
            pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            raise ValueError("Error parsing atom coordinates in the line: " + line)
        atoms.append((label, pos))
    
    return atoms, comment

def cartesian_to_fractional(cart_pos, cell_matrix):
    """
    Converts a Cartesian coordinate into fractional coordinates with respect to cell_matrix.
    
    Given the equation:
         cart_pos = cell_matrix * frac,
    this function solves for frac.
    """
    frac = np.linalg.solve(cell_matrix, cart_pos)
    return frac

def fractional_to_cartesian(frac, cell_matrix):
    """
    Converts fractional coordinates back to Cartesian via:
         cart_pos = cell_matrix * frac.
    """
    cart = np.dot(cell_matrix, frac)
    return cart

def wrap_fractional(frac):
    """
    Wraps the fractional coordinates so they lie within [0, 1).
    
    This is done by performing mod 1 arithmetic.
    """
    return np.mod(frac, 1)

def generate_supercell(wrapped_frac_atoms, cell_matrix):
    """
    Generates a 3x3x3 supercell from the list of wrapped fractional atom positions.

    For each atom, we apply a translation vector (tx, ty, tz) with each component in {-1, 0, 1}.
    The (0, 0, 0) translation is applied first so that the first block of atoms corresponds
    to the original cell.

    Inputs:
      wrapped_frac_atoms: List of tuples (atom_label, fractional_coordinate) where each
                          fractional_coordinate is a numpy array in [0,1)^3.
      cell_matrix: The 3x3 numpy array representing the unit cell vectors.
      
    Returns:
      A list of tuples (atom_label, cartesian_coordinate) for the entire supercell.
    """
    supercell_atoms = []
    # Define translation vectors with (0, 0, 0) first.
    translations = [(0, 0, 0)]
    
    # Use nested loops instead of itertools.product to form the translation list.
    for tx in [-1, 0, 1]:
        for ty in [-1, 0, 1]:
            for tz in [-1, 0, 1]:
                if tx == 0 and ty == 0 and tz == 0:
                    continue  # (0, 0, 0) is already included.
                translations.append((tx, ty, tz))
    
    # Apply each translation to the wrapped fractional coordinates, then convert to Cartesian.
    for shift in translations:
        trans = np.array(shift)
        for label, frac in wrapped_frac_atoms:
            new_frac = frac + trans
            new_cartesian = fractional_to_cartesian(new_frac, cell_matrix)
            supercell_atoms.append((label, new_cartesian))
    
    return supercell_atoms

def create_supercell(cell_filename, xyz_filename):
    """
    Processes the given cell file and xyz file to:
      1. Convert the atomic Cartesian positions into fractional coordinates in the unit cell.
      2. Wrap the fractional coordinates so that they lie within the unit cell (i.e. in [0, 1)).
      3. Generate a 3x3x3 supercell from the wrapped coordinates.
    
    Inputs:
      cell_filename: Filename for the unit cell vectors.
      xyz_filename: Filename for the xyz file of atomic positions.
      
    Returns:
      A tuple with three elements:
        - original_atoms_cart: List of (atom_label, cartesian_coord) for atoms wrapped into the unit cell.
        - supercell_atoms_cart: List of (atom_label, cartesian_coord) for the full 3x3x3 supercell.
          (The first N entries correspond to the original atoms.)
        - cell_matrix: The 3x3 numpy array representing the unit cell vectors.
    """
    cell_matrix = read_cell(cell_filename)
    atoms, _ = read_xyz(xyz_filename)
    
    wrapped_frac_atoms = []
    original_atoms_cart = []
    
    for label, cart in atoms:
        frac = cartesian_to_fractional(cart, cell_matrix)
        frac_wrapped = wrap_fractional(frac)
        wrapped_frac_atoms.append((label, frac_wrapped))
        cart_wrapped = fractional_to_cartesian(frac_wrapped, cell_matrix)
        original_atoms_cart.append((label, cart_wrapped))
    
    supercell_atoms_cart = generate_supercell(wrapped_frac_atoms, cell_matrix)
    
    return original_atoms_cart, supercell_atoms_cart, cell_matrix


def create_supercell_from_atoms(cell_matrix,atomspos):
    """
    Processes the given cell file and xyz file to:
      1. Convert the atomic Cartesian positions into fractional coordinates in the unit cell.
      2. Wrap the fractional coordinates so that they lie within the unit cell (i.e. in [0, 1)).
      3. Generate a 3x3x3 supercell from the wrapped coordinates.

    Inputs:
      cell_filename: Filename for the unit cell vectors.
      xyz_filename: Filename for the xyz file of atomic positions.

    Returns:
      A tuple with three elements:
        - original_atoms_cart: List of (atom_label, cartesian_coord) for atoms wrapped into the unit cell.
        - supercell_atoms_cart: List of (atom_label, cartesian_coord) for the full 3x3x3 supercell.
          (The first N entries correspond to the original atoms.)
        - cell_matrix: The 3x3 numpy array representing the unit cell vectors.
    """
    atoms=atomspos

    wrapped_frac_atoms = []
    original_atoms_cart = []

    for label, cart in atoms:
        frac = cartesian_to_fractional(cart, cell_matrix)
        frac_wrapped = frac
        wrapped_frac_atoms.append((label, frac_wrapped))

    supercell_atoms_cart = generate_supercell(wrapped_frac_atoms, cell_matrix)

    return supercell_atoms_cart


def analyze_supercell_bonds(supercell_atoms, original_atoms, bond_cutoff):
    """
    Determines bonds, molecular connectivity, and the pairwise distance matrix for atoms in a supercell.
    
    This function:
      1. Computes the pairwise Euclidean distance matrix among all atoms in the supercell 
         (ignoring periodic boundaries, since the supercell explicitly contains all images).
      2. Constructs a binary bond matrix of shape (M, M) where:
             bond_matrix[i, j] = 1   if the distance between atoms i and j is <= bond_cutoff
             bond_matrix[i, j] = 0   otherwise.
         (A small tolerance is used to avoid self-bonding.)
      3. Uses a depth-first search (DFS) to assign molecule indices. Starting with the first atom 
         (assigning it molecule index 0), the function “flood fills” all atoms connected by bonds 
         (directly or indirectly) with that same molecule index, then moves to the next unassigned atom.
      4. As the supercell_atoms list has its first N entries corresponding to the original atoms, 
         the molecule assignment for these atoms is similarly extracted.
    
    Inputs:
      supercell_atoms: List of tuples (atom_label, cartesian_coord) for all atoms in the supercell.
      original_atoms:  List of tuples (atom_label, cartesian_coord) for the original (wrapped) atoms.
      bond_cutoff:     A float specifying the cutoff distance below which two atoms are considered bonded.
    
    Returns:
      bond_matrix:                   A 2D numpy integer array of shape (M, M) where M = len(supercell_atoms).
      molecule_assignment:           A 1D numpy integer array of length M, with the molecule index for each atom.
      original_molecule_assignment:  A 1D numpy integer array containing the molecule indices for the original atoms.
      distances:                     A 2D numpy array of shape (M, M) containing the pairwise Euclidean distances.
    """
    # Extract positions from the supercell atom list.
    positions = np.array([pos for (_, pos) in supercell_atoms])
    M = positions.shape[0]
    
    # Compute the pairwise Euclidean distance matrix.
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Build the binary bond matrix with a tolerance to skip self-distances.
    tol = 1e-6
    bond_matrix = ((distances <= bond_cutoff) & (distances > tol)).astype(int)
    
    # Initialize an array to hold the molecule assignments for each atom.
    molecule_assignment = -1 * np.ones(M, dtype=int)
    current_mol = 0
    
    # Use DFS to assign molecule indices.
    for i in range(M):
        if molecule_assignment[i] != -1:
            continue  # Skip already assigned atoms.
        stack = [i]
        while stack:
            j = stack.pop()
            if molecule_assignment[j] == -1:
                molecule_assignment[j] = current_mol
                # Find all neighboring atoms bonded to the current atom.
                neighbors = np.where(bond_matrix[j] == 1)[0]
                for n in neighbors:
                    if molecule_assignment[n] == -1:
                        stack.append(n)
        current_mol += 1
    
    # Extract the molecule assignment for the original atoms.
    original_count = len(original_atoms)
    original_molecule_assignment = molecule_assignment[:original_count]
    
    return bond_matrix, molecule_assignment, original_molecule_assignment, distances



def identify_groups(supercell_atoms, original_atoms, bond_matrix):
    """
    Identify functional-group atom memberships (0/1 flags on *original* atoms).

    Groups returned:
      - nitro:      N bonded to exactly two terminal O atoms (NO2)
      - furazan:    C–C–N–O–N 5-member ring *without* an N-oxide O
      - furoxano:   same 5-member ring *with* a terminal exocyclic O bonded to a ring N (N-oxide)
                  (flags include the ring atoms + the exocyclic O)
      - rings:      furazan ring atoms + furoxano ring atoms (excludes the exocyclic O)
      - furoxano_oxo: only the exocyclic O atom of the furoxano group
    """
    M = len(supercell_atoms)
    N = len(original_atoms)
    elements = [elem for (elem, _) in supercell_atoms]

    def bonded_neighbors(i: int) -> list[int]:
        return [j for j in range(M) if bond_matrix[i, j] == 1]

    flags = {
        "nitro": [0] * N,
        "furazan": [0] * N,
        "furoxano": [0] * N,
        "rings": [0] * N,
        "furoxano_oxo": [0] * N,
    }

    # --- Nitro group detection (NO2) ---
    for i in range(M):
        if elements[i] != "N":
            continue
        neighbors = bonded_neighbors(i)
        oxygen_neighbors = [j for j in neighbors if elements[j] == "O"]
        if len(oxygen_neighbors) != 2:
            continue

        # Both oxygens must be terminal and only bonded to the nitro N.
        if all(bonded_neighbors(o) == [i] for o in oxygen_neighbors):
            for idx in [i] + oxygen_neighbors:
                if idx < N:
                    flags["nitro"][idx] = 1

    # --- 5-member ring detection: C–C–N–O–N (furazan/furoxano ring core) ---
    def find_rings_ccnon():
        rings_out = []
        for i in range(M):
            if elements[i] != "C":
                continue
            for j in bonded_neighbors(i):
                if elements[j] != "C" or j <= i:
                    continue
                for k in bonded_neighbors(j):
                    if elements[k] != "N" or k in (i, j):
                        continue
                    for l in bonded_neighbors(k):
                        if elements[l] != "O" or l in (i, j, k):
                            continue
                        for m in bonded_neighbors(l):
                            if elements[m] != "N" or m in (i, j, k, l):
                                continue
                            # close ring back to i
                            if i in bonded_neighbors(m):
                                rings_out.append([i, j, k, l, m])
        # De-duplicate by atom set
        seen = set()
        uniq = []
        for ring in rings_out:
            key = frozenset(ring)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ring)
        return uniq

    rings = find_rings_ccnon()

    furazan_rings = []
    furoxano_rings = []
    furoxano_exo_oxygens: set[int] = set()

    for ring in rings:
        ring_set = set(ring)
        ring_ns = [idx for idx in ring if elements[idx] == "N"]

        exo_o = None
        for n_idx in ring_ns:
            for nb in bonded_neighbors(n_idx):
                if elements[nb] != "O" or nb in ring_set:
                    continue
                # Terminal oxygen (only bonded to this N) → treat as N-oxide oxygen
                if bonded_neighbors(nb) == [n_idx]:
                    exo_o = nb
                    break
            if exo_o is not None:
                break

        if exo_o is None:
            furazan_rings.append(ring)
        else:
            furoxano_rings.append(ring)
            furoxano_exo_oxygens.add(exo_o)

    # Set flags for furazan ring atoms
    for ring in furazan_rings:
        for idx in ring:
            if idx < N:
                flags["furazan"][idx] = 1
                flags["rings"][idx] = 1

    # Set flags for furoxano ring atoms and exocyclic O
    for ring in furoxano_rings:
        for idx in ring:
            if idx < N:
                flags["furoxano"][idx] = 1
                flags["rings"][idx] = 1
    for o_idx in furoxano_exo_oxygens:
        if o_idx < N:
            flags["furoxano"][o_idx] = 1
            flags["furoxano_oxo"][o_idx] = 1

    return flags

def compute_centers_of_mass(original_atoms, original_molecule_assignment, supercell_atoms, full_molecule_assignment):
    """
    Computes the center of mass (COM) for each molecule, ensuring molecules include all supercell atoms 
    that belong to molecules containing at least one original atom.

    This function:
      1. Checks that each molecule (determined from full_molecule_assignment) contains exactly 12 atoms.
      2. Identifies molecules that contain at least one original atom.
      3. Computes the center of mass for each identified molecule using all atoms in the supercell belonging to it.

    Parameters:
      original_atoms: List of tuples (atom_label, cartesian_coord) for the original (wrapped) atoms.
                      Example: [("C", np.array([0.0, 0.0, 0.0])), ("H", np.array([0.9, 0.0, 0.0]))]
      original_molecule_assignment: NumPy array where original_molecule_assignment[i] is the molecule index 
                                    assigned to original_atoms[i].
      supercell_atoms: List of tuples (atom_label, cartesian_coord) for all atoms in the supercell.
      full_molecule_assignment: NumPy array where full_molecule_assignment[i] is the molecule index 
                                assigned to supercell_atoms[i].
    
    Returns:
      centers: A NumPy array of shape (n_molecules, 3) where the row with index i is the center of mass 
               for molecule i.
    
    Raises:
      ValueError: If any molecule in the full molecule assignment does not contain exactly 12 atoms.
    """
    # Define atomic masses (default values for H, C, O, N).
    mass_dict = {
        "H": 1.008,
        "C": 12.011,
        "O": 15.999,
        "N": 14.007
    }

    # Get unique molecules from the original molecule assignment.
    unique_mols = np.unique(original_molecule_assignment)

    # Verify that each molecule appears 12 times in the supercell.
    for mol in unique_mols:
        count = np.sum(full_molecule_assignment == mol)
        if count != 22:
            print(
                f"Molecule index {mol} contains {count} atoms in the full supercell assignment, expected 22."
            )

    # Initialize accumulation arrays for COM calculations.
    n_mol = len(unique_mols)
    mass_weighted_sum = np.zeros((n_mol, 3))
    mass_total = np.zeros(n_mol)

    # Loop through all atoms in the supercell, accumulating mass-weighted positions.
    for i, (label, pos) in enumerate(supercell_atoms):
        mol_index = full_molecule_assignment[i]

        # Only consider atoms belonging to molecules containing at least one original atom.
        if mol_index in unique_mols:
            mass = mass_dict.get(label, 1.0)  # Default mass = 1.0 if not specified.
            mass_weighted_sum[mol_index] += mass * pos
            mass_total[mol_index] += mass

    # Compute the center of mass for each molecule.
    centers = np.zeros((n_mol, 3))
    for m in range(n_mol):
        if mass_total[m] > 0:
            centers[m] = mass_weighted_sum[m] / mass_total[m]
        else:
            centers[m] = mass_weighted_sum[m]

    return centers


def compute_centers_of_mass_change(
    original_atoms, original_molecule_assignment,
    supercell_atoms, full_molecule_assignment,
    normal_mode_atoms, cell_matrix, tol=0.1
):
    """
    Computes the average displacement magnitude of molecular centers of mass,
    excluding duplicates within `tol` Å after applying periodic boundary conditions.
    Ensures all COMs are wrapped into the central unit cell (origin at 0,0,0).

    cell_matrix: shape (3,3), each column is a lattice vector in Cartesian space
                 (as from np.column_stack((A, B, C))).
    """

    # Atomic masses
    mass_dict = {
        "H": 1.008,
        "C": 12.011,
        "O": 15.999,
        "N": 14.007
    }

    # Step 1: Apply normal mode displacement to original atoms
    shifted_atoms = []
    for i, (atom_i, pos_i) in enumerate(original_atoms):
        _, mode_vec = normal_mode_atoms[i]
        mass = mass_dict[atom_i]
        scale = 1.0 / np.sqrt(mass)
        shifted_pos = pos_i + scale * mode_vec
        shifted_atoms.append((atom_i, shifted_pos))

    # Step 2: Build supercell for shifted atoms
    atomshift_supercell = create_supercell_from_atoms(cell_matrix, shifted_atoms)
    atomorig_supercell =  create_supercell_from_atoms(cell_matrix, original_atoms)
    # Step 3: Compute COMs before and after shift
    centers_start = compute_centers_of_mass(
        original_atoms, original_molecule_assignment,
        atomorig_supercell, full_molecule_assignment
    )
    centers_end = compute_centers_of_mass(
        shifted_atoms, original_molecule_assignment,
        atomshift_supercell, full_molecule_assignment
    )

    # Step 4: Prepare fractional coordinate conversion for column-vector cell_matrix
    inv_cell = np.linalg.inv(cell_matrix)

    def frac_coords(cart):
        """Convert Cartesian → fractional (column-vector cell convention)."""
        return np.dot(cart, inv_cell)

    def cart_coords(frac):
        """Convert fractional → Cartesian (column-vector cell convention)."""
        return np.dot(frac, cell_matrix)

    def wrap_to_central(cart):
        """Wrap Cartesian point into [0,1) fractional space, then back to Cartesian."""
        f = frac_coords(cart) % 1.0
        return cart_coords(f)

    def min_image_dist(c1, c2):
        """Minimum image distance between two Cartesian points."""
        df = frac_coords(c1) - frac_coords(c2)
        df -= np.round(df)  # wrap into [-0.5, 0.5)
        return np.linalg.norm(np.dot(df, cell_matrix))

    # Step 5: Wrap all COMs into central cell
    centers_start = np.array([wrap_to_central(c) for c in centers_start])
    centers_end   = np.array([wrap_to_central(c) for c in centers_end])

    # Step 6: Identify unique COMs (no duplicates within tol)
    unique_indices = []
    for i, c in enumerate(centers_start):
        if not any(min_image_dist(c, centers_start[j]) < tol for j in unique_indices):
            unique_indices.append(i)

    # Step 7: Compute displacements only for unique molecules
    displacements = [
        np.linalg.norm(centers_end[i] - centers_start[i])
        for i in unique_indices
    ]
    


    return np.mean(displacements) if displacements else 0.0




def read_normal_mode_xyz(xyz_file):
    """
    Reads an xyz file containing normal mode displacement vectors.

    Expected file format:
      number_of_atoms
      comment_line
      atom_label    dx    dy    dz
      ...

    Returns:
      A tuple (atoms, comment) where atoms is a list of tuples:
         (atom_label, numpy array([dx, dy, dz]))
    """
    atoms = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError("XYZ file does not contain enough lines.")
    
    try:
        natoms = int(lines[0].strip())
    except Exception as e:
        raise ValueError("The first line must be an integer (number of atoms).") from e
    
    comment = lines[1].strip() if len(lines) > 1 else ""
    
    # Read normal mode displacement vectors
    for line in lines[2:2+natoms]:  # Skip first two header lines
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        label = parts[0]
        try:
            displacement = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            raise ValueError("Error parsing displacement vectors in the line: " + line)
        atoms.append((label, displacement))
    
    return atoms, comment

def analyze_mode_projection(original_atoms, original_molecule_assignment, centers_of_mass, normal_mode_atoms):
    """
    Computes the projection of normal mode vectors onto the center-of-mass direction for each atom.

    Parameters:
      original_atoms: List of tuples (atom_label, cartesian_coord) for the original atoms.
      original_molecule_assignment: NumPy array of molecule indices corresponding to the original atoms.
      centers_of_mass: NumPy array of shape (n_molecules, 3), where row i contains the center of mass for molecule i.
      normal_mode_atoms: List of tuples (atom_label, numpy array([dx, dy, dz])) from read_normal_mode_xyz.

    Returns:
      normalized_sum: The sum of absolute dot products, normalized by the sum of normal mode magnitudes.
    """
    total_dot_product = 0.0
    total_mode_magnitude = 0.0

    for i in range(len(original_atoms)):  # Assume orders match
        atom_label, atom_pos = original_atoms[i]
        _, displacement_vector = normal_mode_atoms[i]  # Retrieve normal mode vector at matching index
        
        mol_index = original_molecule_assignment[i]
        com_vector = (centers_of_mass[mol_index] - atom_pos)/ np.linalg.norm(centers_of_mass[mol_index] - atom_pos)  # Vector from atom to COM

        # Compute dot product (absolute value)
        dot_product = np.abs(np.dot(displacement_vector, com_vector))
        total_dot_product += dot_product

        # Accumulate normal mode vector magnitude
        total_mode_magnitude += np.linalg.norm(displacement_vector)

    # Normalize the sum of dot products by the sum of normal mode vector magnitudes
    normalized_sum = total_dot_product / total_mode_magnitude if total_mode_magnitude > 0 else 0.0

    return normalized_sum, total_dot_product

def analyze_mode_axis(original_atoms, normal_mode_atoms):
    """

    Parameters:
      original_atoms: List of tuples (atom_label, cartesian_coord) for the original atoms.
      normal_mode_atoms: List of tuples (atom_label, numpy array([dx, dy, dz])) from read_normal_mode_xyz.

    Returns:
      normalized_sum: The sum of absolute dot products, normalized by the sum of normal mode magnitudes.
    """
    total_dot_product = 0.0
    total_mode_magnitude = 0.0
    axesset=['x','y','z']
    projectionaxes = {axis: 0.0 for axis in axesset}
    unnormprojectionaxes = {axis: 0.0 for axis in axesset}
    axesdefine=[[1,0,0],[0,1,0],[0,0,1]]

    for j in range(3):
      axisnow=axesdefine[j]
      total_dot_product = 0.0
      total_mode_magnitude = 0.0
      for i in range(len(original_atoms)):  # Assume orders match
        atom_label, atom_pos = original_atoms[i]
        _, displacement_vector = normal_mode_atoms[i]  # Retrieve normal mode vector at matching index


        # Compute dot product (absolute value)

        dot_product = np.abs(np.dot(displacement_vector, axisnow))
        total_dot_product += dot_product

        # Accumulate normal mode vector magnitude
        total_mode_magnitude += np.linalg.norm(displacement_vector)

    # Normalize the sum of dot products by the sum of normal mode vector magnitudes
      normalized_sum = total_dot_product / total_mode_magnitude if total_mode_magnitude > 0 else 0.0
      unnormalized_sum=total_dot_product
      projectionaxes[axesset[j]]=normalized_sum 
      unnormprojectionaxes[axesset[j]]=unnormalized_sum
    

    return projectionaxes,unnormprojectionaxes


def analyze_mode_orthogonal_magnitude(original_atoms, original_molecule_assignment, centers_of_mass, normal_mode_atoms):
    """
    Computes the sum of the magnitudes of normal mode components orthogonal to the radial vector from the center of mass 
    to the atom, then normalizes by the sum of the normal mode magnitudes.

    Parameters:
      original_atoms: List of tuples (atom_label, cartesian_coord) for the original atoms.
      original_molecule_assignment: NumPy array of molecule indices corresponding to the original atoms.
      centers_of_mass: NumPy array of shape (n_molecules, 3), where row i contains the center of mass for molecule i.
      normal_mode_atoms: List of tuples (atom_label, numpy array([dx, dy, dz])) from read_normal_mode_xyz.

    Returns:
      normalized_sum: The sum of orthogonal component magnitudes, normalized by the sum of normal mode magnitudes.
    """
    total_orthogonal_magnitude = 0.0
    total_mode_magnitude = 0.0

    for i in range(len(original_atoms)):  # Assume orders match
        atom_label, atom_pos = original_atoms[i]
        _, displacement_vector = normal_mode_atoms[i]  # Retrieve normal mode vector at matching index

        mol_index = original_molecule_assignment[i]
        com_vector = centers_of_mass[mol_index] - atom_pos  # Vector from atom to COM

        # Normalize the radial vector
        radial_magnitude = np.linalg.norm(com_vector)
        radial_unit_vector = com_vector / radial_magnitude if radial_magnitude > 0 else np.zeros_like(com_vector)

        # Compute projection of normal mode vector onto the normalized radial direction
        projection_radial = np.dot(displacement_vector, radial_unit_vector) * radial_unit_vector

        # Compute orthogonal component by subtracting the radial projection from the normal mode displacement vector
        orthogonal_component = displacement_vector - projection_radial

        # Compute magnitude of orthogonal component and accumulate
        total_orthogonal_magnitude += np.linalg.norm(orthogonal_component)

        # Accumulate normal mode vector magnitude
        total_mode_magnitude += np.linalg.norm(displacement_vector)

    # Normalize the sum of orthogonal magnitudes by the sum of normal mode vector magnitudes
    normalized_sum = total_orthogonal_magnitude / total_mode_magnitude if total_mode_magnitude > 0 else 0.0

    return normalized_sum,total_orthogonal_magnitude



def analyze_spread(norm_mode):
    """
    Analyze the spread of a normal mode displacement pattern.

    Parameters
    ----------
    norm_mode : array-like, shape (N, 3)
        Displacement vectors for each atom in the mode.

    Returns
    -------
    rms : float
        RMS displacement magnitude.
    normalized_rms : float
        RMS displacement normalized by the value if the total displacement
        magnitude was split evenly amongst all atoms.
    """
    disp = np.array([vec for _, vec in norm_mode], dtype=float)
    N = disp.shape[0]


    # Magnitude of displacement for each atom
    mags = np.linalg.norm(disp, axis=1)
    # RMS displacement magnitude
    rms = np.sqrt(np.sum(mags**2))

    # Hypothetical RMS if displacement magnitude were evenly distributed
    total_mag = np.sum(mags)
    equal_share_mag = total_mag / N
    normalized_rms = np.sqrt((equal_share_mag**2) * N)
    return rms/normalized_rms, rms    



def analyze_mode_intermole(original_atoms,
                           original_molecule_assignment,
                           supercell_atoms,
                           full_molecule_assignment,
                           normal_mode_atoms,
                           distance_threshold):
    """
    Computes vibrational coupling contributions and provides both overall
    and atomic-type-specific normalized sums, using label-dependent
    distance offsets in the distance-scaled matrix.

    Returns:
      normalized_threshold_sum_all_atoms,
      normalized_distance_scaled_sum_all_atoms,
      normalized_threshold_sums_by_type,
      normalized_distance_scaled_sums_by_type
    """

    N = len(original_atoms)
    M = len(supercell_atoms)

    # Prepare label‐dependent distance offsets
    distance_offsets = {
        ('C','C'): 1.6, ('C','H'): 1.1, ('H','C'): 1.1,
        ('C','N'): 1.4, ('N','C'): 1.4,
        ('C','O'): 1.4, ('O','C'): 1.4,
        ('H','H'): 0.9,
        ('O','N'): 1.2, ('N','O'): 1.2,
        ('O','H'): 1.0, ('H','O'): 1.0,
        ('N','H'): 1.0, ('H','N'): 1.0,
        ('O','O'): 1.3,
        ('N','N'): 1.2,
    }

    # Accumulators
    atom_types = {label for label, _ in original_atoms}
    max_thr_by_type = {atom: 0.0 for atom in atom_types}
    max_scaled_by_type = {atom: 0.0 for atom in atom_types}
    magnitude_by_type = {atom: 0.0 for atom in atom_types}
    max_thr_vals = np.zeros(N)
    max_scaled_vals = np.zeros(N)
    total_mode_mag = 0.0
    min_inmolec=0
    min_outmolec= 0
    min_inmolec_by_type= {atom: 0.0 for atom in atom_types}
    min_outmolec_by_type= {atom: 0.0 for atom in atom_types}


    for i in range(N):
        atom_i, pos_i = original_atoms[i]
        _, mode_vec = normal_mode_atoms[i]

        local_max_thr = 0.0
        local_max_scaled = 0.0
        local_min_inmolec_dist=10000
        local_min_outmolec_dist=10000
        for j in range(M):
            atom_j, pos_j = supercell_atoms[j]

            # displacement and distance
            disp = pos_j - pos_i
            dist = np.linalg.norm(disp)
            norm_disp = disp/dist if dist > 0 else np.zeros_like(disp)

            # dot product (zero if same molecule)
            dot = abs(np.dot(mode_vec, norm_disp))
            same_molec=10000
            diff_molec=dist
            if original_molecule_assignment[i] == full_molecule_assignment[j]:
                dot = 0.0
                if dist!=0:
                  same_molec=dist
                diff_molec=10000
            # threshold matrix contribution
            thr_val = dot if dist <= distance_threshold else 0.0
            local_max_thr = max(local_max_thr, thr_val)

            # distance‐scaled with label‐dependent offset
            offset = distance_offsets.get((atom_i, atom_j), 1.6)
            



            if dist > 0:
                scaled_val = dot / ((dist - offset) ** 2)
            else:
                scaled_val = 0.0
            local_max_scaled = max(local_max_scaled, scaled_val)
            

            local_min_inmolec_dist=min(same_molec, local_min_inmolec_dist)
            local_min_outmolec_dist=min(diff_molec, local_min_outmolec_dist)




        min_inmolec+=np.linalg.norm(mode_vec)*local_min_inmolec_dist
        min_outmolec+=np.linalg.norm(mode_vec)*local_min_outmolec_dist

        max_thr_vals[i] = local_max_thr
        max_scaled_vals[i] = local_max_scaled
        total_mode_mag += np.linalg.norm(mode_vec)




        max_thr_by_type[atom_i] += local_max_thr
        max_scaled_by_type[atom_i] += local_max_scaled
        
        magnitude_by_type[atom_i] +=  np.linalg.norm(mode_vec)
        
        min_inmolec_by_type[atom_i]+=np.linalg.norm(mode_vec)*local_min_inmolec_dist
        min_outmolec_by_type[atom_i]+=np.linalg.norm(mode_vec)*local_min_outmolec_dist


    norm_min_inmolec=min_inmolec/total_mode_mag
    norm_min_outmolec=min_outmolec/total_mode_mag



    # overall normalized sums
    normval=1
    if total_mode_mag > 0:
        norm_thr_all = (np.sum(max_thr_vals**normval) **(1/normval)) / total_mode_mag
        norm_scaled_all = (np.sum(max_scaled_vals) ** (1/normval)) / total_mode_mag
    else:
        norm_thr_all = 0.0
        norm_scaled_all = 0.0
    
    normval=1
    if total_mode_mag > 0:
        unnorm_thr_all = (np.sum(max_thr_vals**normval) **(1/normval)) 
        unnorm_scaled_all = (np.sum(max_scaled_vals) ** (1/normval)) 
    else:
        unnorm_thr_all = 0.0
        unnorm_scaled_all = 0.0



    # per-atom-type normalized sums
    norm_thr_by_type = {
        atom: val/total_mode_mag if total_mode_mag>0 else 0.0
        for atom, val in max_thr_by_type.items()
    }
    norm_scaled_by_type = {
        atom: val/total_mode_mag if total_mode_mag>0 else 0.0
        for atom, val in max_scaled_by_type.items()
    }
        # per-atom-type normalized sums
    unnorm_thr_by_type = {
        atom: val 
        for atom, val in max_thr_by_type.items()
    }
    unnorm_scaled_by_type = {
        atom: val 
        for atom, val in max_scaled_by_type.items()
    }
    

    norm_inmolec_by_type = {
        atom: val/magnitude_by_type[atom] if magnitude_by_type[atom]>0 else 0.0
        for atom, val in min_inmolec_by_type.items()
    }
    norm_outmolec_by_type = {
        atom: val/magnitude_by_type[atom] if magnitude_by_type[atom]>0 else 0.0
        for atom, val in min_outmolec_by_type.items()
    }



    return norm_thr_all, norm_scaled_all, norm_thr_by_type, norm_scaled_by_type,  unnorm_thr_all, unnorm_scaled_all, unnorm_thr_by_type, unnorm_scaled_by_type, total_mode_mag,magnitude_by_type, norm_min_outmolec,norm_min_inmolec,norm_outmolec_by_type ,norm_inmolec_by_type




def fractional_group_displacements(norm_mode, group_flags: dict[str, list[int]]):
    """
    Compute (fraction_total, fraction_ON) for each group in group_flags.

    - fraction_total: group displacement magnitude / total displacement magnitude
    - fraction_ON:    group displacement magnitude in (O,N) atoms / total (O,N) magnitude
    """
    disp = np.array([vec for _, vec in norm_mode], dtype=float)
    elements = np.array([elem for elem, _ in norm_mode], dtype=object)
    mags = np.linalg.norm(disp, axis=1)

    total_mag = float(np.sum(mags))
    on_mask = np.isin(elements, ["O", "N"])
    on_mag = float(np.sum(mags[on_mask]))

    out: dict[str, tuple[float, float]] = {}
    for name, flags in group_flags.items():
        mask = np.array(flags, dtype=bool)
        group_total = float(np.sum(mags[mask]))
        group_on = float(np.sum(mags[mask & on_mask]))
        out[name] = (
            group_total / total_mag if total_mag > 0 else 0.0,
            group_on / on_mag if on_mag > 0 else 0.0,
        )
    return out

def run_supercell(cell_filename, xyz_filename, xyz_modes_filename,bond_cutoff,distance_threshold):
    """
    A helper function which takes the filenames, executes the supercell creation,
    and prints the resulting original atoms, full supercell atoms, and the unit cell matrix.
    """

    normal_mode_atoms, comment = read_normal_mode_xyz(xyz_modes_filename)

    original_atoms, supercell_atoms, cell_matrix = create_supercell(cell_filename, xyz_filename)
    bond_matrix, molecule_assignment, original_molecule_assignment,distances= analyze_supercell_bonds(supercell_atoms, original_atoms, bond_cutoff)
    group_flags = identify_groups(supercell_atoms, original_atoms, bond_matrix)
    group_fracs = fractional_group_displacements(normal_mode_atoms, group_flags)

    nitro_fraction_total, nitro_fraction_ON = group_fracs["nitro"]
    furazan_fraction_total, furazan_fraction_ON = group_fracs["furazan"]
    furoxano_fraction_total, furoxano_fraction_ON = group_fracs["furoxano"]
    rings_fraction_total, rings_fraction_ON = group_fracs["rings"]
    furoxano_oxo_fraction_total, furoxano_oxo_fraction_ON = group_fracs["furoxano_oxo"]
    orig_molec_centers=compute_centers_of_mass(original_atoms, original_molecule_assignment,supercell_atoms, molecule_assignment)

    com_disp_total_mag=compute_centers_of_mass_change(original_atoms, original_molecule_assignment,supercell_atoms, molecule_assignment,normal_mode_atoms,cell_matrix)

    radialproj, unnormradialproj =analyze_mode_projection(original_atoms, original_molecule_assignment,orig_molec_centers, normal_mode_atoms)
    orthoproj,unnormorthoproj=analyze_mode_orthogonal_magnitude(original_atoms, original_molecule_assignment, orig_molec_centers, normal_mode_atoms)

    axisproj, unnormaxisproj =analyze_mode_axis(original_atoms, normal_mode_atoms)

    
    spread_factor, spread_factor_unnorm= analyze_spread(normal_mode_atoms)


    normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms, normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type, unnormalized_threshold_sum_all_atoms, unnormalized_distance_scaled_sum_all_atoms, unnormalized_threshold_sums_by_type, unnormalized_distance_scaled_sums_by_type, totalmag, magbytype,norm_min_outmolec,norm_min_inmolec,norm_outmolec_by_type ,norm_inmolec_by_type =analyze_mode_intermole(original_atoms, original_molecule_assignment, supercell_atoms, molecule_assignment, normal_mode_atoms, distance_threshold)
 

    return normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type,normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms ,radialproj,orthoproj,unnormalized_threshold_sums_by_type, unnormalized_distance_scaled_sums_by_type,unnormalized_threshold_sum_all_atoms, unnormalized_distance_scaled_sum_all_atoms ,unnormradialproj,unnormorthoproj, totalmag,magbytype,  axisproj, unnormaxisproj,norm_min_outmolec,norm_min_inmolec,norm_outmolec_by_type ,norm_inmolec_by_type,com_disp_total_mag,spread_factor, spread_factor_unnorm,nitro_fraction_total, furazan_fraction_total, furoxano_fraction_total, rings_fraction_total, furoxano_oxo_fraction_total, nitro_fraction_ON, furazan_fraction_ON, furoxano_fraction_ON, rings_fraction_ON, furoxano_oxo_fraction_ON









def parse_trajectories_for_modes(pressureslisttraj,position_xyz,
                                 trajectory_list,
                                 mode_indices,
                                 output_prefix="output",
                                 outdir=None):
    """
    Computes normal mode displacement vectors between frame 0 and the last frame
    of each trajectory.  The provided mode_indices list is used only to label
    outputs (it does not select which frame to read).

    Parameters:
      position_xyz    : Path to a static XYZ file with atom labels & reference positions.
      trajectory_list : List of multi‐frame XYZ trajectory filenames.
      mode_indices    : List of integers, one per trajectory, used to tag the mode in the
                        output filenames (does not index frames).
      output_prefix   : Prefix for all output filenames.

    Returns:
      positions_filename        (str) :
        The path to the saved reference positions .xyz file.
      displacement_filenames (list) :
        Paths to the per‐trajectory displacement .xyz files.
    """
    print(trajectory_list)
    if len(trajectory_list) != len(mode_indices):
        raise ValueError(
            f"Must supply one mode index per trajectory; got "
            f"{len(trajectory_list)} trajectories vs {len(mode_indices)} indices"
        )

    def read_xyz_frame_block(lines, start):
        """Return a block of num_atoms lines (split) starting at `start`."""
        num_atoms = int(lines[start].strip())
        block = lines[start+2 : start+2+num_atoms]
        if len(block) != num_atoms:
            raise ValueError(
                f"Incomplete frame: expected {num_atoms} atoms "
                f"at lines {start+2}–{start+1+num_atoms}"
            )
        return [ln.split() for ln in block]

    def read_all_frames(filename):
        """Read all xyz frames into a list of [ [atom, x, y, z], ... ] blocks."""
        with open(filename) as f:
            lines = f.readlines()

        frames = []
        i = 0
        while i < len(lines):
            try:
                num = int(lines[i].strip())
            except Exception:
                raise ValueError(f"Bad atom count line {i} in {filename!r}")
            frame = read_xyz_frame_block(lines, i)
            frames.append(frame)
            i += 2 + num
        return frames

    # --- Read & write reference positions once ---
    ref_block = read_all_frames(position_xyz)[0]
    atom_labels = [entry[0] for entry in ref_block]
    natoms = len(atom_labels)
    ref_coords = np.array([[float(v) for v in entry[1:]] for entry in ref_block])

    if outdir is None:
        outdir = BASEDIR

    positions_filename = os.path.join(outdir, f"{output_prefix}_P{pressureslisttraj}_positions.xyz")
    with open(positions_filename, "w") as outf:
        outf.write(f"{natoms}\nReference positions\n")
        for lab, coord in zip(atom_labels, ref_coords):
            outf.write(f"{lab} {' '.join(map(str, coord))}\n")

    # --- Process each trajectory & tag by mode_indices ---
    disp_files = []
    mass_dict = {
        "H": 1.008,
        "C": 12.011,
        "O": 15.999,
        "N": 14.007
    }

    for traj, mode_idx in zip(trajectory_list, mode_indices):
        frames = read_all_frames(traj)
        # always use first & last frame
        coords0 = np.array([[float(x) for x in ent[1:]] for ent in frames[0]])
        coordsF = np.array([[float(x) for x in ent[1:]] for ent in frames[25]])

        if coords0.shape != coordsF.shape or coords0.shape[0] != natoms:
            raise ValueError(
                f"Atom count mismatch: reference has {natoms}, "
                f"traj '{traj}' frame counts {coords0.shape[0]}"
            )

        disp = coordsF - coords0
        for kk in range(len(disp)):
             disp[kk] *= np.sqrt(mass_dict[frames[0][kk][0]])
        outname = os.path.join(outdir, f"{output_prefix}_P{pressureslisttraj}_mode{mode_idx}.xyz")
        with open(outname, "w") as outf:
            outf.write(f"{natoms}\nDisplacement mode {mode_idx}\n")
            for lab, vec in zip(atom_labels, disp):
                outf.write(f"{lab} {' '.join(map(str, vec))}\n")

        print(f"✅ Wrote: {outname}")
        disp_files.append(outname)

    return positions_filename, disp_files


def run_analysis_over_pressures(cell_filenames, xyz_filenames, xyz_modes_filenames, bond_cutoff, distance_threshold, pressures, output_prefix="plot"):
    """
    Runs supercell analysis over a range of pressures, collects results, and generates multipanel figures.

    Parameters:
      cell_filenames: List of unit cell data filenames, one per pressure.
      xyz_filenames: List of atomic positions filenames, one per pressure.
      xyz_modes_filenames: List of normal mode vectors filenames, one per pressure.
      bond_cutoff: Cutoff distance for determining molecular connectivity.
      distance_threshold: Cutoff distance for filtering vibrational interactions.
      pressures: List of pressure values to run the analysis at.
      output_prefix: Prefix for saving plot filenames.

    Returns:
      results_dict: Dictionary containing analysis results for each pressure.
    """
    if not (len(cell_filenames) == len(xyz_filenames) == len(xyz_modes_filenames) == len(pressures)):
        raise ValueError("Mismatch in list lengths: Each pressure must have corresponding filenames.")

    results_dict = {
        "pressures": pressures,
        "threshold_sum_all_atoms": [],
        "distance_scaled_sum_all_atoms": [],
        "radial_projection": [],
        "orthogonal_projection": [],
        "threshold_sums_by_type": {},
        "distance_scaled_sums_by_type": {},
        "normalized_threshold_sum_all_atoms": [],
        "normalized_distance_scaled_sum_all_atoms": [],
        "normalized_radial_projection": [],
        "normalized_orthogonal_projection": [],
        "normalized_threshold_sums_by_type": {},
        "normalized_distance_scaled_sums_by_type": {},
        "total_magnitude": [],
        "magnitude_by_type": {},
        "calssical_potential": [],
        "classical_potential_by_type": {},
        "axis_projection": {},
        "unnormaxis_projection": {},
        "norm_minoutmolec":[],
        "norm_mininmolec":[],
        "norm_outmolec_bytype":{},
        "norm_inmolec_bytype":{},
        "com_disp_mag":[],
        "spread":[],
        "spread_unnorm":[],
        "nitro_fraction_total":[],
        "furazan_fraction_total":[],
        "furoxano_fraction_total":[],
        "rings_fraction_total":[],
        "furoxano_oxo_fraction_total":[],
        "nitro_fraction_ON":[],
        "furazan_fraction_ON":[],
        "furoxano_fraction_ON":[],
        "rings_fraction_ON":[],
        "furoxano_oxo_fraction_ON":[]
    }

    # Collect results across pressures
    for idx, pressure in enumerate(pressures):
        print(f"Running analysis at pressure {pressure} using files: {cell_filenames[idx]}, {xyz_filenames[idx]}, {xyz_modes_filenames[idx]}")

        # Run supercell analysis with the corresponding files
        normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type,normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms ,radialproj,orthoproj,unnormalized_threshold_sums_by_type, unnormalized_distance_scaled_sums_by_type,unnormalized_threshold_sum_all_atoms, unnormalized_distance_scaled_sum_all_atoms ,unnormradialproj,unnormorthoproj, totalmag,magbytype, axisproj, unnormaxisproj, norm_min_outmolec,norm_min_inmolec,norm_outmolec_by_type ,norm_inmolec_by_type,com_disp_total_mag,spread_factor, spread_factor_unnorm,nitro_fraction_total, furazan_fraction_total, furoxano_fraction_total, rings_fraction_total, furoxano_oxo_fraction_total, nitro_fraction_ON, furazan_fraction_ON, furoxano_fraction_ON, rings_fraction_ON, furoxano_oxo_fraction_ON = run_supercell(
            cell_filenames[idx], xyz_filenames[idx], xyz_modes_filenames[idx], bond_cutoff, distance_threshold
        )

        # Store results
        results_dict["normalized_threshold_sum_all_atoms"].append(normalized_threshold_sum_all_atoms)
        results_dict["normalized_distance_scaled_sum_all_atoms"].append(normalized_distance_scaled_sum_all_atoms)
        results_dict["normalized_radial_projection"].append(radialproj)
        results_dict["normalized_orthogonal_projection"].append(orthoproj)
                # Store results
        results_dict["threshold_sum_all_atoms"].append(unnormalized_threshold_sum_all_atoms)
        results_dict["distance_scaled_sum_all_atoms"].append(unnormalized_distance_scaled_sum_all_atoms)
        results_dict["radial_projection"].append(unnormradialproj)
        results_dict["orthogonal_projection"].append(unnormorthoproj)
        results_dict["total_magnitude"].append(totalmag)
        results_dict["com_disp_mag"].append(com_disp_total_mag)
        results_dict["norm_minoutmolec"].append(norm_min_outmolec)
        results_dict["norm_mininmolec"].append(norm_min_inmolec)
        results_dict["spread"].append(spread_factor)
        results_dict["spread_unnorm"].append(spread_factor_unnorm) 
        results_dict["nitro_fraction_total"].append(nitro_fraction_total)
        results_dict["furazan_fraction_total"].append(furazan_fraction_total)
        results_dict["furoxano_fraction_total"].append(furoxano_fraction_total)
        results_dict["rings_fraction_total"].append(rings_fraction_total)
        results_dict["furoxano_oxo_fraction_total"].append(furoxano_oxo_fraction_total)
        results_dict["nitro_fraction_ON"].append(nitro_fraction_ON)
        results_dict["furazan_fraction_ON"].append(furazan_fraction_ON)
        results_dict["furoxano_fraction_ON"].append(furoxano_fraction_ON)
        results_dict["rings_fraction_ON"].append(rings_fraction_ON)
        results_dict["furoxano_oxo_fraction_ON"].append(furoxano_oxo_fraction_ON)




        # Store per-atom-type results
        for atom_type in normalized_threshold_sums_by_type:
            if atom_type not in results_dict["threshold_sums_by_type"]:
                results_dict["threshold_sums_by_type"][atom_type] = []
                results_dict["distance_scaled_sums_by_type"][atom_type] = []
                results_dict["magnitude_by_type"][atom_type]=[]
                
                results_dict["norm_outmolec_bytype"][atom_type]=[]
                results_dict["norm_inmolec_bytype"][atom_type]=[]

                results_dict["normalized_threshold_sums_by_type"][atom_type] = []
                results_dict["normalized_distance_scaled_sums_by_type"][atom_type] = []

            results_dict["normalized_threshold_sums_by_type"][atom_type].append(normalized_threshold_sums_by_type[atom_type])
            results_dict["normalized_distance_scaled_sums_by_type"][atom_type].append(normalized_distance_scaled_sums_by_type[atom_type])
            results_dict["threshold_sums_by_type"][atom_type].append(unnormalized_threshold_sums_by_type[atom_type])
            results_dict["distance_scaled_sums_by_type"][atom_type].append(unnormalized_distance_scaled_sums_by_type[atom_type])
            results_dict["magnitude_by_type"][atom_type].append(magbytype[atom_type])

            results_dict["norm_outmolec_bytype"][atom_type].append(norm_outmolec_by_type[atom_type])
            results_dict["norm_inmolec_bytype"][atom_type].append(norm_inmolec_by_type[atom_type])

        for axis in axisproj:
            if axis not in results_dict["axis_projection"]:
               results_dict["axis_projection"][axis]=[]
               results_dict["unnormaxis_projection"][axis]=[]
            results_dict["axis_projection"][axis].append(axisproj[axis])
            results_dict["unnormaxis_projection"][axis].append(unnormaxisproj[axis])


    # Create a single figure with 4 subplots for all atoms (2x2 layout)
 #   fig, axs = plt.subplots(2, 2, figsize=(10, 8))
  #  metrics = ["threshold_sum_all_atoms", "distance_scaled_sum_all_atoms", "radial_projection", "orthogonal_projection"]
  #  titles = ["Normalized Threshold Sum (All Atoms)", "Normalized Distance-Scaled Sum (All Atoms)", "Radial Projection", "Orthogonal Projection"]

   # for ax, metric, title in zip(axs.flatten(), metrics, titles):
 #       ax.plot(results_dict["pressures"], results_dict[metric], marker="o", linestyle="-")
#        ax.set_xlabel("Pressure")
#        ax.set_ylabel(title)
 #       ax.set_title(title + " vs Pressure")
  #      ax.grid(True)

   # plt.tight_layout()
   # plt.savefig(f"{output_prefix}_combined.png")
   # plt.close()
#
    # Create separate multipanel figures for each atomic type
 #   for atom_type in results_dict["threshold_sums_by_type"]:
  #      fig, axs = plt.subplots(1, 2, figsize=(10, 4))
   #     metrics = ["threshold_sums_by_type", "distance_scaled_sums_by_type"]
    #    titles = [f"Threshold Sum ({atom_type})", f"Distance-Scaled Sum ({atom_type})"]

     #   for ax, metric, title in zip(axs.flatten(), metrics, titles):
      #      ax.plot(results_dict["pressures"], results_dict[metric][atom_type], marker="o", linestyle="-")
       #     ax.set_xlabel("Pressure")
       #     ax.set_ylabel(title)
       #     ax.set_title(title + " vs Pressure")
       #     ax.grid(True)

        #plt.tight_layout()
        #plt.savefig(f"{output_prefix}_{atom_type}.png")
        #plt.close()

    return results_dict





def process_cp2k_modes_over_pressures(
    pressures,
    position_xyz,
    unit_cell_files,
    trajectory_list,
    desired_modes,
    bond_cutoff,
    distance_threshold,
    output_prefix="analysis",
    displacement_prefix="output_groups",
    outdir=None,
):
    """
    Runs CP2K mode processing over a range of pressures, extracting atomic positions and vibrational modes,
    and then performs analysis for each mode using the pressure data.

    Parameters:
      pressures: List of pressure values.
      unit_cell_files: List of corresponding unit cell filenames.
      mol_files: List of corresponding CP2K .mol filenames.
      desired_modes: List of integers (or strings) representing the vibration mode indices to extract.
      bond_cutoff: Cutoff distance for determining molecular connectivity.
      distance_threshold: Cutoff distance for filtering vibrational interactions.
      output_prefix: Prefix for saving output files.

    Returns:
      - A dictionary containing processed results for each mode over pressure.
    """
    if not (len(pressures) == len(unit_cell_files) == len(trajectory_list)):
        raise ValueError("Mismatch in list lengths: Each pressure must have corresponding filenames.")

    # Convert mode indices to integers
    desired_modes = [int(mode) for mode in desired_modes]

    # Storage for processed file paths
    all_positions_files = []
    all_mode_files = {mode: [] for mode in desired_modes}
    print(trajectory_list)
    # Step 1: Extract atomic positions and vibrational modes for each pressure
    if outdir is None:
        outdir = BASEDIR

    for idx, pressure in enumerate(pressures):
        print(f"Processing pressure {pressure} with files: {position_xyz[idx]}, {unit_cell_files[idx]}, {trajectory_list[idx]}")
        positions_file, mode_files = parse_trajectories_for_modes(
            pressure,
            position_xyz[idx],
            trajectory_list[idx],
            desired_modes,
            output_prefix=displacement_prefix,
            outdir=outdir,
        )
        
        all_positions_files.append(positions_file)
        for modeint in range(len(desired_modes)):
            all_mode_files[desired_modes[modeint]].append(mode_files[modeint])

    # Step 2: Run analysis for each mode over pressure
    results_dict = {}
    for mode in desired_modes:
        print(f"Running analysis for mode {mode} over pressures...")
        results_dict[mode] = run_analysis_over_pressures(
            unit_cell_files, all_positions_files, all_mode_files[mode],
            bond_cutoff, distance_threshold, pressures, output_prefix=f"{output_prefix}_mode_{mode}"
        )

    return results_dict

def plot_combined_mode_analysis(results_dict, output_filename="combined_modes_plot.png"):
    """
    Creates a 4-panel figure with each subplot showing results across all normal modes.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      output_filename: Name of the file to save the final figure.

    Outputs:
      - A single 4-panel plot where each subplot contains lines for all normal modes, with legends.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ["threshold_sum_all_atoms", "distance_scaled_sum_all_atoms", "radial_projection", "orthogonal_projection"]
    titles = ["Normalized Threshold Sum (All Atoms)", "Normalized Distance-Scaled Sum (All Atoms)", "Radial Projection", "Orthogonal Projection"]

    # Loop through all normal modes and plot them together
    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]

        for ax, metric, title in zip(axs.flatten(), metrics, titles):
            ax.plot(pressures, mode_results[metric], marker="o", linestyle="-", label=f"Mode {mode}")

            ax.set_xlabel("Pressure")
            ax.set_ylabel(title)
            ax.set_title(title + " vs Pressure")
            ax.grid(True)
            ax.legend()

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def plot_combined_mode_analysis(results_dict, output_filename="combined_modes_plot.png"):

    fig, axs = plt.subplots(2, 2, figsize=(8, 3))  # Adjusted aspect ratio for horizontal plots

    # Explicit reordering of subplot locations
    metrics = ["radial_projection", "threshold_sum_all_atoms", "distance_scaled_sum_all_atoms", "orthogonal_projection"]
    titles = ["Radial Projection", "Normalized Threshold Sum (All Atoms)", "Normalized Distance-Weighted Atom-Atom Projection", "Orthogonal Projection"]
    xlabels = ['Normalized Projection', 'Normalized Projection', 'Normalized Weighted Projection', 'Normalized Projection']
    
    legend_handles = []  # Collect legend handles to display only once
    legend_labels = []  # Collect unique labels

    for mode, mode_results in results_dict.items():
        mode_handle = None
        for ax, metric, title, xlabel in zip(axs.flatten(), metrics, titles, xlabels):
            line, = ax.plot(mode_results[metric], range(len(mode_results[metric])), marker="o", linestyle="-", label=f"Mode {mode}")

            ax.set_xlabel(xlabel)  # Now labeling the x-axis
            ax.set_ylabel("")  # Removes y-axis label since values are flipped
            ax.set_title(title)
            ax.grid(True)
            ax.set_yticks([])  # Removes tick marks on Y-axis
            if mode_handle is None:
                mode_handle = line

        if f"Mode {mode}" not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(f"Mode {mode}")

    # Place a single legend outside the figure
    fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.1, 0.5))

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved combined normal mode plot as '{output_filename}'.")
    
    fig, ax = plt.subplots(figsize=(6, 6))  # Square aspect ratio for better scaling

    legend_handles = []  # Collect legend handles to display only once
    legend_labels = []  # Collect unique labels

    for mode, mode_results in results_dict.items():
        line, = ax.plot(mode_results["radial_projection"], mode_results["distance_scaled_sum_all_atoms"],
                        marker="o", linestyle="none", label=f"Mode {mode}")
        
        if f"Mode {mode}" not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(f"Mode {mode}")

    ax.set_xlabel("Radial Projection")
    ax.set_ylabel("Distance-Weighted Atom-Atom Projection")
    ax.set_title("Radial vs. Distance-Weighted Projection")
    ax.grid(True)

    # Place a single legend outside the figure
    ax.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.1, 0.5))

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig("radvsdist.png", bbox_inches="tight")
    plt.close()
    print(f"Saved radial vs. distance-weighted plot as '{output_filename}'.")


def plot_combined_mode_analysis(results_dict, output_filename="combined_modes_plot.png"):
    """
    Creates a 4-panel figure with each subplot showing results across all normal modes.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      output_filename: Name of the file to save the final figure.

    Outputs:
      - A single 4-panel plot where each subplot contains lines for all normal modes, with legends.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ["threshold_sum_all_atoms", "distance_scaled_sum_all_atoms", "radial_projection", "orthogonal_projection"]
    titles = ["Normalized Threshold Sum (All Atoms)", "Normalized Distance-Scaled Sum (All Atoms)", "Radial Projection", "Orthogonal Projection"]

    # Loop through all normal modes and plot them together
    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]

        for ax, metric, title in zip(axs.flatten(), metrics, titles):
            ax.plot(pressures, mode_results[metric], marker="o", linestyle="-", label=f"Mode {mode}")

            ax.set_xlabel("Pressure")
            ax.set_ylabel(title)
            ax.set_title(title + " vs Pressure")
            ax.grid(True)
            ax.legend()

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved combined normal mode plot as '{output_filename}'.")



def plot_atomwise_mode_analysis(results_dict, output_prefix="atomwise_analysis"):
    """
    Creates stacked 4-panel figures for each atomic type, showing vibrational trends across normal modes.

    Parameters:
      results_dict: Dictionary containing processed results for each mode over pressure.
      output_prefix: Prefix for saving output files.

    Outputs:
      - Saves a stacked 4-panel figure for each atomic type, with normal modes plotted together.
    """
    # Identify atomic types from results dictionary
    atomic_types = set()
    for mode_results in results_dict.values():
        atomic_types.update(mode_results["threshold_sums_by_type"].keys())  # Fixed key reference

    # Metrics to plot
    metrics = ["threshold_sums_by_type", "distance_scaled_sums_by_type"]
    titles = ["Threshold Sum", "Distance-Scaled Sum"]

    for atom_type in atomic_types:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Stacked format

        for mode, mode_results in results_dict.items():
            pressures = mode_results["pressures"]

            for ax, metric, title in zip(axs.flatten(), metrics, titles):
                if atom_type in mode_results[metric]:  # Only plot if atom type exists for this mode
                    ax.plot(pressures, mode_results[metric][atom_type], marker="o", linestyle="-", label=f"Mode {mode}")

                ax.set_xlabel("Pressure")
                ax.set_ylabel(title)
                ax.set_title(f"{title} for {atom_type} vs Pressure")
                ax.grid(True)
                ax.legend()

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{atom_type}.png")
        plt.close()
        print(f"Saved atomwise plot for {atom_type} as '{output_prefix}_{atom_type}.png'.")


def run_full_analysis(pressurein,pospath, unit_cell_filepath, mode_mol_filepaths, desired_modes, bond_cutoff, distance_threshold, output_prefix="final_analysis"):
    """
    Runs the full CP2K vibrational mode analysis pipeline, processing extracted modes
    and generating combined multi-mode plots over pressures.

    Parameters:
      pressurein: List of pressure values.
      unit_cell_filepath: List of unit cell filenames, one per pressure.
      mode_mol_filepaths: List of CP2K .mol filenames, one per pressure.
      desired_modes: List of integers (or strings) representing the vibration mode indices to extract.
      bond_cutoff: Cutoff distance for determining molecular connectivity.
      distance_threshold: Cutoff distance for filtering vibrational interactions.
      output_prefix: Prefix for saving output files.

    Outputs:
      - Saves individual normal mode analysis plots.
      - Creates a final combined 4-panel figure with multiple normal mode trends plotted together.
    """
    print("Starting full analysis...")

    # Step 1: Process vibrational modes over pressures
    results_dict = process_cp2k_modes_over_pressures(
        pressurein,
        pospath,
        unit_cell_filepath,
        mode_mol_filepaths,
        desired_modes,
        bond_cutoff,
        distance_threshold,
        output_prefix=output_prefix,
        displacement_prefix=f"output_{output_prefix}",
        outdir=BASEDIR,
    )
    results_filepath = os.path.join(BASEDIR, f"{output_prefix}_results.json")
    with open(results_filepath, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Step 2: Generate a combined visualization for all modes
#    plot_combined_mode_analysis(results_dict, output_filename=f"{output_prefix}_combined_modes.png")
 #   plot_atomwise_mode_analysis(results_dict, output_prefix=f"{output_prefix}_combined_modes_atomwise_analysis")


    print("Full analysis complete. Saved processed results.")
    return results_dict, results_filepath


def _apply_pub_axes_style(ax, x_ticks=(0, 4, 10)):
    # ticks on all 4 sides, facing in
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.2,
        length=5.0,
    )
    # slightly thicker box
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    # pressure ticks only at 0/4/10
    ax.set_xticks(list(x_ticks))


def _apply_pub_axes_style_categorical(ax):
    # ticks on all 4 sides, facing in (no forced numeric xticks)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.2,
        length=5.0,
    )
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)


def _value_at_pressure(mode_results: dict, key: str, pressure_value=0):
    pressures = mode_results.get("pressures", [])
    if pressures and pressure_value in pressures:
        idx = pressures.index(pressure_value)
    else:
        idx = 0
    vals = mode_results[key]
    return vals[idx]


def _plot_minamb_two_category_single_metric(
    results_dict,
    active_modes,
    metric_key,
    ylabel,
    title,
    output_filename,
    *,
    pressure_value=0,
):
    """
    Minamb-only: plot only the selected pressure point, separated into Active/Inactive columns.
    Active modes are drawn with dashed style; inactive with solid style.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    active_set = set(map(str, active_modes))
    x_inactive, x_active = 0.0, 1.0
    ms = 9.0

    for mode, mode_results in results_dict.items():
        m = str(mode)
        if m not in used_colors:
            used_colors[m] = next(color_cycle)

        y = _value_at_pressure(mode_results, metric_key, pressure_value=pressure_value)
        is_active = m in active_set
        x = x_active if is_active else x_inactive
        ax.plot([x], [y], marker="o", markersize=ms, linestyle="None", color=used_colors[m])

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"], fontsize=12)
    ax.set_xlabel(r"$\mathcal{O}(100\,\mathrm{ps})$ Dynamics", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="y", labelsize=11)
    _apply_pub_axes_style_categorical(ax)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def _plot_minamb_two_category_axis_proj(
    results_dict,
    active_modes,
    output_filename,
    *,
    pressure_value=0,
    suptitle="Axis Projections (0 GPa)",
):
    axes = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 1, figsize=(6.8, 8.2), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    active_set = set(map(str, active_modes))
    x_inactive, x_active = 0.0, 1.0
    ms = 8.5

    for row, axis in enumerate(axes):
        ax = axs[row]
        for mode, data in results_dict.items():
            m = str(mode)
            if m not in used_colors:
                used_colors[m] = next(color_cycle)
            if axis not in data.get("axis_projection", {}):
                continue

            pressures = data.get("pressures", [])
            idx = pressures.index(pressure_value) if pressures and pressure_value in pressures else 0
            y = data["axis_projection"][axis][idx]

            is_active = m in active_set
            x = x_active if is_active else x_inactive
            ax.plot([x], [y], marker="o", markersize=ms, linestyle="None", color=used_colors[m])

        ax.set_ylabel({"x": "[1 0 0]", "y": "[0 1 0]", "z": "[0 0 1]"}[axis], fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        _apply_pub_axes_style_categorical(ax)

    axs[-1].set_xlim(-0.5, 1.5)
    axs[-1].set_xticks([0, 1])
    axs[-1].set_xticklabels(["Inactive", "Active"], fontsize=12)
    axs[-1].set_xlabel(r"$\mathcal{O}(100\,\mathrm{ps})$ Dynamics", fontsize=13)
    fig.suptitle(suptitle, fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def _plot_minamb_two_category_group_rows(
    results_dict,
    active_modes,
    rows,
    suptitle,
    output_filename,
    *,
    pressure_value=0,
):
    """
    Minamb-only group figure: 3 rows x 1 column, only the 'all atoms' fraction at 0 GPa,
    displayed in Active/Inactive columns on the x-axis.
    """
    nrows = len(rows)
    fig, axs = plt.subplots(nrows, 1, figsize=(6.8, 2.55 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}
    active_set = set(map(str, active_modes))
    x_inactive, x_active = 0.0, 1.0
    ms = 8.0

    for r, (label, key_total, _key_on_unused) in enumerate(rows):
        ax = axs[r]
        for mode, data in results_dict.items():
            m = str(mode)
            if m not in used_colors:
                used_colors[m] = next(color_cycle)

            y = _value_at_pressure(data, key_total, pressure_value=pressure_value)
            is_active = m in active_set
            x = x_active if is_active else x_inactive
            ax.plot([x], [y], marker="o", markersize=ms, linestyle="None", color=used_colors[m])

        ax.set_ylabel(label, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        _apply_pub_axes_style_categorical(ax)

    axs[-1].set_xlim(-0.5, 1.5)
    axs[-1].set_xticks([0, 1])
    axs[-1].set_xticklabels(["Inactive", "Active"], fontsize=12)
    axs[-1].set_xlabel(r"$\mathcal{O}(100\,\mathrm{ps})$ Dynamics", fontsize=13)

    fig.suptitle(suptitle, fontsize=14, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def _plot_minamb_two_category_localization_totalmag_pair(
    results_dict,
    active_modes,
    output_filename,
    *,
    suptitle,
    pressure_value=0,
):
    """
    Minamb-only pair plot at 0 GPa:
      (a) normalized displacement magnitude, (b) localization,
    both split into Active/Inactive columns on x-axis.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 5.1), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}
    active_set = set(map(str, active_modes))
    x_inactive, x_active = 0.0, 1.0
    ms = 8.2

    # normalize displacement magnitude by max at selected pressure across modes
    mags = []
    for _m, data in results_dict.items():
        mags.append(_value_at_pressure(data, "total_magnitude", pressure_value=pressure_value))
    max_mag = max(mags) if mags else 1.0

    for mode, data in results_dict.items():
        m = str(mode)
        if m not in used_colors:
            used_colors[m] = next(color_cycle)

        is_active = m in active_set
        x = x_active if is_active else x_inactive
        y_mag = _value_at_pressure(data, "total_magnitude", pressure_value=pressure_value) / max_mag
        y_loc = _value_at_pressure(data, "spread", pressure_value=pressure_value)

        axs[0].plot([x], [y_mag], marker="o", markersize=ms, linestyle="None", color=used_colors[m])
        axs[1].plot([x], [y_loc], marker="o", markersize=ms, linestyle="None", color=used_colors[m])

    for ax in axs:
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Inactive", "Active"], fontsize=12)
        ax.set_xlabel(r"$\mathcal{O}(100\,\mathrm{ps})$ Dynamics", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        _apply_pub_axes_style_categorical(ax)

    axs[0].set_ylabel("Normalized displacement magnitude", fontsize=13)
    axs[1].set_ylabel("Normalized RMS displacement", fontsize=13)
    axs[0].set_title("Displacement Magnitude", fontsize=13, pad=8)
    axs[1].set_title("Localization", fontsize=13, pad=8)

    axs[0].annotate("(a)", xy=(0, 1), xycoords="axes fraction", xytext=(-10, 14), textcoords="offset points",
                    ha="left", va="bottom", fontsize=13, fontweight="bold", clip_on=False)
    axs[1].annotate("(b)", xy=(0, 1), xycoords="axes fraction", xytext=(-10, 14), textcoords="offset points",
                    ha="left", va="bottom", fontsize=13, fontweight="bold", clip_on=False)

    fig.suptitle(suptitle, fontsize=14, y=0.992)
    plt.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.985])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def _plot_minamb_two_category_NNdist_min_SI(
    results_dict,
    active_modes,
    output_filename,
    title,
    *,
    pressure_value=0,
):
    """
    Minamb-only NN-distance SI figure at 0 GPa:
      rows: O, N
      cols: Intermolecular, Intramolecular
    Values are split into Active/Inactive x-axis categories.
    """
    atom_types = ["O", "N"]
    fig, axs = plt.subplots(len(atom_types), 2, figsize=(8.2, 5.8), sharex=True)

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}
    active_set = set(map(str, active_modes))
    x_inactive, x_active = 0.0, 1.0
    ms = 7.0

    for mode, data in results_dict.items():
        m = str(mode)
        if m not in used_colors:
            used_colors[m] = next(color_cycle)

        pressures = data.get("pressures", [])
        idx = pressures.index(pressure_value) if pressures and pressure_value in pressures else 0
        is_active = m in active_set
        x = x_active if is_active else x_inactive

        for r, at in enumerate(atom_types):
            if at in data.get("norm_outmolec_bytype", {}):
                y = data["norm_outmolec_bytype"][at][idx]
                axs[r, 0].plot([x], [y], marker="o", markersize=ms, linestyle="None", color=used_colors[m])
            if at in data.get("norm_inmolec_bytype", {}):
                y = data["norm_inmolec_bytype"][at][idx]
                axs[r, 1].plot([x], [y], marker="o", markersize=ms, linestyle="None", color=used_colors[m])

    axs[0, 0].set_title("Intermolecular", fontsize=12)
    axs[0, 1].set_title("Intramolecular", fontsize=12)

    for r, at in enumerate(atom_types):
        axs[r, 0].set_ylabel(f"{at}–X (Å)", fontsize=12)
        for c in [0, 1]:
            axs[r, c].set_xlim(-0.5, 1.5)
            axs[r, c].set_xticks([0, 1])
            axs[r, c].set_xticklabels(["Inactive", "Active"], fontsize=11)
            axs[r, c].tick_params(axis="y", labelsize=11)
            _apply_pub_axes_style_categorical(axs[r, c])

    axs[-1, 0].set_xlabel(r"$\mathcal{O}(100\,\mathrm{ps})$ Dynamics", fontsize=13)
    axs[-1, 1].set_xlabel(r"$\mathcal{O}(100\,\mathrm{ps})$ Dynamics", fontsize=13)
    fig.suptitle(title, fontsize=14, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

def _extract_last_list_assignment(py_text: str, varname: str) -> list[int]:
    """
    Extract the last *uncommented* single-line assignment like:
      desired_modesin=[1,2,3]
      desired_modesin = [1, 2, 3]
    Returns list of ints.
    """
    last_inner = None
    for line in py_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Require assignment at beginning of line (after whitespace)
        if not stripped.startswith(varname):
            continue
        # Avoid cases like "#desired_modesin=..." which are already skipped above
        if "=" not in stripped:
            continue
        rhs = stripped.split("=", 1)[1].strip()
        if "[" not in rhs or "]" not in rhs:
            continue
        inner = rhs[rhs.find("[") + 1 : rhs.rfind("]")]
        last_inner = inner

    if last_inner is None:
        raise ValueError(f"Could not find assignment to {varname!r}")

    vals: list[int] = []
    for tok in last_inner.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    return vals


def _extract_first_str_list_assignment(py_text: str, varname: str) -> list[str]:
    """
    Extract the first *uncommented* single-line assignment like:
      minamb=['172','179',...]
    Returns list of strings (without quotes).
    """
    for line in py_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith(varname):
            continue
        if "=" not in stripped:
            continue
        rhs = stripped.split("=", 1)[1].strip()
        if "[" not in rhs or "]" not in rhs:
            continue
        inner = rhs[rhs.find("[") + 1 : rhs.rfind("]")]
        return [s for s in re.findall(r"['\\\"]([^'\\\"]+)['\\\"]", inner)]

    raise ValueError(f"Could not find assignment to {varname!r}")


def _plot_single_metric(results_dict, highlight_modes, metric_key, ylabel, title, output_filename):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)
        is_hi = str(mode) in highlight_modes
        linestyle = "--" if is_hi else "-"
        lw = 2.6 if is_hi else 1.8

        ax.plot(
            pressures,
            mode_results[metric_key],
            marker="o",
            markersize=5.5,
            linewidth=lw,
            linestyle=linestyle,
            color=used_colors[mode],
        )

    ax.set_xlabel("Pressure (GPa)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    _apply_pub_axes_style(ax)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_normalized_axis_proj(results_dict, highlight_modes, output_filename):
    axes = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    for row, axis in enumerate(axes):
        ax = axs[row]
        for mode, data in results_dict.items():
            if mode not in used_colors:
                used_colors[mode] = next(color_cycle)
            is_hi = str(mode) in highlight_modes
            linestyle = "--" if is_hi else "-"
            lw = 2.6 if is_hi else 1.8
            if axis in data.get("axis_projection", {}):
                ax.plot(
                    data["pressures"],
                    data["axis_projection"][axis],
                    marker="o",
                    markersize=5.0,
                    linewidth=lw,
                    linestyle=linestyle,
                    color=used_colors[mode],
                )
        ax.set_ylabel({"x": "[1 0 0]", "y": "[0 1 0]", "z": "[0 0 1]"}[axis], fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        _apply_pub_axes_style(ax)

    axs[-1].set_xlabel("Pressure (GPa)", fontsize=13)
    fig.suptitle("Axis Projections", fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_group_rows_all_vs_on(results_dict, highlight_modes, rows, suptitle, output_filename):
    nrows = len(rows)
    fig, axs = plt.subplots(nrows, 2, figsize=(8.0, 2.8 * nrows), sharex=True)
    if nrows == 1:
        axs = np.array([axs])  # normalize shape to (1,2)

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)
        is_hi = str(mode) in highlight_modes
        linestyle = "--" if is_hi else "-"
        lw = 2.6 if is_hi else 1.8

        for r, (label, key_total, key_on) in enumerate(rows):
            axs[r, 0].plot(
                pressures,
                mode_results[key_total],
                marker="o",
                markersize=4.8,
                linewidth=lw,
                linestyle=linestyle,
                color=used_colors[mode],
            )
            axs[r, 1].plot(
                pressures,
                mode_results[key_on],
                marker="o",
                markersize=4.8,
                linewidth=lw,
                linestyle=linestyle,
                color=used_colors[mode],
            )

    for r, (label, _key_total, _key_on) in enumerate(rows):
        axs[r, 0].set_ylabel(label, fontsize=12)
        axs[r, 0].tick_params(axis="both", labelsize=11)
        axs[r, 1].tick_params(axis="both", labelsize=11)
        _apply_pub_axes_style(axs[r, 0])
        _apply_pub_axes_style(axs[r, 1])

    axs[0, 0].set_title("All atoms", fontsize=12)
    axs[0, 1].set_title("O & N only", fontsize=12)
    axs[-1, 0].set_xlabel("Pressure (GPa)", fontsize=13)
    axs[-1, 1].set_xlabel("Pressure (GPa)", fontsize=13)

    # reduce white space between suptitle and panels
    fig.suptitle(suptitle, fontsize=14, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_group_rows_single(results_dict, highlight_modes, rows, suptitle, output_filename):
    """
    Single-column group figure (all atoms only).
    Rows correspond to group labels; x-axis is pressure.
    """
    nrows = len(rows)
    fig, axs = plt.subplots(nrows, 1, figsize=(6.8, 2.8 * nrows), sharex=True)
    if nrows == 1:
        axs = np.array([axs])  # normalize to 1D array

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    for mode, mode_results in results_dict.items():
        pressures = mode_results["pressures"]
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)
        is_hi = str(mode) in highlight_modes
        linestyle = "--" if is_hi else "-"
        lw = 2.6 if is_hi else 1.8

        for r, (label, key_total, _key_on_unused) in enumerate(rows):
            axs[r].plot(
                pressures,
                mode_results[key_total],
                marker="o",
                markersize=4.8,
                linewidth=lw,
                linestyle=linestyle,
                color=used_colors[mode],
            )

    for r, (label, _key_total, _key_on_unused) in enumerate(rows):
        axs[r].set_ylabel(label, fontsize=12)
        axs[r].tick_params(axis="both", labelsize=11)
        _apply_pub_axes_style(axs[r])

    axs[-1].set_xlabel("Pressure (GPa)", fontsize=13)

    # reduce white space between suptitle and panels
    fig.suptitle(suptitle, fontsize=14, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_NNdist_min_SI(results_dict, highlight_modes, output_filename, title):
    """
    Publication-style 2x2 panel:
      rows: O, N
      cols: Intermolecular, Intramolecular

    Uses keys:
      - norm_outmolec_bytype
      - norm_inmolec_bytype
    """
    atom_types = ["O", "N"]
    fig, axs = plt.subplots(len(atom_types), 2, figsize=(8.2, 5.8), sharex=True)

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    for mode, data in results_dict.items():
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)
        is_hi = str(mode) in highlight_modes
        linestyle = "--" if is_hi else "-"
        lw = 2.6 if is_hi else 1.8

        pressures = data["pressures"]
        for r, at in enumerate(atom_types):
            if at in data.get("norm_outmolec_bytype", {}):
                axs[r, 0].plot(
                    pressures,
                    data["norm_outmolec_bytype"][at],
                    marker="o",
                    markersize=4.8,
                    linewidth=lw,
                    linestyle=linestyle,
                    color=used_colors[mode],
                )
            if at in data.get("norm_inmolec_bytype", {}):
                axs[r, 1].plot(
                    pressures,
                    data["norm_inmolec_bytype"][at],
                    marker="o",
                    markersize=4.8,
                    linewidth=lw,
                    linestyle=linestyle,
                    color=used_colors[mode],
                )

    axs[0, 0].set_title("Intermolecular", fontsize=12)
    axs[0, 1].set_title("Intramolecular", fontsize=12)

    for r, at in enumerate(atom_types):
        axs[r, 0].set_ylabel(f"{at}–X (Å)", fontsize=12)
        for c in [0, 1]:
            axs[r, c].tick_params(axis="both", labelsize=11)
            _apply_pub_axes_style(axs[r, c])

    axs[-1, 0].set_xlabel("Pressure (GPa)", fontsize=13)
    axs[-1, 1].set_xlabel("Pressure (GPa)", fontsize=13)
    fig.suptitle(title, fontsize=14, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_localization_totalmag_pair(
    results_dict,
    highlight_modes,
    output_filename,
    *,
    left_ylabel="Normalized RMS displacement",
    right_ylabel="Total mass-weighted displacement",
    suptitle=None,
    left_title="Localization",
    right_title="Displacement Magnitude",
):
    """
    Two-panel (left/right) figure for publication:
      (a) localization/spread
      (b) total magnitude
    No overall title (intended for LaTeX captioning).
    """
    # Slightly taller aspect ratio for publication layouts (~20% taller than earlier version)
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 5.1), sharex=True)

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    used_colors = {}

    # Ensure consistent colors across both panels.
    for mode in results_dict.keys():
        if mode not in used_colors:
            used_colors[mode] = next(color_cycle)

    # Normalize displacement magnitude by max value shown in this plot.
    max_mag = max(
        max(v for v in data["total_magnitude"])
        for data in results_dict.values()
        if "total_magnitude" in data and data["total_magnitude"]
    )
    if max_mag == 0:
        max_mag = 1.0

    for mode, data in results_dict.items():
        pressures = data["pressures"]
        is_hi = str(mode) in highlight_modes
        linestyle = "--" if is_hi else "-"
        lw = 2.6 if is_hi else 1.8
        ms = 5.0

        # (a) Displacement magnitude (normalized)
        axs[0].plot(
            pressures,
            [v / max_mag for v in data["total_magnitude"]],
            marker="o",
            markersize=ms,
            linewidth=lw,
            linestyle=linestyle,
            color=used_colors[mode],
        )
        # (b) Localization (spread)
        axs[1].plot(
            pressures,
            data["spread"],
            marker="o",
            markersize=ms,
            linewidth=lw,
            linestyle=linestyle,
            color=used_colors[mode],
        )

    # Update labels/titles to match new ordering
    axs[0].set_ylabel("Normalized displacement magnitude", fontsize=13)
    axs[1].set_ylabel("Normalized RMS displacement", fontsize=13)
    axs[0].set_title("Displacement Magnitude", fontsize=13, pad=8)
    axs[1].set_title("Localization", fontsize=13, pad=8)
    for ax in axs:
        ax.set_xlabel("Pressure (GPa)", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        _apply_pub_axes_style(ax)

    # Panel labels just outside axes (upper-left)
    axs[0].annotate(
        "(a)",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-10, 14),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        clip_on=False,
    )
    axs[1].annotate(
        "(b)",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-10, 14),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        clip_on=False,
    )

    if suptitle:
        # Reduce whitespace between title and axes
        fig.suptitle(suptitle, fontsize=14, y=0.992)
        plt.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.985])
    else:
        plt.tight_layout(w_pad=2.0)
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def make_plots(results_dict, output_prefix="modesout_groups"):
    # --- minpress selection (keep stable, do NOT couple to minamb) ---
    modes_minpress = ["166", "179", "188", "200", "204", "225"]
    highlight_minpress = ["166", "188", "204"]  # as in minpress plottingmodes.py

    # --- minamb selection (read from minamb scripts so it's easy to add modes later, e.g. 239) ---
    minamb_dir = os.path.abspath(os.path.join(BASEDIR, "..", "minamb"))
    minamb_analysis_script = os.path.join(minamb_dir, "legacy_traj_r2scale_massweight.py")
    minamb_plot_script = os.path.join(minamb_dir, "plottingmodes.py")

    try:
        with open(minamb_analysis_script, "r") as f:
            minamb_text = f.read()
        modes_minamb = [str(x) for x in _extract_last_list_assignment(minamb_text, "desired_modesin")]
    except Exception as e:
        raise RuntimeError(f"Failed to read minamb desired modes from {minamb_analysis_script!r}: {e}")

    try:
        with open(minamb_plot_script, "r") as f:
            minamb_plot_text = f.read()
        highlight_minamb = _extract_first_str_list_assignment(minamb_plot_text, "minamb")
    except Exception as e:
        raise RuntimeError(f"Failed to read minamb highlight modes from {minamb_plot_script!r}: {e}")

    results_minpress = {k: v for k, v in results_dict.items() if str(k) in modes_minpress}

    # Minamb "Active" modes (showing O(100 ps) dynamics) for the new 0 GPa Active/Inactive layout.
    active_minamb = ["202", "217", "239"]

    # Ensure active modes are included in the minamb plot set even if the minamb script list wasn't updated yet.
    existing_modes = {str(k) for k in results_dict.keys()}
    for m in active_minamb:
        if m in existing_modes and m not in modes_minamb:
            modes_minamb.append(m)

    results_minamb = {k: v for k, v in results_dict.items() if str(k) in modes_minamb}

    missing_active = [m for m in active_minamb if m not in {str(k) for k in results_minamb.keys()}]
    if missing_active:
        print("⚠️ Active minamb modes missing from results_dict (won't plot):", missing_active)

    out_spread_minpress = os.path.join(BASEDIR, f"{output_prefix}_spread_minpress.png")
    out_spread_minamb = os.path.join(BASEDIR, f"{output_prefix}_spread_minamb.png")
    _plot_single_metric(
        results_minpress,
        highlight_minpress,
        "spread",
        "Normalized RMS displacement",
        "Mode Localization - Pressure Sensitivity",
        out_spread_minpress,
    )
    _plot_minamb_two_category_single_metric(
        results_minamb,
        active_minamb,
        "spread",
        "Normalized RMS displacement",
        "Mode Localization - Ambient Character",
        out_spread_minamb,
    )

    out_mag_minpress = os.path.join(BASEDIR, f"{output_prefix}_total_magnitude_minpress.png")
    out_mag_minamb = os.path.join(BASEDIR, f"{output_prefix}_total_magnitude_minamb.png")
    _plot_single_metric(
        results_minpress,
        highlight_minpress,
        "total_magnitude",
        "Total mass-weighted displacement",
        "Total Mass-Weighted Displacements - Pressure Sensitivity",
        out_mag_minpress,
    )
    _plot_minamb_two_category_single_metric(
        results_minamb,
        active_minamb,
        "total_magnitude",
        "Total mass-weighted displacement",
        "Total Mass-Weighted Displacements - Ambient Character",
        out_mag_minamb,
    )

    # New paired plot (keep individual plots intact)
    out_pair_minpress = os.path.join(BASEDIR, f"{output_prefix}_localization_totalmag_minpress_pair.png")
    out_pair_minamb = os.path.join(BASEDIR, f"{output_prefix}_localization_totalmag_minamb_pair.png")
    plot_localization_totalmag_pair(
        results_minpress,
        highlight_minpress,
        out_pair_minpress,
        suptitle="Mode Character and Pressure Sensitivity",
    )
    _plot_minamb_two_category_localization_totalmag_pair(
        results_minamb,
        active_minamb,
        out_pair_minamb,
        suptitle="Mode Character and Ambient Dynamics",
    )

    out_axis_minpress = os.path.join(BASEDIR, f"{output_prefix}_axis_proj_minpress.png")
    out_axis_minamb = os.path.join(BASEDIR, f"{output_prefix}_axis_proj_minamb.png")
    plot_normalized_axis_proj(results_minpress, highlight_minpress, out_axis_minpress)
    _plot_minamb_two_category_axis_proj(
        results_minamb,
        active_minamb,
        out_axis_minamb,
        suptitle="Axis Projections (0 GPa)",
    )

    out_dist_minpress = os.path.join(BASEDIR, f"{output_prefix}_distance_scaled_minpress.png")
    out_dist_minamb = os.path.join(BASEDIR, f"{output_prefix}_distance_scaled_minamb.png")
    _plot_single_metric(
        results_minpress,
        highlight_minpress,
        "normalized_distance_scaled_sum_all_atoms",
        "Normalized metric",
        "Interatomic Metric - Pressure Sensitivity",
        out_dist_minpress,
    )
    _plot_minamb_two_category_single_metric(
        results_minamb,
        active_minamb,
        "normalized_distance_scaled_sum_all_atoms",
        "Normalized metric",
        "Interatomic Metric - Ambient Character",
        out_dist_minamb,
    )

    # New group-analysis layouts
    rows_1 = [
        ("Furazan Groups", "furazan_fraction_total", "furazan_fraction_ON"),
        ("Nitro Groups", "nitro_fraction_total", "nitro_fraction_ON"),
        ("Furoxan Groups", "furoxano_fraction_total", "furoxano_fraction_ON"),
    ]
    out_groups1_minpress = os.path.join(BASEDIR, f"{output_prefix}_group_rows_furazan_nitro_furoxano_minpress.png")
    out_groups1_minamb = os.path.join(BASEDIR, f"{output_prefix}_group_rows_furazan_nitro_furoxano_minamb.png")
    plot_group_rows_single(
        results_minpress,
        highlight_minpress,
        rows_1,
        "Chemical Group Character - Pressure Sensitivity",
        out_groups1_minpress,
    )
    _plot_minamb_two_category_group_rows(
        results_minamb,
        active_minamb,
        rows_1,
        "Chemical Group Character - Ambient Dynamics",
        out_groups1_minamb,
    )

    rows_2 = [
        ("Rings", "rings_fraction_total", "rings_fraction_ON"),
        ("Nitro Groups", "nitro_fraction_total", "nitro_fraction_ON"),
        ("Furoxan O", "furoxano_oxo_fraction_total", "furoxano_oxo_fraction_ON"),
    ]
    out_groups2_minpress = os.path.join(BASEDIR, f"{output_prefix}_group_rows_rings_nitro_furoxanoO_minpress.png")
    out_groups2_minamb = os.path.join(BASEDIR, f"{output_prefix}_group_rows_rings_nitro_furoxanoO_minamb.png")
    plot_group_rows_single(
        results_minpress,
        highlight_minpress,
        rows_2,
        "Chemical Group Character - Pressure Sensitivity",
        out_groups2_minpress,
    )
    _plot_minamb_two_category_group_rows(
        results_minamb,
        active_minamb,
        rows_2,
        "Chemical Group Character - Ambient Dynamics",
        out_groups2_minamb,
    )

    # Recreate the NN distance SI plots (as produced by the older plotting script)
    # Use the legacy filenames for compatibility.
    out_nn_press_si = os.path.join(BASEDIR, "modesout_NNdist_min_press_SI.png")
    out_nn_amb_si = os.path.join(BASEDIR, "modesout_NNdist_min_amb_SI.png")
    plot_NNdist_min_SI(
        results_minpress,
        highlight_minpress,
        out_nn_press_si,
        "Nearest-Neighbor Distances - Pressure Sensitivity",
    )
    _plot_minamb_two_category_NNdist_min_SI(
        results_minamb,
        active_minamb,
        out_nn_amb_si,
        "Nearest-Neighbor Distances - Ambient Character (0 GPa)",
    )




if __name__ == "__main__":
    # Legacy direct execution. Prefer: python run_group_analysis.py --data-dir ...
    import sys

    repo_root = os.path.abspath(os.path.join(BASEDIR, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from run_group_analysis import main as cli_main

    raise SystemExit(cli_main())
