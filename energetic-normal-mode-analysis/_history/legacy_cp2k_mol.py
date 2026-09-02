"""
Legacy normal-mode analysis (CP2K .mol vibrational output).

Prefer the trajectory-based canonical workflow:
  python run_group_analysis.py --data-dir /path/to/BNFFanalysis/minpress

This module remains for historical CP2K .mol parsing experiments.
"""

import numpy as np
import matplotlib.pyplot as plt


pressurein=[0]
unit_cell_filepathin=['cell_filename.xyz'] 


mode_mol_filepathsin=['RDX-cellopt-VIBRATIONS-1.mol']

# Modes you want as integers
desired_modesin=[50,75,100,125,150,200,250 ,300,400,450]

bond_cutoffin=1.6
distance_thresholdin=2.2
prefixin="modes"


def read_cell(cell_file):
    """
    Reads unit cell vectors from a file.
    
    Expected formats:
      Option 1 (3 lines):
         A x1 y1 z1
         B x2 y2 z2
         C x3 y3 z3
      Option 2 (1 line with 12 tokens):
         A x1 y1 z1 B x2 y2 z2 C x3 y3 z3
    
    Returns:
      A 3x3 numpy array (cell_matrix) whose columns are the lattice vectors A, B, and C.
    """
    with open(cell_file, 'r') as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        if line.strip():
            tokens.extend(line.strip().split())
    
    if len(tokens) == 12:
        try:
            A = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            B = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
            C = np.array([float(tokens[9]), float(tokens[10]), float(tokens[11])])
        except ValueError:
            raise ValueError("Error parsing cell file tokens.")
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
        if count != 12:
            print(
                f"Molecule index {mol} contains {count} atoms in the full supercell assignment, expected 12."
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

    return normalized_sum

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

    return normalized_sum




def analyze_mode_intermole(original_atoms, original_molecule_assignment, supercell_atoms, full_molecule_assignment, normal_mode_atoms, distance_threshold):
    """
    Computes vibrational coupling contributions and provides both overall and atomic-type-specific normalized sums.

    This function:
      1. Constructs three NxM matrices:
         - Standard coupling matrix (absolute dot product).
         - Threshold matrix (applies distance cutoff).
         - Distance-scaled matrix (divides values by atom-pair distance).
      2. Extracts maximum values per original atom separately for **threshold matrix** and **distance-scaled matrix**.
      3. Computes overall normalized sum across all atoms (ignoring atomic type).
      4. **Separates normalized sums for each atomic type**.

    Parameters:
      original_atoms: List of tuples (atom_label, cartesian_coord) for the original atoms.
      original_molecule_assignment: NumPy array of molecule indices corresponding to the original atoms.
      supercell_atoms: List of tuples (atom_label, cartesian_coord) for all atoms in the supercell.
      full_molecule_assignment: NumPy array of molecule indices corresponding to the supercell atoms.
      normal_mode_atoms: List of tuples (atom_label, numpy array([dx, dy, dz])) from read_normal_mode_xyz.
      distance_threshold: Maximum allowed distance for interactions; entries are set to zero if exceeded.

    Returns:
      coupling_matrix: NxM NumPy array containing the absolute dot product, zeroed out for atoms in the same molecule.
      threshold_matrix: NxM NumPy array with additional filtering where distant atom pairs are also set to zero.
      distance_scaled_matrix: NxM NumPy array where values from coupling_matrix are divided by distance,
                              with entries set to zero if the distance is zero.
      normalized_threshold_sum_all_atoms: The total normalized sum across all atoms.
      normalized_distance_scaled_sum_all_atoms: The total normalized sum for distance-scaled contributions across all atoms.
      normalized_threshold_sums_by_type: Dictionary containing normalized threshold sums for each atomic type.
      normalized_distance_scaled_sums_by_type: Dictionary containing normalized distance-scaled sums for each atomic type.
    """
    N = len(original_atoms)
    M = len(supercell_atoms)
    
    # Initialize NxM matrices
    coupling_matrix = np.zeros((N, M))
    threshold_matrix = np.zeros((N, M))
    distance_scaled_matrix = np.zeros((N, M))

    # Identify unique atomic types and their indices
    atom_types = {label for label, _ in original_atoms}
    max_threshold_values_by_type = {atom: 0.0 for atom in atom_types}
    max_distance_scaled_values_by_type = {atom: 0.0 for atom in atom_types}

    max_threshold_values = np.zeros(N)
    max_distance_scaled_values = np.zeros(N)
    total_mode_magnitude = 0.0

    for i in range(N):  # Loop over original atoms
        atom_label, original_pos = original_atoms[i]
        _, normal_mode_vector = normal_mode_atoms[i]  # Retrieve corresponding normal mode displacement
        
        max_threshold_coupling = 0.0
        max_distance_scaled_coupling = 0.0

        for j in range(M):  # Loop over supercell atoms
            _, supercell_pos = supercell_atoms[j]

            # Compute displacement vector
            displacement_vector = supercell_pos - original_pos

            # Compute distance magnitude
            distance_magnitude = np.linalg.norm(displacement_vector)

            # Normalize the displacement vector
            if distance_magnitude > 0:
                normalized_displacement = displacement_vector / distance_magnitude
            else:
                normalized_displacement = np.zeros_like(displacement_vector)

            # Compute absolute dot product **before modifying for threshold**
            original_dot_product = np.abs(np.dot(normal_mode_vector, normalized_displacement))

            # If the atoms belong to the same molecule, set entry to zero
            if original_molecule_assignment[i] == full_molecule_assignment[j]:
                original_dot_product = 0.0

            coupling_matrix[i, j] = original_dot_product

            # Apply threshold-based filtering **without modifying original_dot_product**
            threshold_dot_product = original_dot_product
            if distance_magnitude > distance_threshold:
                threshold_dot_product = 0.0

            threshold_matrix[i, j] = threshold_dot_product
            max_threshold_coupling = max(max_threshold_coupling, threshold_dot_product)

            # Apply distance scaling **using the original dot product**
            scaled_value = original_dot_product / (distance_magnitude**2) if distance_magnitude > 0 else 0.0
            distance_scaled_matrix[i, j] = scaled_value
            max_distance_scaled_coupling = max(max_distance_scaled_coupling, scaled_value)

        max_threshold_values[i] = max_threshold_coupling
        max_distance_scaled_values[i] = max_distance_scaled_coupling
        total_mode_magnitude += np.linalg.norm(normal_mode_vector)

        # Store values by atomic type
        max_threshold_values_by_type[atom_label] += max_threshold_coupling
        max_distance_scaled_values_by_type[atom_label] += max_distance_scaled_coupling

    # Compute overall normalized sums across all atoms
    normalized_threshold_sum_all_atoms = np.sum(max_threshold_values) / total_mode_magnitude if total_mode_magnitude > 0 else 0.0
    normalized_distance_scaled_sum_all_atoms = np.sum(max_distance_scaled_values) / total_mode_magnitude if total_mode_magnitude > 0 else 0.0
    normalized_threshold_sum_all_atoms = np.sqrt(np.sum(np.square(max_threshold_values))) / total_mode_magnitude if total_mode_magnitude > 0 else 0.0
    normalized_distance_scaled_sum_all_atoms = np.sqrt(np.sum(np.square(max_distance_scaled_values))) / total_mode_magnitude if total_mode_magnitude > 0 else 0.0



    # Compute normalized sums by atomic type (but use total magnitude from **all atoms**)
    normalized_threshold_sums_by_type = {
        atom: max_threshold_values_by_type[atom] / total_mode_magnitude if total_mode_magnitude > 0 else 0.0
        for atom in atom_types
    }

    normalized_distance_scaled_sums_by_type = {
        atom: max_distance_scaled_values_by_type[atom] / total_mode_magnitude if total_mode_magnitude > 0 else 0.0
        for atom in atom_types
    }

    return normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms, normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type



def run_supercell(cell_filename, xyz_filename, xyz_modes_filename,bond_cutoff,distance_threshold):
    """
    A helper function which takes the filenames, executes the supercell creation,
    and prints the resulting original atoms, full supercell atoms, and the unit cell matrix.
    """

    normal_mode_atoms, comment = read_normal_mode_xyz(xyz_modes_filename)

    original_atoms, supercell_atoms, cell_matrix = create_supercell(cell_filename, xyz_filename)
    bond_matrix, molecule_assignment, original_molecule_assignment,distances= analyze_supercell_bonds(supercell_atoms, original_atoms, bond_cutoff)
    orig_molec_centers=compute_centers_of_mass(original_atoms, original_molecule_assignment,supercell_atoms, molecule_assignment)
    radialproj=analyze_mode_projection(original_atoms, original_molecule_assignment,orig_molec_centers, normal_mode_atoms)
    orthoproj=analyze_mode_orthogonal_magnitude(original_atoms, original_molecule_assignment, orig_molec_centers, normal_mode_atoms)

    normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms, normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type =analyze_mode_intermole(original_atoms, original_molecule_assignment, supercell_atoms, molecule_assignment, normal_mode_atoms, distance_threshold)
    return normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type,normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms ,radialproj,orthoproj


def parse_cp2k_mol_file(mol_filename, desired_modes, output_prefix="output"):
    """
    Parses a CP2K .mol file, extracts atomic positions and selected vibrational modes, 
    and saves XYZ files for positions and modes, including frequencies in comments.

    Parameters:
      mol_filename: Path to the .mol file.
      desired_modes: List of integers (or strings) representing the vibration mode indices to extract.
      output_prefix: Prefix for output filenames.

    Outputs:
      - An XYZ file for atomic positions.
      - Separate XYZ files for each selected vibrational mode, including the corresponding frequency in the comment line.
    """
    # Convert mode indices to integers if they are given as strings
    desired_modes = [int(mode) for mode in desired_modes]

    with open(mol_filename, 'r') as file:
        lines = file.readlines()

    # Initialize storage
    atomic_positions = []
    vibrations = {}
    frequencies = []
    current_vibration = None
    parsing_coords = False
    parsing_modes = False
    parsing_freq = False

    for line in lines:
        line = line.strip()

        if "[FREQ]" in line:
            parsing_freq = True
            parsing_coords = False
            parsing_modes = False
            continue
        elif "[FR-COORD]" in line:
            parsing_coords = True
            parsing_freq = False
            parsing_modes = False
            continue
        elif "[FR-NORM-COORD]" in line:
            parsing_modes = True
            parsing_coords = False
            parsing_freq = False  # Ensure frequency parsing stops here
            continue
        elif "[" in line:  # Stop parsing when a new section starts
            parsing_coords = False
            parsing_modes = False
            parsing_freq = False
            continue

        if parsing_freq:
            frequencies.extend(line.split())  # Collect frequency values

        if parsing_coords:
            parts = line.split()
            if len(parts) == 4:  # Expecting format: atom_label x y z
                atomic_positions.append(parts)

        if parsing_modes:
            if "vibration" in line:
                mode_number = int(line.split()[1])  # Extract mode number
                current_vibration = mode_number
                vibrations[current_vibration] = []
            elif current_vibration and len(line.split()) == 3:
                vibrations[current_vibration].append(line.split())

    # Save atomic positions to XYZ file
    with open(f"{output_prefix}_positions.xyz", 'w') as xyz_file:
        xyz_file.write(f"{len(atomic_positions)}\nCP2K Atomic Positions\n")
        for atom in atomic_positions:
            xyz_file.write(f"{' '.join(atom)}\n")

    # Save vibrational modes to XYZ files with frequency in the comment
    for mode in desired_modes:
        if mode in vibrations:
            frequency = frequencies[mode - 1] if mode - 1 < len(frequencies) else "Unknown"
            with open(f"{output_prefix}_mode_{mode}.xyz", 'w') as xyz_file:
                xyz_file.write(f"{len(vibrations[mode])}\nFrequency: {frequency} cm⁻¹\n")
                for i, displacement in enumerate(vibrations[mode]):
                    xyz_file.write(f"{atomic_positions[i][0]} {' '.join(displacement)}\n")

    print(f"Saved atomic positions and selected modes to XYZ files with prefix '{output_prefix}', including frequencies.")





def parse_cp2k_mol_file(mol_filename, desired_modes, output_prefix="output"):
    """
    Parses a CP2K .mol file, extracts atomic positions and selected vibrational modes, 
    converts positions from Bohr to Angstroms, and saves XYZ files.

    Parameters:
      mol_filename: Path to the .mol file.
      desired_modes: List of integers (or strings) representing the vibration mode indices to extract.
      output_prefix: Prefix for output filenames.

    Outputs:
      - An XYZ file for atomic positions in Angstroms.
      - Separate XYZ files for each selected vibrational mode, including the corresponding frequency in the comment line.
    
    Returns:
      - positions_filename: The filename of the atomic positions XYZ file.
      - normal_mode_filenames: A list of filenames for each extracted vibrational mode.
    """
    # Convert mode indices to integers if they are given as strings
    desired_modes = [int(mode) for mode in desired_modes]

    BOHR_TO_ANGSTROM = 0.529177  # Conversion factor

    with open(mol_filename, 'r') as file:
        lines = file.readlines()

    # Initialize storage
    atomic_positions = []
    vibrations = {}
    frequencies = []
    current_vibration = None
    parsing_coords = False
    parsing_modes = False
    parsing_freq = False

    for line in lines:
        line = line.strip()

        if "[FREQ]" in line:
            parsing_freq = True
            parsing_coords = False
            parsing_modes = False
            continue
        elif "[FR-COORD]" in line:
            parsing_coords = True
            parsing_freq = False
            parsing_modes = False
            continue
        elif "[FR-NORM-COORD]" in line:
            parsing_modes = True
            parsing_coords = False
            parsing_freq = False  # Ensure frequency parsing stops here
            continue
        elif "[" in line:  # Stop parsing when a new section starts
            parsing_coords = False
            parsing_modes = False
            parsing_freq = False
            continue

        if parsing_freq:
            frequencies.extend(line.split())  # Collect frequency values

        if parsing_coords:
            parts = line.split()
            if len(parts) == 4:  # Expecting format: atom_label x y z
                atom_label, x, y, z = parts
                x, y, z = [float(coord) * BOHR_TO_ANGSTROM for coord in (x, y, z)]  # Convert to Angstroms
                atomic_positions.append([atom_label, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

        if parsing_modes:
            if "vibration" in line:
                mode_number = int(line.split()[1])  # Extract mode number
                current_vibration = mode_number
                vibrations[current_vibration] = []
            elif current_vibration and len(line.split()) == 3:
                vibrations[current_vibration].append(line.split())

    # Save atomic positions to XYZ file
    positions_filename = f"{output_prefix}_positions.xyz"
    with open(positions_filename, 'w') as xyz_file:
        xyz_file.write(f"{len(atomic_positions)}\nCP2K Atomic Positions (Converted to Angstroms)\n")
        for atom in atomic_positions:
            xyz_file.write(f"{' '.join(atom)}\n")

    # Save vibrational modes to XYZ files with frequency in the comment
    normal_mode_filenames = []
    for mode in desired_modes:
        if mode in vibrations:
            frequency = frequencies[mode - 1] if mode - 1 < len(frequencies) else "Unknown"
            mode_filename = f"{output_prefix}_mode_{mode}.xyz"
            normal_mode_filenames.append(mode_filename)
            with open(mode_filename, 'w') as xyz_file:
                xyz_file.write(f"{len(vibrations[mode])}\nFrequency: {frequency} cm⁻¹\n")
                for i, displacement in enumerate(vibrations[mode]):
                    xyz_file.write(f"{atomic_positions[i][0]} {' '.join(displacement)}\n")

    print(f"Saved atomic positions (converted to Angstroms) and selected modes to XYZ files with prefix '{output_prefix}', including frequencies.")

    return positions_filename, normal_mode_filenames



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
        "distance_scaled_sums_by_type": {}
    }

    # Collect results across pressures
    for idx, pressure in enumerate(pressures):
        print(f"Running analysis at pressure {pressure} using files: {cell_filenames[idx]}, {xyz_filenames[idx]}, {xyz_modes_filenames[idx]}")

        # Run supercell analysis with the corresponding files
        normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type, normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms, radialproj, orthoproj = run_supercell(
            cell_filenames[idx], xyz_filenames[idx], xyz_modes_filenames[idx], bond_cutoff, distance_threshold
        )

        # Store results
        results_dict["threshold_sum_all_atoms"].append(normalized_threshold_sum_all_atoms)
        results_dict["distance_scaled_sum_all_atoms"].append(normalized_distance_scaled_sum_all_atoms)
        results_dict["radial_projection"].append(radialproj)
        results_dict["orthogonal_projection"].append(orthoproj)

        # Store per-atom-type results
        for atom_type in normalized_threshold_sums_by_type:
            if atom_type not in results_dict["threshold_sums_by_type"]:
                results_dict["threshold_sums_by_type"][atom_type] = []
                results_dict["distance_scaled_sums_by_type"][atom_type] = []
            results_dict["threshold_sums_by_type"][atom_type].append(normalized_threshold_sums_by_type[atom_type])
            results_dict["distance_scaled_sums_by_type"][atom_type].append(normalized_distance_scaled_sums_by_type[atom_type])

    # Create a single figure with 4 subplots for all atoms (2x2 layout)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    metrics = ["threshold_sum_all_atoms", "distance_scaled_sum_all_atoms", "radial_projection", "orthogonal_projection"]
    titles = ["Normalized Threshold Sum (All Atoms)", "Normalized Distance-Scaled Sum (All Atoms)", "Radial Projection", "Orthogonal Projection"]

    for ax, metric, title in zip(axs.flatten(), metrics, titles):
        ax.plot(results_dict["pressures"], results_dict[metric], marker="o", linestyle="-")
        ax.set_xlabel("Pressure")
        ax.set_ylabel(title)
        ax.set_title(title + " vs Pressure")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_combined.png")
    plt.close()

    # Create separate multipanel figures for each atomic type
    for atom_type in results_dict["threshold_sums_by_type"]:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        metrics = ["threshold_sums_by_type", "distance_scaled_sums_by_type"]
        titles = [f"Threshold Sum ({atom_type})", f"Distance-Scaled Sum ({atom_type})"]

        for ax, metric, title in zip(axs.flatten(), metrics, titles):
            ax.plot(results_dict["pressures"], results_dict[metric][atom_type], marker="o", linestyle="-")
            ax.set_xlabel("Pressure")
            ax.set_ylabel(title)
            ax.set_title(title + " vs Pressure")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{atom_type}.png")
        plt.close()

    return results_dict





def process_cp2k_modes_over_pressures(pressures, unit_cell_files, mol_files, desired_modes, bond_cutoff, distance_threshold, output_prefix="analysis"):
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
    if not (len(pressures) == len(unit_cell_files) == len(mol_files)):
        raise ValueError("Mismatch in list lengths: Each pressure must have corresponding filenames.")

    # Convert mode indices to integers
    desired_modes = [int(mode) for mode in desired_modes]

    # Storage for processed file paths
    all_positions_files = []
    all_mode_files = {mode: [] for mode in desired_modes}

    # Step 1: Extract atomic positions and vibrational modes for each pressure
    for idx, pressure in enumerate(pressures):
        print(f"Processing pressure {pressure} with files: {unit_cell_files[idx]}, {mol_files[idx]}")

        positions_file, mode_files = parse_cp2k_mol_file(mol_files[idx], desired_modes, output_prefix=f"{output_prefix}_P{pressure}")

        all_positions_files.append(positions_file)
        for mode in desired_modes:
            all_mode_files[mode].append(f"{output_prefix}_P{pressure}_mode_{mode}.xyz")

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


def run_full_analysis(pressurein, unit_cell_filepath, mode_mol_filepaths, desired_modes, bond_cutoff, distance_threshold, output_prefix="final_analysis"):
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
        pressurein, unit_cell_filepath, mode_mol_filepaths, desired_modes, bond_cutoff, distance_threshold, output_prefix
    )

    # Step 2: Generate a combined visualization for all modes
    plot_combined_mode_analysis(results_dict, output_filename=f"{output_prefix}_combined_modes.png")
    plot_atomwise_mode_analysis(results_dict, output_prefix=f"{output_prefix}_combined_modes_atomwise_analysis")


    print("Full analysis complete. Saved all plots and processed results.")




if __name__ == "__main__":
    run_full_analysis(
        pressurein,
        unit_cell_filepathin,
        mode_mol_filepathsin,
        desired_modesin,
        bond_cutoffin,
        distance_thresholdin,
        output_prefix="final_analysis" + prefixin,
    )
