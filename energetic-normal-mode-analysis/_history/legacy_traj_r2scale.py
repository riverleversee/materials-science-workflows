import numpy as np
import matplotlib.pyplot as plt
import json

pressurein=[0,4,10]
unit_cell_filepathin=['optimized_cell0GPa.cell','optimized_cell4GPa.cell','optimized_cell10GPa.cell'] 
position_pathin=['0GPa/AnimationFiles/anime_120.xyz','4GPa/AnimationFiles/anime_120.xyz','10GPa/AnimationFiles/anime_120.xyz']


# Modes you want as integers
desired_modesin=[120,166,172,179,183,188,189,200,202,210,225,246,261]

#100ps amb modes
#desired_modesin=[172,179,189,200,210]
desired_modesin=[172,179,189,200,210,217]

#pressure modes
desired_modesin=[166,179,188,200,204,225]
#all modes
#desired_modesin=[120,166,172,179,183,188,189,200,202,210,225,246,261]


pressurepaths=['0GPa/AnimationFiles/','4GPa/AnimationFiles/','10GPa/AnimationFiles/']
trajstart='anime_'
trajend='.xyz'
mode_mol_filepathsin = [
    ['' for _ in range(len(desired_modesin))]
    for _ in range(len(pressurepaths))
]


for presspathint in range(len(pressurepaths)):
    for modeint in range(len(desired_modesin)):
        print(pressurepaths[presspathint])
        mode_mol_filepathsin[presspathint][modeint]=pressurepaths[presspathint]+trajstart+str(desired_modesin[modeint])+trajend
        print(mode_mol_filepathsin[1])
        print(presspathint)
print(mode_mol_filepathsin)
bond_cutoffin=1.6
distance_thresholdin=3.2
prefixin="modes"


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



def read_cell(cell_file):
    """
    Reads unit cell vectors from a file, without using the 're' module.

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

    # Try to find a "Lattice=" line and parse it manually
    for line in lines:
        if 'Lattice=' in line:
            # locate the first quote after 'Lattice='
            eq_pos = line.find('Lattice=')
            first_quote = line.find('"', eq_pos)
            if first_quote == -1:
                raise ValueError("Lattice data not properly quoted.")
            second_quote = line.find('"', first_quote + 1)
            if second_quote == -1:
                raise ValueError("Lattice data closing quote not found.")
            content = line[first_quote + 1:second_quote].strip().split()
            if len(content) != 9:
                raise ValueError("Expected 9 components for 3 lattice vectors.")
            try:
                vals = list(map(float, content))
            except ValueError:
                raise ValueError("Lattice values must be valid floats.")
            A = np.array(vals[0:3])
            B = np.array(vals[3:6])
            C = np.array(vals[6:9])
            cell_matrix = np.column_stack((A, B, C))
            print(cell_matrix)
            return cell_matrix

    # Fallback: tokenize all non-blank lines
    tokens = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            tokens.extend(stripped.split())

    # Option 2: a single line with 12 tokens
    if len(tokens) == 12:
        try:
            A = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            B = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
            C = np.array([float(tokens[9]), float(tokens[10]), float(tokens[11])])
        except ValueError:
            raise ValueError("Tokens must be valid floats in the 12-token format.")
        return np.column_stack((A, B, C))

    # Option 1: three separate lines, each with at least 4 columns
    vectors = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                vectors.append(vec)
            except ValueError:
                raise ValueError(f"Cannot parse vector from line: {line.strip()}")
        if len(vectors) == 3:
            break

    if len(vectors) < 3:
        raise ValueError("The cell file does not contain three valid lattice vectors.")

    A, B, C = vectors
    return np.column_stack((A, B, C))


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



def identify_nitro_and_furazan_atoms(supercell_atoms, original_atoms, bond_matrix):
    """
    Identifies atoms involved in nitro groups and furazan rings.

    Parameters:
      supercell_atoms: List of (element, position) tuples for all atoms in the supercell.
      original_atoms:  List of (element, position) tuples for original atoms.
      bond_matrix:     Binary numpy array of shape (M, M) indicating bonded pairs.

    Returns:
      nitro_flags:     List of 0/1 flags for original atoms involved in nitro groups.
      furazan_flags:   List of 0/1 flags for original atoms involved in furazan rings.
    """
    M = len(supercell_atoms)
    N = len(original_atoms)
    elements = [elem for (elem, _) in supercell_atoms]

    nitro_flags = [0] * N
    furazan_flags = [0] * N
    furazan_flags_NO = [0] * N
    # Helper: get bonded neighbors
    def bonded_neighbors(i):
        return [j for j in range(M) if bond_matrix[i, j] == 1]

    # --- Nitro Group Detection ---
    for i in range(M):
        if elements[i] != "N":
            continue
        neighbors = bonded_neighbors(i)
        oxygen_neighbors = [j for j in neighbors if elements[j] == "O"]
        if len(oxygen_neighbors) == 2:
            valid = True
            for o in oxygen_neighbors:
                o_neighbors = bonded_neighbors(o)
                if len(o_neighbors) != 1 or o_neighbors[0] != i:
                    valid = False
                    break
            if valid:
                for idx in [i] + oxygen_neighbors:
                    if idx < N:
                        nitro_flags[idx] = 1
    print(np.sum(nitro_flags))
    # --- Furazan Ring Detection ---
    # Look for 6-membered rings with pattern C–C–N–O–N
    def find_rings():
        rings = []
        for i in range(M):
            if elements[i] != "C":
                continue
            for j in bonded_neighbors(i):
                if elements[j] != "C" or j <= i:
                    continue
                for k in bonded_neighbors(j):
                    if elements[k] != "N" or k in [i, j]:
                        continue
                    for l in bonded_neighbors(k):
                        if elements[l] != "O" or l in [i, j, k]:
                            continue
                        for m in bonded_neighbors(l):
                            if elements[m] != "N" or m in [i, j, k, l]:
                                continue
                            for n in bonded_neighbors(m):
                                if n == i and len(set([i, j, k, l, m])) == 5:
                                    rings.append([i, j, k, l, m])
        return rings

    furazan_rings = find_rings()
    for ring in furazan_rings:
        for idx in ring:
            if idx < N and elements[idx] in ["N", "O","C"]:
                furazan_flags[idx] = 1
    for ring in furazan_rings:
        for idx in ring:
            if idx < N and elements[idx] in ["N", "O"]:
                furazan_flags_NO[idx] = 1

    print(np.sum(furazan_flags))              

    return nitro_flags, furazan_flags , furazan_flags_NO

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




def fractional_group_displacement(norm_mode, initro, furazan, furazan_NO):
    """
    Computes fractional displacement contributions from nitro and furazan atoms,
    normalized by total displacement and by total displacement in O/N atoms.

    Parameters
    ----------
    norm_mode : list of (element, displacement_vector), shape (N, 3)
        Displacement vectors for each atom in the mode.
    initro : list of int
        List of 0/1 flags indicating nitro group involvement for each atom.
    furazan : list of int
        List of 0/1 flags indicating furazan ring involvement for each atom.

    Returns
    -------
    nitro_fraction_total : float
        Fraction of total displacement magnitude from nitro atoms.
    furazan_fraction_total : float
        Fraction of total displacement magnitude from furazan atoms.
    nitro_fraction_ON : float
        Fraction of O/N displacement magnitude from nitro atoms.
    furazan_fraction_ON : float
        Fraction of O/N displacement magnitude from furazan atoms.
    """
    disp = np.array([vec for _, vec in norm_mode], dtype=float)
    elements = [elem for elem, _ in norm_mode]
    mags = np.linalg.norm(disp, axis=1)

    total_mag = np.sum(mags)
    ON_mag = np.sum([mag for mag, elem in zip(mags, elements) if elem in ["O", "N"]])

    nitro_mag = np.sum([mag for mag, flag in zip(mags, initro) if flag])
    furazan_mag = np.sum([mag for mag, flag in zip(mags, furazan) if flag])
    furazan_mag_NO = np.sum([mag for mag, flag in zip(mags, furazan_NO) if flag])
    nitro_fraction_total = nitro_mag / total_mag if total_mag > 0 else 0.0
    furazan_fraction_total = furazan_mag / total_mag if total_mag > 0 else 0.0

    nitro_fraction_ON = nitro_mag / ON_mag if ON_mag > 0 else 0.0
    furazan_fraction_ON = furazan_mag_NO / ON_mag if ON_mag > 0 else 0.0

    return nitro_fraction_total, furazan_fraction_total, nitro_fraction_ON, furazan_fraction_ON

def run_supercell(cell_filename, xyz_filename, xyz_modes_filename,bond_cutoff,distance_threshold):
    """
    A helper function which takes the filenames, executes the supercell creation,
    and prints the resulting original atoms, full supercell atoms, and the unit cell matrix.
    """

    normal_mode_atoms, comment = read_normal_mode_xyz(xyz_modes_filename)

    original_atoms, supercell_atoms, cell_matrix = create_supercell(cell_filename, xyz_filename)
    bond_matrix, molecule_assignment, original_molecule_assignment,distances= analyze_supercell_bonds(supercell_atoms, original_atoms, bond_cutoff)
    initro,furazan, furazan_NO=identify_nitro_and_furazan_atoms(supercell_atoms, original_atoms, bond_matrix)
     
    nitro_fraction_total, furazan_fraction_total, nitro_fraction_ON, furazan_fraction_ON=fractional_group_displacement(normal_mode_atoms, initro, furazan, furazan_NO)
    orig_molec_centers=compute_centers_of_mass(original_atoms, original_molecule_assignment,supercell_atoms, molecule_assignment)

    com_disp_total_mag=compute_centers_of_mass_change(original_atoms, original_molecule_assignment,supercell_atoms, molecule_assignment,normal_mode_atoms,cell_matrix)

    radialproj, unnormradialproj =analyze_mode_projection(original_atoms, original_molecule_assignment,orig_molec_centers, normal_mode_atoms)
    orthoproj,unnormorthoproj=analyze_mode_orthogonal_magnitude(original_atoms, original_molecule_assignment, orig_molec_centers, normal_mode_atoms)

    axisproj, unnormaxisproj =analyze_mode_axis(original_atoms, normal_mode_atoms)

    
    spread_factor, spread_factor_unnorm= analyze_spread(normal_mode_atoms)


    normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms, normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type, unnormalized_threshold_sum_all_atoms, unnormalized_distance_scaled_sum_all_atoms, unnormalized_threshold_sums_by_type, unnormalized_distance_scaled_sums_by_type, totalmag, magbytype,norm_min_outmolec,norm_min_inmolec,norm_outmolec_by_type ,norm_inmolec_by_type =analyze_mode_intermole(original_atoms, original_molecule_assignment, supercell_atoms, molecule_assignment, normal_mode_atoms, distance_threshold)
 

    return normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type,normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms ,radialproj,orthoproj,unnormalized_threshold_sums_by_type, unnormalized_distance_scaled_sums_by_type,unnormalized_threshold_sum_all_atoms, unnormalized_distance_scaled_sum_all_atoms ,unnormradialproj,unnormorthoproj, totalmag,magbytype,  axisproj, unnormaxisproj,norm_min_outmolec,norm_min_inmolec,norm_outmolec_by_type ,norm_inmolec_by_type,com_disp_total_mag,spread_factor, spread_factor_unnorm,nitro_fraction_total, furazan_fraction_total, nitro_fraction_ON, furazan_fraction_ON 









def parse_trajectories_for_modes(pressureslisttraj,position_xyz,
                                 trajectory_list,
                                 mode_indices,
                                 output_prefix="output"):
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

    positions_filename = f"{output_prefix}_P{pressureslisttraj}_positions.xyz"
    with open(positions_filename, "w") as outf:
        outf.write(f"{natoms}\nReference positions\n")
        for lab, coord in zip(atom_labels, ref_coords):
            outf.write(f"{lab} {' '.join(map(str, coord))}\n")

    # --- Process each trajectory & tag by mode_indices ---
    disp_files = []
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


        outname = f"{output_prefix}_P{pressureslisttraj}_mode{mode_idx}.xyz"
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
        "nitro_fraction_ON":[],
        "furazan_fraction_ON":[]
    }

    # Collect results across pressures
    for idx, pressure in enumerate(pressures):
        print(f"Running analysis at pressure {pressure} using files: {cell_filenames[idx]}, {xyz_filenames[idx]}, {xyz_modes_filenames[idx]}")

        # Run supercell analysis with the corresponding files
        normalized_threshold_sums_by_type, normalized_distance_scaled_sums_by_type,normalized_threshold_sum_all_atoms, normalized_distance_scaled_sum_all_atoms ,radialproj,orthoproj,unnormalized_threshold_sums_by_type, unnormalized_distance_scaled_sums_by_type,unnormalized_threshold_sum_all_atoms, unnormalized_distance_scaled_sum_all_atoms ,unnormradialproj,unnormorthoproj, totalmag,magbytype, axisproj, unnormaxisproj, norm_min_outmolec,norm_min_inmolec,norm_outmolec_by_type ,norm_inmolec_by_type,com_disp_total_mag,spread_factor, spread_factor_unnorm,nitro_fraction_total, furazan_fraction_total, nitro_fraction_ON, furazan_fraction_ON= run_supercell(
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
        results_dict["nitro_fraction_ON"].append(nitro_fraction_ON)
        results_dict["furazan_fraction_ON"].append(furazan_fraction_ON)




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





def process_cp2k_modes_over_pressures(pressures,  position_xyz ,  unit_cell_files,  trajectory_list, desired_modes, bond_cutoff, distance_threshold, output_prefix="analysis"):
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
    for idx, pressure in enumerate(pressures):
        print(f"Processing pressure {pressure} with files: {position_xyz[idx]}, {unit_cell_files[idx]}, {trajectory_list[idx]}")
        positions_file, mode_files =parse_trajectories_for_modes(pressure,position_xyz[idx], trajectory_list[idx], desired_modes, output_prefix="output")
        
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
        pressurein, pospath, unit_cell_filepath, mode_mol_filepaths, desired_modes, bond_cutoff, distance_threshold, output_prefix
    )
    results_filepath = f"{output_prefix}_results.json"
    with open(results_filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Step 2: Generate a combined visualization for all modes
#    plot_combined_mode_analysis(results_dict, output_filename=f"{output_prefix}_combined_modes.png")
 #   plot_atomwise_mode_analysis(results_dict, output_prefix=f"{output_prefix}_combined_modes_atomwise_analysis")


    print("Full analysis complete. Saved all plots and processed results.")




run_full_analysis(pressurein, position_pathin,unit_cell_filepathin, mode_mol_filepathsin, desired_modesin, bond_cutoffin, distance_thresholdin, output_prefix="final_analysis"+prefixin)
