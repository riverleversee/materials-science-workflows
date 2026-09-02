#!/usr/bin/env python3
"""Run the user's strain calculation snippet: Green-Lagrange with rotated cells."""
import numpy as np


def calculate_strain(A, A_prime):
    """
    Calculates Green-Lagrange strain tensor E.
    A: Initial cell matrix (columns are lattice vectors)
    A_prime: Deformed cell matrix
    """
    # Deformation Gradient F = A_prime * A^-1
    F = A_prime @ np.linalg.inv(A)
    # Green-Lagrange Strain E = 0.5 * (F^T * F - I)
    E = 0.5 * (F.T @ F - np.eye(3))
    return E


# 1. Define initial cell (Unit Cube)
A_orig = np.eye(3)

# 2. Define distorted cell (10% stretch in the X direction)
A_prime_orig = np.array(
    [
        [1.1, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# 3. Define a 45-degree rotation around the Z-axis
theta = np.radians(45)
R = np.array(
    [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ]
)

# 4. Apply rotation to BOTH the initial and distorted cells
A_rot = R @ A_orig
A_prime_rot = R @ A_prime_orig

# 5. Calculate strain for both cases
E_orig = calculate_strain(A_orig, A_prime_orig)
E_rot = calculate_strain(A_rot, A_prime_rot)

print("Original Strain Tensor (Stretch along X):")
print(np.round(E_orig, 4))

print("\nStrain Tensor with 45-degree rotation:")
print(np.round(E_rot, 4))

# Comparison
identical = np.allclose(E_orig, E_rot)
print(f"\nAre the strain tensors identical? {identical}")

# Extra: show that E_rot = R @ E_orig @ R^T (same physical tensor, different components)
E_orig_rotated = R @ E_orig @ R.T
print("\nR @ E_orig @ R^T (strain tensor expressed in rotated frame):")
print(np.round(E_orig_rotated, 4))
print("E_rot == R @ E_orig @ R^T?", np.allclose(E_rot, E_orig_rotated))
