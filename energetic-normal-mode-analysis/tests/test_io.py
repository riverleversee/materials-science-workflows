"""Unit tests for lattice / XYZ I/O (no trajectory data required)."""

from __future__ import annotations

import os
import unittest

import numpy as np

from mode_analysis.io import read_cell, read_xyz


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CELL = os.path.join(REPO_ROOT, "Ambient_DNTF_UnitCell.xyz")


class TestIo(unittest.TestCase):
    def test_read_cell_from_sample_unit_cell(self) -> None:
        matrix = read_cell(SAMPLE_CELL)
        self.assertEqual(matrix.shape, (3, 3))
        # Lattice from extended XYZ header (Angstrom).
        np.testing.assert_allclose(matrix[:, 0], [6.738, 0.0, 0.0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(matrix[:, 1], [0.0, 10.971, 0.0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(matrix[:, 2], [0.0, 0.0, 15.262], rtol=0, atol=1e-6)

    def test_read_xyz_from_sample_unit_cell(self) -> None:
        atoms, comment = read_xyz(SAMPLE_CELL)
        self.assertEqual(len(atoms), 88)
        self.assertIn("Lattice=", comment)
        labels = {label for label, _ in atoms}
        self.assertTrue(labels.issubset({"O", "N", "C", "H"}))


if __name__ == "__main__":
    unittest.main()
