"""Smoke tests: imports and CLI help without trajectory data."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSmoke(unittest.TestCase):
    def test_driver_import_has_no_side_effects(self) -> None:
        driver_path = os.path.join(
            REPO_ROOT,
            "BNFFanalysis",
            "minpress",
            "ModeAnalysis_River_Traj_r2scale_massweight_groups.py",
        )
        spec = importlib.util.spec_from_file_location("nma_groups_driver_test", driver_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "run_full_analysis"))
        self.assertTrue(hasattr(module, "make_plots"))

    def test_cli_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "run_group_analysis.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--data-dir", proc.stdout)


if __name__ == "__main__":
    unittest.main()
