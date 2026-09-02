#!/usr/bin/env python3
"""
Generate per-axis CP2K ramp bash scripts and copypaste submission list.

Inputs:
  uniax/deltaPramp/axes_manifest.json

Outputs:
  uniax/deltaPramp/pIso_<pIsoBar>bar/axis_<group>/run_axis.sh
  uniax/deltaPramp/pIso_<pIsoBar>bar/axis_<group>/run_axis.sbatch
  uniax/deltaPramp/pIso_<pIsoBar>bar/copypastesubmit.txt
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

GPA_TO_BAR = 10000.0


def main() -> None:
    delta_pramp = Path(__file__).resolve().parent
    workflow_root = delta_pramp.parents[1]

    manifest_path = Path(
        os.environ.get("AXES_MANIFEST", delta_pramp / "axes_manifest.json")
    ).resolve()
    pressure_gpa = float(os.environ.get("PRESSURE_GPA", "15.0"))
    p_iso_bar = int(round(pressure_gpa * GPA_TO_BAR))
    p_iso_label = f"pIso_{p_iso_bar}bar"

    slurm_account = os.environ.get("SLURM_ACCOUNT", "ucb-general")
    slurm_partition = os.environ.get("SLURM_PARTITION", "acpu")
    slurm_qos = os.environ.get("SLURM_QOS", "cpu-normal")
    # Optional; Alpine no longer requires --constraint=ib for acpu
    slurm_constraint = os.environ.get("SLURM_CONSTRAINT", "").strip()
    conda_env = os.environ.get(
        "CONDA_ENV", f"/projects/{os.environ.get('USER', 'rile5166')}/software/dftb_22"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    axes: List[Dict[str, Any]] = json.loads(manifest_path.read_text())
    if not isinstance(axes, list) or not axes:
        raise ValueError("Manifest must be a non-empty list")

    out_base = delta_pramp / p_iso_label
    out_base.mkdir(parents=True, exist_ok=True)

    run_script = delta_pramp / "run_ramp_workflow.py"
    centered_fd = delta_pramp / "uniax_param_centered_fd.sh"
    opt_script = workflow_root / "scfhel/uniax_surrogate_optimizer_test.py"

    if not run_script.is_file():
        raise FileNotFoundError(f"Ramp driver missing: {run_script}")

    lines: List[str] = []
    for ax in axes:
        group_id = int(ax["group_id"])
        path_cell = Path(ax["path_cell"]).resolve()
        path_coords = Path(ax["path_coords"]).resolve()

        axis_dir = out_base / f"axis_{group_id}"
        axis_dir.mkdir(parents=True, exist_ok=True)
        sh_path = axis_dir / "run_axis.sh"
        sbatch_path = axis_dir / "run_axis.sbatch"

        sh = f"""#!/bin/bash
set -euo pipefail

export WORKFLOW_ROOT="{workflow_root}"
export PRESSURE_GPA="{pressure_gpa}"
export AXIS_OUT_DIR="{axis_dir}"
export START_CELL="{path_cell}"
export START_COORDS="{path_coords}"

export OPT_SCRIPT="${{OPT_SCRIPT:-{opt_script}}}"
export CENTERED_FD_SCRIPT="${{CENTERED_FD_SCRIPT:-{centered_fd}}}"

# shellcheck source=/dev/null
source "${{WORKFLOW_ROOT}}/cp2k_env.sh"

export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-${{SLURM_CPUS_PER_TASK:-2}}}}"
export SLURM_NTASKS="${{SLURM_NTASKS:-8}}"

echo "=== run_axis.sh starting on $(hostname) ==="
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-}}"
echo "PRESSURE_GPA=${{PRESSURE_GPA}}"
echo "AXIS_OUT_DIR=${{AXIS_OUT_DIR}}"
echo "which python3 -> $(command -v python3 || echo 'python3 not found')"

python3 "{run_script}"
"""
        sh_path.write_text(sh)
        sh_path.chmod(0o755)

        constraint_line = (
            f"#SBATCH --constraint={slurm_constraint}\n" if slurm_constraint else ""
        )
        sbatch = f"""#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos={slurm_qos}
#SBATCH --partition={slurm_partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cp2k-ramp-g{group_id}
{constraint_line}#SBATCH --output=cp2k-ramp-g{group_id}.%j.out
#SBATCH --account={slurm_account}

set -euo pipefail

module purge
module load slurm/alpine
module load gcc/11.2.0 openmpi/4.1.1
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "{conda_env}"

export WORKFLOW_ROOT="{workflow_root}"
source "${{WORKFLOW_ROOT}}/cp2k_env.sh"
export APPTAINER_CMD="${{APPTAINER_CMD:-/usr/bin/apptainer}}"
cd "{axis_dir}"
bash ./run_axis.sh
"""
        sbatch_path.write_text(sbatch)
        sbatch_path.chmod(0o755)

        lines.append(f"sbatch {axis_dir.name}/run_axis.sbatch")

    (out_base / "copypastesubmit.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} axis jobs under {out_base}")


if __name__ == "__main__":
    main()
