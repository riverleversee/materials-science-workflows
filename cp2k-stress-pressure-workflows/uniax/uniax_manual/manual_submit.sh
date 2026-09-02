#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=uniax-manual-launch
#SBATCH --output=uniax_launch.%j.out
#SBATCH --account=your_slurm_account

# --- Workflow paths (see cluster.env.example) ---
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WORKFLOW_ROOT}/cp2k_env.sh"


# Launcher: submits all 3 axis jobs (separate scripts for cluster runtime).
# Run each axis individually: cd aaxis && sbatch manual_submit.sh
# Or submit all from here: sbatch manual_submit.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Submitting manual cell optimization for all 3 axes..."
jid_a=$(sbatch --parsable aaxis/manual_submit.sh)
jid_b=$(sbatch --parsable baxis/manual_submit.sh)
jid_c=$(sbatch --parsable cbaxis/manual_submit.sh)
echo "  a-axis:  $jid_a"
echo "  b-axis:  $jid_b"
echo "  cb-axis: $jid_c"
echo "To run a single axis: cd aaxis && sbatch manual_submit.sh"