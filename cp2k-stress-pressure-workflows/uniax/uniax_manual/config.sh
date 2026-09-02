#!/bin/bash
# Manual cell optimization workflow configuration

# Fraction of full Newton step (0.3 = 30%)
STEP_FRACTION="${STEP_FRACTION:-0.3}"

# Max manual cell + geo opt cycles (safety limit; loop exits when stress converged)
MAX_MANUAL_CYCLES="${MAX_MANUAL_CYCLES:-100}"

# Finite-difference perturbation for lengths a,b,c (absolute [Å])
FD_DELTA_LENGTH_ANG="${FD_DELTA_LENGTH_ANG:-0.005}"

# Finite-difference perturbation for angles α,β,γ (degrees, absolute)
# Chosen so L*δθ ≈ 0.005 Å for L~6 Å: δθ ≈ 0.05°
FD_DELTA_ANGLE="${FD_DELTA_ANGLE:-0.05}"

# Stress convergence tolerance [bar]; max |sigma - target| per component
STRESS_TOL_BAR="${STRESS_TOL_BAR:-2000}"

# Trust radius: max change per cell step (limits absurd steps from bad Jacobian)
MAX_DELTA_LENGTH_ANG="${MAX_DELTA_LENGTH_ANG:-0.1}"
MAX_DELTA_ANGLE_DEG="${MAX_DELTA_ANGLE_DEG:-1.0}"
