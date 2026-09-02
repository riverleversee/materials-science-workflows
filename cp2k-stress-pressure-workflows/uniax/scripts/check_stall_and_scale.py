#!/usr/bin/env python3
"""
Check if fixed-atom cell opt stalled (last 3 steps pressure change < 500 bar)
and pressure deviation > 1000 bar. If so, compute isotropic scale factor from
hydroopt dP/dV and apply to cell.
"""
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


def read_cell_volume(cell_path):
    """Read cell file and return volume (|a.(bxc)|)."""
    path = Path(cell_path)
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        nums = []
        for x in parts:
            try:
                nums.append(float(x))
            except ValueError:
                pass
        if len(nums) >= 3:
            rows.append(nums[-3:])
        if len(rows) == 3:
            break
    if len(rows) != 3 or np is None:
        return None
    cell = np.array(rows, dtype=float)
    return abs(np.linalg.det(cell))


def parse_pressures_from_output(out_path):
    """
    Parse CP2K run output for pressure at each CELL_OPT step.
    Tries: 1) OPT| Internal pressure [bar] lines (most reliable)
    2) Stress tensor diagonal (s_xx + s_yy + s_zz)/3.
    Returns list of (step_index, pressure) for last steps.
    """
    path = Path(out_path)
    if not path.exists():
        return []
    text = path.read_text(errors='replace')
    pressures = []

    # Primary: parse "Internal pressure [bar]" or "OPT| Internal pressure [bar]"
    for i, line in enumerate(text.splitlines()):
        if 'INTERNAL PRESSURE' in line.upper() and '[BAR]' in line.upper():
            nums = re.findall(r'-?\d+\.?\d*(?:[Ee][-+]?\d+)?', line)
            if nums:
                try:
                    p = float(nums[-1])
                    if -1e7 < p < 1e7:
                        pressures.append((len(pressures), p))
                except (ValueError, IndexError):
                    pass

    if len(pressures) >= 3:
        return pressures

    # Fallback: parse stress tensor blocks
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'STRESS' in line.upper() and 'TENSOR' in line.upper():
            block = []
            for k in range(1, 8):  # allow more lines for header row
                if i + k >= len(lines):
                    break
                nums = re.findall(r'-?\d+\.?\d*(?:[Ee][-+]?\d+)?', lines[i + k])
                if len(nums) >= 3:
                    try:
                        row = [float(nums[0]), float(nums[1]), float(nums[2])]
                        if all(-1e7 < x < 1e7 for x in row):  # plausible bar values
                            block.append(row)
                    except (ValueError, IndexError):
                        break
                elif block:
                    break
            if len(block) >= 3:
                p = (block[0][0] + block[1][1] + block[2][2]) / 3.0
                pressures.append((len(pressures), p))
                i += 3
        i += 1
    return pressures


def main():
    if len(sys.argv) < 7:
        print("Usage: check_stall_and_scale.py <run_out> <cell_file> <target_p_bar> <p_gpa> <hydro_dir> <stage_num>", file=sys.stderr)
        sys.exit(2)
    run_out = Path(sys.argv[1])
    cell_file = Path(sys.argv[2])
    target_p_bar = float(sys.argv[3])
    p_gpa = int(float(sys.argv[4]))
    hydro_dir = Path(sys.argv[5])
    stage_num = int(sys.argv[6])

    # Get pressures from output
    pressures = parse_pressures_from_output(run_out)
    if len(pressures) < 3:
        sys.exit(0)  # Not enough data, skip

    last3 = [pr[1] for pr in pressures[-3:]]
    max_change = max(abs(last3[i] - last3[j]) for i in range(3) for j in range(3))
    current_p = last3[-1]
    deviation = abs(current_p - target_p_bar)

    if max_change >= 500 or deviation <= 1000:
        sys.exit(0)  # Not stalled or within tolerance

    # Stalled and large deviation - compute scale factor from hydroopt
    p_lo = max(1, p_gpa - 1)
    hydro_p = Path(hydro_dir) / f"{p_gpa}gpa" / "final_cell.cell"
    hydro_plo = Path(hydro_dir) / f"{p_lo}gpa" / "final_cell.cell"
    if not hydro_p.exists() or not hydro_plo.exists():
        print(f"Warning: hydroopt cells not found for {p_gpa} and {p_lo} GPa", file=sys.stderr)
        sys.exit(0)

    v_p = read_cell_volume(hydro_p)
    v_plo = read_cell_volume(hydro_plo)
    if v_p is None or v_plo is None:
        sys.exit(0)
    dP = 10000  # 1 GPa = 10000 bar
    dV = v_plo - v_p  # higher P = smaller V, so v_plo > v_p
    if abs(dV) < 1e-10:
        sys.exit(0)
    # When V increases by (v_plo - v_p), P decreases by 10000. So dP/dV = -10000/(v_plo - v_p).
    # dV_needed = dP_desired / (dP/dV) = (current - target) * (v_plo - v_p) / 10000
    dP_dV = dP / dV  # 10000/(v_plo - v_p) > 0; used as |dP|/|dV| for magnitude

    # Undershot (current < target): need higher P -> smaller V -> dV_needed < 0, scale < 1
    # Overshot (current > target): need lower P -> larger V -> dV_needed > 0, scale > 1
    # Aim to overshoot: go past target by the same deviation (e.g. 2500 bar undershot -> aim 2500 bar overshot)
    dP_desired = current_p - target_p_bar  # positive when overshot, negative when undershot
    dV_needed = 2 * dP_desired / dP_dV  # double the correction to overshoot by same deviation
    v_current = read_cell_volume(cell_file)
    if v_current is None or v_current < 1e-10:
        sys.exit(0)
    v_new = v_current + dV_needed
    if v_new <= 0:
        sys.exit(0)
    scale = (v_new / v_current) ** (1.0 / 3.0)
    # Limit scale to reasonable range
    scale = max(0.9, min(1.1, scale))

    # Sanity: undershot -> scale < 1; overshot -> scale > 1
    if (dP_desired < 0 and scale > 1.001) or (dP_desired > 0 and scale < 0.999):
        print(f"Warning: scale direction inconsistent (dP_desired={dP_desired:.0f}, scale={scale:.4f})", file=sys.stderr)
        sys.exit(0)

    # Apply isotropic scaling to cell
    if np is None:
        sys.exit(0)
    rows = []
    for line in cell_file.read_text(errors='replace').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        nums = []
        prefix = []
        for x in parts:
            try:
                nums.append(float(x))
            except ValueError:
                prefix.append(x)
        if len(nums) >= 3:
            scaled = [nums[-3] * scale, nums[-2] * scale, nums[-1] * scale]
            if prefix and prefix[0].upper() in ('A', 'B', 'C'):
                rows.append((prefix[0], scaled))
            else:
                rows.append((None, scaled))
    if len(rows) < 3:
        sys.exit(0)
    with open(cell_file, 'w') as f:
        for name, vec in rows[:3]:
            if name:
                f.write(f"{name:<3} {vec[0]:18.15f} {vec[1]:18.15f} {vec[2]:18.15f}\n")
            else:
                f.write(f"     {vec[0]:18.15f} {vec[1]:18.15f} {vec[2]:18.15f}\n")
    print(f"  [Stall correction] Applied isotropic scale {scale:.6f} (deviation {deviation:.0f} bar)")
    sys.exit(1)  # Signal that correction was applied


if __name__ == '__main__':
    main()
