#!/bin/bash

# 1. Find the latest log file
FILE=$(ls QM_cellopt-optimized_cell.dat-1_*.Log 2>/dev/null | sort -V | tail -n 1)

if [ -z "$FILE" ]; then
    echo "Error: No matching log file found."
    exit 1
fi

echo "Extracting high-precision data from: $FILE"

# 2. Extract raw values (last occurrence)
A=$(grep "|a| =" "$FILE" | tail -n 1 | awk '{print $NF}')
B=$(grep "|b| =" "$FILE" | tail -n 1 | awk '{print $NF}')
C=$(grep "|c| =" "$FILE" | tail -n 1 | awk '{print $NF}')

ALPHA=$(grep "alpha \[degree\]:" "$FILE" | tail -n 1 | awk '{print $NF}')
BETA=$(grep "beta  \[degree\]:" "$FILE" | tail -n 1 | awk '{print $NF}')
GAMMA=$(grep "gamma \[degree\]:" "$FILE" | tail -n 1 | awk '{print $NF}')

# 3. Calculate Vectors using bc (Vector A on X-axis, B in XY-plane)
# Note: s() is sine, c() is cosine. Input must be in Radians.
VECTORS=$(bc -l << EOF
scale=15
p=4*a(1)  # Pi
ra = $ALPHA * p / 180
rb = $BETA * p / 180
rg = $GAMMA * p / 180

# Vector A components
ax = $A; ay = 0; az = 0

# Vector B components
bx = $B * c(rg); by = $B * s(rg); bz = 0

# Vector C components
cx = $C * c(rb)
cy = $C * (c(ra) - (c(rb) * c(rg))) / s(rg)
cz = sqrt($C^2 - cx^2 - cy^2)

print "A ", ax, " ", ay, " ", az, "\n"
print "B ", bx, " ", by, " ", bz, "\n"
print "C ", cx, " ", cy, " ", cz, "\n"
EOF
)

# 4. Format and Save
echo "$VECTORS" | awk '{printf "%-3s %15.10f %15.10f %15.10f\n", $1, $2, $3, $4}' > init_cell.cell

echo "Done! High-precision cell saved to init_cell.cell."
