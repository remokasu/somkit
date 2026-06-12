#!/bin/bash
# Regenerate SOM_PAK golden files for SPEC-0001 FR-8 layer-B conformance tests.
#
# Runs the SOM_PAK 3.1 binaries (Kohonen, Helsinki University of Technology) on
# ex.dat to produce reference codebooks (.cod) and the quantization error. These
# outputs are committed under test/golden/ so CI compares against them without a
# C toolchain (ADR-0001 golden-file strategy).
#
# Parameters follow SOM_PAK's command.sh (hexa / bubble, randinit seed 123,
# vsom data-order seed 1), with the second phase shortened to rlen=2000 to keep
# CI light (the numerical match was verified to hold; see SPEC-0001 FR-8).
#
# Usage: SP=/path/to/SOM_PAK bash test/golden/regenerate.sh
set -eu

SP="${SP:-$HOME/remokasu/SOM_PAK}"
HERE="$(cd "$(dirname "$0")" && pwd)"

XDIM=10
YDIM=10
TOPOL=hexa
NEIGH=bubble
INIT_SEED=123     # randinit reference-vector RNG seed
ORDER_SEED=1      # vsom data presentation-order RNG seed

cp "$SP/ex.dat" "$HERE/ex.dat"

"$SP/randinit" -din "$HERE/ex.dat" -cout "$HERE/ex_init.cod" \
    -xdim "$XDIM" -ydim "$YDIM" -topol "$TOPOL" -neigh "$NEIGH" -rand "$INIT_SEED"

# Phase 1 (coarse ordering): start from the random init.
cp "$HERE/ex_init.cod" "$HERE/ex_phase1.cod"
"$SP/vsom" -din "$HERE/ex.dat" -cin "$HERE/ex_phase1.cod" -cout "$HERE/ex_phase1.cod" \
    -rlen 1000 -alpha 0.05 -radius 10 -rand "$ORDER_SEED"

# Phase 2 (fine tuning): continue from phase 1.
cp "$HERE/ex_phase1.cod" "$HERE/ex_trained.cod"
"$SP/vsom" -din "$HERE/ex.dat" -cin "$HERE/ex_trained.cod" -cout "$HERE/ex_trained.cod" \
    -rlen 2000 -alpha 0.02 -radius 3 -rand "$ORDER_SEED"

"$SP/qerror" -din "$HERE/ex.dat" -cin "$HERE/ex_trained.cod" > "$HERE/qerror.txt"

# --- vcal label calibration golden (SPEC-0002 FR-5) ---
# animal.dat is labelled (16 dims, 16 samples); train then calibrate labels.
ANIMAL_XDIM=5
ANIMAL_YDIM=5
"$SP/randinit" -din "$HERE/animal.dat" -cout "$HERE/animal.cod" \
    -xdim "$ANIMAL_XDIM" -ydim "$ANIMAL_YDIM" -topol hexa -neigh bubble -rand "$INIT_SEED"
"$SP/vsom" -din "$HERE/animal.dat" -cin "$HERE/animal.cod" -cout "$HERE/animal.cod" \
    -rlen 1000 -alpha 0.05 -radius 5 -rand "$ORDER_SEED"
"$SP/vcal" -numlabs 1 -din "$HERE/animal.dat" -cin "$HERE/animal.cod" -cout "$HERE/animal_cal.cod"

# --- U-matrix golden (SPEC-0003): SOM_PAK umat reference values ---
# umat writes (int)(100 * uvalue[x][y]) per cell to the PS (umat.c:596) on the
# (2*xdim-1)x(2*ydim-1) interpolated grid, y-outer/x-inner, no swap. We extract
# those ints into a plain matrix (rows=y, cols=x) so compute_umatrix_pak can be
# compared C-independently. Covers animal.cod (5x5 -> 9x9) and ex_trained.cod
# (10x10 -> 19x19).
for base in animal ex_trained; do
    "$SP/umat" -cin "$HERE/$base.cod" -ps -o "/tmp/${base}_umat.ps"
    python3 - "$HERE" "$base" <<'PY'
import re, sys
here, base = sys.argv[1], sys.argv[2]
rows = []
for line in open(f"/tmp/{base}_umat.ps"):
    if line.startswith("XSH"):
        vals = re.findall(r"(\d+) H", line)
        if vals:
            rows.append(vals)
assert rows, f"{base}: no value rows parsed (SOM_PAK PS format changed?)"
w = max(len(r) for r in rows)
rows = [r for r in rows if len(r) == w]  # value rows only (w H-blocks)
# Guard against silently writing a broken golden if the PS format drifts:
# the interpolated grid is (2n-1) in each axis (odd), and square for n x n maps.
assert w % 2 == 1, f"{base}: parsed width {w} is not odd (expected 2*dim-1)"
assert len(rows) == w, f"{base}: got {len(rows)}x{w}, expected a square grid"
with open(f"{here}/{base}_umat_pak.txt", "w") as f:
    f.write(f"# SOM_PAK umat golden (int 100*uvalue, rows=y, cols=x) for {base}.cod\n")
    for r in rows:
        f.write(" ".join(r) + "\n")
PY
done

# --- visual golden (SPEC-0004 FR-2): per-sample BMU coords + qerror ---
# ex_trained.cod is unlabeled; animal_cal.cod is vcal-labeled so the .vis
# also exercises the BMU-label column.
"$SP/visual" -din "$HERE/ex.dat" -cin "$HERE/ex_trained.cod" -dout "$HERE/ex_trained.vis"
"$SP/visual" -din "$HERE/animal.dat" -cin "$HERE/animal_cal.cod" -dout "$HERE/animal_cal.vis"

# --- snapshot golden (SPEC-0004 FR-3): intermediate codebook at step 500 ---
# vsom saves when (le % interval == 0) && (le > 0), so rlen=1000/interval=500
# yields exactly one snapshot (ex_snap_00500.cod).
"$SP/vsom" -din "$HERE/ex.dat" -cin "$HERE/ex_init.cod" -cout /tmp/ex_snap_final.cod \
    -rlen 1000 -alpha 0.05 -radius 10 -rand "$ORDER_SEED" \
    -snapfile "$HERE/ex_snap_%05d.cod" -snapinterval 500

echo "Golden files regenerated in $HERE"
