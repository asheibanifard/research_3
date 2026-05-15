#!/bin/bash
# Evaluate 3DGS with MIP rendering for run_fewest_ellipsoids

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

RUN_DIR="$ROOT/logs/3dgs/run15/20260515_100118/"
CKPT="$ROOT/logs/3dgs/run15/20260515_100118/best_20260515_100118.pth
"
OUT_PREFIX="$ROOT/results/run15/run15"

# Create output directory
mkdir -p "$ROOT/results/run13"

echo "=== Evaluating 3DGS: run_fewest_ellipsoids ==="
echo "Checkpoint: $CKPT"
echo "Output prefix: $OUT_PREFIX"
echo ""

# Run evaluation with MIP rendering and save reconstructions
python "$SCRIPT_DIR/3dgs_eval.py" \
    --run_dir "$RUN_DIR" \
    --ckpt "$CKPT" \
    --out "${OUT_PREFIX}_recon.tif" \
    --mip \
    --mip_out "$OUT_PREFIX" \
    --plot "$OUT_PREFIX"

echo ""
echo "=== Evaluation Complete ==="
echo "Outputs saved to results/ directory:"
echo "  - ${OUT_PREFIX}_recon.tif (full 3D reconstruction)"
echo "  - ${OUT_PREFIX}_mip_xy.tif (Z-axis MIP projection)"
echo "  - ${OUT_PREFIX}_mip_xz.tif (Y-axis MIP projection)"  
echo "  - ${OUT_PREFIX}_mip_yz.tif (X-axis MIP projection)"
echo "  - ${OUT_PREFIX}_z.png, _y.png, _x.png (slice visualizations)"
echo "  - ${OUT_PREFIX}_mip_xy.png, _mip_xz.png, _mip_yz.png (MIP visualizations)"
