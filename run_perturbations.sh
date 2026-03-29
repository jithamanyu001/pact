#!/bin/bash
set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
NNODES=1
NPROC=4
LOG_DIR="logs/perturbation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "args/gsm_perturb_zero.yaml"
    "args/gsm_perturb_noise_0.01.yaml"
    "args/gsm_perturb_noise_0.05.yaml"
    "args/gsm_perturb_noise_0.1.yaml"
    "args/gsm_perturb_noise_0.5.yaml"
    "args/gsm_perturb_noise_1.0.yaml"
    "args/gsm_perturb_random.yaml"
    "args/gsm_perturb_swap.yaml"
)

# ─── Run ─────────────────────────────────────────────────────────────────────
TOTAL=${#CONFIGS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

echo "========================================="
echo " PaCT Perturbation Experiments"
echo " Logging to: $LOG_DIR"
echo " Total experiments: $TOTAL"
echo "========================================="

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME=$(basename "$CONFIG" .yaml)
    LOG_FILE="$LOG_DIR/${NAME}.log"
    IDX=$((i + 1))

    echo ""
    echo "-----------------------------------------"
    echo "[$IDX/$TOTAL] Running: $NAME"
    echo "  Config: $CONFIG"
    echo "  Log:    $LOG_FILE"
    echo "-----------------------------------------"

    START_TIME=$(date +%s)

    if torchrun --nnodes "$NNODES" --nproc_per_node "$NPROC" run.py "$CONFIG" \
        > "$LOG_FILE" 2>&1; then
        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        echo "  [PASS] $NAME completed in ${ELAPSED}s"
        PASSED=$((PASSED + 1))
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        echo "  [FAIL] $NAME failed after ${ELAPSED}s (exit code: $EXIT_CODE)"
        echo "  Check log: $LOG_FILE"
        echo "  Last 5 lines:"
        tail -5 "$LOG_FILE" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$NAME")
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo " Summary"
echo "========================================="
echo " Total:  $TOTAL"
echo " Passed: $PASSED"
echo " Failed: $FAILED"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo " Failed experiments:"
    for f in "${FAILED_LIST[@]}"; do
        echo "   - $f"
    done
fi
echo " Logs:   $LOG_DIR/"
echo "========================================="

exit $FAILED
