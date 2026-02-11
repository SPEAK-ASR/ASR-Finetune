#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_optuna_parallel.sh — Launch parallel Optuna workers (one per GPU)
#
# Usage:
#   bash run_optuna_parallel.sh          # auto-detect GPUs, 50 total trials
#   bash run_optuna_parallel.sh 100      # auto-detect GPUs, 100 total trials
#   bash run_optuna_parallel.sh 100 4    # force 4 workers, 100 total trials
#
# Each worker runs main.py with task=optuna_optimize, pinned to a single GPU
# via CUDA_VISIBLE_DEVICES.  All workers share the same Optuna study through
# a JournalFileStorage file (logs/optuna_journal.log).
# ---------------------------------------------------------------------------
set -euo pipefail

TOTAL_TRIALS="${1:-50}"
NUM_GPUS="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "ERROR: No GPUs detected. Pass the count manually: $0 <trials> <gpus>"
    exit 1
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
    echo "INFO: Only 1 GPU detected. Running all $TOTAL_TRIALS trials sequentially."
    echo "      (You can also run directly: python main.py)"
fi

TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + NUM_GPUS - 1) / NUM_GPUS ))  # ceil division

echo "============================================================"
echo "Optuna parallel launcher"
echo "  Total trials : $TOTAL_TRIALS"
echo "  GPUs         : $NUM_GPUS"
echo "  Trials/worker: $TRIALS_PER_WORKER"
echo "============================================================"

# Remove stale journal so we start a fresh study
rm -f logs/optuna_journal.log

PIDS=()

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching worker on GPU $GPU_ID ..."

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    OPTUNA_TRIALS_PER_WORKER="$TRIALS_PER_WORKER" \
        python main.py \
        > "logs/optuna_worker_gpu${GPU_ID}.log" 2>&1 &

    PIDS+=($!)
done

echo "All workers launched. PIDs: ${PIDS[*]}"
echo "Logs: logs/optuna_worker_gpu<N>.log"
echo "Waiting for all workers to finish ..."

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "Worker PID $PID exited with error"
        FAIL=1
    fi
done

if [[ "$FAIL" -eq 1 ]]; then
    echo "Some workers failed — check logs for details."
    exit 1
fi

echo "============================================================"
echo "All workers finished. Results: logs/optuna_results.json"
echo "============================================================"
