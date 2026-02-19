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
#
# Workers run fully detached (survives terminal close).
# To stop all workers:  bash stop_optuna.sh
# To monitor progress:  tail -f logs/optuna_worker_gpu0.log
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

# Ensure logs directory exists
mkdir -p logs

# Remove stale journal and PID file so we start a fresh study
rm -f logs/optuna_journal.log logs/optuna_workers.pid

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching worker on GPU $GPU_ID ..."

    # setsid creates a new process group so the worker is fully independent.
    # nohup keeps it alive after the terminal closes.
    # The PGID equals the PID of the setsid child, so we can kill -PGID later.
    setsid nohup bash -c "
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        export OPTUNA_TRIALS_PER_WORKER=${TRIALS_PER_WORKER}
        exec python main.py
    " >> "logs/optuna_worker_gpu${GPU_ID}.log" 2>&1 &

    echo $! >> logs/optuna_workers.pid
done

# Detach all launched jobs from this shell so closing the terminal won't
# send SIGHUP to the workers.
disown -a

echo "============================================================"
echo "All $NUM_GPUS worker(s) launched in the background."
echo ""
echo "  PID file : logs/optuna_workers.pid"
echo "  Logs     : logs/optuna_worker_gpu<N>.log"
echo ""
echo "  Monitor  : tail -f logs/optuna_worker_gpu0.log"
echo "  Status   : ps -p \$(paste -sd, logs/optuna_workers.pid)"
echo "  Stop all : bash stop_optuna.sh"
echo "============================================================"
