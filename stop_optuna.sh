#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stop_optuna.sh — Stop all running Optuna workers launched by
#                  run_optuna_parallel.sh
#
# Usage:
#   bash stop_optuna.sh           # graceful SIGTERM, then SIGKILL after 15 s
#   bash stop_optuna.sh --force   # immediate SIGKILL
# ---------------------------------------------------------------------------
set -euo pipefail

PID_FILE="logs/optuna_workers.pid"
FORCE="${1:-}"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found at $PID_FILE — nothing to stop."
    exit 0
fi

mapfile -t PIDS < "$PID_FILE"

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "PID file is empty — nothing to stop."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Stopping ${#PIDS[@]} Optuna worker(s) ..."

# Kill each worker's entire process group (negative PID = kill whole group).
# This catches any subprocesses (e.g. torch dataloader workers) as well.
for PID in "${PIDS[@]}"; do
    if [[ -z "$PID" ]]; then continue; fi

    if [[ "$FORCE" == "--force" ]]; then
        echo "  SIGKILL -> process group of PID $PID"
        kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
    else
        echo "  SIGTERM -> process group of PID $PID"
        kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
    fi
done

if [[ "$FORCE" != "--force" ]]; then
    echo "Waiting up to 15 seconds for workers to exit gracefully ..."
    DEADLINE=$(( $(date +%s) + 15 ))

    ALL_GONE=0
    while [[ $(date +%s) -lt $DEADLINE ]]; do
        ALIVE=0
        for PID in "${PIDS[@]}"; do
            [[ -z "$PID" ]] && continue
            kill -0 "$PID" 2>/dev/null && ALIVE=1
        done
        if [[ $ALIVE -eq 0 ]]; then
            ALL_GONE=1
            break
        fi
        sleep 1
    done

    if [[ $ALL_GONE -eq 0 ]]; then
        echo "Some workers still alive — sending SIGKILL ..."
        for PID in "${PIDS[@]}"; do
            [[ -z "$PID" ]] && continue
            kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
        done
    fi
fi

rm -f "$PID_FILE"
echo "Done. All Optuna workers stopped."
