#!/bin/bash

# start.sh — Detached training launcher for Whisper ASR fine-tuning
#
# Usage:
#   ./start.sh          Start training in the background
#   ./start.sh start    Same as above
#   ./start.sh stop     Kill the running training process
#   ./start.sh status   Show whether training is running
#   ./start.sh logs     Tail the live training log (Ctrl+C to detach)

set -e

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/training.log"
PID_FILE="$SCRIPT_DIR/.training.pid"
ACCELERATE_CONFIG="$SCRIPT_DIR/accelerate_config.yaml"

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
step()    { echo -e "${BLUE}==>${NC} ${GREEN}$1${NC}"; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_check_pid() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    return 1
}

_require_env() {
    local missing=0
    for var in HF_TOKEN WANDB_API_KEY; do
        if [ -z "${!var}" ]; then
            # Try loading from .env as a fallback
            if [ -f "$SCRIPT_DIR/.env" ]; then
                # shellcheck disable=SC1090
                set -a; source "$SCRIPT_DIR/.env"; set +a
            fi
        fi
        if [ -z "${!var}" ]; then
            error "$var is not set. Export it or add it to .env"
            missing=1
        fi
    done
    [ "$missing" -eq 0 ] || exit 1
}

_require_venv() {
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        error "Virtual environment not found at $VENV_DIR"
        error "Run ./setup.sh first to create it."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_start() {
    # Guard: already running?
    if pid=$(_check_pid); then
        warn "Training is already running (PID $pid)."
        warn "Use './start.sh logs' to follow it, or './start.sh stop' to kill it."
        exit 0
    fi

    step "Validating prerequisites..."
    _require_env
    _require_venv

    mkdir -p "$LOG_DIR"

    step "Activating virtual environment..."
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    success "venv: $(which python) — $(python --version 2>&1)"

    step "Configuring multi-GPU environment..."
    export PYTHONUNBUFFERED=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # Surface NCCL errors quickly
    export NCCL_TIMEOUT=1800             # 30-min timeout for checkpoint saves
    export TORCH_NCCL_BLOCKING_WAIT=0   # Non-blocking collectives

    if [ ! -f "$ACCELERATE_CONFIG" ]; then
        error "Accelerate config not found: $ACCELERATE_CONFIG"
        exit 1
    fi

    step "Launching training (detached)..."
    info  "Log file         : $LOG_FILE"
    info  "PID file         : $PID_FILE"
    info  "Accelerate config: $ACCELERATE_CONFIG"
    info  "Command          : accelerate launch --config_file $ACCELERATE_CONFIG main.py"
    echo  ""

    # Launch detached; nohup keeps the process alive after terminal close
    nohup accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        main.py \
        >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    success "Training started in the background (PID $(cat "$PID_FILE"))"
    echo ""
    info "Tail logs with:  ./start.sh logs"
    info "Check status:    ./start.sh status"
    info "Stop training:   ./start.sh stop"
}

cmd_stop() {
    if pid=$(_check_pid); then
        step "Stopping training (PID $pid)..."
        # Kill the entire process group so all torchrun worker processes die too
        kill -- "-$pid" 2>/dev/null || kill "$pid"
        rm -f "$PID_FILE"
        success "Training stopped."
    else
        warn "No running training process found."
    fi
}

cmd_status() {
    if pid=$(_check_pid); then
        success "Training is RUNNING (PID $pid)"
        echo ""
        info "Process tree:"
        ps --ppid "$pid" -o pid,pcpu,pmem,stat,cmd 2>/dev/null || true
        echo ""
        if [ -f "$LOG_FILE" ]; then
            info "Last 5 log lines:"
            tail -n 5 "$LOG_FILE"
        fi
    else
        warn "Training is NOT running."
        if [ -f "$LOG_FILE" ]; then
            info "Last log entry:"
            tail -n 3 "$LOG_FILE"
        fi
    fi
}

cmd_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        error "Log file not found: $LOG_FILE"
        error "Start training first with: ./start.sh"
        exit 1
    fi
    info "Tailing $LOG_FILE  (Ctrl+C to detach)"
    echo ""
    tail -f "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "${1:-start}" in
    start)   cmd_start  ;;
    stop)    cmd_stop   ;;
    status)  cmd_status ;;
    logs)    cmd_logs   ;;
    *)
        echo "Usage: $0 {start|stop|status|logs}"
        exit 1
        ;;
esac
