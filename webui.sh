#!/usr/bin/env bash
#
# evo2Mac Web UI control script.
#
#   ./webui.sh start      # launch the Gradio app in the background
#   ./webui.sh stop       # stop it
#   ./webui.sh restart    # stop then start
#   ./webui.sh status      # is it running? on what URL?
#   ./webui.sh logs       # tail the server log (Ctrl-C to stop tailing)
#
# It runs the server detached, tracks it with a PID file, waits until the
# server is actually serving before returning, and installs gradio into the
# conda env if it's missing. Override any of these via env vars:
#
#   EVO2MAC_ENV   conda env name      (default: evo2Mac)
#   EVO2MAC_HOST  bind address        (default: 127.0.0.1)
#   EVO2MAC_PORT  port                (default: 7860)
#   EVO2MAC_SHARE 1 = public share    (default: 0)
#
# Examples:
#   EVO2MAC_PORT=8000 ./webui.sh start
#   ./webui.sh restart

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${EVO2MAC_ENV:-evo2Mac}"
HOST="${EVO2MAC_HOST:-127.0.0.1}"
PORT="${EVO2MAC_PORT:-7860}"
SHARE="${EVO2MAC_SHARE:-0}"

PID_FILE="$REPO_ROOT/.webui.pid"
LOG_FILE="$REPO_ROOT/webui.log"
URL="http://${HOST}:${PORT}"

log()  { printf "\033[1;34m[webui]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[webui]\033[0m %s\n" "$*" >&2; }

# --- conda -------------------------------------------------------------------

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    echo "conda"; return 0
  fi
  for cand in \
    "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
    "$HOME/miniforge3/bin/conda" \
    "/opt/miniforge3/bin/conda"; do
    [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  done
  return 1
}

CONDA="$(find_conda)" || { err "conda not found. Run ./install.sh first."; exit 1; }

# Verify the env exists.
if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  err "conda env '$ENV_NAME' not found. Run ./install.sh first."
  exit 1
fi

run_in_env() { "$CONDA" run -n "$ENV_NAME" "$@"; }

# --- helpers -----------------------------------------------------------------

# True if a recorded PID is alive.
is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid; pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# True once the server is accepting connections on the port. Uses bash's
# /dev/tcp built-in — no python/conda dependency (conda run does not reliably
# propagate the inner exit code, which would make this always look "open").
port_open() {
  (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null
}

ensure_gradio() {
  if run_in_env python -c "import gradio" >/dev/null 2>&1; then
    return 0
  fi
  log "gradio not installed in '$ENV_NAME' — installing..."
  run_in_env pip install "gradio>=4.0,<6" >/dev/null
  log "gradio installed."
}

# --- commands ----------------------------------------------------------------

cmd_start() {
  if is_running; then
    log "already running (pid $(cat "$PID_FILE")) at $URL"
    return 0
  fi
  # Stale PID file or someone else on the port?
  if port_open; then
    err "port $PORT is already in use (not by us). Set EVO2MAC_PORT or free it."
    exit 1
  fi

  ensure_gradio

  log "starting web UI on $URL (env: $ENV_NAME)..."
  # inbrowser is handled by webapp.py; we just background it and capture logs.
  EVO2MAC_HOST="$HOST" EVO2MAC_PORT="$PORT" EVO2MAC_SHARE="$SHARE" \
    nohup "$CONDA" run -n "$ENV_NAME" python -u "$REPO_ROOT/webapp.py" \
    >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"

  # Wait up to ~40s for the server to come up.
  for _ in $(seq 1 80); do
    if ! is_running; then
      err "server exited during startup. Last log lines:"
      tail -n 15 "$LOG_FILE" >&2
      rm -f "$PID_FILE"
      exit 1
    fi
    if port_open; then
      log "ready -> $URL  (pid $(cat "$PID_FILE"))"
      log "logs: ./webui.sh logs    stop: ./webui.sh stop"
      return 0
    fi
    sleep 0.5
  done

  err "timed out waiting for the server. Last log lines:"
  tail -n 15 "$LOG_FILE" >&2
  exit 1
}

cmd_stop() {
  if ! is_running; then
    log "not running."
    rm -f "$PID_FILE"
    return 0
  fi
  local pid; pid="$(cat "$PID_FILE")"
  log "stopping (pid $pid)..."
  # Kill the process group so the conda-run wrapper and its python child die too.
  kill "$pid" 2>/dev/null || true
  pkill -P "$pid" 2>/dev/null || true
  # Belt-and-suspenders: kill any webapp.py still bound, in case of a stale tree.
  pkill -f "python -u $REPO_ROOT/webapp.py" 2>/dev/null || true
  for _ in $(seq 1 20); do
    is_running || break
    sleep 0.25
  done
  if is_running; then
    err "did not stop gracefully; sending SIGKILL."
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  log "stopped."
}

cmd_status() {
  if is_running; then
    local pid; pid="$(cat "$PID_FILE")"
    if port_open; then
      log "running (pid $pid) and serving at $URL"
    else
      log "process alive (pid $pid) but not yet serving on $PORT — still starting?"
    fi
  else
    log "not running."
    [[ -f "$LOG_FILE" ]] && log "last log: $LOG_FILE"
  fi
}

cmd_logs() {
  [[ -f "$LOG_FILE" ]] || { err "no log file yet ($LOG_FILE). Start it first."; exit 1; }
  log "tailing $LOG_FILE (Ctrl-C to stop)..."
  tail -n 40 -f "$LOG_FILE"
}

usage() {
  cat <<EOF
evo2Mac Web UI control

Usage: ./webui.sh <command>

  start     launch the Gradio app in the background (opens your browser)
  stop      stop the running app
  restart   stop then start
  status    show whether it's running and the URL
  logs      tail the server log

Env overrides: EVO2MAC_ENV, EVO2MAC_HOST, EVO2MAC_PORT, EVO2MAC_SHARE
Current:       env=$ENV_NAME host=$HOST port=$PORT share=$SHARE
URL:           $URL
EOF
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  ""|-h|--help|help) usage ;;
  *) err "unknown command: $1"; echo; usage; exit 1 ;;
esac
