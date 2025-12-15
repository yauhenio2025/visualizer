#!/bin/bash
# Start the visualizer notification listener

export VISUALIZER_NTFY_TOPIC="${VISUALIZER_NTFY_TOPIC:-visualizer-${USER}-$(echo $USER | md5sum | cut -c1-4)}"

PID_FILE="/tmp/visualizer_listener.pid"

if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Listener already running (PID: $pid)"
        exit 0
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nohup "$SCRIPT_DIR/visualizer-listener.sh" > /dev/null 2>&1 &
echo "Visualizer listener started (topic: $VISUALIZER_NTFY_TOPIC)"
