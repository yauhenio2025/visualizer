#!/bin/bash
# Wrapper script to run MCP server with environment loaded
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment
if [[ -f "$SCRIPT_DIR/.env.mcp" ]]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env.mcp" | xargs)
fi

# Activate venv and run
source "$SCRIPT_DIR/venv/bin/activate"
exec python "$SCRIPT_DIR/mcp_server.py" "$@"
