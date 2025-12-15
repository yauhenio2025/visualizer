#!/bin/bash
# Visualizer MCP Server Installer
# Sets up the MCP server for Claude Code
#
# Prerequisites:
#   - Python 3.10+
#   - Claude Code CLI installed
#   - Visualizer API running (http://localhost:5847)
#   - Analyzer API running (http://localhost:8847)
#
# Usage: ./install-mcp-server.sh

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Visualizer MCP Server Installer                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$INSTALL_DIR")"
NTFY_TOPIC="visualizer-${USER}-$(echo $USER | md5sum | cut -c1-4)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }
info() { echo -e "${BLUE}ℹ${NC} $1"; }

# Check prerequisites
echo "Checking prerequisites..."

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    success "Python found: $PYTHON_VERSION"
else
    error "Python 3 not found. Please install Python 3.10+"
fi

# Claude CLI
if command -v claude &> /dev/null; then
    success "Claude CLI found"
else
    error "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
fi

echo ""

# Set up Python virtual environment
echo "Setting up Python environment..."
cd "$INSTALL_DIR"

if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    success "Created virtual environment"
else
    success "Virtual environment exists"
fi

# Activate and install dependencies
source venv/bin/activate
pip install --quiet --upgrade pip

# Install MCP server dependencies
pip install --quiet fastmcp requests pydantic

success "Dependencies installed"

echo ""

# Create environment file for MCP server
echo "Configuring environment..."
ENV_FILE="$INSTALL_DIR/.env.mcp"

cat > "$ENV_FILE" << EOF
# Visualizer MCP Server Configuration
VISUALIZER_API_URL=https://visualizer-tw4i.onrender.com
ANALYZER_API_URL=https://analyzer-3wsg.onrender.com
VISUALIZER_NTFY_TOPIC=$NTFY_TOPIC
VISUALIZER_OUTPUT_DIR=~/visualizer-results

# Optional: Set default API keys (or provide per-request)
# GEMINI_API_KEY=your_gemini_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
EOF

success "Created $ENV_FILE"

# Create wrapper script that loads env and runs MCP server
WRAPPER_SCRIPT="$INSTALL_DIR/run-mcp-server.sh"

cat > "$WRAPPER_SCRIPT" << 'EOF'
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
EOF

chmod +x "$WRAPPER_SCRIPT"
success "Created wrapper script"

echo ""

# Register MCP server with Claude Code
echo "Registering MCP server with Claude Code..."

# Check if already registered
if claude mcp list 2>/dev/null | grep -q "visualizer"; then
    info "MCP server 'visualizer' already registered"
    echo "Updating registration..."
    claude mcp remove visualizer 2>/dev/null || true
fi

# Register the MCP server
claude mcp add visualizer \
    --transport stdio \
    -- "$WRAPPER_SCRIPT"

success "MCP server registered"

echo ""

# Install notification listener
echo "Installing notification listener..."
LISTENER_SCRIPT="$INSTALL_DIR/visualizer-listener.sh"
HOOKS_DIR="$HOME/.claude/hooks/active"

mkdir -p "$HOOKS_DIR"

# Copy listener to hooks directory
cp "$LISTENER_SCRIPT" "$HOOKS_DIR/"
chmod +x "$HOOKS_DIR/visualizer-listener.sh"

success "Notification listener installed"

# Add hook to start listener on session start (if not already present)
SETTINGS_FILE="$HOME/.claude/settings.json"

if [[ -f "$SETTINGS_FILE" ]]; then
    if ! grep -q "visualizer-listener" "$SETTINGS_FILE" 2>/dev/null; then
        info "To auto-start the listener, add this to your SessionStart hooks in $SETTINGS_FILE:"
        echo ""
        echo "  {\"type\": \"command\", \"command\": \"$HOOKS_DIR/visualizer-listener.sh\"}"
        echo ""
    else
        success "Listener hook already configured"
    fi
fi

echo ""

# Create start listener script
START_LISTENER="$INSTALL_DIR/start-listener.sh"

cat > "$START_LISTENER" << 'EOFSTART'
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
EOFSTART

chmod +x "$START_LISTENER"
success "Created start-listener.sh"

echo ""

# Verify installation
echo "Verifying installation..."

# Test that the MCP server can start
info "Testing MCP server startup..."
timeout 5 "$WRAPPER_SCRIPT" --help 2>/dev/null && success "MCP server starts correctly" || warn "Could not verify MCP server"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  MCP Server:     $WRAPPER_SCRIPT"
echo "  Config:         $ENV_FILE"
echo "  Install Dir:    $INSTALL_DIR"
echo "  Notification:   $NTFY_TOPIC"
echo ""
echo "  The MCP server is now available in Claude Code."
echo "  Start a new Claude Code session to use it."
echo ""
echo "  Available tools:"
echo "    - get_ai_recommendations    Get AI-powered engine suggestions"
echo "    - list_available_engines    Browse all analysis engines"
echo "    - submit_analysis           Start analysis with selected engines"
echo "    - check_job_status          Monitor job progress"
echo "    - get_results               Download completed results"
echo "    - list_bundles              View engine bundles"
echo "    - list_pipelines            View analysis pipelines"
echo ""
echo "  Quick start:"
echo "    1. Start the notification listener:"
echo "       $START_LISTENER"
echo "    2. Open Claude Code and try:"
echo "       'Get AI recommendations for document.pdf'"
echo ""
echo "  Services (deployed on Render - always available):"
echo "    - Visualizer: https://visualizer-tw4i.onrender.com"
echo "    - Analyzer: https://analyzer-3wsg.onrender.com"
echo ""
echo "  Test with: claude"
echo ""
