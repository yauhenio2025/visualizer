#!/bin/bash
# Package the Visualizer MCP Server for distribution
# Creates a zip file with all necessary files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$HOME"  # Output to home directory for easy sharing
PACKAGE_NAME="visualizer-mcp-server"

echo "Packaging Visualizer MCP Server..."
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="$TEMP_DIR/$PACKAGE_NAME"
mkdir -p "$PACKAGE_DIR"

# Copy files (complete package with installer and notifications)
echo "Copying files..."
cp "$SCRIPT_DIR/mcp_server.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/job_poller.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/run-mcp-server.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/.env.mcp.template" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/.env.mcp" "$PACKAGE_DIR/" 2>/dev/null || true  # Include actual keys if present
cp "$SCRIPT_DIR/README.md" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/install-mcp-server.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/visualizer-listener.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/start-listener.sh" "$PACKAGE_DIR/"

# Make scripts executable
chmod +x "$PACKAGE_DIR/run-mcp-server.sh"
chmod +x "$PACKAGE_DIR/job_poller.py"
chmod +x "$PACKAGE_DIR/install-mcp-server.sh"
chmod +x "$PACKAGE_DIR/visualizer-listener.sh"
chmod +x "$PACKAGE_DIR/start-listener.sh"

# Create zip
ZIP_FILE="$OUTPUT_DIR/${PACKAGE_NAME}.zip"
rm -f "$ZIP_FILE"

cd "$TEMP_DIR"
zip -r "$ZIP_FILE" "$PACKAGE_NAME"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "Package created: $ZIP_FILE"
echo ""
echo "Contents:"
unzip -l "$ZIP_FILE"
echo ""
echo "Size: $(du -h "$ZIP_FILE" | cut -f1)"
echo ""
echo "Share this file with your friend!"
