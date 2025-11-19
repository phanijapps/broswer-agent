#!/bin/bash
set -e

echo "=== Starting Services ==="

# Start Xvnc server directly
echo "Starting Xvnc..."
Xvnc :1 -screen 0 1920x1080x24 -rfbport 5901 -SecurityTypes None &
XVNC_PID=$!

sleep 3

# Start window manager
echo "Starting XFCE..."
export DISPLAY=:1
startxfce4 &

sleep 3

# Start noVNC websockify proxy
echo "Starting noVNC..."
websockify --web=/usr/share/novnc 6901 localhost:5901 &
WEBSOCK_PID=$!

sleep 5

echo "VNC services started"

# Start Playwright MCP SSE server
echo "Starting Playwright MCP on http://0.0.0.0:8931"
cd /home/headless
export HOME=/home/headless
export USER=headless
export DISPLAY=:1
export npm_config_cache=/home/headless/.npm

npx -y @playwright/mcp@latest \
  --port=8931 \
  --host=0.0.0.0 \
  --user-data-dir=/home/headless/.playwright-profile \
  --output-dir=/home/headless/screenshots \
  --shared-browser-context \
  --no-sandbox &

MCP_PID=$!

echo ""
echo "========================================"
echo "All services running!"
echo "  VNC:     http://localhost:6901"
echo "  MCP:     http://localhost:8931"
echo "  Display: :1"
echo "========================================"
echo ""

# Keep container running
tail -f /dev/null
