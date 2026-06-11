#!/bin/bash
# Installs the StockSight SMS agent on macOS as a hidden, auto-starting LaunchAgent.
# Run ONCE:  bash scripts/install_sms_agent_mac.sh
# It starts at every login, runs with no window, and restarts itself if it dies.
#
# Uninstall:  launchctl unload ~/Library/LaunchAgents/com.stocksight.smsagent.plist
#             rm ~/Library/LaunchAgents/com.stocksight.smsagent.plist
set -e

# Repo = parent of this script's directory.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO/src/sms_agent.py"
LABEL="com.stocksight.smsagent"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

[ -f "$SCRIPT" ] || { echo "Cannot find $SCRIPT"; exit 1; }

# macOS Homebrew Python is externally managed (PEP 668), so a system-wide
# pip install fails. Use a repo-local virtualenv and point launchd at it.
VENV="$REPO/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating virtualenv at $VENV ..."
  python3 -m venv "$VENV" || { echo "Could not create venv"; exit 1; }
fi
PY="$VENV/bin/python"
echo "Installing dependencies into the venv ..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r "$REPO/requirements.txt" || { echo "pip install failed"; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$SCRIPT</string></array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$REPO/reports/sms_agent.mac.log</string>
  <key>StandardErrorPath</key><string>$REPO/reports/sms_agent.mac.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <!-- Leave Windows as the primary responder for untargeted texts so the two
         daemons do not both reply. This Mac answers only @mac-addressed texts.
         To make the Mac the primary instead, change the value below to mac. -->
    <key>STOCKSIGHT_PRIMARY</key><string>windows</string>
  </dict>
</dict></plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "Installed and loaded $LABEL."
echo "Runs hidden at every login. Log: $REPO/reports/sms_agent.mac.log"
echo "Stop:    launchctl unload $PLIST"
echo "Start:   launchctl load $PLIST"
