#!/bin/bash
# run_kdm.sh
# KDM Trading System — One command launcher
# Usage: ./run_kdm.sh           (defaults to BTC)
#        ./run_kdm.sh GOLD
#        ./run_kdm.sh NASDAQ

ASSET=${1:-BTC}

echo ""
echo "╔══════════════════════════════════════╗"
echo "║       KDM TRADING SYSTEM             ║"
echo "║       Asset: $ASSET                  ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Install it first."
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install ccxt pandas scikit-learn joblib yfinance --quiet

# Start dashboard in background
echo "🌐 Starting dashboard on http://localhost:5050 ..."
python3 dashboard.py &
DASHBOARD_PID=$!

sleep 1

# Open browser automatically
open http://localhost:5050 2>/dev/null || xdg-open http://localhost:5050 2>/dev/null

# Start bot
echo "🤖 Starting KDM bot for $ASSET..."
echo "   Press Ctrl+C to stop"
echo ""
python3 main.py $ASSET

# Cleanup on exit
kill $DASHBOARD_PID 2>/dev/null
echo "Bot stopped."