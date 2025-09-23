#!/bin/bash
# Simple script to kill all pipeline processes

echo "🔍 Searching for pipeline processes..."

# Find and kill all run_pipeline.py processes
pids=$(pgrep -f "run_pipeline.py")

if [ -z "$pids" ]; then
    echo "✅ No pipeline processes found."
else
    echo "❌ Found pipeline processes with PIDs: $pids"
    echo "🛑 Killing processes..."

    # Try graceful termination first
    for pid in $pids; do
        echo "  Sending SIGTERM to PID $pid..."
        kill -TERM "$pid" 2>/dev/null
    done

    # Wait a moment
    sleep 2

    # Check if any are still running and force kill
    remaining=$(pgrep -f "run_pipeline.py")
    if [ ! -z "$remaining" ]; then
        echo "🔥 Force killing remaining processes..."
        for pid in $remaining; do
            echo "  Sending SIGKILL to PID $pid..."
            kill -KILL "$pid" 2>/dev/null
        done
    fi

    echo "✅ All pipeline processes terminated."
fi

echo "🧹 Cleanup complete."