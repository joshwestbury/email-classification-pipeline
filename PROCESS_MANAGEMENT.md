# Pipeline Process Management

This document describes the process management and cleanup mechanisms implemented for the email taxonomy pipeline.

## 🚀 Enhanced Pipeline Features

### Automatic Process Cleanup

The pipeline now includes several mechanisms to ensure proper process termination:

1. **Signal Handlers**: Both `run_pipeline.py` and `pipeline/pipeline.py` now handle SIGINT (Ctrl+C) and SIGTERM signals gracefully
2. **Automatic Exit**: Pipeline processes automatically terminate when complete
3. **State Saving**: Partial results are saved if the pipeline is interrupted
4. **Resource Cleanup**: Memory and file handles are properly released

### Signal Handling

```python
# The pipeline now handles these signals:
SIGINT   # Ctrl+C - Graceful shutdown
SIGTERM  # System termination - Graceful shutdown
```

## 🛠️ Process Management Tools

### 1. Pipeline Monitor (`pipeline_monitor.py`)

A comprehensive tool for monitoring and managing pipeline processes:

```bash
# List all running pipeline processes
uv run python pipeline_monitor.py --list

# Kill all pipeline processes (graceful)
uv run python pipeline_monitor.py --kill-all

# Force kill all pipeline processes
uv run python pipeline_monitor.py --kill-all --force

# Monitor processes continuously
uv run python pipeline_monitor.py --monitor

# Clean up old output directories
uv run python pipeline_monitor.py --cleanup-outputs
```

### 2. Quick Kill Script (`kill_pipelines.sh`)

Simple bash script for immediate process termination:

```bash
# Make executable (first time only)
chmod +x kill_pipelines.sh

# Kill all pipeline processes
./kill_pipelines.sh
```

## 🔧 Implementation Details

### Enhanced `run_pipeline.py`

- Added signal handlers for SIGINT and SIGTERM
- Proper exception handling with specific exit codes
- Cleanup function registered with `atexit`
- Forces clean exit with `sys.exit(0)`

### Enhanced `pipeline/pipeline.py`

- Signal handlers at the class level
- State saving functionality for interrupted pipelines
- Resource cleanup in `_cleanup()` method
- Explicit process termination after completion

### Exit Codes

The pipeline now uses standard exit codes:

- `0`: Success
- `1`: General error
- `130`: Interrupted by SIGINT (Ctrl+C)

## 🚨 Troubleshooting

### Stale System Reminders

If you see system reminders about running processes but `pipeline_monitor.py --list` shows none, the reminders are stale and can be ignored.

### Force Killing Processes

If processes don't respond to graceful termination:

```bash
# Use the force option
uv run python pipeline_monitor.py --kill-all --force

# Or use the kill script which tries both methods
./kill_pipelines.sh
```

### Checking Process Status

```bash
# Manual check for pipeline processes
ps aux | grep run_pipeline.py

# Using the monitor tool
uv run python pipeline_monitor.py --list
```

## 📋 Best Practices

1. **Always use the provided tools** rather than manual `kill` commands
2. **Check for running processes** before starting new pipelines
3. **Use the monitor tool** for long-running pipeline supervision
4. **Clean up old outputs** regularly using the cleanup feature

## 🔄 Future Improvements

Potential enhancements for process management:

- Process heartbeat monitoring
- Automatic restart on failure
- Process queuing system
- Resource usage alerts
- Integration with system service managers