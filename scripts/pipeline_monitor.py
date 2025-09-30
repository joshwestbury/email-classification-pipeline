#!/usr/bin/env python3
"""
Pipeline Process Monitor

Utility to monitor and clean up pipeline processes.
Can detect stale processes and ensure proper cleanup.
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path so imports work from scripts/ directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from typing import List, Dict, Optional

class PipelineMonitor:
    """Monitor and manage pipeline processes."""

    def __init__(self):
        self.pipeline_processes: List[Dict] = []

    def find_pipeline_processes(self) -> List[Dict]:
        """Find all running pipeline processes."""
        try:
            # Use ps to find python processes running run_pipeline.py
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                check=True
            )

            processes = []
            for line in result.stdout.split('\n'):
                if 'run_pipeline.py' in line and 'python' in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        pid = int(parts[1])
                        cpu = parts[2]
                        mem = parts[3]
                        start_time = parts[8]
                        cmd = ' '.join(parts[10:])

                        processes.append({
                            'pid': pid,
                            'cpu': cpu,
                            'mem': mem,
                            'start_time': start_time,
                            'command': cmd
                        })

            return processes

        except (subprocess.CalledProcessError, ValueError, IndexError) as e:
            print(f"Error finding processes: {e}")
            return []

    def kill_process(self, pid: int, force: bool = False) -> bool:
        """Kill a specific process by PID."""
        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
            print(f"Sent {'SIGKILL' if force else 'SIGTERM'} to process {pid}")

            # Wait a moment and check if process is gone
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if process still exists
                return False
            except OSError:
                return True  # Process is gone

        except OSError as e:
            if e.errno == 3:  # No such process
                print(f"Process {pid} not found (already terminated)")
                return True
            print(f"Error killing process {pid}: {e}")
            return False

    def kill_all_pipeline_processes(self, force: bool = False) -> None:
        """Kill all running pipeline processes."""
        processes = self.find_pipeline_processes()

        if not processes:
            print("No pipeline processes found.")
            return

        print(f"Found {len(processes)} pipeline process(es):")
        for proc in processes:
            print(f"  PID {proc['pid']}: {proc['command']}")

        for proc in processes:
            success = self.kill_process(proc['pid'], force)
            if success:
                print(f"✅ Successfully terminated PID {proc['pid']}")
            else:
                print(f"❌ Failed to terminate PID {proc['pid']}")
                if not force:
                    print(f"  Trying force kill...")
                    self.kill_process(proc['pid'], force=True)

    def monitor_processes(self, interval: int = 5) -> None:
        """Monitor pipeline processes continuously."""
        print(f"Monitoring pipeline processes (checking every {interval}s)...")
        print("Press Ctrl+C to stop monitoring")

        try:
            while True:
                processes = self.find_pipeline_processes()

                if processes:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Found {len(processes)} process(es):")
                    for proc in processes:
                        print(f"  PID {proc['pid']} | CPU: {proc['cpu']}% | "
                              f"Mem: {proc['mem']}% | Started: {proc['start_time']}")
                        print(f"    Command: {proc['command']}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No pipeline processes running")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    def cleanup_outputs(self, older_than_hours: int = 24) -> None:
        """Clean up old output directories."""
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            print("No outputs directory found.")
            return

        cutoff_time = time.time() - (older_than_hours * 3600)
        cleaned = 0

        for item in outputs_dir.iterdir():
            if item.is_dir() and item.stat().st_mtime < cutoff_time:
                try:
                    import shutil
                    shutil.rmtree(item)
                    print(f"Removed old output directory: {item}")
                    cleaned += 1
                except Exception as e:
                    print(f"Failed to remove {item}: {e}")

        print(f"Cleaned up {cleaned} old output directories.")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pipeline Process Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List running pipeline processes
    python pipeline_monitor.py --list

    # Kill all pipeline processes
    python pipeline_monitor.py --kill-all

    # Force kill all pipeline processes
    python pipeline_monitor.py --kill-all --force

    # Monitor processes continuously
    python pipeline_monitor.py --monitor

    # Clean up old outputs
    python pipeline_monitor.py --cleanup-outputs
        """,
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all running pipeline processes"
    )

    parser.add_argument(
        "--kill-all", "-k",
        action="store_true",
        help="Kill all running pipeline processes"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Use force kill (SIGKILL) instead of graceful termination"
    )

    parser.add_argument(
        "--monitor", "-m",
        action="store_true",
        help="Monitor pipeline processes continuously"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds (default: 5)"
    )

    parser.add_argument(
        "--cleanup-outputs",
        action="store_true",
        help="Clean up old output directories"
    )

    parser.add_argument(
        "--older-than-hours",
        type=int,
        default=24,
        help="Remove outputs older than N hours (default: 24)"
    )

    args = parser.parse_args()

    monitor = PipelineMonitor()

    if args.list:
        processes = monitor.find_pipeline_processes()
        if processes:
            print(f"Found {len(processes)} pipeline process(es):")
            for proc in processes:
                print(f"  PID {proc['pid']} | CPU: {proc['cpu']}% | "
                      f"Mem: {proc['mem']}% | Started: {proc['start_time']}")
                print(f"    Command: {proc['command']}")
        else:
            print("No pipeline processes found.")

    elif args.kill_all:
        monitor.kill_all_pipeline_processes(force=args.force)

    elif args.monitor:
        monitor.monitor_processes(interval=args.interval)

    elif args.cleanup_outputs:
        monitor.cleanup_outputs(older_than_hours=args.older_than_hours)

    else:
        # Default action: list processes
        processes = monitor.find_pipeline_processes()
        if processes:
            print(f"Found {len(processes)} pipeline process(es):")
            for proc in processes:
                print(f"  PID {proc['pid']}: {proc['command']}")
        else:
            print("No pipeline processes found.")

if __name__ == "__main__":
    main()