"""
TensorBoard utilities for breast cancer detection pipeline.

This module provides utilities for:
- Starting TensorBoard server
- Monitoring training progress
- Viewing experiment comparisons
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, List
import webbrowser
import time

logger = logging.getLogger(__name__)


def start_tensorboard(
    log_dir: str = "results/tensorboard_logs",
    port: int = 6006,
    host: str = "localhost",
    auto_open: bool = True,
) -> Optional[subprocess.Popen]:
    """
    Start TensorBoard server

    Args:
        log_dir: Directory containing TensorBoard logs
        port: Port to run TensorBoard on
        host: Host to bind to
        auto_open: Whether to automatically open browser

    Returns:
        Subprocess object for the TensorBoard server
    """
    log_path = Path(log_dir)

    if not log_path.exists():
        logger.warning(f"TensorBoard log directory {log_dir} does not exist")
        return None

    # Check if port is available
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        sock.close()
    except OSError:
        logger.warning(
            f"Port {port} is already in use. TensorBoard may already be running."
        )

    # Start TensorBoard
    cmd = [
        "tensorboard",
        "--logdir",
        str(log_path),
        "--port",
        str(port),
        "--host",
        host,
        "--reload_interval",
        "5",
    ]

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        logger.info(f"TensorBoard started on http://{host}:{port}")
        logger.info(f"Log directory: {log_path.absolute()}")

        if auto_open:
            # Wait a moment for server to start
            time.sleep(2)
            webbrowser.open(f"http://{host}:{port}")

        return process

    except Exception as e:
        logger.error(f"Failed to start TensorBoard: {e}")
        return None


def stop_tensorboard(process: subprocess.Popen):
    """Stop TensorBoard server"""
    if process:
        process.terminate()
        logger.info("TensorBoard stopped")


def get_tensorboard_url(port: int = 6006, host: str = "localhost") -> str:
    """Get TensorBoard URL"""
    return f"http://{host}:{port}"


def list_experiments(log_dir: str = "results/tensorboard_logs") -> List[str]:
    """List available TensorBoard experiments"""
    log_path = Path(log_dir)

    if not log_path.exists():
        return []

    experiments = []
    for item in log_path.iterdir():
        if item.is_dir() and item.name.startswith("version_"):
            experiments.append(item.name)

    return sorted(experiments)


def print_tensorboard_instructions():
    """Print instructions for using TensorBoard"""
    print("\n" + "=" * 60)
    print("📊 TENSORBOARD MONITORING INSTRUCTIONS")
    print("=" * 60)
    print("1. Start TensorBoard server:")
    print("   python -m utils.tensorboard_utils start")
    print()
    print("2. Or manually start TensorBoard:")
    print("   tensorboard --logdir results/tensorboard_logs --port 6006")
    print()
    print("3. Open your browser and go to:")
    print("   http://localhost:6006")
    print()
    print("4. Available tabs:")
    print("   📈 SCALARS: Training/validation metrics over time")
    print("   🏗️  GRAPHS: Model architecture visualization")
    print("   📊 DISTRIBUTIONS: Parameter distributions")
    print("   📋 HISTOGRAMS: Parameter histograms")
    print("   🖼️  IMAGES: Sample predictions (if logged)")
    print()
    print("5. Key metrics to monitor:")
    print("   - train_loss: Training loss")
    print("   - val_loss: Validation loss")
    print("   - train_pf1: Training probabilistic F1")
    print("   - val_pf1: Validation probabilistic F1")
    print("   - learning_rate: Learning rate over time")
    print("=" * 60)


def main():
    """Main function for TensorBoard utilities"""
    import argparse

    parser = argparse.ArgumentParser(description="TensorBoard utilities")
    parser.add_argument(
        "action", choices=["start", "list", "help"], help="Action to perform"
    )
    parser.add_argument(
        "--log-dir",
        default="results/tensorboard_logs",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--port", type=int, default=6006, help="Port to run TensorBoard on"
    )
    parser.add_argument("--host", default="localhost", help="Host to bind to")

    args = parser.parse_args()

    if args.action == "start":
        process = start_tensorboard(args.log_dir, args.port, args.host)
        if process:
            try:
                print(f"TensorBoard running on http://{args.host}:{args.port}")
                print("Press Ctrl+C to stop")
                process.wait()
            except KeyboardInterrupt:
                stop_tensorboard(process)
                print("\nTensorBoard stopped")

    elif args.action == "list":
        experiments = list_experiments(args.log_dir)
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("No experiments found")

    elif args.action == "help":
        print_tensorboard_instructions()


if __name__ == "__main__":
    main()
