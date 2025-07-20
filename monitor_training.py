#!/usr/bin/env python3
"""
Comprehensive training monitoring script for breast cancer detection.

This script provides easy access to:
- TensorBoard monitoring
- MLFlow experiment tracking
- Training progress monitoring
- Model performance analysis
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.tensorboard_utils import start_tensorboard, print_tensorboard_instructions
from utils.mlflow_utils import compare_experiments, get_best_run


def print_banner():
    """Print monitoring banner"""
    print("\n" + "=" * 80)
    print("🔬 BREAST CANCER DETECTION - TRAINING MONITORING")
    print("=" * 80)


def check_training_status():
    """Check if training is currently running"""
    # Check for active Python processes running main.py
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py"], capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            print(f"🎯 Training is currently running (PIDs: {', '.join(pids)})")
            return True
        else:
            print("⏸️  No active training detected")
            return False
    except Exception:
        print("❓ Could not check training status")
        return False


def start_monitoring(port: int = 6006, auto_open: bool = True):
    """Start TensorBoard monitoring"""
    print_banner()
    print("📊 Starting TensorBoard monitoring...")

    # Check if training is running
    training_active = check_training_status()

    if training_active:
        print("✅ Training is active - you can monitor progress in real-time!")
    else:
        print("💡 No active training - you can view previous experiments")

    # Start TensorBoard
    process = start_tensorboard(
        log_dir="results/tensorboard_logs", port=port, auto_open=auto_open
    )

    if process:
        print(f"\n🌐 TensorBoard is running at: http://localhost:{port}")
        print("📈 Monitor your training progress in real-time!")
        print("\nKey metrics to watch:")
        print("  • train_loss / val_loss - Loss curves")
        print("  • train_pf1 / val_pf1 - Probabilistic F1 scores")
        print("  • learning_rate - Learning rate schedule")
        print("  • train_uncertainty / val_uncertainty - Model uncertainty")

        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping TensorBoard...")
            process.terminate()
            print("✅ TensorBoard stopped")
    else:
        print("❌ Failed to start TensorBoard")


def show_mlflow_experiments(experiment_name: str = "breast-cancer-detection"):
    """Show MLFlow experiment results"""
    print_banner()
    print("🔬 MLFlow Experiment Analysis")
    print("=" * 50)

    try:
        # Get experiment comparison
        df = compare_experiments(experiment_name)

        if df.empty:
            print("📭 No experiments found in MLFlow")
            return

        print(f"📊 Found {len(df)} experiments:")
        print()

        # Show recent experiments
        for _, run in df.head(5).iterrows():
            print(f"🏃 Run: {run.get('run_name', 'Unknown')}")
            print(f"   📅 Started: {run.get('start_time', 'Unknown')}")
            print(f"   📈 Best PF1: {run.get('metrics.val_pf1', 'N/A')}")
            print(f"   📉 Loss: {run.get('metrics.val_loss', 'N/A')}")
            print()

        # Get best run
        best_run = get_best_run(experiment_name)
        if best_run:
            print("🏆 BEST EXPERIMENT:")
            print(f"   Run: {best_run.get('run_name', 'Unknown')}")
            print(f"   PF1: {best_run.get('metrics.val_pf1', 'N/A')}")
            print(f"   Loss: {best_run.get('metrics.val_loss', 'N/A')}")

        print(f"\n🌐 View all experiments: mlflow ui")

    except Exception as e:
        print(f"❌ Error accessing MLFlow: {e}")


def show_training_instructions():
    """Show comprehensive training instructions"""
    print_banner()
    print("📚 TRAINING & MONITORING GUIDE")
    print("=" * 50)

    print("\n🚀 STARTING TRAINING:")
    print("   poetry run python main.py --mode train --config configs/default.yaml")

    print("\n📊 MONITORING OPTIONS:")
    print("   1. Real-time monitoring:")
    print("      python monitor_training.py --tensorboard")
    print()
    print("   2. View experiment history:")
    print("      python monitor_training.py --mlflow")
    print()
    print("   3. Manual TensorBoard:")
    print("      tensorboard --logdir results/tensorboard_logs --port 6006")
    print()
    print("   4. MLFlow UI:")
    print("      mlflow ui")

    print("\n📈 KEY METRICS TO MONITOR:")
    print("   • val_pf1: Validation Probabilistic F1 (main metric)")
    print("   • val_loss: Validation loss")
    print("   • train_loss: Training loss")
    print("   • learning_rate: Learning rate schedule")
    print("   • train_uncertainty: Model uncertainty")

    print("\n⚠️  TRAINING STATUS:")
    print("   • Green progress bars = Training is working")
    print("   • DICOM warnings = Normal (files loaded with force=True)")
    print("   • No progress = Check logs/main.log for errors")

    print("\n🔍 TROUBLESHOOTING:")
    print("   • If training stops: Check logs/main.log")
    print("   • If no progress: Verify data paths in config")
    print("   • If DICOM errors: Normal (handled automatically)")
    print("   • If memory issues: Reduce batch_size in config")


def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(
        description="Breast Cancer Detection Training Monitor"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Start TensorBoard monitoring"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="Show MLFlow experiment results"
    )
    parser.add_argument("--status", action="store_true", help="Check training status")
    parser.add_argument(
        "--help-training", action="store_true", help="Show training instructions"
    )
    parser.add_argument(
        "--port", type=int, default=6006, help="TensorBoard port (default: 6006)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser"
    )

    args = parser.parse_args()

    if args.status:
        print_banner()
        check_training_status()

    elif args.tensorboard:
        start_monitoring(port=args.port, auto_open=not args.no_browser)

    elif args.mlflow:
        show_mlflow_experiments()

    elif args.help_training:
        show_training_instructions()

    else:
        # Default: show help
        print_banner()
        print("🎯 TRAINING MONITORING TOOL")
        print("=" * 50)
        print("\nAvailable commands:")
        print("  --tensorboard    Start TensorBoard monitoring")
        print("  --mlflow         Show MLFlow experiment results")
        print("  --status         Check if training is running")
        print("  --help-training  Show comprehensive instructions")
        print("\nExamples:")
        print("  python monitor_training.py --tensorboard")
        print("  python monitor_training.py --mlflow")
        print("  python monitor_training.py --status")


if __name__ == "__main__":
    main()
