"""
MLFlow utilities for breast cancer detection pipeline.

This module provides comprehensive MLFlow integration for:
- Experiment tracking
- Model versioning
- Metrics logging
- Artifact management
- Hyperparameter tracking
"""

import os
import logging
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MLFlowTracker:
    """MLFlow experiment tracker for breast cancer detection"""

    def __init__(
        self,
        experiment_name: str = "breast-cancer-detection",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLFlow tracker

        Args:
            experiment_name: Name of the MLFlow experiment
            tracking_uri: MLFlow tracking server URI (optional)
            artifact_location: Location to store artifacts (optional)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.artifact_location = artifact_location or "./mlruns"

        # Setup MLFlow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                experiment_name, artifact_location=self.artifact_location
            )

        mlflow.set_experiment(experiment_name)

        self.run = None
        self.logger = logging.getLogger(__name__)

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLFlow run"""
        if run_name is None:
            run_name = f"breast-cancer-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.run = mlflow.start_run(run_name=run_name, tags=tags or {})
        self.logger.info(f"Started MLFlow run: {run_name}")
        return self.run

    def end_run(self):
        """End the current MLFlow run"""
        if self.run:
            mlflow.end_run()
            self.logger.info("Ended MLFlow run")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        mlflow.log_params(params)
        self.logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
        self.logger.info(f"Logged {len(metrics)} metrics")

    def log_model(self, model, model_name: str = "breast-cancer-model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, model_name)
        self.logger.info(f"Logged model: {model_name}")

    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifacts (files/directories)"""
        mlflow.log_artifacts(local_path, artifact_path)
        self.logger.info(f"Logged artifacts from: {local_path}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a single artifact file"""
        mlflow.log_artifact(local_path, artifact_path)
        self.logger.info(f"Logged artifact: {local_path}")

    def log_cross_validation_results(
        self, cv_results: List[Dict[str, float]], fold: int
    ):
        """Log cross-validation results for a specific fold"""
        # Log fold-specific metrics
        fold_metrics = {
            f"fold_{fold}_val_pf1": cv_results["best_val_pf1"],
            f"fold_{fold}_val_loss": cv_results.get("val_loss", 0.0),
            f"fold_{fold}_train_loss": cv_results.get("train_loss", 0.0),
        }
        self.log_metrics(fold_metrics, step=fold)

        # Log model checkpoint path
        if "best_model_path" in cv_results:
            self.log_artifact(cv_results["best_model_path"], f"models/fold_{fold}")

    def log_final_results(self, final_results: Dict[str, Any]):
        """Log final training results"""
        # Log final metrics
        final_metrics = {
            "final_mean_pf1": final_results.get("mean_pf1", 0.0),
            "final_std_pf1": final_results.get("std_pf1", 0.0),
            "final_mean_auc": final_results.get("mean_auc", 0.0),
            "final_std_auc": final_results.get("std_auc", 0.0),
        }
        self.log_metrics(final_metrics)

        # Log results CSV
        if "results_path" in final_results:
            self.log_artifact(final_results["results_path"], "results")

    def log_config(self, config: Dict[str, Any]):
        """Log configuration as JSON artifact"""
        config_path = "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.log_artifact(config_path, "config")
        os.remove(config_path)

    def log_training_plots(self, plots_dir: str):
        """Log training plots and visualizations"""
        if os.path.exists(plots_dir):
            self.log_artifacts(plots_dir, "plots")

    def log_model_summary(self, model_summary: str):
        """Log model architecture summary"""
        summary_path = "model_summary.txt"
        with open(summary_path, "w") as f:
            f.write(model_summary)
        self.log_artifact(summary_path, "model_info")
        os.remove(summary_path)


class MLFlowCallback:
    """PyTorch Lightning callback for MLFlow integration"""

    def __init__(self, tracker: MLFlowTracker):
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)

    def on_train_start(self, trainer, pl_module):
        """Called when training starts"""
        self.logger.info("MLFlow callback: Training started")

    def on_train_end(self, trainer, pl_module):
        """Called when training ends"""
        # Log final model
        self.tracker.log_model(pl_module.model, "final-model")
        self.logger.info("MLFlow callback: Training ended, model logged")

    def on_validation_end(self, trainer, pl_module):
        """Called when validation ends"""
        # Log validation metrics
        metrics = trainer.callback_metrics
        if metrics:
            self.tracker.log_metrics(metrics, step=trainer.current_epoch)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when checkpoint is saved"""
        # Log checkpoint path
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        if checkpoint_path:
            self.tracker.log_artifact(checkpoint_path, "checkpoints")


def setup_mlflow_experiment(
    experiment_name: str = "breast-cancer-detection",
    tracking_uri: Optional[str] = None,
) -> MLFlowTracker:
    """Setup MLFlow experiment and return tracker"""
    tracker = MLFlowTracker(experiment_name, tracking_uri)
    return tracker


def log_experiment_summary(tracker: MLFlowTracker, summary: Dict[str, Any]):
    """Log experiment summary"""
    summary_path = "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    tracker.log_artifact(summary_path, "summary")
    os.remove(summary_path)


def compare_experiments(experiment_name: str, metric: str = "val_pf1") -> pd.DataFrame:
    """Compare experiments and return results DataFrame"""
    mlflow.set_experiment(experiment_name)

    # Get all runs
    runs = mlflow.search_runs()

    if runs.empty:
        return pd.DataFrame()

    # Extract metrics
    results = []
    for _, run in runs.iterrows():
        run_data = {
            "run_id": run["run_id"],
            "run_name": run["tags.mlflow.runName"],
            "start_time": run["start_time"],
            "end_time": run["end_time"],
            "status": run["status"],
        }

        # Add metrics
        for col in run.columns:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                run_data[metric_name] = run[col]

        results.append(run_data)

    return pd.DataFrame(results)


def get_best_run(experiment_name: str, metric: str = "val_pf1") -> Optional[Dict]:
    """Get the best run based on a metric"""
    runs_df = compare_experiments(experiment_name, metric)

    if runs_df.empty:
        return None

    # Find best run
    metric_col = f"metrics.{metric}"
    if metric_col in runs_df.columns:
        best_idx = runs_df[metric_col].idxmax()
        return runs_df.loc[best_idx].to_dict()

    return None
