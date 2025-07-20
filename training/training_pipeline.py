import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    ProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import optuna
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.models import create_model, MultiViewFusionModel
from utils.dataset import (
    create_dataloaders,
    MedicalImageDataset,
    create_multi_view_transforms,
)
from utils.constants import (
    SEED,
    DEVICE,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    N_WORKERS,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    MULTI_VIEW_CONFIG,
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
)
from utils.loss_function import ProbabilisticF1Loss, FocalLoss, CombinedLoss
from utils.mlflow_utils import MLFlowTracker, MLFlowCallback, setup_mlflow_experiment


# Removed problematic custom callback class


class ProbabilisticF1Metric:
    """Implementation of probabilistic F1 score for evaluation"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.reset()

    def reset(self):
        self.p_tp = 0.0
        self.p_fp = 0.0
        self.tp = 0.0
        self.fn = 0.0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with batch predictions and targets

        Args:
            predictions: Predicted probabilities [batch_size]
            targets: Ground truth labels [batch_size]
        """
        # Convert to numpy for easier computation - handle BFloat16
        preds = predictions.detach().cpu().float().numpy()
        targs = targets.detach().cpu().float().numpy()

        # Probabilistic True Positives
        self.p_tp += np.sum(preds * targs)

        # Probabilistic False Positives
        self.p_fp += np.sum(preds * (1 - targs))

        # True Positives and False Negatives (for recall denominator)
        self.tp += np.sum((preds > 0.5) * targs)
        self.fn += np.sum((preds <= 0.5) * targs)

    def compute(self) -> float:
        """Compute probabilistic F1 score"""
        # Probabilistic Precision
        p_precision = (
            self.p_tp / (self.p_tp + self.p_fp) if (self.p_tp + self.p_fp) > 0 else 0.0
        )

        # Probabilistic Recall
        p_recall = self.p_tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

        # Probabilistic F1
        if p_precision + p_recall > 0:
            p_f1 = (
                (1 + self.beta**2)
                * (p_precision * p_recall)
                / ((self.beta**2 * p_precision) + p_recall)
            )
        else:
            p_f1 = 0.0

        return p_f1


class BreastCancerLightningModule(pl.LightningModule):
    """PyTorch Lightning module for breast cancer detection"""

    def __init__(
        self,
        model_config: Dict = None,
        training_config: Dict = None,
        loss_config: Dict = None,
    ):
        super().__init__()

        self.model_config = model_config or MODEL_CONFIG
        self.training_config = training_config or TRAINING_CONFIG
        self.loss_config = loss_config or {
            "type": "combined",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        }

        # Initialize model
        self.model = create_model(
            model_type="multiview_fusion",
            backbone_name=self.model_config["backbone"],
            pretrained=self.model_config["pretrained"],
            num_classes=self.model_config["num_classes"],
            dropout=self.model_config["dropout"],
            fusion_method=MULTI_VIEW_CONFIG["fusion_method"],
        )

        # Initialize loss function
        self._setup_loss_function()

        # Metrics
        self.train_pf1 = ProbabilisticF1Metric()
        self.val_pf1 = ProbabilisticF1Metric()
        self.train_auc = 0.0
        self.val_auc = 0.0

        # Save hyperparameters
        self.save_hyperparameters()

    def _setup_loss_function(self):
        """Setup the loss function based on configuration"""
        loss_type = self.loss_config.get("type", "combined")

        if loss_type == "pf1":
            self.criterion = ProbabilisticF1Loss()
        elif loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=self.loss_config.get("focal_alpha", 0.25),
                gamma=self.loss_config.get("focal_gamma", 2.0),
            )
        elif loss_type == "combined":
            self.criterion = CombinedLoss(
                focal_alpha=self.loss_config.get("focal_alpha", 0.25),
                focal_gamma=self.loss_config.get("focal_gamma", 2.0),
                pf1_weight=self.loss_config.get("pf1_weight", 0.3),
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, cc_view, mlo_view, patient_metadata=None):
        return self.model(cc_view, mlo_view, patient_metadata)

    def training_step(self, batch, batch_idx):
        cc_view, mlo_view, targets = batch

        # Forward pass
        logits, uncertainty = self(cc_view, mlo_view)

        # Calculate loss
        loss = self.criterion(logits.squeeze(), targets.squeeze())

        # Calculate probabilities
        probs = torch.sigmoid(logits).squeeze()

        # Update metrics
        self.train_pf1.update(probs, targets.squeeze())

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_uncertainty", uncertainty.mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        cc_view, mlo_view, targets = batch

        # Forward pass
        logits, uncertainty = self(cc_view, mlo_view)

        # Calculate loss
        loss = self.criterion(logits.squeeze(), targets.squeeze())

        # Calculate probabilities
        probs = torch.sigmoid(logits).squeeze()

        # Update metrics
        self.val_pf1.update(probs, targets.squeeze())

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_uncertainty", uncertainty.mean(), prog_bar=True)

        return {
            "val_loss": loss,
            "probs": probs,
            "targets": targets.squeeze(),
            "uncertainty": uncertainty,
        }

    def on_train_epoch_end(self):
        """Compute and log training metrics at epoch end"""
        train_pf1 = self.train_pf1.compute()
        self.log("train_pf1", train_pf1, prog_bar=True)
        self.train_pf1.reset()

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end"""
        val_pf1 = self.val_pf1.compute()
        self.log("val_pf1", val_pf1, prog_bar=True)
        self.val_pf1.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""

        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
            eps=1e-8,
        )

        # Learning rate scheduler
        scheduler_type = self.training_config.get("scheduler", "cosine_with_warmup")

        if scheduler_type == "cosine_with_warmup":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.training_config.get("T_max", 20),
                T_mult=self.training_config.get("T_mult", 2),
                eta_min=self.training_config.get("eta_min", 1e-6),
            )
        elif scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.training_config["learning_rate"],
                epochs=self.training_config["max_epochs"],
                steps_per_epoch=len(self.train_dataloader()),
            )
        else:
            scheduler = None

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_pf1",
                    "interval": "epoch",
                },
            }
        else:
            return {"optimizer": optimizer}


class TrainingPipeline:
    """Advanced training pipeline for breast cancer detection"""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        model_dir: str = MODEL_DIR,
        results_dir: str = RESULTS_DIR,
        config: Dict = None,
        use_mlflow: bool = True,
        experiment_name: str = "breast-cancer-detection",
    ):

        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.config = config or {}
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup MLFlow if enabled
        self.mlflow_tracker = None
        if self.use_mlflow:
            self.mlflow_tracker = setup_mlflow_experiment(experiment_name)
            self.mlflow_tracker.start_run(
                run_name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                tags={"type": "training", "model": "multiview-fusion"},
            )
            # Log configuration
            self.mlflow_tracker.log_config(self.config)

        # Setup logging
        self._setup_logging()

        # Load data
        self.train_df = None
        self.test_df = None
        self._load_data()

        # Set random seed
        pl.seed_everything(SEED)

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.results_dir / "training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _load_data(self):
        """Load training and test data"""
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"

        if train_path.exists():
            self.train_df = pd.read_csv(train_path)
            self.logger.info(f"Loaded training data: {len(self.train_df)} samples")
        else:
            self.logger.warning("Training data not found")

        if test_path.exists():
            self.test_df = pd.read_csv(test_path)
            self.logger.info(f"Loaded test data: {len(self.test_df)} samples")
        else:
            self.logger.warning("Test data not found")

    def _create_callbacks(self) -> List[pl.Callback]:
        """Create training callbacks"""
        callbacks = []

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename="best_model_{epoch:02d}_{val_pf1:.4f}",
            monitor="val_pf1",
            mode="max",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_pf1",
            mode="max",
            patience=self.config.get("patience", TRAINING_CONFIG["patience"]),
            verbose=True,
        )
        callbacks.append(early_stopping)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        return callbacks

    def _create_loggers(self) -> List:
        """Create logging callbacks"""
        loggers = []

        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=self.results_dir, name="tensorboard_logs"
        )
        loggers.append(tb_logger)

        # WandB logger (if configured)
        if wandb.run is not None:
            wandb_logger = WandbLogger(
                project="breast-cancer-detection", log_model=True
            )
            loggers.append(wandb_logger)

        return loggers

    def train_single_fold(
        self, train_idx: List[int], val_idx: List[int], fold: int = 0
    ) -> Dict[str, float]:
        """Train a single fold"""

        self.logger.info(f"Training fold {fold}")

        # Split data
        train_data = self.train_df.iloc[train_idx]
        val_data = self.train_df.iloc[val_idx]

        # Create dataloaders
        train_loader = create_dataloaders(
            metadata_df=train_data,
            data_dir=str(self.data_dir / "train_images"),
            batch_size=self.config.get("batch_size", TRAIN_BATCH_SIZE),
            is_train=True,
            patient_level=True,
            num_workers=self.config.get("num_workers", N_WORKERS),
        )

        val_loader = create_dataloaders(
            metadata_df=val_data,
            data_dir=str(self.data_dir / "train_images"),
            batch_size=self.config.get("val_batch_size", VAL_BATCH_SIZE),
            is_train=False,
            patient_level=True,
            num_workers=self.config.get("num_workers", N_WORKERS),
        )

        # Create model
        model = BreastCancerLightningModule(
            model_config=self.config.get("model_config", MODEL_CONFIG),
            training_config=self.config.get("training_config", TRAINING_CONFIG),
            loss_config=self.config.get("loss_config", {}),
        )

        # Print training start info
        print(
            f"\n🚀 Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"📊 Model: MultiViewFusionModel")
        print(f"🔧 Device: CPU")
        print(f"📈 Precision: {self.config.get('precision', 16)}")
        print("=" * 80)

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.get("max_epochs", TRAINING_CONFIG["max_epochs"]),
            min_epochs=self.config.get("min_epochs", TRAINING_CONFIG["min_epochs"]),
            accelerator="auto",
            devices=1,
            precision=self.config.get("precision", 16),
            callbacks=self._create_callbacks(),
            logger=self._create_loggers(),
            gradient_clip_val=self.config.get("gradient_clip_val", 1.0),
            deterministic=True,
            enable_progress_bar=True,
            enable_model_summary=True,
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Get best validation score
        best_score = trainer.callback_metrics.get("val_pf1", 0.0)

        # Print training completion info
        print("=" * 80)
        print(f"✅ Fold {fold} completed!")
        print(f"📊 Best validation PF1: {best_score:.4f}")
        print("=" * 80)

        self.logger.info(f"Fold {fold} completed. Best val_pf1: {best_score:.4f}")

        return {
            "fold": fold,
            "best_val_pf1": best_score,
            "best_model_path": trainer.checkpoint_callback.best_model_path,
        }

    def train_cross_validation(self, n_folds: int = 5) -> List[Dict[str, float]]:
        """Train with cross-validation"""

        if self.train_df is None:
            raise ValueError("Training data not loaded")

        print(f"\n🎯 Starting {n_folds}-fold cross-validation training")
        print(f"📊 Total samples: {len(self.train_df)}")
        print(f"👥 Unique patients: {self.train_df['patient_id'].nunique()}")
        print("=" * 80)

        # Create cross-validation splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        # Get patient-level labels for stratification
        patient_labels = self.train_df.groupby("patient_id")["cancer"].first().values

        fold_results = []
        cv_start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(range(len(patient_labels)), patient_labels)
        ):
            fold_start_time = time.time()
            print(f"\n🔄 FOLD {fold + 1}/{n_folds}")
            print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")

            self.logger.info(f"Starting fold {fold + 1}/{n_folds}")

            # Convert patient indices to image indices
            train_patients = self.train_df["patient_id"].unique()[train_idx]
            val_patients = self.train_df["patient_id"].unique()[val_idx]

            train_img_idx = self.train_df[
                self.train_df["patient_id"].isin(train_patients)
            ].index.tolist()
            val_img_idx = self.train_df[
                self.train_df["patient_id"].isin(val_patients)
            ].index.tolist()

            print(f"📈 Train samples: {len(train_img_idx)}")
            print(f"🔍 Val samples: {len(val_img_idx)}")

            # Train fold
            fold_result = self.train_single_fold(train_img_idx, val_img_idx, fold)
            fold_results.append(fold_result)

            # Log fold results
            fold_time = time.time() - fold_start_time
            print(f"✅ Fold {fold + 1} completed in {fold_time/60:.1f} minutes")
            print(f"📊 Best val_pf1: {fold_result['best_val_pf1']:.4f}")

            # Log to MLFlow if enabled
            if self.mlflow_tracker:
                self.mlflow_tracker.log_cross_validation_results(fold_result, fold + 1)

        # Log cross-validation results
        cv_scores = [result["best_val_pf1"] for result in fold_results]
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        total_time = time.time() - cv_start_time

        print("\n" + "=" * 80)
        print("🎉 CROSS-VALIDATION COMPLETED!")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"📊 Mean val_pf1: {mean_score:.4f} ± {std_score:.4f}")
        print(f"📈 Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
        print("=" * 80)

        self.logger.info(f"Cross-validation completed:")
        self.logger.info(f"Mean val_pf1: {mean_score:.4f} ± {std_score:.4f}")
        self.logger.info(f"Individual fold scores: {cv_scores}")

        # Log final results to MLFlow
        if self.mlflow_tracker:
            final_results = {
                "mean_pf1": mean_score,
                "std_pf1": std_score,
                "fold_scores": cv_scores,
                "total_time": total_time,
            }
            self.mlflow_tracker.log_final_results(final_results)

        return fold_results

    def hyperparameter_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""

        def objective(trial):
            # Define hyperparameter search space
            config = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-3, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-5, 1e-3, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "focal_alpha": trial.suggest_float("focal_alpha", 0.1, 0.5),
                "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
                "pf1_weight": trial.suggest_float("pf1_weight", 0.1, 0.5),
            }

            # Update config
            self.config.update(config)

            # Train with cross-validation
            fold_results = self.train_cross_validation(
                n_folds=3
            )  # Use fewer folds for speed

            # Return mean validation score
            return np.mean([result["best_val_pf1"] for result in fold_results])

        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Best hyperparameters: {study.best_params}")
        self.logger.info(f"Best validation score: {study.best_value:.4f}")

        return study.best_params


def main():
    """Main training function"""

    # Initialize training pipeline
    pipeline = TrainingPipeline()

    # Run hyperparameter optimization
    best_params = pipeline.hyperparameter_optimization(n_trials=20)

    # Update config with best parameters
    pipeline.config.update(best_params)

    # Train final model with cross-validation
    fold_results = pipeline.train_cross_validation(n_folds=5)

    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(
        pipeline.results_dir / "cross_validation_results.csv", index=False
    )

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
