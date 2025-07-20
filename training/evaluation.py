import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
import optuna
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.models import create_model, MultiViewFusionModel
from utils.dataset import create_dataloaders, MedicalImageDataset
from utils.constants import (
    SEED,
    DEVICE,
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    N_WORKERS,
)

# ProbabilisticF1Metric is defined in training_pipeline.py, not in loss_function.py


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
        # Convert to numpy for easier computation
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()

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


class ModelEvaluator:
    """Advanced model evaluator for breast cancer detection"""

    def __init__(
        self, model_path: str, data_dir: str = DATA_DIR, results_dir: str = RESULTS_DIR
    ):

        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Load model and data
        self.model = None
        self.train_df = None
        self.test_df = None
        self._load_model_and_data()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.results_dir / "evaluation.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _load_model_and_data(self):
        """Load trained model and data"""
        # Load model
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=DEVICE)
            self.model = create_model(model_type="multiview_fusion")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(DEVICE)
            self.model.eval()
            self.logger.info(f"Loaded model from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Load data
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"

        if train_path.exists():
            self.train_df = pd.read_csv(train_path)
            self.logger.info(f"Loaded training data: {len(self.train_df)} samples")

        if test_path.exists():
            self.test_df = pd.read_csv(test_path)
            self.logger.info(f"Loaded test data: {len(self.test_df)} samples")

    def evaluate_single_fold(
        self, train_idx: List[int], val_idx: List[int], fold: int = 0
    ) -> Dict[str, float]:
        """Evaluate a single fold"""

        self.logger.info(f"Evaluating fold {fold}")

        # Split data
        train_data = self.train_df.iloc[train_idx]
        val_data = self.train_df.iloc[val_idx]

        # Create dataloader
        val_loader = create_dataloaders(
            metadata_df=val_data,
            data_dir=str(self.data_dir / "train_images"),
            batch_size=VAL_BATCH_SIZE,
            is_train=False,
            patient_level=True,
            num_workers=N_WORKERS,
            shuffle=False,
        )

        # Evaluate
        predictions, targets, uncertainties = self._predict_on_dataloader(val_loader)

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets, uncertainties)

        self.logger.info(f"Fold {fold} evaluation completed")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

        return metrics

    def _predict_on_dataloader(
        self, dataloader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions on a dataloader"""

        predictions = []
        targets = []
        uncertainties = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Making predictions"):
                cc_view, mlo_view, batch_targets = batch
                cc_view = cc_view.to(DEVICE)
                mlo_view = mlo_view.to(DEVICE)

                # Forward pass
                logits, uncertainty = self.model(cc_view, mlo_view)

                # Convert to probabilities
                probs = torch.sigmoid(logits).cpu().numpy()

                predictions.extend(probs.flatten())
                targets.extend(batch_targets.numpy().flatten())
                uncertainties.extend(uncertainty.cpu().numpy().flatten())

        return np.array(predictions), np.array(targets), np.array(uncertainties)

    def _calculate_metrics(
        self, predictions: np.ndarray, targets: np.ndarray, uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""

        metrics = {}

        # Probabilistic F1 Score
        pf1_metric = ProbabilisticF1Metric()
        pf1_metric.update(torch.from_numpy(predictions), torch.from_numpy(targets))
        metrics["pf1"] = pf1_metric.compute()

        # ROC AUC
        metrics["roc_auc"] = roc_auc_score(targets, predictions)

        # Average Precision Score
        metrics["avg_precision"] = average_precision_score(targets, predictions)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(targets, predictions)
        metrics["pr_auc"] = auc(recall, precision)

        # Binary classification metrics
        binary_predictions = (predictions > 0.5).astype(int)
        cm = confusion_matrix(targets, binary_predictions)

        tn, fp, fn, tp = cm.ravel()
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["f1"] = (
            2
            * (metrics["precision"] * metrics["recall"])
            / (metrics["precision"] + metrics["recall"])
            if (metrics["precision"] + metrics["recall"]) > 0
            else 0.0
        )
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Uncertainty metrics
        metrics["mean_uncertainty"] = np.mean(uncertainties)
        metrics["uncertainty_correlation"] = np.corrcoef(predictions, uncertainties)[
            0, 1
        ]

        # Calibration metrics
        metrics["calibration_error"] = self._calculate_calibration_error(
            predictions, targets
        )

        return metrics

    def _calculate_calibration_error(
        self, predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate calibration error"""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        calibration_error = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(targets[in_bin])
                bin_confidence = np.mean(predictions[in_bin])
                calibration_error += (
                    np.sum(in_bin) * (bin_confidence - bin_accuracy) ** 2
                )

        return np.sqrt(calibration_error / len(predictions))

    def evaluate_cross_validation(self, n_folds: int = 5) -> Dict[str, Any]:
        """Evaluate with cross-validation"""

        if self.train_df is None:
            raise ValueError("Training data not loaded")

        # Create cross-validation splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        # Get patient-level labels for stratification
        patient_labels = self.train_df.groupby("patient_id")["cancer"].first().values

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(range(len(patient_labels)), patient_labels)
        ):
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

            # Evaluate fold
            fold_metrics = self.evaluate_single_fold(train_img_idx, val_img_idx, fold)
            fold_metrics["fold"] = fold
            fold_results.append(fold_metrics)

        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results)

        # Save results
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(
            self.results_dir / "cross_validation_results.csv", index=False
        )

        # Create summary
        summary_df = pd.DataFrame([cv_results])
        summary_df.to_csv(self.results_dir / "cv_summary.csv", index=False)

        return cv_results

    def _aggregate_cv_results(
        self, fold_results: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Aggregate cross-validation results"""

        # Calculate mean and std for each metric
        metrics = [
            "pf1",
            "roc_auc",
            "avg_precision",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "specificity",
        ]

        aggregated = {}
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_values"] = values

        # Add fold results
        aggregated["fold_results"] = fold_results

        return aggregated

    def create_evaluation_plots(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
        save_path: str = None,
    ):
        """Create comprehensive evaluation plots"""

        if save_path is None:
            save_path = self.results_dir / "evaluation_plots.png"

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Model Evaluation Results", fontsize=16)

        # ROC Curve
        fpr, tpr, _ = roc_curve(targets, predictions)
        auc_score = roc_auc_score(targets, predictions)
        axes[0, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
        axes[0, 0].plot([0, 1], [0, 1], "k--", label="Random")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets, predictions)
        pr_auc = auc(recall, precision)
        axes[0, 1].plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision-Recall Curve")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Calibration Plot
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accuracies = []
        bin_confidences = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(targets[in_bin])
                bin_confidence = np.mean(predictions[in_bin])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)

        axes[0, 2].plot(
            bin_confidences, bin_accuracies, "o-", label="Model Calibration"
        )
        axes[0, 2].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        axes[0, 2].set_xlabel("Mean Predicted Probability")
        axes[0, 2].set_ylabel("Fraction of Positives")
        axes[0, 2].set_title("Calibration Plot")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Uncertainty vs Prediction
        axes[1, 0].scatter(predictions, uncertainties, alpha=0.6)
        axes[1, 0].set_xlabel("Predicted Probability")
        axes[1, 0].set_ylabel("Uncertainty")
        axes[1, 0].set_title("Uncertainty vs Prediction")
        axes[1, 0].grid(True)

        # Prediction Distribution
        axes[1, 1].hist(
            predictions[targets == 0],
            bins=30,
            alpha=0.7,
            label="Negative",
            density=True,
        )
        axes[1, 1].hist(
            predictions[targets == 1],
            bins=30,
            alpha=0.7,
            label="Positive",
            density=True,
        )
        axes[1, 1].set_xlabel("Predicted Probability")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Prediction Distribution")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Confusion Matrix
        binary_predictions = (predictions > 0.5).astype(int)
        cm = confusion_matrix(targets, binary_predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 2])
        axes[1, 2].set_xlabel("Predicted")
        axes[1, 2].set_ylabel("Actual")
        axes[1, 2].set_title("Confusion Matrix")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Evaluation plots saved to {save_path}")

    def generate_submission(
        self, test_dataloader, output_path: str = None
    ) -> pd.DataFrame:
        """Generate submission file for the competition"""

        if output_path is None:
            output_path = self.results_dir / "submission.csv"

        predictions, _, uncertainties = self._predict_on_dataloader(test_dataloader)

        # Create submission DataFrame
        submission_data = []

        # Assuming test_dataloader.dataset has access to metadata
        for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
            # You'll need to adapt this based on your test data structure
            patient_id = f"test_{i:05d}"
            submission_data.append(
                {"prediction_id": f"{i}-L", "cancer": pred}  # Assuming left view
            )
            submission_data.append(
                {"prediction_id": f"{i}-R", "cancer": pred}  # Assuming right view
            )

        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)

        self.logger.info(f"Submission file saved to {output_path}")

        return submission_df


def main():
    """Main evaluation function"""

    # Find the best model
    model_dir = Path(MODEL_DIR)
    model_files = list(model_dir.glob("*.ckpt"))

    if not model_files:
        print("No model files found!")
        return

    # Use the most recent model
    best_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Using model: {best_model}")

    # Initialize evaluator
    evaluator = ModelEvaluator(str(best_model))

    # Run cross-validation evaluation
    cv_results = evaluator.evaluate_cross_validation(n_folds=5)

    # Print results
    print("\nCross-Validation Results:")
    print("=" * 50)
    for key, value in cv_results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
