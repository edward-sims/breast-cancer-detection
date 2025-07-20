#!/usr/bin/env python3
"""
Main training script for breast cancer detection using Vision Transformers and multi-view fusion.

This script orchestrates the entire pipeline:
1. Data preprocessing and augmentation
2. Model training with cross-validation
3. Evaluation and submission generation
4. Hyperparameter optimization (optional)

Usage:
    python main.py --mode train --config configs/default.yaml
    python main.py --mode evaluate --model_path models/best_model.pth
    python main.py --mode predict --model_path models/best_model.pth --data_path data/test_images
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Any
import yaml
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from preprocessing.preprocessing import MedicalImagePreprocessor
from training.training_pipeline import TrainingPipeline
from training.evaluation import ModelEvaluator
from inference.inference_pipeline import BreastCancerInferenceEngine
from utils.constants import SEED, DEVICE, DATA_DIR, MODEL_DIR, RESULTS_DIR


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/main.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("BREAST CANCER DETECTION PIPELINE STARTED")
    logger.info("=" * 60)
    logger.info(f"Log level set to: {log_level}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Default configuration
        return {
            "data": {
                "data_dir": DATA_DIR,
                "train_csv": "train.csv",
                "test_csv": "test.csv",
                "image_size": [512, 512],
                "batch_size": 8,
                "num_workers": 4,
            },
            "model": {
                "backbone": "vit_base_patch16_224",
                "pretrained": True,
                "num_classes": 1,
                "dropout": 0.1,
                "fusion_method": "attention",
            },
            "training": {
                "epochs": 50,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "patience": 15,
                "n_folds": 5,
                "loss_type": "combined",
            },
            "preprocessing": {
                "augmentations": ["normalize", "contrast_enhancement"],
                "output_dir": "data/processed",
            },
        }


def run_preprocessing(config: Dict[str, Any], logger: logging.Logger):
    """Run data preprocessing pipeline"""
    logger.info("=" * 40)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 40)

    logger.info(f"Data directory: {config['data']['data_dir']}")
    logger.info(f"Output directory: {config['preprocessing']['output_dir']}")
    logger.info(f"Augmentations: {config['preprocessing']['augmentations']}")

    try:
        logger.info("Creating MedicalImagePreprocessor...")
        preprocessor = MedicalImagePreprocessor(
            data_dir=config["data"]["data_dir"],
            output_dir=config["preprocessing"]["output_dir"],
            config=config["preprocessing"],
        )
        logger.info("MedicalImagePreprocessor created successfully!")

        # Process training data
        logger.info("Processing training dataset...")
        train_stats = preprocessor.process_dataset("train")
        logger.info(
            f"Training data processed: {train_stats.get('statistics', {}).get('total_images', 0)} images"
        )
        logger.info(f"Training processing stats: {train_stats.get('statistics', {})}")

        # Process test data
        logger.info("Processing test dataset...")
        test_stats = preprocessor.process_dataset("test")
        logger.info(
            f"Test data processed: {test_stats.get('statistics', {}).get('total_images', 0)} images"
        )
        logger.info(f"Test processing stats: {test_stats.get('statistics', {})}")

        # Create visualization report
        logger.info("Creating visualization report...")
        preprocessor.create_visualization_report(
            {"train": train_stats, "test": test_stats},
            output_path="data/processed/preprocessing_report.html",
        )
        logger.info("Visualization report created successfully!")

        logger.info("=" * 40)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 40)

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def run_training(config: Dict[str, Any], logger: logging.Logger):
    """Run model training pipeline"""
    logger.info("Starting model training...")

    # Create training pipeline with MLFlow integration
    pipeline = TrainingPipeline(
        data_dir=config["data"]["data_dir"],
        model_dir=MODEL_DIR,
        results_dir=RESULTS_DIR,
        config=config,
        use_mlflow=config.get("logging", {}).get("use_mlflow", True),
        experiment_name=config.get("logging", {}).get(
            "experiment_name", "breast-cancer-detection"
        ),
    )

    try:
        # Run cross-validation training
        cv_results = pipeline.train_cross_validation(
            n_folds=config["training"]["n_folds"]
        )

        # Log results
        logger.info("Cross-validation training completed!")
        for fold, metrics in enumerate(cv_results):
            logger.info(f"Fold {fold + 1}: PF1 = {metrics['best_val_pf1']:.4f}")

        # Calculate average metrics
        avg_pf1 = sum(m["best_val_pf1"] for m in cv_results) / len(cv_results)
        logger.info(f"Average PF1: {avg_pf1:.4f}")

        # End MLFlow run if active
        if pipeline.mlflow_tracker:
            pipeline.mlflow_tracker.end_run()

        return cv_results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # End MLFlow run if active
        if pipeline.mlflow_tracker:
            pipeline.mlflow_tracker.end_run()
        raise


def run_evaluation(config: Dict[str, Any], model_path: str, logger: logging.Logger):
    """Run model evaluation"""
    logger.info(f"Starting model evaluation with {model_path}...")

    evaluator = ModelEvaluator(
        model_path=model_path,
        data_dir=config["data"]["data_dir"],
        results_dir=RESULTS_DIR,
    )

    # Evaluate with cross-validation
    cv_results = evaluator.evaluate_cross_validation(
        n_folds=config["training"]["n_folds"]
    )

    logger.info("Evaluation completed!")
    logger.info(
        f"Average PF1: {cv_results['mean_pf1']:.4f} ± {cv_results['std_pf1']:.4f}"
    )
    logger.info(
        f"Average AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}"
    )

    return cv_results


def run_inference(
    config: Dict[str, Any], model_path: str, data_path: str, logger: logging.Logger
):
    """Run inference on test data"""
    logger.info(f"Starting inference with {model_path}...")

    # Create inference engine
    engine = BreastCancerInferenceEngine(
        model_path=model_path,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Process test directory
    results_df = engine.predict_directory(
        data_dir=data_path, output_path="results/test_predictions.csv"
    )

    logger.info(f"Inference completed! Processed {len(results_df)} samples")
    logger.info(f"Results saved to: results/test_predictions.csv")

    return results_df


def run_hyperparameter_optimization(config: Dict[str, Any], logger: logging.Logger):
    """Run hyperparameter optimization"""
    logger.info("Starting hyperparameter optimization...")

    pipeline = TrainingPipeline(
        data_dir=config["data"]["data_dir"],
        model_dir=MODEL_DIR,
        results_dir=RESULTS_DIR,
        config=config,
    )

    # Run optimization
    best_params = pipeline.hyperparameter_optimization(n_trials=50)

    logger.info("Hyperparameter optimization completed!")
    logger.info(f"Best parameters: {best_params}")

    return best_params


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Breast Cancer Detection Training Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "evaluate", "predict", "optimize"],
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (for evaluate/predict modes)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to test data (for predict mode)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Log level: {args.log_level}")

    # Set random seed
    logger.info(f"Setting random seed to {SEED}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        logger.info("CUDA seeds set")

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    logger.info(f"Configuration loaded successfully from {args.config}")
    logger.info(f"Config keys: {list(config.keys())}")

    # Create necessary directories
    logger.info("Creating necessary directories...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("Directories created successfully")

    try:
        logger.info(f"Starting {args.mode} mode...")

        if args.mode == "preprocess":
            run_preprocessing(config, logger)

        elif args.mode == "train":
            run_training(config, logger)

        elif args.mode == "evaluate":
            if not args.model_path:
                raise ValueError("Model path is required for evaluation mode")
            run_evaluation(config, args.model_path, logger)

        elif args.mode == "predict":
            if not args.model_path:
                raise ValueError("Model path is required for predict mode")
            if not args.data_path:
                args.data_path = os.path.join(config["data"]["data_dir"], "test_images")
            run_inference(config, args.model_path, args.data_path, logger)

        elif args.mode == "optimize":
            run_hyperparameter_optimization(config, logger)

        logger.info(f"{args.mode} mode completed successfully!")

    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
