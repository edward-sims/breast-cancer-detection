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
import pydicom
import cv2
from PIL import Image
import json
import time
from datetime import datetime
import threading
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.models import create_model, MultiViewFusionModel
from utils.dataset import create_medical_transforms
from utils.constants import (
    SEED,
    DEVICE,
    ROWS,
    COLS,
    CHANNELS,
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
)


class BreastCancerInferenceEngine:
    """Production-ready inference engine for breast cancer detection"""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4,
    ):

        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Setup logging
        self._setup_logging()

        # Load model and configuration
        self.model = None
        self.config = {}
        self.transforms = None
        self._load_model_and_config()

        # Initialize queues for batch processing
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.processing_thread = None
        self.is_running = False

        # Performance tracking
        self.inference_times = []
        self.total_predictions = 0

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("inference.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _load_model_and_config(self):
        """Load trained model and configuration"""
        # Load configuration
        if self.config_path and self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")

        # Load model
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=DEVICE)

            # Create model
            self.model = create_model(
                model_type="multiview_fusion",
                backbone_name=self.config.get("backbone", "vit_base_patch16_224"),
                pretrained=False,
                num_classes=self.config.get("num_classes", 1),
                dropout=self.config.get("dropout", 0.1),
                fusion_method=self.config.get("fusion_method", "attention"),
            )

            # Load state dict
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(DEVICE)
            self.model.eval()

            self.logger.info(f"Loaded model from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Setup transforms
        self.transforms = create_medical_transforms(
            image_size=(ROWS, COLS), is_train=False
        )

    def preprocess_dicom(self, dicom_path: str) -> np.ndarray:
        """Preprocess DICOM image for inference"""
        try:
            # Load DICOM
            dcm = pydicom.dcmread(dicom_path)

            # Extract pixel data
            image = dcm.pixel_array

            # Handle different bit depths
            if dcm.BitsAllocated == 16:
                image = image.astype(np.float32) / 65535.0
            else:
                image = image.astype(np.float32) / 255.0

            # Convert to 3-channel if grayscale
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            # Normalize to [0, 1]
            image = np.clip(image, 0, 1)

            return image

        except Exception as e:
            self.logger.error(f"Error preprocessing DICOM {dicom_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((ROWS, COLS, CHANNELS), dtype=np.float32)

    def preprocess_image_pair(
        self, cc_path: str, mlo_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess CC and MLO image pair"""

        # Load and preprocess images
        cc_image = self.preprocess_dicom(cc_path)
        mlo_image = self.preprocess_dicom(mlo_path)

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(image=cc_image, image1=mlo_image)
            cc_image = transformed["image"]
            mlo_image = transformed["image1"]

        # Convert to tensors
        cc_tensor = torch.from_numpy(cc_image.transpose(2, 0, 1)).float().unsqueeze(0)
        mlo_tensor = torch.from_numpy(mlo_image.transpose(2, 0, 1)).float().unsqueeze(0)

        return cc_tensor, mlo_tensor

    def predict_single(
        self, cc_path: str, mlo_path: str, patient_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make prediction for a single patient"""

        start_time = time.time()

        try:
            # Preprocess images
            cc_tensor, mlo_tensor = self.preprocess_image_pair(cc_path, mlo_path)

            # Move to device
            cc_tensor = cc_tensor.to(DEVICE)
            mlo_tensor = mlo_tensor.to(DEVICE)

            # Make prediction
            with torch.no_grad():
                logits, uncertainty = self.model(
                    cc_tensor, mlo_tensor, patient_metadata
                )

                # Convert to probabilities
                probability = torch.sigmoid(logits).cpu().numpy().item()
                uncertainty = uncertainty.cpu().numpy().item()

            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_predictions += 1

            # Create result
            result = {
                "probability": probability,
                "uncertainty": uncertainty,
                "inference_time": inference_time,
                "timestamp": datetime.now().isoformat(),
                "patient_metadata": patient_metadata or {},
            }

            return result

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return {
                "probability": 0.0,
                "uncertainty": 1.0,
                "inference_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "patient_metadata": patient_metadata or {},
            }

    def predict_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """Make predictions for a batch of patients"""

        start_time = time.time()
        results = []

        try:
            # Prepare batch tensors
            cc_tensors = []
            mlo_tensors = []
            patient_metadatas = []

            for item in batch_data:
                cc_path = item["cc_path"]
                mlo_path = item["mlo_path"]
                patient_metadata = item.get("patient_metadata")

                cc_tensor, mlo_tensor = self.preprocess_image_pair(cc_path, mlo_path)
                cc_tensors.append(cc_tensor)
                mlo_tensors.append(mlo_tensor)
                patient_metadatas.append(patient_metadata)

            # Stack tensors
            cc_batch = torch.cat(cc_tensors, dim=0).to(DEVICE)
            mlo_batch = torch.cat(mlo_tensors, dim=0).to(DEVICE)

            # Make predictions
            with torch.no_grad():
                logits, uncertainties = self.model(cc_batch, mlo_batch)

                # Convert to probabilities
                probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
                uncertainties = uncertainties.cpu().numpy().flatten()

            # Create results
            batch_time = time.time() - start_time
            for i, (prob, unc, metadata) in enumerate(
                zip(probabilities, uncertainties, patient_metadatas)
            ):
                result = {
                    "probability": prob,
                    "uncertainty": unc,
                    "inference_time": batch_time / len(batch_data),
                    "timestamp": datetime.now().isoformat(),
                    "patient_metadata": metadata or {},
                    "batch_index": i,
                }
                results.append(result)

            # Update performance tracking
            self.inference_times.append(batch_time)
            self.total_predictions += len(batch_data)

        except Exception as e:
            self.logger.error(f"Error during batch inference: {e}")
            # Return error results for all items
            for item in batch_data:
                result = {
                    "probability": 0.0,
                    "uncertainty": 1.0,
                    "inference_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "patient_metadata": item.get("patient_metadata", {}),
                }
                results.append(result)

        return results

    def start_batch_processing(self):
        """Start background batch processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._batch_processing_loop
            )
            self.processing_thread.start()
            self.logger.info("Started batch processing thread")

    def stop_batch_processing(self):
        """Stop background batch processing thread"""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
            self.logger.info("Stopped batch processing thread")

    def _batch_processing_loop(self):
        """Background thread for batch processing"""
        current_batch = []

        while self.is_running:
            try:
                # Get item from queue with timeout
                try:
                    item = self.input_queue.get(timeout=1.0)
                    current_batch.append(item)
                except:
                    # Timeout - process current batch if not empty
                    if current_batch:
                        results = self.predict_batch(current_batch)
                        for result in results:
                            self.output_queue.put(result)
                        current_batch = []
                    continue

                # Process batch if full or if we've been waiting
                if len(current_batch) >= self.batch_size:
                    results = self.predict_batch(current_batch)
                    for result in results:
                        self.output_queue.put(result)
                    current_batch = []

            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                # Clear current batch on error
                current_batch = []

    async def predict_async(
        self, cc_path: str, mlo_path: str, patient_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Asynchronous prediction"""

        loop = asyncio.get_event_loop()

        # Run prediction in thread pool
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            result = await loop.run_in_executor(
                executor, self.predict_single, cc_path, mlo_path, patient_metadata
            )

        return result

    def predict_directory(self, data_dir: str, output_path: str = None) -> pd.DataFrame:
        """Predict on all patients in a directory"""

        data_dir = Path(data_dir)
        if output_path is None:
            output_path = str(self.results_dir / "predictions.csv")

        results = []

        # Find all patient directories
        patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

        self.logger.info(f"Found {len(patient_dirs)} patient directories")

        for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
            try:
                # Find CC and MLO images
                cc_files = list(patient_dir.glob("*CC*.dcm"))
                mlo_files = list(patient_dir.glob("*MLO*.dcm"))

                if not cc_files or not mlo_files:
                    self.logger.warning(f"No CC/MLO images found in {patient_dir}")
                    continue

                # Use first CC and MLO images
                cc_path = str(cc_files[0])
                mlo_path = str(mlo_files[0])

                # Make prediction
                result = self.predict_single(cc_path, mlo_path)

                # Add patient info
                result["patient_id"] = patient_dir.name
                result["cc_path"] = cc_path
                result["mlo_path"] = mlo_path

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing {patient_dir}: {e}")
                continue

        # Create DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)

        self.logger.info(f"Predictions saved to {output_path}")

        return results_df

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""

        if not self.inference_times:
            return {}

        stats = {
            "total_predictions": self.total_predictions,
            "mean_inference_time": np.mean(self.inference_times),
            "std_inference_time": np.std(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "throughput_predictions_per_second": self.total_predictions
            / np.sum(self.inference_times),
        }

        return stats

    def save_model_info(self, output_path: str = None):
        """Save model information and configuration"""

        if output_path is None:
            output_path = self.results_dir / "model_info.json"

        model_info = {
            "model_path": str(self.model_path),
            "config": self.config,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "device": str(DEVICE),
            "image_size": (ROWS, COLS),
            "channels": CHANNELS,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "performance_stats": self.get_performance_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=2)

        self.logger.info(f"Model info saved to {output_path}")


class InferenceAPI:
    """Simple API wrapper for the inference engine"""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.engine = BreastCancerInferenceEngine(model_path, config_path)

    def predict(
        self, cc_path: str, mlo_path: str, patient_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make prediction"""
        return self.engine.predict_single(cc_path, mlo_path, patient_metadata)

    def predict_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """Make batch predictions"""
        return self.engine.predict_batch(batch_data)

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return self.engine.get_performance_stats()


def main():
    """Main inference function"""

    # Find the best model
    model_dir = Path(MODEL_DIR)
    model_files = list(model_dir.glob("*.ckpt"))

    if not model_files:
        print("No model files found!")
        return

    # Use the most recent model
    best_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Using model: {best_model}")

    # Initialize inference engine
    engine = BreastCancerInferenceEngine(str(best_model))

    # Example: predict on test directory
    test_dir = Path(DATA_DIR) / "test_images"
    if test_dir.exists():
        print(f"Making predictions on {test_dir}")
        results_df = engine.predict_directory(str(test_dir))

        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total predictions: {len(results_df)}")
        print(f"Mean probability: {results_df['probability'].mean():.4f}")
        print(f"Mean uncertainty: {results_df['uncertainty'].mean():.4f}")

        # Save model info
        engine.save_model_info()

        # Print performance stats
        stats = engine.get_performance_stats()
        print(f"\nPerformance Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")

    else:
        print(f"Test directory not found: {test_dir}")

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
