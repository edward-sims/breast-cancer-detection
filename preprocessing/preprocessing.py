import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pydicom
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import ROWS, COLS, CHANNELS, DATA_DIR, SEED


class MedicalImagePreprocessor:
    """Advanced preprocessing pipeline for medical imaging data"""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        output_dir: str = "data/processed",
        config: Dict = None,
    ):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Statistics tracking
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "file_sizes": [],
            "processing_times": [],
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "preprocessing.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def load_dicom_metadata(self, dicom_path: str) -> Dict[str, Any]:
        """Extract metadata from DICOM file"""
        try:
            # Try to read with force=True for files without proper DICOM headers
            dcm = pydicom.dcmread(dicom_path, force=True)

            metadata = {
                "patient_id": getattr(dcm, "PatientID", "Unknown"),
                "study_date": getattr(dcm, "StudyDate", "Unknown"),
                "modality": getattr(dcm, "Modality", "Unknown"),
                "manufacturer": getattr(dcm, "Manufacturer", "Unknown"),
                "bits_allocated": getattr(dcm, "BitsAllocated", 16),
                "pixel_spacing": getattr(dcm, "PixelSpacing", [1.0, 1.0]),
                "window_center": getattr(dcm, "WindowCenter", 0),
                "window_width": getattr(dcm, "WindowWidth", 1),
                "image_size": (
                    dcm.pixel_array.shape if hasattr(dcm, "pixel_array") else None
                ),
            }

            return metadata

        except Exception as e:
            self.logger.error(f"Error loading DICOM metadata from {dicom_path}: {e}")
            return {}

    def preprocess_dicom_image(
        self, dicom_path: str, target_size: Tuple[int, int] = (ROWS, COLS)
    ) -> Optional[np.ndarray]:
        """Preprocess a single DICOM image"""
        try:
            # Load DICOM with force=True for files without proper headers
            dcm = pydicom.dcmread(dicom_path, force=True)

            # Extract pixel data
            image = dcm.pixel_array

            # Handle different bit depths
            if hasattr(dcm, "BitsAllocated") and dcm.BitsAllocated == 16:
                image = image.astype(np.float32) / 65535.0
            else:
                image = image.astype(np.float32) / 255.0

            # Apply windowing if available
            if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
                window_center = float(dcm.WindowCenter)
                window_width = float(dcm.WindowWidth)

                # Apply windowing
                min_val = window_center - window_width / 2
                max_val = window_center + window_width / 2
                image = np.clip(image, min_val, max_val)
                image = (image - min_val) / (max_val - min_val)

            # Normalize to [0, 1]
            image = np.clip(image, 0, 1)

            # Resize image
            if image.shape != target_size:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

            # Convert to 3-channel if grayscale
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            return image

        except Exception as e:
            self.logger.error(f"Error preprocessing DICOM {dicom_path}: {e}")
            return None

    def apply_medical_augmentations(
        self, image: np.ndarray, augmentations: List[str] = None
    ) -> np.ndarray:
        """Apply medical imaging specific augmentations"""

        if augmentations is None:
            augmentations = ["normalize", "contrast_enhancement"]

        processed_image = image.copy()

        for aug in augmentations:
            if aug == "normalize":
                # Histogram normalization
                processed_image = self._histogram_normalization(processed_image)

            elif aug == "contrast_enhancement":
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                processed_image = self._clahe_enhancement(processed_image)

            elif aug == "noise_reduction":
                # Gaussian blur for noise reduction
                processed_image = cv2.GaussianBlur(processed_image, (3, 3), 0)

            elif aug == "edge_enhancement":
                # Unsharp masking for edge enhancement
                processed_image = self._unsharp_masking(processed_image)

        return processed_image

    def _histogram_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram normalization"""
        # Convert to grayscale for histogram calculation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate histogram
        hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 1])

        # Calculate cumulative distribution
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()

        # Apply histogram equalization
        equalized = np.interp(gray.flatten(), bins[:-1], cdf_normalized)
        equalized = equalized.reshape(gray.shape)

        # Convert back to 3-channel if needed
        if len(image.shape) == 3:
            equalized = np.stack([equalized] * 3, axis=-1)

        return equalized

    def _clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement"""
        # Convert to grayscale for CLAHE
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Convert to uint8 for CLAHE
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_uint8)

        # Convert back to float
        enhanced = enhanced.astype(np.float32) / 255.0

        # Convert back to 3-channel if needed
        if len(image.shape) == 3:
            enhanced = np.stack([enhanced] * 3, axis=-1)

        return enhanced

    def _unsharp_masking(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for edge enhancement"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Create blurred version
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)

        # Calculate unsharp mask
        unsharp_mask = gray - blurred

        # Apply mask
        enhanced = gray + 0.5 * unsharp_mask
        enhanced = np.clip(enhanced, 0, 1)

        # Convert back to 3-channel if needed
        if len(image.shape) == 3:
            enhanced = np.stack([enhanced] * 3, axis=-1)

        return enhanced

    def process_patient_directory(
        self, patient_dir: Path, patient_id: str
    ) -> Dict[str, Any]:
        """Process all images for a single patient"""

        patient_data = {
            "patient_id": patient_id,
            "images": [],
            "metadata": {},
            "processing_stats": {},
        }

        # Find all DICOM files
        dicom_files = list(patient_dir.glob("*.dcm"))

        if not dicom_files:
            self.logger.warning(f"No DICOM files found in {patient_dir}")
            return patient_data

        # Process each image
        for dicom_file in dicom_files:
            try:
                # Load metadata
                metadata = self.load_dicom_metadata(str(dicom_file))

                # Preprocess image
                processed_image = self.preprocess_dicom_image(str(dicom_file))

                if processed_image is not None:
                    # Apply augmentations
                    enhanced_image = self.apply_medical_augmentations(processed_image)

                    # Save processed image
                    output_path = (
                        self.output_dir
                        / patient_id
                        / f"{dicom_file.stem}_processed.npy"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, enhanced_image)

                    # Update patient data
                    patient_data["images"].append(
                        {
                            "original_path": str(dicom_file),
                            "processed_path": str(output_path),
                            "metadata": metadata,
                            "shape": enhanced_image.shape,
                        }
                    )

                    self.stats["processed_images"] += 1
                else:
                    self.stats["failed_images"] += 1

                self.stats["total_images"] += 1

            except Exception as e:
                self.logger.error(f"Error processing {dicom_file}: {e}")
                self.stats["failed_images"] += 1
                self.stats["total_images"] += 1

        return patient_data

    def process_dataset(self, dataset_type: str = "train") -> Dict[str, Any]:
        """Process entire dataset"""

        dataset_dir = self.data_dir / f"{dataset_type}_images"

        if not dataset_dir.exists():
            self.logger.error(f"Dataset directory not found: {dataset_dir}")
            return {}

        self.logger.info(f"Processing {dataset_type} dataset from {dataset_dir}")

        # Find all patient directories
        patient_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        dataset_data = {"dataset_type": dataset_type, "patients": {}, "statistics": {}}

        # Process each patient
        for patient_dir in tqdm(
            patient_dirs, desc=f"Processing {dataset_type} patients"
        ):
            patient_id = patient_dir.name

            try:
                patient_data = self.process_patient_directory(patient_dir, patient_id)
                dataset_data["patients"][patient_id] = patient_data

            except Exception as e:
                self.logger.error(f"Error processing patient {patient_id}: {e}")

        # Calculate dataset statistics
        dataset_data["statistics"] = self._calculate_dataset_statistics(dataset_data)

        # Save dataset info
        dataset_info_path = self.output_dir / f"{dataset_type}_dataset_info.json"
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_data, f, indent=2, default=str)

        self.logger.info(
            f"Dataset processing completed. Info saved to {dataset_info_path}"
        )

        return dataset_data

    def _calculate_dataset_statistics(
        self, dataset_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics"""

        stats = {
            "total_patients": len(dataset_data["patients"]),
            "total_images": 0,
            "image_shapes": [],
            "file_sizes": [],
            "modalities": [],
            "manufacturers": [],
        }

        for patient_id, patient_data in dataset_data["patients"].items():
            stats["total_images"] += len(patient_data["images"])

            for image_data in patient_data["images"]:
                # Image shapes
                if "shape" in image_data:
                    stats["image_shapes"].append(image_data["shape"])

                # File sizes
                if os.path.exists(image_data["processed_path"]):
                    file_size = os.path.getsize(image_data["processed_path"])
                    stats["file_sizes"].append(file_size)

                # Metadata
                metadata = image_data.get("metadata", {})
                if "modality" in metadata:
                    stats["modalities"].append(metadata["modality"])
                if "manufacturer" in metadata:
                    stats["manufacturers"].append(metadata["manufacturer"])

        # Calculate summary statistics
        if stats["image_shapes"]:
            stats["unique_shapes"] = list(set(map(tuple, stats["image_shapes"])))

        if stats["file_sizes"]:
            stats["file_size_stats"] = {
                "mean": np.mean(stats["file_sizes"]),
                "std": np.std(stats["file_sizes"]),
                "min": np.min(stats["file_sizes"]),
                "max": np.max(stats["file_sizes"]),
            }

        if stats["modalities"]:
            stats["modality_counts"] = (
                pd.Series(stats["modalities"]).value_counts().to_dict()
            )

        if stats["manufacturers"]:
            stats["manufacturer_counts"] = (
                pd.Series(stats["manufacturers"]).value_counts().to_dict()
            )

        return stats

    def create_visualization_report(
        self, dataset_data: Dict[str, Any], output_path: str = None
    ):
        """Create comprehensive visualization report"""

        if output_path is None:
            output_path = self.output_dir / "preprocessing_report.html"

        # Create HTML report
        html_content = self._generate_html_report(dataset_data)

        with open(output_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Visualization report saved to {output_path}")

    def _generate_html_report(self, dataset_data: Dict[str, Any]) -> str:
        """Generate HTML report with visualizations"""

        # Extract statistics safely
        train_stats = dataset_data.get("train", {}).get("statistics", {})
        test_stats = dataset_data.get("test", {}).get("statistics", {})

        total_patients = train_stats.get("total_patients", 0) + test_stats.get(
            "total_patients", 0
        )
        total_images = train_stats.get("total_images", 0) + test_stats.get(
            "total_images", 0
        )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical Image Preprocessing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .stat-card {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Medical Image Preprocessing Report</h1>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="stats">
                    <div class="stat-card">
                        <h3>Total Patients</h3>
                        <p>{total_patients}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Total Images</h3>
                        <p>{total_images}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Training Patients</h3>
                        <p>{train_stats.get('total_patients', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Test Patients</h3>
                        <p>{test_stats.get('total_patients', 0)}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Processing Results</h2>
                <div class="stats">
                    <div class="stat-card">
                        <h3>Training Images</h3>
                        <p>{train_stats.get('total_images', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Test Images</h3>
                        <p>{test_stats.get('total_images', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Processing Status</h3>
                        <p>Completed</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html


def main():
    """Main preprocessing function"""

    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor()

    # Process training dataset
    train_data = preprocessor.process_dataset("train")

    # Process test dataset if available
    test_data = preprocessor.process_dataset("test")

    # Create visualization report
    preprocessor.create_visualization_report(train_data)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
