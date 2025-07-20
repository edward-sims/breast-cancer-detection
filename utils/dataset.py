import os
import pydicom
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

from utils.constants import ROWS, COLS, CHANNELS, AUGMENTATION_CONFIG


class MedicalImageDataset(Dataset):
    """Advanced medical imaging dataset for breast cancer detection"""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_dir: str,
        transforms: Optional[A.Compose] = None,
        is_train: bool = True,
        patient_level: bool = True,
    ):
        """
        Initialize the dataset

        Args:
            metadata_df: DataFrame with image metadata
            data_dir: Directory containing DICOM images
            transforms: Albumentations transforms
            is_train: Whether this is training data
            patient_level: Whether to return patient-level data
        """
        self.metadata_df = metadata_df
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.is_train = is_train
        self.patient_level = patient_level

        # Group by patient if patient-level
        if patient_level:
            self.patient_groups = self._group_by_patient()
        else:
            self.patient_groups = None

        self.logger = logging.getLogger(__name__)

    def _group_by_patient(self) -> List[Dict]:
        """Group images by patient for patient-level prediction"""
        patient_groups = []

        for patient_id in self.metadata_df["patient_id"].unique():
            patient_data = self.metadata_df[
                self.metadata_df["patient_id"] == patient_id
            ]

            # Separate CC and MLO views
            cc_images = patient_data[patient_data["view"] == "CC"]
            mlo_images = patient_data[patient_data["view"] == "MLO"]

            # Ensure we have both views
            if len(cc_images) > 0 and len(mlo_images) > 0:
                patient_groups.append(
                    {
                        "patient_id": patient_id,
                        "cc_images": cc_images.to_dict("records"),
                        "mlo_images": mlo_images.to_dict("records"),
                        "cancer": (
                            patient_data["cancer"].iloc[0]
                            if "cancer" in patient_data.columns
                            else None
                        ),
                        "age": patient_data["age"].iloc[0],
                        "sex": patient_data["sex"].iloc[0],
                    }
                )

        return patient_groups

    def _load_dicom_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess DICOM image"""
        try:
            # Load DICOM file with force=True to handle missing headers
            dcm = pydicom.dcmread(image_path, force=True)

            # Extract pixel data
            image = dcm.pixel_array

            # Handle different bit depths
            if hasattr(dcm, "BitsAllocated") and dcm.BitsAllocated == 16:
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
            self.logger.warning(f"Error loading DICOM {image_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((ROWS, COLS, CHANNELS), dtype=np.float32)

    def _load_image_pair(
        self, cc_record: Dict, mlo_record: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load CC and MLO image pair"""

        # Construct image paths
        cc_path = (
            self.data_dir / cc_record["patient_id"] / f"{cc_record['image_id']}.dcm"
        )
        mlo_path = (
            self.data_dir / mlo_record["patient_id"] / f"{mlo_record['image_id']}.dcm"
        )

        # Load images
        cc_image = self._load_dicom_image(str(cc_path))
        mlo_image = self._load_dicom_image(str(mlo_path))

        return cc_image, mlo_image

    def __len__(self) -> int:
        if self.patient_level:
            return len(self.patient_groups) if self.patient_groups is not None else 0
        else:
            return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Get a sample from the dataset"""

        if self.patient_level:
            return self._get_patient_sample(idx)
        else:
            return self._get_image_sample(idx)

    def _get_patient_sample(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get patient-level sample with both views"""

        if self.patient_groups is None:
            raise ValueError("Patient groups not initialized")
        patient_data = self.patient_groups[idx]

        # Select random CC and MLO images for this patient
        cc_record = np.random.choice(patient_data["cc_images"])
        mlo_record = np.random.choice(patient_data["mlo_images"])

        # Load image pair
        cc_image, mlo_image = self._load_image_pair(cc_record, mlo_record)

        # Apply transforms
        if self.transforms:
            # Apply same transforms to both views for consistency
            transformed = self.transforms(image=cc_image, image1=mlo_image)
            cc_image = transformed["image"]
            mlo_image = transformed["image1"]

        # Convert to tensors
        cc_tensor = torch.from_numpy(cc_image.transpose(2, 0, 1)).float()
        mlo_tensor = torch.from_numpy(mlo_image.transpose(2, 0, 1)).float()

        # Create label tensor
        if patient_data["cancer"] is not None:
            label = torch.tensor([patient_data["cancer"]], dtype=torch.float32)
        else:
            label = torch.tensor(
                [0.0], dtype=torch.float32
            )  # Placeholder for test data

        return cc_tensor, mlo_tensor, label

    def _get_image_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single image sample"""

        record = self.metadata_df.iloc[idx]

        # Construct image path
        image_path = self.data_dir / record["patient_id"] / f"{record['image_id']}.dcm"

        # Load image
        image = self._load_dicom_image(str(image_path))

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Create label tensor
        if "cancer" in record:
            label = torch.tensor([record["cancer"]], dtype=torch.float32)
        else:
            label = torch.tensor([0.0], dtype=torch.float32)

        return image_tensor, label


def create_medical_transforms(
    image_size: Tuple[int, int], is_train: bool = True, config: Dict = None
) -> A.Compose:
    """Create medical imaging specific transforms"""

    if config is None:
        config = AUGMENTATION_CONFIG

    if is_train:
        transforms = A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1]),
                # Geometric transforms
                A.ShiftScaleRotate(
                    shift_limit=config.get("shift_limit", 0.1),
                    scale_limit=config.get("scale_limit", 0.1),
                    rotate_limit=config.get("rotation_limit", 15),
                    p=config.get("p", 0.5),
                ),
                # Intensity transforms
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=config.get("brightness_limit", 0.2),
                            contrast_limit=config.get("contrast_limit", 0.2),
                            p=1.0,
                        ),
                        A.HueSaturationValue(p=1.0),
                        A.RGBShift(p=1.0),
                    ],
                    p=0.5,
                ),
                # Noise and blur
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=7, p=1.0),
                        A.MedianBlur(blur_limit=5, p=1.0),
                    ],
                    p=0.3,
                ),
                # Medical imaging specific
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=1.0),
                    ],
                    p=0.2,
                ),
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                    p=1.0,
                ),
            ]
        )
    else:
        transforms = A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                    p=1.0,
                ),
            ]
        )

    return transforms


def create_multi_view_transforms(
    image_size: Tuple[int, int], is_train: bool = True
) -> Dict[str, A.Compose]:
    """Create transforms for multi-view fusion"""

    base_transforms = create_medical_transforms(image_size, is_train)

    # For multi-view, we want consistent transforms across views
    if is_train:
        # Create a shared transform that applies the same augmentation to both views
        multi_view_transform = A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1]),
                # Geometric transforms (same for both views)
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Intensity transforms (same for both views)
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.HueSaturationValue(p=1.0),
                        A.RGBShift(p=1.0),
                    ],
                    p=0.5,
                ),
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                    p=1.0,
                ),
            ],
            additional_targets={"image1": "image"},
        )
    else:
        multi_view_transform = A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0,
                    p=1.0,
                ),
            ],
            additional_targets={"image1": "image"},
        )

    return {
        "train_transforms": multi_view_transform,
        "val_transforms": multi_view_transform,
    }


def create_dataloaders(
    metadata_df: pd.DataFrame,
    data_dir: str,
    batch_size: int,
    image_size: Tuple[int, int] = (ROWS, COLS),
    is_train: bool = True,
    patient_level: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloaders for the medical imaging dataset"""

    # Create transforms
    if patient_level:
        transforms = create_multi_view_transforms(image_size, is_train)
        train_transforms = transforms["train_transforms"]
    else:
        train_transforms = create_medical_transforms(image_size, is_train)

    # Create dataset
    dataset = MedicalImageDataset(
        metadata_df=metadata_df,
        data_dir=data_dir,
        transforms=train_transforms,
        is_train=is_train,
        patient_level=patient_level,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )

    return dataloader


# Legacy dataset for compatibility
class BiteDataset(Dataset):
    """Legacy dataset for backward compatibility"""

    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        # Read image
        image = cv2.cvtColor(cv2.imread(self.data.iloc[index, 0]), cv2.COLOR_BGR2RGB)

        # Convert if not the right shape
        if image.shape != (ROWS, COLS, CHANNELS):
            image = image.transpose(1, 0, 2)

        # Perform augmentation
        if self.transforms is not None:
            image = self.transforms(image=image)["image"].transpose(2, 0, 1)

        label = torch.FloatTensor(self.data.iloc[index, 1:].values.astype(np.int64))

        return image, label

    def __len__(self):
        return len(self.data)


def generate_transforms(image_size):
    """Legacy transform generation for compatibility"""

    train_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
        ]
    )

    return {"train_transforms": train_transform, "val_transforms": val_transform}


def generate_dataloaders(hparams, train_data, val_data, transforms):
    """Legacy dataloader generation for compatibility"""

    train_dataset = BiteDataset(
        data=train_data, transforms=transforms["train_transforms"]
    )

    val_dataset = BiteDataset(data=val_data, transforms=transforms["train_transforms"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.train_batch_size,
        shuffle=True,
        num_workers=hparams.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.val_batch_size,
        shuffle=False,
        num_workers=hparams.n_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader
