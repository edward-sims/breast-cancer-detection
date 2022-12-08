import os
from time import time

# Third party libraries
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A

from torch.utils.data import DataLoader, Dataset

from utils.constants import ROWS, COLS, CHANNELS


class BiteDataset(Dataset):
    """ 
    ADD DOCSTRING
    """

    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        start_time = time()
        # Read image
        image = cv2.cvtColor(
            cv2.imread(self.data.iloc[index, 0]), 
            cv2.COLOR_BGR2RGB
        )
        
        # Convert if not the right shape
        if image.shape != (ROWS, COLS, CHANNELS):
            image = image.transpose(1, 0, 2)

        # Perform augmentation
        if self.transforms is not None:
            image = self.transforms(image=image)["image"].transpose(2, 0, 1)

        label = torch.FloatTensor(self.data.iloc[index, 1:].values.astype(np.int64))

        return image, label, time() - start_time

    def __len__(self):
        return len(self.data)


def generate_transforms(image_size):
    """
    ADD DOCSTRING
    """
    
    train_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            #A.RandomBrightnessContrast(p=0.5),
            #A.OneOf([A.MotionBlur(), A.MedianBlur(), A.GaussianBlur()], p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )
    

    return {"train_transforms": train_transform, "val_transforms": val_transform}


def generate_dataloaders(hparams, train_data, val_data, transforms):
    """
    ADD DOCSTRING
    """
    
    train_dataset = BiteDataset(
        data=train_data, transforms=transforms["train_transforms"]
    )
    
    val_dataset = BiteDataset(
        data=val_data, transforms=transforms["train_transforms"]
    )
    
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