# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Image processing
import cv2
import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa

# Modelling
from sklearn.model_selection import train_test_split
import torch

# Miscallaneous
import hashlib
import time
from tqdm import tqdm
import warnings
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

# Local
from utils.constants import *



def hash_files(data_dir_path):
    """
    Renames files in data directory to its file hash.
    
    Parameters
    ----------
    data_dir_path : str
        Path to data dictionary
    """
    def sha1_file(file_path):
        """
        Create hashed name for file path
        """
        f = open(file_path, "rb")
        r = hashlib.sha1(f.read()).hexdigest()
        f.close()
        return r

    # List data directory
    data_dir = os.listdir(data_dir_path)

    # List sub directories in data directory
    for label_dir_path in data_dir:
        label_dir_path = os.path.join(data_dir_path, label_dir_path)

        # Get full relative file paths for images
        for img_name in os.listdir(label_dir_path):
            img_name_path = os.path.join(label_dir_path, img_name)
            
            # Rename image file with its hash
            if img_name_path.endswith(".jpg"):
                hexh = f"{os.path.join(label_dir_path, sha1_file(img_name_path))}.jpg"
                print(f"Renamed {img_name_path} to {hexh}")
                os.rename(img_name_path, hexh)
                
                
def create_metadata(data_dir_path: str) -> pd.DataFrame:
    """
    Parses through data directory and sub-directories to create a metadata csv,
    containing paths, names and labels.
    
    Parameters
    ----------
    data_dir_path : str
        Relative path to data directory.
        
    Returns
    -------
    pd.DataFrame
        Dataframe containing relevant metadata collected from data directory.
    """
    
    data_dir = os.listdir(data_dir_path)

    # Create empty dictionary 
    data_metadata = pd.DataFrame(columns=["img_name", "img_path", "label"])
    
    for label in data_dir:
        label_dir_path = os.path.join(data_dir_path, label)
        
        if os.path.isdir(label_dir_path):
            for img_name in os.listdir(label_dir_path):
                img_name_path = os.path.join(label_dir_path, img_name)
                
                if ".jpg" in img_name_path:
                    # Add to metadata
                    data_metadata = pd.concat(
                        [
                            data_metadata,
                            pd.DataFrame.from_dict(
                                {
                                    "img_name": [img_name], 
                                    "img_path": [img_name_path], 
                                    "label": [label]
                                }
                            )
                        ], ignore_index=True
                    )            
    
    return data_metadata


def read_images(data_dir_path: str, 
                rows: int=ROWS, 
                cols: int=COLS, 
                channels: int=CHANNELS, 
                write_images: bool=False, 
                output_data_dir_path: str=None,
                verbose: bool=VERBOSE)->np.array:
    """
    Reads all images in all labels/sub-directories in data_dir_path into np.array.
    
    Parameters
    -----------
    data_dir_path : str
        Path to image directory.
    rows : int
        Row (height) dimension to resize images to.
    cols : int
        Columns (width) dimension to resize images to.
    channels : int
        Number of image channels, e.g. RGB are 3 channels, GREY is 2 channels.
    write_images : bool
        Specifies whether to write new images to specified data directory
    output_data_dir_path : str
        If write_images is True, directory path to write new images to.
    verbose : bool
        If True, prints verbose logging.

    
    Returns
    -------
    np.array
        Array containing all images in directory.
    """
    
    assert len(os.listdir(data_dir_path)) > 0, "Empty parent directory."
    
    img_array = np.empty(shape=(0, ROWS, COLS, CHANNELS), dtype=np.uint8)
    
    if verbose:
        print(f"Reading images from: {data_dir_path}")
        print(f"Rows set to {rows}")
        print(f"Columns set to {cols}")
        print(f"Channels set to {channels}")
        if write_images:
            print("\nWriting images to disk!")
            print(f"Writing images to: {output_data_dir_path}")
            time.sleep(4)
        else:
            print(f"Writing images is set to: {write_images}")
        print("Reading images...")
    
    # Loop through all labels and images
    for label in os.listdir(data_dir_path):
        # Label/sub dir path
        label_path = os.path.join(data_dir_path, label)

        # Check for directory
        if os.path.isdir(label_path) & (label != "."):
            for img_name in tqdm(os.listdir(label_path)):
                # Individual image path
                img_path = os.path.join(label_path, img_name)
                # Read image, resize and append to array
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (cols, rows))
                img_expand = np.expand_dims(img, axis=0)
                
                # Create directories and write images if specified
                if write_images:
                    label_path_write = os.path.join(output_data_dir_path, label)
                    img_path_write = os.path.join(label_path_write, img_name)

                    if os.path.isdir(label_path_write):
                        cv2.imwrite(img_path_write, img)
                    else:
                        os.mkdir(label_path_write)
                        cv2.imwrite(img_path_write, img)
                
                img_array = np.append(img_array, img_expand, axis=0)
            
    time.sleep(1)
    if verbose:
        print("Image reading complete.")
        print(f"Image array shape: {img_array.shape}")
    
    return img_array


def display_img(img: np.array=None) -> None:
    """
    View multiple images stored in files, stacking vertically

    Parameters
    ----------
    img : np.array 
        Numpy array containing image to display.
            
    Returns
    -------
    plt.figure
        Plotted image.
    """
    plt.figure()
    plt.imshow(img)
    
    
def get_train_test_split(metadata_df: pd.DataFrame(), 
                         test_size: float=TEST_SIZE,
                         verbose: bool=VERBOSE):
    """
    Get indices and labels for train and test splits.

    Parameters
    ----------
    metadata_df : np.array 
        DataFrame containing metadata (labels included).
    test_size : float 
        Proportion of test images to return.
    verbose : bool
        If True, prints verbose logging.

    Returns
    -------
    train_idx: list
        List of indimosquito', 'horsefly', 'antces for training split.
    test_idx: list
        List of indices for testing split.
    y_train: np.array
        Array of labels for training split.
    y_test: np.array
        Array of labels for testing split.
    
    """
    # Split train/test images
    train_idx, test_idx = train_test_split(
        metadata_df.index, 
        shuffle=True, 
        random_state=SEED, 
        test_size=test_size,
        stratify=metadata_df["label"]
    )
    train_idx, test_idx = list(train_idx), list(test_idx)
    y_train, y_test = metadata_df["label"][train_idx], metadata_df["label"][test_idx]

    if verbose:
        print(f"{len(train_idx)} train images")
        print(f"{len(test_idx)} test images")
        print("\nTRAIN IMAGE COUNTS\n" + "-"*18)
        print(y_train.value_counts())
        print("\nTEST IMAGE COUNTS\n" + "-"*18)
        print(y_test.value_counts())

    # Convert labels to np.array
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    return train_idx, test_idx, y_train, y_test
    

# Can't think of a better way to save the augmentations dictionary than this
# Given that it doesn't just contain simple objects, but actual functions
augs = {
    "Fliplr": {"aug": iaa.Fliplr, "args": {"p": 1.0}}, 
    "Flipud": {"aug": iaa.Flipud,"args": {"p": 1.0}}, 
    "GaussianBlur": {"aug": iaa.GaussianBlur,"args": {"sigma": 2.0}}, 
    "AverageBlur": {"aug": iaa.AverageBlur,"args": {"k": 5.0}}, 
    "MotionBlur": {"aug": iaa.MotionBlur,"args": {"k": 8}}, 
    "MultiplyBrightness": {"aug": iaa.MultiplyBrightness,"args": {"mul": 0.8}}, 
    "MultiplyHue": {"aug": iaa.MultiplyHue,"args": {"mul": 1.2}}, 
    "MultiplySaturation": {"aug": iaa.MultiplySaturation,"args": {"mul": 0.4}}, 
    "Grayscale": {"aug": iaa.Grayscale,"args": {"alpha": 1}}, 
    "GammaContrast": {"aug": iaa.GammaContrast,"args": {"gamma": 0.6}}, 
    "SigmoidContrast": {"aug": iaa.SigmoidContrast,"args": {"gain": 2.0}}, 
    "LinearContrast": {"aug": iaa.LinearContrast,"args": {"alpha": 0.6}}, 
    "Affine": {"aug": iaa.Affine,"args": {"scale": 0.8}}, 
    "ScaleX": {"aug": iaa.ScaleX,"args": {"scale": 0.8}}, 
    "ScaleY": {"aug": iaa.ScaleY,"args": {"scale": 0.8}}, 
    "TranslateX": {"aug": iaa.TranslateX,"args": {"percent": 0.1}}, 
    "TranslateY": {"aug": iaa.TranslateY,"args": {"percent": 0.1}}, 
    "Rotate": {"aug": iaa.Rotate,"args": {"rotate": 90.0}}, 
    "ShearX": {"aug": iaa.ShearX,"args": {"shear": 20.0}}, 
    "ShearY": {"aug": iaa.ShearY,"args": {"shear": 20.0}},
    "GaussianNoise": {"aug": iaa.imgcorruptlike.GaussianNoise,"args": {"severity": 1}}, 
    "ShotNoise": {"aug": iaa.imgcorruptlike.ShotNoise,"args": {"severity": 1}}, 
    "ImpulseNoise": { "aug": iaa.imgcorruptlike.ImpulseNoise,"args": {"severity": 1}}, 
    "SpeckleNoise": {"aug": iaa.imgcorruptlike.SpeckleNoise,"args": {"severity": 1}}, 
    "DefocusBlur": {"aug": iaa.imgcorruptlike.DefocusBlur,"args": {"severity": 2}}, 
    "ZoomBlur": {"aug": iaa.imgcorruptlike.ZoomBlur,"args": {"severity": 1}}, 
    "Contrast": {"aug": iaa.imgcorruptlike.Contrast,"args": {"severity": 1}}, 
    "Brightness": {"aug": iaa.imgcorruptlike.Brightness,"args": {"severity": 1}},
    "Saturate": {"aug": iaa.imgcorruptlike.Saturate,"args": {"severity": 1}}, 
    "Solarize": {"aug": iaa.Solarize,"args": {"threshold": 1}}, 
    "EnhanceColor": {"aug": iaa.pillike.EnhanceColor,"args": {"factor": 0.5}}, 
    "EnhanceContrast": {"aug": iaa.pillike.EnhanceContrast,"args": {"factor": 0.7}}, 
    "EnhanceBrightness": {"aug": iaa.pillike.EnhanceBrightness,"args": {"factor": 0.7}}, 
    "EnhanceSharpness": {"aug": iaa.pillike.EnhanceSharpness,"args": {"factor": 4.0}}, 
    "AdditiveGaussianNoise": {"aug": iaa.AdditiveGaussianNoise,"args": {"loc": 2.0}}, 
    "AdditiveLaplaceNoise": {"aug": iaa.AdditiveLaplaceNoise,"args": {"loc": 2.0}}, 
    "AdditivePoissonNoise": {"aug": iaa.AdditivePoissonNoise,"args": {"lam": 10.0}}, 
    "Cutout": {"aug": iaa.Cutout,"args": {"nb_iterations": 1,"size": 0.2,"fill_mode": "gaussian","fill_per_channel": True}}, 
    "Dropout": {"aug": iaa.Dropout,"args": {"p": 0.02,"per_channel": 0.2}}, 
    "CoarseDropout": {"aug": iaa.CoarseDropout,"args": {"size_percent": 0.4}}, 
    "ImpulseNoise": {"aug": iaa.ImpulseNoise,"args": {"p": 0.05}}, 
    "SaltAndPepper": {"aug": iaa.SaltAndPepper,"args": {"p": 0.03}}
}

def get_augs(imgs_raw: np.array, 
             labels_raw: np.array, 
             augs: dict,
             keep_originals: bool=True, 
             verbose: bool=True) -> np.array:
    """
    Reads raw images and returns array containing augmented images.
    
    Parameters
    ----------
    imgs_raw : np.array
        Array of raw images to augment.
    labels_raw : np.array
        Array of raw labels, retaining order in imgs_raw.
    augs : dict
        Dictionary of augmentations to perform, including arguments, in specified format.        
    keep_originals : bool
        If True, appends augmented images to original array, otherwise only returns augmented images.
    verbose : bool
        If True, prints verbose logging.
    
    Returns
    -------
    imgs_aug : np.array
        Array containing augmented images. 
    labels_aug : np.array
        Array containing augmented labels, replicating order in imgs_aug.
    """
    # Unpack augmentations
    augs_list = []
    for aug_name in augs:
        # Get augmentation
        aug = augs[aug_name]["aug"]
        # Get args dictionary for augmentation
        args = augs[aug_name]["args"]
        augs_list.append(aug(**args))
    num_augs = len(augs_list)
    
    
    if keep_originals == True:
        # Create augmentations and add to array with original images
        imgs_aug = imgs_raw.copy()
        for aug in augs_list:
            imgs_aug = np.concatenate(
                (
                    imgs_aug, 
                    aug(images=imgs_raw)
                ),
                axis=0
            )
        
        # Count number of augmentations
        labels_aug = np.concatenate(
            (
                labels_raw,
                np.array([labels_raw for i in range(num_augs)]).flatten()
            ),
            axis=0
        )

    elif keep_originals == False:
        # Create augmentations and add to array without original images
        imgs_aug = np.array([]).reshape(0, imgs_raw.shape[0], imgs_raw.shape[1], imgs_raw.shape[2])
        for aug in augs_list:
            imgs_aug = np.concatenate(
                (
                    imgs_aug, 
                    aug(images=imgs_raw)
                ),
                axis=0
            )
        
        # Count number of augmentations
        labels_aug = np.array([labels_raw for i in range(num_augs)]).flatten()
        
    # Logging
    if verbose:
        print(f"Used augs: {list(augs)}")
        print(f"Created {imgs_aug.shape[0] - imgs_raw.shape[0]} augmentations.")
        print(f"Image array shape: {imgs_aug.shape}")
        print(f"Labels array shape: {labels_aug.shape}")
    
    return imgs_aug, labels_aug


def seed_reproducer(seed=SEED):
    """
    Reproducer for pytorch experiment.
    
    Parameters
    ----------
    seed: int, optional (default = 14)
        Random seed.
        
    Example
    ----------
    seed_reproducer(seed=14).
    
    Returns
    ----------
    None
    
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        

def mkdir(path: str):
    """Create directory.
     Create directory if it is not exist, else do nothing.
     Parameters
     ----------
     path: str
        Path of your directory.
     Examples
     --------
     mkdir("data/raw/train/")
     """
    try:
        if path is None:
            pass
        else:
            os.stat(path)
    except Exception:
        os.makedirs(path)
        
        
def init_logger(log_name, log_dir=LOG_DIR):
    """
    ADD DOCSTRING
    """

    mkdir(
        os.path.join(log_dir, log_name)
    )

    if log_name not in Logger.manager.loggerDict:
        logging.root.handlers.clear()
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s] %(filename)s[%(lineno)4s] : %(levelname)s  %(message)s"
        formatter = logging.Formatter(format_str, datefmt)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            file_info_handler = TimedRotatingFileHandler(
                filename=os.path.join(log_dir, log_name, "%s.log" % log_name), when="D", backupCount=7
            )
            file_info_handler.setFormatter(formatter)
            file_info_handler.setLevel(logging.INFO)
            logger.addHandler(file_info_handler)

    logger = logging.getLogger(log_name)

    return logger

