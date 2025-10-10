from typing import Tuple, Optional
import numpy as np
import os
import pickle
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from config import mean, PROJECT_ROOT
from pathlib import Path

def unpickle(file):
    """Unpickle CIFAR-10 batch files"""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def compute_cifar100_stats_from_pickles(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean and standard deviation for CIFAR-10 dataset
    directly from pickle files (more efficient than loading individual images).
    
    Args:
        data_dir: Path to directory containing CIFAR-10 pickle files
        
    Returns:
        Tuple of (mean, std) where each is a numpy array of shape (3,) for RGB channels
    """
    print("Computing CIFAR-100 dataset statistics from pickle files...")
    # Lists to store all pixel values for each channel
    all_data = []

    # Process training data
    print("Loading training data...")
    train_file = os.path.join(data_dir, f'train')
    train_dict = unpickle(train_file)
    batch_data = train_dict[b'data']  # Shape: (50000, 3072)
    all_data.append(batch_data)
    print(f"Loaded training data")

    # Process test data
    print("Loading test data...")
    test_file = os.path.join(data_dir, 'test')
    test_dict = unpickle(test_file)
    test_data = test_dict[b'data']  # Shape: (10000, 3072)
    all_data.append(test_data)
    print("Loaded test data")

    # Combine all data
    combined_data = np.vstack(all_data)  # Shape: (60000, 3072)
    print(f"Combined data shape: {combined_data.shape}")

    # Reshape to separate RGB channels
    # CIFAR-10 0format: first 1024 values = red, next 1024 = green, last 1024 = blue
    red_channel = combined_data[:, :1024]      # Shape: (60000, 1024)
    green_channel = combined_data[:, 1024:2048]  # Shape: (60000, 1024)
    blue_channel = combined_data[:, 2048:3072]   # Shape: (60000, 1024)

    # Compute statistics for each channel
    # Convert to float32 and normalize to [0, 1] range
    red_mean = np.mean(red_channel.astype(np.float32) / 255.0)
    red_std = np.std(red_channel.astype(np.float32) / 255.0)
    
    green_mean = np.mean(green_channel.astype(np.float32) / 255.0)
    green_std = np.std(green_channel.astype(np.float32) / 255.0)
    
    blue_mean = np.mean(blue_channel.astype(np.float32) / 255.0)
    blue_std = np.std(blue_channel.astype(np.float32) / 255.0)
    
    # Combine into arrays
    mean = np.array([red_mean, green_mean, blue_mean])
    std = np.array([red_std, green_std, blue_std])

    print("\nCIFAR-100 Dataset Statistics:")
    print(f"Mean (RGB): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std  (RGB): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    
    return mean, std

def compute_cifar100_stats_from_images(images_dir: str, 
                                     csv_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean and standard deviation for CIFAR-10 dataset
    from individual image files.
    
    Args:
        images_dir: Path to directory containing train and test subdirectories with images
        csv_file: Optional path to CSV file containing image paths. If None, scans directories.
        
    Returns:
        Tuple of (mean, std) where each is a numpy array of shape (3,) for RGB channels
    """
    print("Computing CIFAR-10 dataset statistics from image files...")
    
    # Collect all image paths
    image_paths = []
    
    if csv_file and os.path.exists(csv_file):
        # Load from CSV
        df = pd.read_csv(csv_file)
        image_paths = [os.path.join(images_dir, path) for path in df['image_path']]
    else:
        # Scan directories
        train_dir = os.path.join(images_dir, 'train')
        test_dir = os.path.join(images_dir, 'test')
        
        for directory in [train_dir, test_dir]:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(directory, filename))
    
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # Initialize accumulators for online computation of mean and std
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    
    # Process images in batches to manage memory
    batch_size = 1000
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        for j, img_path in enumerate(batch_paths):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                
                # Flatten spatial dimensions, keep channel dimension
                img_flat = img_array.reshape(-1, 3)  # Shape: (1024, 3) for 32x32 images
                
                # Update accumulators
                pixel_sum += np.sum(img_flat, axis=0)
                pixel_sum_sq += np.sum(img_flat ** 2, axis=0)
                total_pixels += img_flat.shape[0]
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if (i + len(batch_paths)) % 5000 == 0:
            print(f"Processed {i + len(batch_paths)} images...")
    
    # Compute final statistics
    mean = pixel_sum / total_pixels
    # Var = E[X²] - E[X]²
    variance = (pixel_sum_sq / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)
    
    print("\nCIFAR-10 Dataset Statistics:")
    print(f"Mean (RGB): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std  (RGB): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"Total pixels processed: {total_pixels:,}")
    
    return mean, std

def get_transforms(type="train", mean=None, std=None):
    transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    if type == "train":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=(-0.0625, 0.0625),
                scale_limit=(-0.1, 0.1),
                rotate_limit=(-45, 45),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="largest_box",
                p=0.5
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(16, 16),
                hole_width_range=(16, 16),
                fill=mean,
                p=0.5
            )
        ]+ transforms
    else:
        return transforms

    return transforms

def serialize_transforms(transform_compose):
    """
    Convert torchvision transforms to a serializable format
    
    Args:
        transform_compose: torchvision.transforms.Compose object
        
    Returns:
        list: List of transform dictionaries with parameters
    """
    if not isinstance(transform_compose, A.Compose):
        return str(transform_compose)
    
    serialized_transforms = []
    
    for transform in transform_compose.transforms:
        transform_info = {
            "name": transform.__class__.__name__,
            "module": transform.__class__.__module__
        }
        
        # Extract parameters for common transforms
        if hasattr(transform, '__dict__'):
            params = {}
            for key, value in transform.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    # Convert non-serializable types to strings
                    if isinstance(value, (int, float, str, bool, list, tuple)):
                        params[key] = value
                    elif value is None:
                        params[key] = None
                    else:
                        params[key] = str(value)
            transform_info["parameters"] = params
        
        serialized_transforms.append(transform_info)
    
    return serialized_transforms    

def get_relative_path(path):
    """
    Convert absolute path to relative path from project root
    
    Args:
        path: Path object or string path
        
    Returns:
        str: Relative path from project root
    """
    try:
        path = Path(path)
        if path.is_absolute():
            # Try to get relative path from project root
            relative_path = path.relative_to(PROJECT_ROOT)
            return str(relative_path)
        else:
            # Already relative
            return str(path)
    except ValueError:
        # Path is not under project root, return just the filename
        return path.name if hasattr(path, 'name') else str(path)
