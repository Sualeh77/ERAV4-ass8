from pathlib import Path
import torch

# Computed with custom script
mean = tuple([0.50736207, 0.4866896, 0.44108862])
std = tuple([0.26748818, 0.2565931, 0.2763085])

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

train_labels_csv_path = PROJECT_ROOT / "dataset" / "cifar100_images" / "train_labels.csv"
test_labels_csv_path = PROJECT_ROOT / "dataset" / "cifar100_images" / "test_labels.csv"
train_img_dir = PROJECT_ROOT / "dataset" / "cifar100_images"
test_img_dir = PROJECT_ROOT / "dataset" / "cifar100_images"
model_path = PROJECT_ROOT / "models" / "cifar100_fully_cnn.pth"
logs_dir = PROJECT_ROOT / "logs"