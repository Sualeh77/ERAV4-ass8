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

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# device = "cpu"

input_size = (1, 3, 32, 32)

num_workers: int = 8,
val_split: float = 0.1

learning_rate = 1.13e-01 # Found with LR finder
weight_decay = 1e-4
epochs = 50
batch_size = 128
experiment_name = "experiment1"

scheduler_type = 'one_cycle_policy'

lr_finder_kwargs = {
            'start_lr': 1e-7,
            'end_lr': 10,
            'num_iter': 1000,
            'step_mode': 'exp'
        }

onecycle_kwargs = {
            'lr_strategy': 'manual',  # 'conservative', 'manual'
            'pct_start': 0.2,
            'anneal_strategy': 'cos',
            'div_factor': 100.0,
            'final_div_factor': 1000.0
        }