import lightning as L
from torch.utils.data import DataLoader
from dataset import Cifar100Dataset
from config import train_labels_csv_path, test_labels_csv_path, train_img_dir, test_img_dir, device
import torch
from torch.utils.data import Subset

class Cifar100DataModule(L.LightningDataModule):
    """
    Lightning DataModule for Cifar100 dataset
    
    This handles all data operations:
    - Setup datasets
    - Create train/val/test dataloaders
    - Handle data transformations
    """
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,  # 10% of training data for validation
        pin_memory: bool = True,
        train_transforms = None,
        test_transforms = None,
    ):
        super().__init__()
        # Store hyperparameters - Lightning will log these automatically
        # EXCLUDE transforms from hyperparameters to avoid conflicts
        self.save_hyperparameters(ignore=['train_transforms', 'test_transforms'])
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        
        # Define transforms
        self.train_transforms = train_transforms
        
        self.test_transforms = test_transforms
        
        # Will be set in setup()
        self.cifar100_train = None
        self.cifar100_val = None
        self.cifar100_test = None

    def prepare_data(self):
        """
        Called once to prepare data (download, etc.)
        Use this for operations that should be done on only one GPU in distributed training
        """
        # In my case, data is already downloaded and prepared
        # This is where you'd put download logic if needed
        print("📁 Data already prepared")

    def setup(self, stage: str = None):
        """
        Called on every GPU in distributed training
        Setup datasets for train/val/test
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Create full training dataset
            full_train_dataset = Cifar100Dataset(
                annotations_file=train_labels_csv_path,
                img_dir=train_img_dir,
                transform=self.train_transforms
            )

            # Create validation dataset WITHOUT augmentation
            full_val_dataset = Cifar100Dataset(
                annotations_file=train_labels_csv_path,
                img_dir=train_img_dir,
                transform=self.test_transforms  # Clean transforms, no augmentation
            )
        
            # Split the indices
            total_size = len(full_train_dataset)
            train_size = int((1 - self.val_split) * total_size)
            # val_size = total_size - train_size

            # Generate indices for splitting
            # indices = list(range(total_size))
            torch.manual_seed(42)  # For reproducibility
            train_indices = torch.randperm(total_size)[:train_size].tolist()
            val_indices = torch.randperm(total_size)[train_size:].tolist()

            # Create subsets with proper transforms
            self.cifar100_train = Subset(full_train_dataset, train_indices)
            self.cifar100_val = Subset(full_val_dataset, val_indices)

        if stage == 'test' or stage is None:
            # Test dataset
            self.cifar100_test = Cifar100Dataset(
                annotations_file=test_labels_csv_path,
                img_dir=test_img_dir,
                transform=self.test_transforms
            )
        
        # Print dataset splits - only print what exists
        print(f"📊 Dataset splits:")
        if hasattr(self, 'cifar100_train') and self.cifar100_train is not None:
            print(f"   Train: {len(self.cifar100_train)} samples (with augmentation)")
        if hasattr(self, 'cifar100_val') and self.cifar100_val is not None:
            print(f"   Val:   {len(self.cifar100_val)} samples (without augmentation)")
        if hasattr(self, 'cifar100_test') and self.cifar100_test is not None:
            print(f"   Test:  {len(self.cifar100_test)} samples")

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.cifar100_train,
            batch_size=self.batch_size,
            shuffle=True,  # Always shuffle training data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.cifar100_val,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.cifar100_test,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle test data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
