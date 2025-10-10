from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
import torch

class Cifar100Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB color image
        # CONVERT PIL Image to numpy array for Albumentations
        image = np.array(image)  # Convert PIL Image to numpy array

        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            # Albumentations expects: transform(image=numpy_array)
            transformed = self.transform(image=image)
            image = transformed['image']  # This will be a tensor after ToTensorV2()
            # Ensure tensor is on the right device (let Lightning handle device placement)
            if isinstance(image, torch.Tensor):
                image = image.float()  # Ensure float32 type
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label