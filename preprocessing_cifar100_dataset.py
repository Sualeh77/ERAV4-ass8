import os
import pickle
import numpy as np
from PIL import Image
import pandas as pd

def unpickle(file, file_encoding='bytes'):
    """Unpickle CIFAR-10 batch files"""
    with open(file, 'rb') as fo:
        meta_dict = pickle.load(fo, encoding=file_encoding)
    return meta_dict

def reshape_cifar_image(flat_image):
    """
    Reshape flat CIFAR-100 image array to 32x32x3 RGB format
    Input: 3072-element array (1024 red + 1024 green + 1024 blue)
    Output: 32x32x3 numpy array
    """
    # Split into RGB channels
    red = flat_image[:1024].reshape(32, 32)
    green = flat_image[1024:2048].reshape(32, 32)
    blue = flat_image[2048:3072].reshape(32, 32)
    
    # Stack channels to create RGB image
    rgb_image = np.stack([red, green, blue], axis=2) # Axis=2 for z-axis
    return rgb_image

def load_cifar_file(file):  
    """Load CIFAR-100 train file"""
    data_dict = unpickle(file)
    
    # Extract data and labels (keys are bytes in Python 3)
    images = data_dict[b'data']
    fine_labels = data_dict[b'fine_labels']
    coarse_labels = data_dict[b'coarse_labels']
    
    return images, fine_labels, coarse_labels

def load_cifar_meta(meta_file):
    """Load CIFAR-10 metadata file to get label names"""
    meta_dict = unpickle(meta_file)
    fine_label_names = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]
    coarse_label_names = [name.decode('utf-8') for name in meta_dict[b'coarse_label_names']]
    return fine_label_names, coarse_label_names

def convert_cifar100_to_images(data_dir, output_dir):
    """Convert CIFAR-10 pickled files to individual image files"""
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load label names
    meta_file = os.path.join(data_dir, 'meta')
    fine_label_names, coarse_label_names = load_cifar_meta(meta_file)
    print(f"CIFAR-100 classes: {fine_label_names}")
    
    # Lists to store image paths and labels for CSV files
    train_data = []
    test_data = []
    
    # Process training data (train file)
    print("Processing training data...")
    train_image_counter = 0
    
    train_file = os.path.join(data_dir, f'train')
    print(f"Processing train file...")
    
    images, fine_labels, coarse_labels = load_cifar_file(train_file)
    
    for i, (flat_image, fine_label, coarse_label) in enumerate(zip(images, fine_labels, coarse_labels)):
        # Reshape flat image to RGB format
        rgb_image = reshape_cifar_image(flat_image)
        
        # Create filename with class name
        fine_class_name = fine_label_names[fine_label]
        coarse_class_name = coarse_label_names[coarse_label]
        filename = f"train_{train_image_counter:05d}_class_{fine_label}_{fine_class_name}_{coarse_label}_{coarse_class_name}.png"
        filepath = os.path.join(train_dir, filename)
        
        # Convert numpy array to PIL Image and save
        img = Image.fromarray(rgb_image, mode='RGB')
        img.save(filepath)
        
        # Add to data list
        train_data.append({
            'image_path': f"train/{filename}",
            'fine_label': int(fine_label),
            'class_name': fine_class_name,
            'coarse_label': int(coarse_label),
            'coarse_class_name': coarse_class_name,
            'absolute_path': filepath
        })
        
        train_image_counter += 1
        
        if train_image_counter % 5000 == 0:
            print(f"Processed {train_image_counter} training images")
    
    # Process test data
    print("Processing test data...")
    test_file = os.path.join(data_dir, 'test')
    test_images, test_fine_labels, test_coarse_labels = load_cifar_file(test_file)
    
    for i, (flat_image, fine_label, coarse_label) in enumerate(zip(test_images, test_fine_labels, test_coarse_labels)):
        # Reshape flat image to RGB format
        rgb_image = reshape_cifar_image(flat_image)
        
        # Create filename with class name
        fine_class_name = fine_label_names[fine_label]
        coarse_class_name = coarse_label_names[coarse_label]
        filename = f"test_{i:05d}_class_{fine_label}_{fine_class_name}_{coarse_label}_{coarse_class_name}.png"
        filepath = os.path.join(test_dir, filename)
        
        # Convert numpy array to PIL Image and save
        img = Image.fromarray(rgb_image, mode='RGB')
        img.save(filepath)
        
        # Add to data list
        test_data.append({
            'image_path': f"test/{filename}",
            'fine_label': int(fine_label),
            'class_name': fine_class_name,
            'coarse_label': int(coarse_label),
            'coarse_class_name': coarse_class_name,
            'absolute_path': filepath
        })
        
        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1} test images")
    
    # Create CSV files with image paths and labels
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Save CSV files
    train_csv_path = os.path.join(output_dir, 'train_labels.csv')
    test_csv_path = os.path.join(output_dir, 'test_labels.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"\nConversion completed!")
    print(f"Training images: {len(train_data)} saved in {train_dir}")
    print(f"Test images: {len(test_data)} saved in {test_dir}")
    print(f"Training labels CSV: {train_csv_path}")
    print(f"Test labels CSV: {test_csv_path}")
    
    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"Training set - Total: {len(train_data)}")
    print("Training set - Label distribution:")
    train_label_counts = train_df['fine_label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        print(f"  {label} ({fine_label_names[label]}): {count}")
    
    print(f"\nTest set - Total: {len(test_data)}")  
    print("Test set - Label distribution:")
    test_label_counts = test_df['fine_label'].value_counts().sort_index()
    for label, count in test_label_counts.items():
        print(f"  {label} ({fine_label_names[label]}): {count}")
    
    return train_df, test_df

# Usage
if __name__ == "__main__":
    # Define paths
    data_dir = "dataset/cifar-100-python"
    output_dir = "dataset/cifar100_images"
    
    # Convert CIFAR-10 data
    train_df, test_df = convert_cifar100_to_images(data_dir, output_dir)
    
    # Display first few entries
    print("\nFirst 5 training entries:")
    print(train_df.head())
    
    print("\nFirst 5 test entries:")
    print(test_df.head())