# CIFAR-100 CNN Training with PyTorch Lightning

A comprehensive deep learning project for training a Convolutional Neural Network (Resnet) on the CIFAR-100 dataset using PyTorch Lightning, featuring advanced training techniques including OneCycle learning rate scheduling, data augmentation with Albumentations, and comprehensive logging.

## 🏆 Experiment Results

### **Best Performance Achieved**
| Metric | Value |
|--------|-------|
| **Best Val Accuracy** | **82.42%** |
| **Training Time** | 16.5 minutes |
| **Total Epochs** | 50 |
| **Model Parameters** | 11,227,812 |

### Learning Curve Highlights
- **Epoch 1**: Val Acc: 18.5%
- **Epoch 10**: Val Acc: 39.68%
- **Epoch 20**: Val Acc: 59.7%
- **Epoch 30**: Val Acc: 70.12%
- **Epoch 44**: Val Acc: 76.52
- **Epoch 50**: Val Acc: **82.42%** (Best)

## 🚀 Key Features

### Advanced Training Techniques
- **OneCycle Learning Rate Scheduling** with LR Finder
- **Albumentations Data Augmentation** (HorizontalFlip, ShiftScaleRotate, CoarseDropout)
- **PyTorch Lightning** for clean, scalable training
- **MPS (Apple Silicon) GPU Support**
- **Comprehensive Logging** (TensorBoard + Text logs)

### Model Architecture
- **Resnet18** 

### Data Pipeline
- **CIFAR-100 Dataset**: 50K training + 10K test images (32×32 RGB)
- **Custom Dataset Class** with Albumentations integration
- **Computed Dataset Statistics**: Mean=[0.50736207, 0.4866896, 0.44108862], Std=[0.26748818, 0.2565931, 0.2763085]
- **Train/Val Split**: 90%/10% with proper transforms

## 📁 Project Structure

```
train_cnn_on_cifar10/
├── 📊 Dataset & Preprocessing
│ ├── dataset/
│ │ ├── cifar-100-python/ # Original CIFAR-100 pickle files
│ │ └── cifar100_images/ # Processed images + CSV labels
│ ├── preprocessing_cifar100_dataset.py # CIFAR-100 to images converter
│ ├── dataset.py # Custom PyTorch Dataset class
│ └── utils.py # Data transforms & statistics
│
├── 🧠 Model & Training
│ ├── model.py # CNN architecture definitions
│ ├── lit_module.py # Lightning Module (model + training logic)
│ ├── lit_datamodule.py # Lightning DataModule (data handling)
│ ├── lit_train.py # Main training script
│ └── config.py # Configuration & hyperparameters
│
├── 📈 Advanced Training Features
│ ├── lr_finder_utils.py # LR Finder & OneCycle utilities
│ ├── text_logging_callback.py # Comprehensive logging callback
│ └── main.py # Alternative training entry point
│
├── 📋 Configuration & Dependencies
│ ├── pyproject.toml # UV package management
│ ├── uv.lock # Dependency lock file
│
├── 📊 Experiment Results
│ └── logs/experiment1/
│ ├── training.log # Detailed training logs
│ ├── metrics.json # Complete metrics history
│ ├── model_info.txt # Model architecture details
│ ├── lr_finder.png # LR Finder plot
│ └── lightning_checkpoints/ # Model checkpoints
│
└── 📓 Development
├── temp.ipynb # Jupyter notebook for experiments
└── README.md # This file


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- UV package manager
- Apple Silicon Mac (for MPS support) or CUDA GPU

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd train_cnn_cifar100

# Install dependencies with UV
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Dependencies
- **PyTorch Lightning 2.5.5+**: Training framework
- **Albumentations 2.0.11+**: Data augmentation
- **TensorBoard 2.20.0+**: Logging and visualization
- **torchinfo**: Model architecture summary
- **torch-lr-finder**: Learning rate finder
- **Rich**: Beautiful progress bars

## 🚀 Usage

### 1. Data Preparation
```bash
python preprocessing_cifar10_dataset.py
```

### 2. Training
```bash
# Run training with Lightning
python lightning_train.py
```

### 3. Monitoring Training
```bash
# View TensorBoard logs
uv run tensorboard --logdir logs/experiment1/lightning_logs/

# Check text logs
tail -f logs/experiment1/training.log

# View metrics
cat logs/experiment1/metrics.json
```

## ⚙️ Configuration

### Key Hyperparameters (config.py)
```python
# Model & Training
learning_rate = 1.13e-01  # Found with LR Finder
weight_decay = 1e-4
epochs = 50
batch_size = 64

# OneCycle Scheduler
scheduler_type = 'one_cycle_policy'
onecycle_kwargs = {
    'pct_start': 0.2,           # 20% warmup
    'anneal_strategy': 'cos',   # Cosine annealing
    'div_factor': 100.0,         # Initial LR = max_lr/10
    'final_div_factor': 1000.0   # Final LR = initial_lr/100
}

Data Augmentation
- HorizontalFlip (p=0.5)
- ShiftScaleRotate (shift=±6.25%, scale=±10%, rotate=±45°)
- CoarseDropout (16×16 holes)
```

## 🏗️ Model Architecture

### Resnet18

~~~
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 100]                  --
├─Conv2d: 1-1                            [1, 64, 16, 16]           9,408
├─BatchNorm2d: 1-2                       [1, 64, 16, 16]           128
├─ReLU: 1-3                              [1, 64, 16, 16]           --
├─MaxPool2d: 1-4                         [1, 64, 8, 8]             --
├─Sequential: 1-5                        [1, 64, 8, 8]             --
│    └─BasicBlock: 2-1                   [1, 64, 8, 8]             --
│    │    └─Conv2d: 3-1                  [1, 64, 8, 8]             36,864
│    │    └─BatchNorm2d: 3-2             [1, 64, 8, 8]             128
│    │    └─ReLU: 3-3                    [1, 64, 8, 8]             --
│    │    └─Conv2d: 3-4                  [1, 64, 8, 8]             36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 8, 8]             128
│    │    └─ReLU: 3-6                    [1, 64, 8, 8]             --
│    └─BasicBlock: 2-2                   [1, 64, 8, 8]             --
│    │    └─Conv2d: 3-7                  [1, 64, 8, 8]             36,864
│    │    └─BatchNorm2d: 3-8             [1, 64, 8, 8]             128
│    │    └─ReLU: 3-9                    [1, 64, 8, 8]             --
│    │    └─Conv2d: 3-10                 [1, 64, 8, 8]             36,864
│    │    └─BatchNorm2d: 3-11            [1, 64, 8, 8]             128
│    │    └─ReLU: 3-12                   [1, 64, 8, 8]             --
├─Sequential: 1-6                        [1, 128, 4, 4]            --
│    └─BasicBlock: 2-3                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-13                 [1, 128, 4, 4]            73,728
│    │    └─BatchNorm2d: 3-14            [1, 128, 4, 4]            256
│    │    └─ReLU: 3-15                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-16                 [1, 128, 4, 4]            147,456
│    │    └─BatchNorm2d: 3-17            [1, 128, 4, 4]            256
│    │    └─Sequential: 3-18             [1, 128, 4, 4]            8,448
│    │    └─ReLU: 3-19                   [1, 128, 4, 4]            --
│    └─BasicBlock: 2-4                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-20                 [1, 128, 4, 4]            147,456
│    │    └─BatchNorm2d: 3-21            [1, 128, 4, 4]            256
│    │    └─ReLU: 3-22                   [1, 128, 4, 4]            --
│    │    └─Conv2d: 3-23                 [1, 128, 4, 4]            147,456
│    │    └─BatchNorm2d: 3-24            [1, 128, 4, 4]            256
│    │    └─ReLU: 3-25                   [1, 128, 4, 4]            --
├─Sequential: 1-7                        [1, 256, 2, 2]            --
│    └─BasicBlock: 2-5                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-26                 [1, 256, 2, 2]            294,912
│    │    └─BatchNorm2d: 3-27            [1, 256, 2, 2]            512
│    │    └─ReLU: 3-28                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-29                 [1, 256, 2, 2]            589,824
│    │    └─BatchNorm2d: 3-30            [1, 256, 2, 2]            512
│    │    └─Sequential: 3-31             [1, 256, 2, 2]            33,280
│    │    └─ReLU: 3-32                   [1, 256, 2, 2]            --
│    └─BasicBlock: 2-6                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-33                 [1, 256, 2, 2]            589,824
│    │    └─BatchNorm2d: 3-34            [1, 256, 2, 2]            512
│    │    └─ReLU: 3-35                   [1, 256, 2, 2]            --
│    │    └─Conv2d: 3-36                 [1, 256, 2, 2]            589,824
│    │    └─BatchNorm2d: 3-37            [1, 256, 2, 2]            512
│    │    └─ReLU: 3-38                   [1, 256, 2, 2]            --
├─Sequential: 1-8                        [1, 512, 1, 1]            --
│    └─BasicBlock: 2-7                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-39                 [1, 512, 1, 1]            1,179,648
│    │    └─BatchNorm2d: 3-40            [1, 512, 1, 1]            1,024
│    │    └─ReLU: 3-41                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-42                 [1, 512, 1, 1]            2,359,296
│    │    └─BatchNorm2d: 3-43            [1, 512, 1, 1]            1,024
│    │    └─Sequential: 3-44             [1, 512, 1, 1]            132,096
│    │    └─ReLU: 3-45                   [1, 512, 1, 1]            --
│    └─BasicBlock: 2-8                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-46                 [1, 512, 1, 1]            2,359,296
│    │    └─BatchNorm2d: 3-47            [1, 512, 1, 1]            1,024
│    │    └─ReLU: 3-48                   [1, 512, 1, 1]            --
│    │    └─Conv2d: 3-49                 [1, 512, 1, 1]            2,359,296
│    │    └─BatchNorm2d: 3-50            [1, 512, 1, 1]            1,024
│    │    └─ReLU: 3-51                   [1, 512, 1, 1]            --
├─AdaptiveAvgPool2d: 1-9                 [1, 512, 1, 1]            --
├─Linear: 1-10                           [1, 100]                  51,300
==========================================================================================
Total params: 11,227,812
Trainable params: 11,227,812
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 37.07
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.81
Params size (MB): 44.91
Estimated Total Size (MB): 45.74
==========================================================================================
~~~

## 📊 Training Features

### 1. OneCycle Learning Rate Scheduling
- **LR Finder**: Automatically finds optimal learning rate (2.35E-04)
- **Warmup Phase**: 10% of training with increasing LR
- **Annealing Phase**: 90% of training with cosine decay
- **Benefits**: Faster convergence, better generalization

### 2. Advanced Data Augmentation
```python
# Training Transforms (Albumentations)
- HorizontalFlip(p=0.5)
- ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45°, p=0.5)
- CoarseDropout(holes=1, size=16×16, p=0.5)
- Normalize(mean, std)

# Validation Transforms
- Normalize(mean, std)
```

### 3. Comprehensive Logging
- **TensorBoard**: Metrics, model graph, hyperparameters
- **Text Logs**: Detailed training progress with timestamps
- **JSON Metrics**: Complete training history for analysis
- **Model Checkpoints**: Best models saved automatically

### 4. MPS (Apple Silicon) Support
- **Optimized for M1/M2/M3**: Native GPU acceleration
- **Device Compatibility**: Automatic fallback to CPU
- **Memory Efficient**: Optimized DataLoader settings

## 🎯 Results Analysis

### Training Progression
The model shows excellent learning progression:
1. **Fast Initial Learning**: 33% → 70% accuracy in first 10 epochs
2. **Steady Improvement**: Consistent gains through epoch 30
3. **Fine-tuning Phase**: Gradual improvements to 87.2% peak
4. **Stable Convergence**: Maintains high performance in final epochs

### OneCycle Benefits
- **Faster Convergence**: Reached 80%+ accuracy by epoch 20
- **Better Generalization**: 4.3% gap between best val (87.2%) and test (82.9%)
- **Stable Training**: Smooth learning curve without major fluctuations

### Model Efficiency
- **Parameter Efficient**: 180K params achieving 87.2% validation accuracy
- **Fast Training**: 21.9 minutes for 50 epochs on Apple Silicon
- **Good Generalization**: Reasonable val/test gap indicates proper regularization

## 🔧 Troubleshooting

### Common Issues

1. **MPS Device Errors**
   ```bash
   # Force CPU training if MPS issues occur
   # In lightning_train.py, change accelerator="mps" to accelerator="cpu"
   ```

2. **TensorBoard NumPy Compatibility**
   ```bash
   # Use project's TensorBoard
   uv run tensorboard --logdir logs/experiment1/lightning_logs/
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in config.py
   batch_size = 64  
   ```

### Performance Tips
- **Batch Size**: Increase to 128 if you have more GPU memory
- **Learning Rate**: Use LR Finder to optimize for your setup
- **Data Workers**: Adjust `num_workers` based on your CPU cores
- **Precision**: Use `precision="16-mixed"` for faster training on supported GPUs

## 📚 References & Inspiration

- **OneCycle Learning**: [Super-Convergence Paper](https://arxiv.org/abs/1708.07120)
- **PyTorch Lightning**: [Official Documentation](https://lightning.ai/docs/pytorch/)
- **Albumentations**: [Data Augmentation Library](https://albumentations.ai/)
- **CIFAR-10**: [Original Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**🎉 Achieved 82.42% validation accuracy on CIFAR-100 in 50 epochs!**