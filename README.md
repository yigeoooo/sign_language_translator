# Sign Language Translation System

A sign language recognition and translation system based on Leap Motion 2, supporting bilingual translation (Chinese-English) and voice output.

## Project Overview

This project is a complete sign language recognition training framework that provides an end-to-end solution from data collection, preprocessing, model training to real-time inference. The system can:

- **Data Collection**: Collect sign language data using Leap Motion device
- **Data Preprocessing**: Feature extraction, data cleaning, and standardization
- **Model Training**: Support multiple deep learning model architectures
- **Real-time Inference**: Real-time sign language recognition and Chinese-English translation

---

## Environment Setup

- **Python Version: 3.8 (Required)**

- **Leap Motion 2 Environment Configuration:**  
  * Refer to the official Python reconstruction configuration environment. Since Leap Motion's SDK is based on C language for secondary development, Python requires special environment configuration.
  * Refer to the official Github project: [leapc-python-bindings](https://github.com/ultraleap/leapc-python-bindings).
  * When the environment is configured according to the process, run visualiser.py in the project. If you see the following effect, it proves that the environment configuration is successful. (Prerequisites: having a Leap Motion 2 device and it is connected)</br>
  ![Image](/tests/video.gif "Leap Motion 2 Test")

## Project Directory Structure

```
tests/                           # Test documents
train/
├── data_collector.py           # Data collector
├── data_preprocessor.py        # Data preprocessor
├── model_definition.py         # Model definitions
├── enhanced_trainer.py         # Enhanced trainer
├── trainer.py                  # Trainer
├── inference.py                # Inference engine
└── data/                       # Data directory
    ├── raw/                    # Raw data files
    │   └── ...
    ├── annotations/            # Annotation files
    │   └── ...
    ├── processed/              # Processed data
    │   └── ...
    ├── models/                 # Trained models
    │   └── ...
    └── gesture_labels.json     # Gesture label configuration
```

## Quick Start

### 1. Data Collection

First, collect sign language data:

```bash
cd train
python data_collector.py
```

**Operation Instructions:**
- Press number keys (0-9) to start recording corresponding gestures
- Press spacebar to stop recording
- Press 'q' to quit

**Default Gesture Labels:**
- 1: 你好 (hello)
- 2: 谢谢 (thank you)
- 3: 再见 (goodbye)
- 4: 是 (yes)
- 5: 不 (no)
- 6: 我 (I)
- 7: 你 (you)
- 8: 爱 (love)
- 9: 家 (home)
- 0: 水 (water)

### 2. Data Preprocessing

Process the collected raw data:

```bash
python data_preprocessor.py
```

This will generate:
- Standardized feature data
- Train/validation/test set splits
- Data statistics and visualizations

### 3. Model Training

Choose model architecture for training:

```bash
# Using LSTM model
python -c "
from trainer import HandGestureTrainer
from data_preprocessor import HandGesturePreprocessor

# Load data
preprocessor = HandGesturePreprocessor()
data_splits = preprocessor.load_processed_data('data/processed/processed_data_latest.pkl')

# Create trainer
trainer = HandGestureTrainer(model_type='lstm')
trainer.prepare_data(data_splits)
trainer.build_model(hidden_dim=128, num_layers=2)
trainer.setup_training(learning_rate=0.001)

# Start training
trainer.train(epochs=100)

# Evaluate model
results = trainer.evaluate()
trainer.plot_training_history()
"
```

**Supported Model Types:**
- `lstm`: LSTM Recurrent Neural Network
- `gru`: GRU Gated Recurrent Unit
- `transformer`: Transformer Attention Model
- `cnn_lstm`: CNN-LSTM Hybrid Model
- `attention_lstm`: LSTM with Attention Mechanism
- `resnet1d`: 1D Residual Network
- `multitask`: Multi-task Learning Model

### 4. Real-time Inference

Start real-time sign language recognition:

```bash
python inference.py
```

**Hotkeys:**
- `q`: Quit the program
- `s`: Save prediction history
- `c`: Clear prediction history
- `h`: Show help information

## Model Performance Comparison

| Model Type | Parameters | Training Speed | Accuracy | Inference Speed | Features |
|-----------|------------|----------------|-----------|-----------------|----------|
| LSTM | Medium | Fast | 85-90% | Fast | Basic sequence model |
| GRU | Medium | Fast | 85-88% | Fast | Simpler LSTM |
| Transformer | High | Slow | 90-95% | Medium | Attention mechanism |
| CNN-LSTM | Medium-High | Medium | 88-92% | Medium | Feature extraction + sequence |
| Attention-LSTM | Medium-High | Medium | 88-91% | Medium | LSTM with attention |
| ResNet1D | Medium-High | Medium | 87-90% | Medium | Residual connections |
| MultiTask | High | Slow | 90-93% | Slow | Multi-task learning |

## Configuration Instructions

### Data Collection Configuration

Modify in `data_collector.py`:

```python
# Frames per gesture recording
max_frames_per_gesture = 60

# Gesture label configuration
gesture_labels = {
    "1": {"chinese": "你好", "english": "hello"},
    # Add more gestures...
}
```

### Preprocessing Configuration

Modify in `data_preprocessor.py`:

```python
feature_config = {
    "sequence_length": 30,      # Sequence length
    "palm_features": True,      # Include palm features
    "arm_features": True,       # Include arm features
    "digit_features": True,     # Include finger features
    "velocity_features": True,  # Include velocity features
    "angle_features": True,     # Include angle features
    "distance_features": True   # Include distance features
}
```

### Training Configuration

Adjustable during training:

```python
# Model parameters
trainer.build_model(
    hidden_dim=128,         # Hidden layer dimension
    num_layers=2,           # Number of layers
    dropout=0.3,            # Dropout rate
    bidirectional=True      # Bidirectional (LSTM/GRU)
)

# Training parameters
trainer.setup_training(
    learning_rate=0.001,    # Learning rate
    optimizer_type="adam",  # Optimizer type
    scheduler_type="cosine", # Learning rate scheduler
    use_early_stopping=True, # Use early stopping
    patience=10             # Early stopping patience
)
```

### Inference Configuration

Modify in `inference.py`:

```python
# Gesture buffer configuration
gesture_buffer = GestureBuffer(
    max_length=30,          # Maximum buffer length
    min_length=10,          # Minimum prediction length
    motion_threshold=0.1,   # Motion detection threshold
    stillness_duration=1.0  # Stillness duration
)

# Confidence tracker configuration
confidence_tracker = ConfidenceTracker(
    window_size=5,          # Sliding window size
    threshold=0.7           # Confidence threshold
)
```

## Advanced Usage

### Custom Models

Create custom model architectures:

```python
import torch.nn as nn
from model_definition import ModelFactory

class CustomModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Define your model architecture
        
    def forward(self, x):
        # Define forward pass
        return output

# Register to factory
ModelFactory.register_model("custom", CustomModel)
```

### Multi-task Training

Train multi-task models:

```python
trainer = HandGestureTrainer(model_type="multitask")
trainer.prepare_multitask_data(data_splits)
trainer.build_model(
    num_gesture_classes=10,
    num_chinese_classes=10,
    num_english_classes=10
)
trainer.train(epochs=100)
```

### Data Augmentation

Add data augmentation during preprocessing:

```python
# Add in data_preprocessor.py
def augment_sequence(self, sequence):
    # Time warping
    # Noise addition
    # Rotation transformation
    return augmented_sequence
```

### Model Ensemble

Use multiple models for ensemble prediction:

```python
models = [
    load_model("best_lstm_model.pth"),
    load_model("best_transformer_model.pth"),
    load_model("best_cnn_lstm_model.pth")
]

# Ensemble prediction
ensemble_prediction = ensemble_predict(models, input_data)
```

## Performance Optimization

### Training Optimization

1. **Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
# Use autocast in training loop
```

2. **Learning Rate Scheduling**:
```python
# Use cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

3. **Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Inference Optimization

1. **Model Quantization**:
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

2. **Batch Inference**:
```python
# Collect multiple gesture sequences for batch inference
batch_predictions = model(batch_input)
```

## Common Issues

### Q1: Leap Motion device cannot connect

**Solutions:**
1. Ensure Leap Motion drivers are properly installed
2. Check if USB connection is stable
3. Restart Leap Motion service
4. Check Device Manager for unknown devices

### Q2: Out of memory during training

**Solutions:**
1. Reduce batch size
2. Use gradient accumulation
3. Reduce model parameters
4. Use mixed precision training

### Q3: Low recognition accuracy

**Solutions:**
1. Increase training data volume
2. Adjust feature extraction parameters
3. Try different model architectures
4. Adjust preprocessing hyperparameters
5. Use data augmentation techniques

### Q4: Voice playback not working

**Solutions:**
1. Check if pyttsx3 is properly installed
2. Confirm system has available TTS engine
3. Check audio device settings
4. Try different voice engines

### Q5: High real-time inference latency

**Solutions:**
1. Use GPU acceleration
2. Reduce model complexity
3. Optimize feature extraction process
4. Use model quantization
5. Adjust buffer size

## Related Resources

- [Leap Motion Developer Documentation](https://developer.leapmotion.com/)
- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [Sign Language Recognition Papers](https://github.com/topics/sign-language-recognition)
- [Deep Learning Best Practices](https://www.deeplearningbook.org/)

## Authors
  * yigeoooo
  * XXK

---

**Note**: This project is for learning and research purposes only. Commercial use should comply with relevant license term