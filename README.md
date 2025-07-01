# Image Classification CNN

Convolutional Neural Network for CIFAR-10 image classification using TensorFlow/Keras.

## Model Architecture
![CNN Architecture](https://via.placeholder.com/400x200?text=CNN+Architecture+Diagram)
- 3 Convolutional Blocks with BatchNorm and MaxPooling
- Progressive Dropout (0.2 → 0.5)
- 128-unit Dense Classifier
- Adam Optimizer

## Performance
**Test Accuracy**: 83.27%  
**Test Loss**: 0.5796  
![Training History](training_performance.png)

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Matplotlib
- NumPy

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
