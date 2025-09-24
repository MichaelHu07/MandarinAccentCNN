# Introduction

Classify Audio Files Using ResNet Convolutional Neural Network

## Training Process

1. Load data using Librosa (audio processing package)
2. Check if processed spectrograms are present in the data cache
    - If not present: Remove Silence (Split speaking sections by decibel thresholds)
    - Generate a Spectrogram numpy array using the Short-time Fourier Transform
    - Save in Data\Stft folder
3. Initialize AccentDataset Object
4. Freeze pretrained Resnet18 Hidden layers
5. Fine-tune ResNet18 input and output layer
    - loss-function: Cross Entropy Loss
    - optimizer: Stochastic Gradient Descent
6. Run specified number of Epochs
7. Load Trained model, validate on test dataset
8. Print Final Accuracy.

## Default Model Settings

Epochs: 2

Batches: 1 

Train/Test Split: 80/20

        - Convolutional Layer 1
        
Kernel Size: 3x3

Padding: 1

        - Optimizer: Stochastic Gradient Descent
        
Learning rate: 0.001

Momentum: 0.9

        - Spectrogram Settings
        
Sample Rate: 16000 

Chunk duration: 1 

Decibel Threshold: 30 

