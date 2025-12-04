#IMPORTANT

This project was developed using **Python 3.12.6**.
Download **cuda 13.0**

# Introduction

Classify Regional Mandarin Accents from Audio using Tencent Wav2Vec2 Transformer Model.

## Dependencies/packages

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Process

1. Load raw audio data using Librosa.
2. Check if processed audio chunks are present in the LMDB data cache (`Data\LMDB_WAV`).
    - If not present: Remove Silence (Split speaking sections by decibel thresholds).
    - Chunk audio into 4-second segments (64000 samples).
    - Store raw waveforms in LMDB database for average O(1) access using B+ tree retrieval.
3. Initialize `AccentDataset` Object.
4. Load pretrained `TencentGameMate/chinese-wav2vec2-base` from Hugging Face.
5. Freeze Feature Extractor (CNN) layers; fine-tune Transformer Encoder and Classification Head.
    - Loss Function: Cross Entropy Loss (computed internally).
    - Optimizer: AdamW (Adaptive Moment Estimation).
6. Iterate through the dataset `epoch` number of times using Gradient Accumulation.
7. Track running loss and validate on the test dataset after each epoch.
8. Generate and save training curves (`training_curves.png`) and final classification report.

## Default Model Settings

Epochs: 5 # Increased for fine-tuning

Batches: 16 # (Effective batch size via Gradient Accumulation)

Train/Test Split: 80/20

- **Transformer Settings**
    
    Model: TencentGameMate/chinese-wav2vec2-base
    
    Frozen Layers: Feature Extractor (7-layer CNN)
    
    Fine-tuned Layers: Transformer Encoder (12 layers) + Classifier Head

- **Optimizer**
    
    Type: AdamW
    
    Learning rate: 1e-5
    
    Weight Decay: 0.01

- **Audio Processing**
    
    Sample Rate: 16000 Hz
    
    Chunk duration: 4.0 seconds (64000 samples)
    
    Hop length: 1.5 seconds (32000 samples - 50% overlap)
    
    Decibel Threshold: 30 (In silence removal processing)

## DATA

Data sourced from: https://magichub.com/datasets/mandarin-heavy-accent-conversational-speech-corpus/

Preprocessed audio segments are cached in `data\LMDB_WAV` for efficient training.