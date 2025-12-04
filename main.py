#pytorch imports

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from dataloader import load_data, collate_fn
from transformers import Wav2Vec2ForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt

def plot_metrics(train_losses, train_accs, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    full_dataset = load_data()
    num_labels = full_dataset.label_length()
    print(f"Number of labels: {num_labels}")

    split_percentage = 0.8
    train_size = int(split_percentage * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        pin_memory=True
    )

    model_name = "TencentGameMate/chinese-wav2vec2-base"
    print(f"Loading model: {model_name}")
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    except OSError:
        print(f"Could not load {model_name}, trying a generic Chinese Wav2Vec2 or falling back to facebook/wav2vec2-base-960h")
        model_name = "ant-research/wav2vec2-base-chinese" # Alternative
        try:
             model = Wav2Vec2ForSequenceClassification.from_pretrained("ant-research/wav2vec2-base-chinese", num_labels=num_labels)
        except:
             print("Fallback to facebook/wav2vec2-base")
             model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)

    model = model.to(device)
    
    model.freeze_feature_encoder()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    epochs = 5
    train_losses = []
    train_accs = []
    val_accs = []
    
    accumulation_steps = 4

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * accumulation_steps
            running_loss += current_loss
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {current_loss:.4f}')

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_accs.append(val_acc)
        print(f"Epoch {epoch + 1} Validation Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

    print("Finished training")
    
    PATH = os.path.join(os.getcwd(), "MODELS", "wav2vec2_accents.pth")
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print(f"Model saved to {PATH}")

    plot_path = os.path.join(os.getcwd(), "training_curves.png")
    plot_metrics(train_losses, train_accs, val_accs, plot_path)
    print(f"Training curves saved to {plot_path}")
    
    print("\nFinal Evaluation Report:")
    print(classification_report(val_labels, val_preds, zero_division=0))

