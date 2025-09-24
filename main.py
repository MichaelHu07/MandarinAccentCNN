#pytorch imports

import torch
from torch.nn import Conv2d
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import random_split
import os
from dataloader import load_data, collate_fn




if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = resnet18(weights = ResNet18_Weights.DEFAULT)
    full_dataset = load_data()

    for param in model.parameters():
        param.requires_grad = False

    split_percentage = 0.8

    train_size = int(split_percentage * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.conv1 = Conv2d(
        in_channels = 1,
        out_channels = 64,
        kernel_size = 3,
        padding = 1
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(512, full_dataset.label_length())
    model.fc.requires_grad = True

    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(train_dataloader,0):
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:  # print every 50 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0

    print("Finished training")

    PATH = os.path.join(os.getcwd(), "MODELS", "resnet18_accents.pth")
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)

    model.load_state_dict(torch.load(PATH))


    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            spectrograms, labels = data
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            outputs = model(spectrograms)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct/total}%")
