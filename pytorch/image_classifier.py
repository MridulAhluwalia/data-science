#!/usr/bin/env python3
import os
import pdb

import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms

# ----------------------------------------
TRAIN_DATA_DIR = "data/train"
TEST_DATA_DIR = "data/test"
IMAGE_SIZE = 128
IMG_CHANNELS = 3
BATCH_SIZE = 64
FEATURE_DIM = 64
N_CLASSES = 6
LR = 0.001
N_EPOCHS = 100

# ----------------------------------------


def load_datasets(
    train_data_dir, test_data_dir, image_size, img_channels, batch_size
):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # pixel range 0-255 to 0-1, numpy to tensor
            transforms.Normalize(
                [0.5 for _ in range(img_channels)],
                [0.5 for _ in range(img_channels)],
            ),  # range 0-1 to -1-1; column -> rgb channels, row -> mean and sd
            # new_pixel = (old_pixel - mean) / sd
        ]
    )

    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader


# ----------------------------------------


class Net(nn.Module):
    """
    To keep image size same:
        kernel_size = 3
        padding = 1
        stride = 1

    To half image size:
        kernel_size = 4
        padding = 1
        stride = 2
    """

    def __init__(
        self,
        features_dim=64,
        img_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        num_classes=2,
    ):
        super(Net, self).__init__()

        # output size ater conv filter
        # ((width_or_height - kernel_size + 2 * padding) / stride) + 1

        self.net = nn.Sequential(
            # input image shape: (batch_size, n_channels, height, width)
            # (64, 3, 128, 128)
            nn.Conv2d(img_channels, features_dim, kernel_size, stride, padding),
            nn.LeakyReLU(0.1, inplace=True),
            # output image shape: (batch_size, features_dim, height /2, width /2)
            # (64, 64, 64, 64)
            nn.Conv2d(
                features_dim, features_dim * 2, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # output image shape: (batch_size, features_dim * 2, height /2, width /2)
            # (64, 128, 32, 32)
            nn.Conv2d(
                features_dim * 2, features_dim * 4, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # output image shape: (batch_size, features_dim * 4, height /2, width /2)
            # (64, 256, 16, 16)
            nn.Conv2d(
                features_dim * 4, features_dim * 8, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # output image shape: (64, 512, 8, 8)
        )

        self.fc = nn.Linear(
            in_features=features_dim * 8 * 64,
            out_features=num_classes
            # in_features = out_channel * imhe_height * image_width
        )

    def forward(self, image):
        output = self.net(image)
        output = output.view(-1, 512 * 64)
        return self.fc(output)


# ----------------------------------------


def train(n_epochs):
    best_accuracy = 0.0
    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []

    for epoch in range(n_epochs):
        model.train()

        train_loss = 0.0
        train_accuracy = 0.0
        test_accuracy = 0.0

        # train model
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_accuracy += int(torch.sum(prediction == labels.data))

        train_accuracy = train_accuracy / total
        train_loss = train_loss / total

        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss)

        model.eval()

        # calculate test accuracy
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(labels)

                outputs = model(images)
                _, prediction = torch.max(outputs.data, 1)
                total += labels.size(0)
                test_accuracy += int(torch.sum(prediction == labels.data))

        test_accuracy = test_accuracy / total
        test_accuracy_list.append(test_accuracy)

        print("-" * 75)
        print(
            f"Epoch: {epoch}, \
            Train Loss: {train_loss}, \
            Train Accuracy: {train_accuracy}, \
            Test Accuracy: {test_accuracy}"
        )
        print("-" * 75)

        # save the best performing model
        if test_accuracy > best_accuracy:
            save_model(model, "models/checkpoint.model")
            best_accuracy = test_accuracy

    return train_accuracy_list, train_loss_list, test_accuracy_list


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def save_plot(x, y, path):
    fig = px.line(x, y)
    fig.write_image(path)


# ----------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = load_datasets(
        TRAIN_DATA_DIR, TEST_DATA_DIR, IMAGE_SIZE, IMG_CHANNELS, BATCH_SIZE
    )
    model = Net(
        features_dim=FEATURE_DIM,
        img_channels=IMG_CHANNELS,
        num_classes=N_CLASSES,
    ).to(device)

    print(summary(model, (IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)))

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    # train model
    train_accuracy_list, train_loss_list, test_accuracy_list = train(N_EPOCHS)
