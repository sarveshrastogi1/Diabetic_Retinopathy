import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import os

# Load your dataset
class ClotDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Filter out rows where the corresponding image doesn't exist
        self.df = df[df['ID'].apply(lambda x: 
            os.path.exists(os.path.join(self.image_dir, f"{x}.jpg")))].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.df.iloc[idx, 0]}.jpg")
        image = Image.open(img_name).convert("RGB")
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data paths and preprocessing
df = pd.read_csv('/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/segmentation_mebo/clot_images/data.csv')
image_dir = '/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/segmentation_mebo/clot_images/images'

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['clots'], random_state=42)

# Create datasets with only available images
train_dataset = ClotDataset(train_df, image_dir, transform=transform)
val_dataset = ClotDataset(val_df, image_dir, transform=transform)

# Adjust class weights for the filtered dataset
available_train_labels = train_dataset.df['clots'].values
class_sample_count = np.array([len(available_train_labels[available_train_labels == t]) for t in [0, 1]])
weights = 1. / class_sample_count
samples_weights = np.array([weights[t] for t in available_train_labels])

# WeightedRandomSampler for the training set
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

# Model setup using ResNet18 with a binary classification head
model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 1)  # Adjust for binary classification

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    predicted = torch.round(torch.sigmoid(outputs))
    correct = (predicted == labels).float().sum()
    return correct / labels.size(0)

# Training loop with accuracy display
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).view(-1)  # Ensure output is 1D
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, labels).item()

    train_loss = running_loss / len(train_loader)
    train_acc = running_acc / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

    # Validation phase with accuracy display
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).view(-1)  # Ensure output is 1D
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += calculate_accuracy(outputs, labels).item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

print("Training Complete")
