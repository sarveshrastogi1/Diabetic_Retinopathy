import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets
from torch.utils.data import DataLoader, random_split
from albumentations import Compose, Normalize, Resize, HorizontalFlip, RandomBrightnessContrast, RandomRotate90
from albumentations.pytorch import ToTensorV2
from timm import create_model
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 5

# --- DATA AUGMENTATION ---
def get_train_transforms():
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        HorizontalFlip(p=0.3),
        RandomRotate90(p=0.3),
        RandomBrightnessContrast(p=0.1),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_test_transforms():
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# --- DATASET LOADER ---
class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        self.aug_transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = datasets.folder.default_loader(path)
        if self.aug_transform:
            image = self.aug_transform(image=np.array(image))['image']
        return image, label

# --- ENSEMBLE MODEL ---
class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, NUM_CLASSES)
        self.vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)

    def forward(self, x):
        out1 = self.efficientnet(x)
        out2 = self.vit(x)
        return (out1 + out2) / 2

# --- LABEL SMOOTHING CROSS-ENTROPY ---
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = self.log_softmax(logits)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / n_classes
        loss = (-targets * log_probs).mean()
        return loss

# --- EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None

    def step(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

# --- TRAINING LOOP ---
def train_model(model, dataloaders, optimizer, scheduler, criterion, num_epochs, save_path='best_ensemble_model.pth'):
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f'Saved Best Model with Acc: {best_acc:.4f}')

            if phase == 'val' and early_stopping.step(epoch_loss):
                return

        scheduler.step()

    print(f'Best val Acc: {best_acc:.4f}')

# --- MAIN FUNCTION ---
def main():
    root_folder = '/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/segmentation_mebo/colored_images'

    dataset = AlbumentationsDataset(root_folder, transform=get_train_transforms())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.aug_transform = get_test_transforms()

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }

    model = EnsembleModel().to(DEVICE)

    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.5, 2.0]).to(DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(dataloaders['train']), epochs=EPOCHS)

    train_model(model, dataloaders, optimizer, scheduler, criterion, EPOCHS, save_path='best_ensemble_model.pth')

if __name__ == '__main__':
    main()
