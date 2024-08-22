from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import json
import torch
from transformers import DINOModel, DINOConfig, AutoFeatureExtractor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn, optim
from tqdm import tqdm

# Custom dataset class for loading images and labels
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Hyperparameters
batch_size = 32
learning_rate = 1e-4
num_epochs = 10
num_classes = 7  # Adjust this to the number of classes in your dataset

# Load feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vitb16')
model = DINOModel.from_pretrained('facebook/dino-vitb16')

# Replace the classifier head
model.classifier = nn.Linear(model.config.hidden_size, num_classes)

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DINO expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

train_image_paths = "data/data_imagenette/train"
classes_path = pathlib.Path('data/data_imagnette/imagenette_labels.json')
        
with classes_path.open('r') as f:
    classes = json.load(f)

# Prepare dataset (you'll need to fill in your own image paths and labels)
train_dataset = FaceDataset(image_paths=train_image_paths, labels=train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "face_classification_dino_vitb16.pth")