import torch
from torch import nn
import json
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as transforms
from ViTransformer import ViT

with open('config.json', 'r') as json_file:
    config = json.load(json_file)
    
TRAIN_INPUT = config['DATA']['TRAIN_INPUT']
TEST_INPUT = config['DATA']['TEST_INPUT']
CLASSES = config['DATA']['CLASSES']
PATCH_SIZE = int(config['PARAMETERS']['patch_size'])
HIDDEN_SIZE = int(config['PARAMETERS']['hidden_size'])
NUM_ATTENTION_HEADS = int(config['PARAMETERS']['num_attention_heads'])
NUM_CHANNELS = int(config['PARAMETERS']['num_channels'])
NUM_CLASSES = int(config['PARAMETERS']['num_classes'])
INTERMEDIATE_SIZE = int(config['PARAMETERS']['intermediate_size'])
IMAGE_SIZE = int(config['PARAMETERS']['image_size'])
EXP_NAME = config['PARAMETERS']['exp_name']
BATCH_SIZE = int(config['PARAMETERS']['batch_size'])
EPOCHS = int(config['PARAMETERS']['epochs'])
LR = float(config['PARAMETERS']['lr'])
SAVE_MODEL_EVERY = int(config['PARAMETERS']['save_model_every'])


def visualize_images():
    trainset = torchvision.datasets.ImageFolder(root=TRAIN_INPUT, transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])) 
    # classes = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise")
    # classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = CLASSES
    # Pick 30 samples randomly
    indices = torch.randperm(len(trainset))[:30]
    # images = [np.asarray(trainset[i][0]) for i in indices]
    images = [np.transpose(np.asarray(trainset[i][0]), (1, 2, 0)) for i in range(len(trainset))]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    model.eval()
    # Load random images
    num_images = 30
    testset = torchvision.datasets.ImageFolder(root=TEST_INPUT, transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    # classes = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise")
    # classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = CLASSES
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.transpose(np.asarray(testset[i][0]), (1, 2, 0)) for i in range(len(testset))]
    labels = [testset[i][1] for i in indices]
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images])
    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    logits = model(images) 
    attention_maps = model().get_last_attnetion
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((IMAGE_SIZE, IMAGE_SIZE)), np.zeros((IMAGE_SIZE, IMAGE_SIZE))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((IMAGE_SIZE, IMAGE_SIZE)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()        