import math
import torch
from torch import nn
import torchvision
import pandas as pd
import numpy as np
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as transforms
import configparser
from ViTransformer import ViT

configs = configparser.ConfigParser()
configs.read('config.ini')
config = {
    "TRAIN_INPUT": configs["DATA"]['TRAIN_INPUT'],
    "TEST_INPUT": configs["DATA"]['TEST_INPUT'],
    "patch_size": configs.getint("PARAMETERS", "patch_size"),
    "hidden_size": configs.getint("PARAMETERS", "hidden_size"),
    "num_hidden_layers": configs.getint("PARAMETERS", "num_hidden_layers"),
    "num_attention_heads": configs.getint("PARAMETERS", "num_attention_heads"),
    "intermediate_size": configs.getint("PARAMETERS", "intermediate_size"),
    "hidden_dropout_prob": configs.getfloat("PARAMETERS", "hidden_dropout_prob"),
    "attention_probs_dropout_prob": configs.getfloat("PARAMETERS", "attention_probs_dropout_prob"),
    "initializer_range": configs.getfloat("PARAMETERS", "initializer_range"),
    "image_size": configs.getint("PARAMETERS", "image_size"),
    "num_classes": configs.getint("PARAMETERS", "num_classes"),
    "num_channels": configs.getint("PARAMETERS", "num_channels"),
    "qkv_bias": configs.getboolean("PARAMETERS", "qkv_bias"),
    "use_faster_attention": configs.getboolean("PARAMETERS", "use_faster_attention")
}
print(config["patch_size"])
# IMG_SIZE = config["PARAMETERS"]["image_size"]
# TRAIN_INPUT = config["DATA"]['TRAIN_INPUT']
# TEST_INPUT = config["DATA"]['TEST_INPUT']
def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    # Define additional transforms for performance improvement
    additional_transforms = [
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomErasing(p=0.9, scale=(0.02, 0.2)),
    ]

    # Update the train_transform with additional transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config["image_size"], config["image_size"]), antialias=True),
        transforms.RandomApply(additional_transforms, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((config["image_size"], config["image_size"]), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2, antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Update the test_transform with additional transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config["image_size"], config["image_size"]), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root=config["TRAIN_INPUT"], transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root=config["TEST_INPUT"], transform=test_transform)

    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        trainset = torch.utils.data.random_split(trainset, [train_sample_size, len(trainset) - train_sample_size])[0]

    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        testset = torch.utils.data.random_split(testset, [test_sample_size, len(testset) - test_sample_size])[0]

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = trainset.classes
    return trainloader, testloader, classes

            
        
        
        
        
def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViT(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    trainset = torchvision.datasets.ImageFolder(root=config["TRAIN_INPUT"], transform=transforms.Compose([
            transforms.Resize((config["image_size"],config["image_size"]), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])) 
    classes = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise")
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
    testset = torchvision.datasets.ImageFolder(root=config["TEST_INPUT"], transform=transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"]), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    classes = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise")
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.transpose(np.asarray(testset[i][0]), (1, 2, 0)) for i in range(len(testset))]
    labels = [testset[i][1] for i in indices]
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((config["image_size"], config["image_size"]), antialias=True),
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
    attention_maps = F.interpolate(attention_maps, size=(config["image_size"], config["image_size"]), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((config["image_size"], config["image_size"])), np.zeros((config["image_size"], config["image_size"]))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((config["image_size"], config["image_size"])), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()        
        
        
        
        
        
        
        
        
        
exp_name = 'vit-with-25-epochs' #@param {type:"string"}
batch_size = 32 #@param {type: "integer"}
epochs = 25 #@param {type: "integer"}
lr = 1e-2  #@param {type: "number"}
save_model_every = 0 #@param {type: "integer"}

import torch
from torch import nn, optim
#This is not needed for kaggle, may be necessary later
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# These are not hard constraints, but are used to prevent misconfigurations
assert int(config["hidden_size"]) % int(config["num_attention_heads"]) == 0
assert eval(config['intermediate_size']) == 4 * int(config['hidden_size']) 
assert int(config['image_size']) % int(config['patch_size']) == 0




class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.model = model

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        # Save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # Get predictions
                logits  = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def main():
    # Training parameters
    save_model_every_n_epochs = save_model_every
    # Load the FER2013 dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer
    model = ViT(config,
                img_size = IMG_SIZE,
                patch_size= 16,
                in_chans= 1,
                num_classes= 7,
                embed_dim=768)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
#     optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay = 1e-2, momentum = 0.9)
    # linear_classifier = LinearClassifier()
    # linear_classifier.train()
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, optimizer, loss_fn, exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == '__main__':
    main()