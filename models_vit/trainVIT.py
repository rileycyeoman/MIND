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
from torch import optim
import torchvision.transforms as transforms
from ViTransformer import ViT

# config = configparser.ConfigParser()
# config.read('config.ini')
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

# Make the output satisfying
class TextColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.RandomApply(additional_transforms, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2, antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Update the test_transform with additional transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #use these for FER2013
    # trainset = torchvision.datasets.ImageFolder(root=TRAIN_INPUT, transform=train_transform)
    # testset = torchvision.datasets.ImageFolder(root=TEST_INPUT, transform=test_transform)
    
    # use these for CIFAR10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    
    # trainset = torchvision.datasets.Imagenette(root='./data', train=True,
    #                                         download=True, transform=train_transform, size = "320px")
    # testset = torchvision.datasets.Imagenette(root='./data', train=False,
    #                                     download=True, transform=test_transform, size = "320px")
    
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        # trainset = torch.utils.data.random_split(trainset, [train_sample_size, len(trainset) - train_sample_size])[0]
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        # testset = torch.utils.data.random_split(testset, [test_sample_size, len(testset) - test_sample_size])[0]
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

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
    copyfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), copyfile)


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
    model = ViT()
    copyfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(copyfile))
    return config, model, train_losses, test_losses, accuracies



        
        
        
        
        
        

# import torch
# from torch import nn, optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using the {TextColors.OKGREEN}{torch.cuda.get_device_name(0)}{TextColors.ENDC} for training.")
# These are not hard constraints, but are used to prevent misconfigurations
assert HIDDEN_SIZE % NUM_ATTENTION_HEADS == 0, "The size of "
assert INTERMEDIATE_SIZE == 4 * HIDDEN_SIZE, "The intermediate size is not 4 times the embedding/hidden size."
assert IMAGE_SIZE % PATCH_SIZE == 0, "Image size is not divisible by patch size"




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
            print(f"{TextColors.HEADER}Epoch:{TextColors.ENDC} {i+1}") 
            print(f"{TextColors.OKBLUE}Train loss:{TextColors.ENDC} {train_loss:.4f}")
            print(f"{TextColors.OKCYAN}Test loss:{TextColors.ENDC} {test_loss:.4f}") 
            print(f"{TextColors.OKGREEN}Accuracy:{TextColors.ENDC} {accuracy:.4f}\n")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        # Save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        scaler = torch.cuda.amp.GradScaler()
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images), labels)
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
    save_model_every_n_epochs = SAVE_MODEL_EVERY
    # Load the FER2013 dataset
    trainloader, testloader, _ = prepare_data(batch_size=BATCH_SIZE)
    # Create the model, optimizer, loss function and trainer
    model = ViT(img_size = IMAGE_SIZE,
                patch_size= PATCH_SIZE,
                in_chans= NUM_CHANNELS,
                num_classes= NUM_CLASSES,
                embed_dim= HIDDEN_SIZE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr = LR, weight_decay = 1e-2, momentum = 0.9)
    # linear_classifier = LinearClassifier()
    # linear_classifier.train()
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, optimizer, loss_fn, EXP_NAME, device=device)
    trainer.train(trainloader, testloader, EPOCHS, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == '__main__':
    main()