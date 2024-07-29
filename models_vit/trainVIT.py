import torch
from torch import nn
import torch.utils
import json, os, time
from torch.nn import functional as F
from torch import optim
from ViTransformer import ViT
from utils import DataHandler
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
DATASET =  config['PARAMETERS']['dataset_name']
# Make the output satisfying
class TextColors:
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    ENDC = "\033[0m"

        
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



        
        
        
    
        


# from torch import nn, optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using the {TextColors.PURPLE}{torch.cuda.get_device_name(0)}{TextColors.ENDC} for training.")
# These are not hard constraints, but are used to prevent misconfigurations
assert HIDDEN_SIZE % NUM_ATTENTION_HEADS == 0, "The size of hidden layer is not divisible by the number of attention heads"
assert INTERMEDIATE_SIZE == 4 * HIDDEN_SIZE, "The intermediate size is not 4 times the embedding/hidden size."
assert IMAGE_SIZE % PATCH_SIZE == 0, "Image size is not divisible by patch size"



class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, classifier, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.classifier = classifier.to(device)
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        
        
    def pretrain(self, pretrainloader, epochs, save_model_every_n_epochs=0):
        """
        Pre-train the model for the specified number of epochs.
        """
        for epoch in range(epochs):
            pretrain_loss, pretrain_accuracy = self.train_epoch(pretrainloader)
            print(f"Pretrain Epoch: {epoch+1}, Loss: {pretrain_loss:.4f}, Accuracy: {pretrain_accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (epoch+1) % save_model_every_n_epochs == 0 and epoch+1 != epochs:
                print('\tSave pretrain checkpoint at epoch', epoch+1)
                save_checkpoint(self.exp_name + '_pretrain', self.model, epoch+1)
        save_checkpoint(self.exp_name + '_pretrain_final', self.model, epochs)

    

    def finetune(self, finetuneloader, epochs, save_model_every_n_epochs=0):
        """
        Fine-tune the model for the specified number of epochs.
        """
        for epoch in range(epochs):
            finetune_loss, finetune_accuracy = self.train_epoch(finetuneloader)
            print(f"Fine-tune Epoch: {epoch+1}, Loss: {finetune_loss:.4f}, Accuracy: {finetune_accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (epoch+1) % save_model_every_n_epochs == 0 and epoch+1 != epochs:
                print('\tSave finetune checkpoint at epoch', epoch+1)
                save_checkpoint(self.exp_name + '_finetune', self.model, epoch+1)
        save_checkpoint(self.exp_name + '_finetune_final', self.model, epochs)
    
        
    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        t0 = time.time()
        # Train the model
        for i in range(epochs):
            train_loss, train_accuracy = self.train_epoch(trainloader)
            test_accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(train_accuracy)
            t1 = time.time()
            print(f"{TextColors.LIGHT_BLUE}Epoch:{TextColors.ENDC} {i+1}") 
            print(f"{TextColors.BLUE}Train loss:{TextColors.ENDC} {train_loss:.4f}")
            print(f"{TextColors.CYAN}Test loss:{TextColors.ENDC} {test_loss:.4f}") 
            print(f"{TextColors.GREEN}Accuracy:{TextColors.ENDC} {test_accuracy:.4f}\n")
            print(f"Time Elapsed from Last Epoch: {t1 - t0}")
            t0 = t1
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
        self.classifier.train()
        total_loss = 0
        correct = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            logits = self.classifier(self.model(images))
            # logits = self.classifier(self.model.get_intermediate_layers(images))
            loss = self.loss_fn(logits, labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
            # Calculate the number of correct predictions
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(trainloader.dataset)
        return total_loss / len(trainloader.dataset), accuracy

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        self.classifier.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            #go through entire batch
            for batch in testloader:
                #send each to GPU
                batch = [t.to(self.device) for t in batch]
                # Get predictions
                images, labels = batch
                
                features  = self.model(images)
                logits = self.classifier(features)
                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)
                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss

# class LinearClassifier(nn.Module):
#     """Linear layer to train on top of frozen features"""
#     def __init__(self, dim, num_labels=NUM_CLASSES):
#         super(LinearClassifier, self).__init__()
#         self.num_labels = num_labels
#         self.linear = nn.Linear(dim, num_labels)
#         self.linear.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear.bias.data.zero_()

#     def forward(self, x):
#         # flatten
#         x = x.view(x.size(0), -1)

#         # linear layer
#         return self.linear(x)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)

    def forward(self, x):
        return self.linear(x)

def main():
    # Training parameters
    save_model_every_n_epochs = SAVE_MODEL_EVERY
    # Load the dataset
    data_loader = DataHandler(batch_size=BATCH_SIZE, dataset_name= DATASET ,num_workers=4, train_sample_size= None, test_sample_size = None)
    trainloader, testloader, _ = data_loader.prepare_data()
    # Create the model, optimizer, loss function and trainer
    model = ViT(img_size = IMAGE_SIZE,
                patch_size= PATCH_SIZE,
                in_chans= NUM_CHANNELS,
                num_classes= NUM_CLASSES,
                embed_dim= HIDDEN_SIZE)
    # classifier = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
    classifier = LinearClassifier(HIDDEN_SIZE, num_labels = NUM_CLASSES)
    optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=LR, weight_decay=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr = LR, weight_decay = 1e-2, momentum = 0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, optimizer, classifier, loss_fn, EXP_NAME, device=device)
    print("==============Pre-training MIND==============")
    # trainer.pretrain(pretrainloader, PRETRAIN_EPOCHS, save_model_every_n_epochs=save_model_every_n_epochs)
    print("==============Training MIND==============")
    trainer.train(trainloader, testloader, EPOCHS, save_model_every_n_epochs=save_model_every_n_epochs)
    print("==============Fine-tuning MIND==============")
    # trainer.finetune(finetuneloader, FINETUNE_EPOCHS, save_model_every_n_epochs=save_model_every_n_epochs)
    print("==============Training done==============")

if __name__ == '__main__':
    main()