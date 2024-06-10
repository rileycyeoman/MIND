#Imports
import math
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel


TRAIN_INPUT = '/home/yeoman/research/train'
TEST_INPUT = '/home/yeoman/research/test'
IMG_SIZE = 48


def drop_path(x, drop_prob: float = 0., training: bool = False):
    return x
    # if drop_prob == 0. or not training:
    #     return x
    # keep_prob = 1 - drop_prob
    # shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # random_tensor.floor_()  # binarize
    # output = x.div(keep_prob) * random_tensor
    # return output

# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(
        self,
        dim:             int,
        num_heads:       int = 6,
        qkv_bias:        bool = False,
        qk_norm:         bool = False,
        attn_drop:       float = 0.,
        proj_drop:       float = 0.,
        norm_layer:      nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(
        self, 
        config, 
        img_size: int = 48, 
        patch_size:int = 16,
        in_channels:int = 3) -> None: 
        super().__init__()
        self.img_size = config["image_size"] #(h,w)
        self.patch_size = config["patch_size"] # 16
        self.in_channels = config["num_channels"] # 1 or 3
        self.hidden_size = config["hidden_size"] #usually 768
        
        self.num_patches = (self.img_size // self.patch_size) ** 2 # (h * w)/p^2
        self.proj = nn.Conv2d(in_channels= self.in_channels, 
                              out_channels=self.hidden_size, 
                              kernel_size=self.patch_size, 
                              stride=self.patch_size) # (C, Hid, kernel size, stride)

    def forward(self, x):
        B, C, H, W = x.shape #not necessary but good for showing what's what
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings. 
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
                
        

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x







class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """
    def __init__(self, config):
        super().__init__()
        self.fc1= nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.drop = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Block(nn.Module):
    """
    A single transformer block.
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.config = config
        dim = config["hidden_size"]
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(config)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # self.drop_path = DropPath(drop_path if drop_path > 0. else nn.Identity)
        
    def forward(self, x, return_attention = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        # x = x + self.drop_path(y)
        # x = x + self.drop_path(self.mlp(self.norm1(x)))
        x = x + y
        x = x + self.mlp(self.norm1(x))
        return x

class ViT(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config, img_size=[48], patch_size=16, in_chans=3, num_classes=7, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs): #TODO: FIX NORM LAYER BEING CALLED OUT EARLIER?
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = self.num_features = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.patch_embed = PatchEmbeddings(config)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(embed_dim)
        # Create a linear layer to project the encoder's output to the number of classes
        # self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        # Initialize the weights
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
            
            
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    
    
    def prepare_tokens(self, x):
            B, nc, w, h = x.shape
            x = self.patch_embed(x)  # patch linear embedding

            # add the [CLS] token to the embed patch tokens
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # add positional encoding to each token
            x = x + self.interpolate_pos_encoding(x, w, h)

            return self.pos_drop(x)
        
        
        

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output



class LinearClassifier(nn.Module):
    def __init__ (self, dim, num_labels = 7):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)



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
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.RandomApply(additional_transforms, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2, antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Update the test_transform with additional transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root=TRAIN_INPUT, transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root=TEST_INPUT, transform=test_transform)

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
    trainset = torchvision.datasets.ImageFolder(root=TRAIN_INPUT, transform=transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
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
    testset = torchvision.datasets.ImageFolder(root=TEST_INPUT, transform=transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
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
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
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
    attention_maps = F.interpolate(attention_maps, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((IMG_SIZE, IMG_SIZE)), np.zeros((IMG_SIZE, IMG_SIZE))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((IMG_SIZE, IMG_SIZE)), attention_maps[i].cpu()), axis=1)
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


config = {
    "patch_size": 16,  # Input image size: 48x48 -> 8x8 patches
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 4 * 768, # 4 * hidden_size
    "hidden_dropout_prob": 0.5,
    "attention_probs_dropout_prob": 0.5,
    "initializer_range": 0.02,
    "image_size": IMG_SIZE,
    "num_classes": 7, # num_classes for FER2013
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0




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
        
        
        
        
        
        
    
        
        
        
        


