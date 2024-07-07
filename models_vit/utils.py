import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import json
from torchvision import transforms
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
CLASSES = config['DATA']['CLASSES']

# Borrowing from https://github.com/lucidrains/linformer/blob/master/linformer/

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class LinformerSelfAttention(nn.Module):
    def __init__(self, 
                 dim: int, 
                 seq_len: int, 
                 k : int = 256,
                 heads : int = 8, 
                 dim_head : int = None, 
                 one_kv_head : bool = False, 
                 share_kv : bool = False, 
                 dropout : float = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len <= self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # allow for variable sequence lengths (less than maximum sequence length) by slicing projections

        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)








###### Data Handling ######


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.classes = subset.dataset.classes
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DataHandler:
    def __init__(self,
                root_dir : str = './data',
                dataset_name : str = 'NHF',
                transform = None,
                download : bool = True,
                train : bool = True,
                batch_size : int = 32, 
                num_workers : int = 4, 
                train_sample_size : int = None, 
                test_sample_size  : int = None, 
                image_size : int = 32,
                num_channels : int = 3,
                )-> None:
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.root = root_dir
        self.train = train
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.image_size = image_size
        self.train_transform = self.get_train_transform()
        self.test_transform = self.get_test_transform()
        self.num_channels = num_channels
    def get_train_transform(self):
        additional_transforms = [
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomErasing(p=0.9, scale=(0.02, 0.2)),
        ]

        return transforms.Compose([
            # transforms.Grayscale(num_output_channels= 1),
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.RandomApply(additional_transforms, p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((self.image_size, self.image_size), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2, antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.5,) * 1, (0.5,) * 1)
        ])
    def get_test_transform(self):
        return transforms.Compose([
            # transforms.Grayscale(num_output_channels= 1),
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            # transforms.Normalize((0.5,) * 1, (0.5,) * 1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])




    def get_dataset(self, train=True):
        if self.dataset_name == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=train, download=self.download, transform=self.train_transform if train else self.test_transform)
        # Add more datasets as needed
        elif self.dataset_name == 'FER2013':
            train_input = "/home/yeoman/research/train" 
            test_input = "/home/yeoman/research/test"  
            if train:
                dataset = torchvision.datasets.ImageFolder(root=train_input, transform=self.train_transform)
            else:
                dataset = torchvision.datasets.ImageFolder(root=test_input, transform=self.test_transform)
        
        
        
        elif self.dataset_name == 'NHF':
            
            train_input = '/home/yeoman/MIND/models_vit/data/train'
            test_input = '/home/yeoman/MIND/models_vit/data/test'
            if train:
                dataset = torchvision.datasets.ImageFolder(root=train_input, transform=self.train_transform)
            else:
                dataset = torchvision.datasets.ImageFolder(root=test_input, transform=self.test_transform)
            self.classes = dataset.classes
        
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return dataset
    def get_data_loader(self, dataset, train=True):
        if train and self.train_sample_size is not None:
            indices = torch.randperm(len(dataset))[:self.train_sample_size]
            dataset = Subset(dataset, indices)
        elif not train and self.test_sample_size is not None:
            indices = torch.randperm(len(dataset))[:self.test_sample_size]
            dataset = Subset(dataset, indices)

        return DataLoader(dataset, 
                          batch_size=self.batch_size, 
                          shuffle=train, 
                          num_workers=self.num_workers, 
                          drop_last=not train, 
                          pin_memory=True)

    def prepare_data(self):
        trainset = self.get_dataset(train=True)
        testset = self.get_dataset(train=False)

        trainloader = self.get_data_loader(trainset, train=True)
        testloader = self.get_data_loader(testset, train=False)

        classes = trainset.dataset.classes if isinstance(trainset, Subset) else trainset.classes
        return trainloader, testloader, classes