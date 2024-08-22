import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import json
from PIL import Image
import pathlib
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


#DINO Utilities
#Borrowing from jankrepl from overfitted https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/dino/utils.py
# class MultiCropWrapper(nn.Module):
#     def __init__(
#         self,
#         backbone,
#         head
#     ):
#         super(MultiCropWrapper, self).__init__()
#         backbone.fc, backbone.head = nn.Identity(), nn.Identity()
#         self.backbone = backbone
#         self = head
        
#         def forward(self, x):
            
#             if not isinstance(x,list):
#                 x = [x]
#             idx_crops = torch.cumsum(torch.unique_consecutive(
#                 torch.tensor([inp.shape[-1] for inp in x])
#             )[1], 0)
#             n_crops = len(x)
#             concat = torch.cat(x, dim = 0)
#             cls_embedding = self.backbone(concat)
#             logits = self.new_head(cls_embedding)
#             chunks = logits.chunk(n_crops)
#             return chunks


#Temporarily stealing from Meta
class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class Loss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        teacher_temp:float = 0.04,
        student_temp:float = 0.1,
        center_momentum:float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim)) #Buffer that is not updated by optimizer, dimension of [1,output features]
        
        
        def forward(self, student_output, teacher_output):
            #Normalizations from 3.1 on the original paper (prior to softmax)
            student_temp = [s/self.student_temp for s in student_output] #this naming is misleading, it's not the temperature, rather the temp applied to the output
            teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

            student_sm = [F.log_softmax(s, dim = -1) for s in student_temp]
            teacher_sm = [F.log_softmax(t,dim=-1).detach() for t in teacher_temp]

            total_loss = 0
            n_loss_terms = 0
            
            for t_ix, t in enumerate(teacher_sm):
                for s_ix, s in enumerate(student_sm):
                    if t_ix == s_ix: #Ensure that only differing views are compared
                        continue
                    loss = torch.sum(-t * s, dim = -1) #sum dot product of outputs across feature dimension
                    total_loss += loss.mean()
                    n_loss_terms += 1
            
            total_loss /= n_loss_terms
            self.update_center(teacher_output)
            return total_loss

        #equation 4 from paper
        @torch.no_grad()
        def update_center(self, teacher_output): 
            batch_center = torch.cat(teacher_output).mean(dim = 0, keepdim=True) #(1, out_dim)
            self.center = self.center * self.center_momentum + (1 - self.center_momentum) * batch_center
        
#Prevent gradient from becoming too large, clip is maximum allowed norm of gradient
def clip_gradents(model, clip = 2.0):
    #Cycle through model parameters
    for p in model.parameters():
        #Don't both with parameters that have no gradient to begin with
        if p.grad is not None:
            #Compute L2 norm of parameter
            param_norm = p.grad.data.norm(2)
            #Divide norm by parameter norm, if param norm becomes too high, scale it by coefficient
            clip_coef = clip/(param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)

###### Data Handling ######
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
                image_size : int = 224,
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
        
        
    

    def prepare_data(self):
        
        
        path_dataset_train = pathlib.Path("data/data_imagenette/train")
        path_dataset_val = pathlib.Path("data/data_imagenette/val")
        classes_path = pathlib.Path('data/data_imagnette/imagenette_labels.json')
        
        with classes_path.open('r') as f:
            classes = json.load(f)
        
        
        
        
        
        
        transform_aug = DataAugmentation(size=224, n_local_crops = 2)
        transform_plain = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize((224, 224)),
            ]
        )

        dataset_train_aug = nn.ImageFolder(path_dataset_train, transform=transform_aug)
        dataset_train_plain = nn.ImageFolder(path_dataset_train, transform=transform_plain)
        dataset_val_plain = nn.ImageFolder(path_dataset_val, transform=transform_plain)
        
        aug_loader = DataLoader(
        dataset_train_aug,
        batch_size= 32,
        shuffle=True,
        drop_last=True,
        num_workers= 4,
        pin_memory=True,
        )
        train_loader = DataLoader(
            dataset_train_plain,
            batch_size= 32,
            drop_last= False,
            num_workers= 4,
        )
        val_loader = DataLoader(
            dataset_val_plain,
            batch_size = 32,
            drop_last = False,
            num_workers = 4,
        )
        
        val_subset_loader = DataLoader( #TODO find out what this does
            dataset_val_plain,
            batch_size= 32,
            drop_last = False,
            sampler = nn.utils.SubsetRandomSampler(list(range(0, len(dataset_val_plain), 50))),
            num_workers = 4,
        )
        
        return train_loader, aug_loader, val_loader



        
#Create crops of input images
class DataAugmentation:
    def __init__(self,
            global_crops_scale: tuple[float, float] =(0.4, 1), #40% or above for global 
            local_crops_scale : tuple[float, float] = (0.05, 0.4), #40% or below for local
            n_local_crops: int  = 8,
            size: int = 224
            ) -> None:
        self.n_local_crops = n_local_crops
        RandomGaussianBlur = lambda p: transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2))], p=p)
        #Mess with colors and axes
        flip_and_jitter = transforms.Compose( 
            [
                transforms.RandomHorizontalFlip(p=0.5), #flip on y-axis
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue = 0.1
                        ),
                    ]
                ),
                transforms.RandomGrayscale(p=0.2)
            ]
        )
        #Image normalization
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                #(mean of channel(i)), (stdev of channel(i))
                #Values of RGB are scaled to (original value - mean(i))/stdev(i)
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )


        self.global_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size, #expected output size of crop
                    scale = global_crops_scale, #upper and lower bounds for the crop area
                    interpolation=Image.BICUBIC #Bicubic interpolation for approxamite pixel values
                ),
                flip_and_jitter,
                RandomGaussianBlur(1.0), #always blur
                normalize
            ],
        )

        self.global_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size, #expected output size of crop
                    scale = global_crops_scale, #upper and lower bounds for the crop area
                    interpolation=Image.BICUBIC #Bicubic interpolation for approxamite pixel values
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                transforms.RandomSolarize(170, p = 0.2),
                normalize
            ],
        )

        self.local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size, #expected output size of crop
                    scale = local_crops_scale, #upper and lower bounds for the crop area
                    interpolation=Image.BICUBIC #Bicubic interpolation for approxamite pixel values
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.5), #always blur
                normalize
            ],
        )
                
                
    def __call__(self, img):
        #create empty list for all crops to reside
        all_crops = []
        #apply global crops to images and add to list
        all_crops.append(self.global_1(img))
        all_crops.append(self.global_2(img))
        #apply local crops n times across image
        all_crops.extend([self.local(img) for _ in range(self.n_local_crops)])
        
        return all_crops
                

#Connect heads to MLP network (may not be used #TODO)
class Head(nn.Module):
    def __init__(
        self,
        in_dim : int,
        out_dim: int,
        hidden_dim : int = 512,
        bottleneck_dim : int = 256 ,
        n_layers: int = 3,
        norm_last_layer: bool = False
    )-> None:
        super().__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
            
        else:
            layers = [nn.Linear(in_dim, bottleneck_dim)]
            layers.append(nn.GELU)
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
            
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)).weight_g.data.fill_(1)
        
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init(m.weight, std = 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim = -1, p = 2)
        return self.last_layer(x)
    