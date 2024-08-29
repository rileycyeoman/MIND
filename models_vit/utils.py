import math
import torch
from torch import nn
import torch.distributed
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import json
from PIL import Image
import pathlib
import numpy as np
from torchvision import transforms
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
CLASSES = config['DATA']['CLASSES']
EPOCHS = int(config['PARAMETERS']['epochs'])


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


#DINO utilities

# #Temporarily stealing from Meta
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


class DINOLoss(nn.Module):
    def __init__(self, 
                 out_dim, 
                 ncrops: int = 10, 
                 warmup_teacher_temp: float = 0.04, 
                 teacher_temp:float = 0.04,
                 warmup_teacher_temp_epochs:int = 0, 
                 nepochs = EPOCHS, 
                 student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        torch.distributed.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * torch.distributed.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
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
        
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)).weight_g.data.fill_(1)
        
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
    