#Imports
import math
import torch
from torch import nn
import math
import torch.nn.functional as F
import json
import utils
from performer_pytorch import SelfAttention as PerformerAttention


with open('config.json', 'r') as json_file:
    config = json.load(json_file)

PATCH_SIZE = int(config['PARAMETERS']['patch_size'])
HIDDEN_SIZE = int(config['PARAMETERS']['hidden_size'])
NUM_HIDDEN_LAYERS = int(config['PARAMETERS']['num_hidden_layers'])
NUM_ATTENTION_HEADS = int(config['PARAMETERS']['num_attention_heads'])
INTERMEDIATE_SIZE = int(config['PARAMETERS']['intermediate_size'])
HIDDEN_DROPOUT_PROB = float(config['PARAMETERS']['hidden_dropout_prob'])
ATTENTION_PROBS_DROPOUT_PROB = float(config['PARAMETERS']['attention_probs_dropout_prob'])
INITIALIZER_RANGE = float(config['PARAMETERS']['initializer_range'])
IMAGE_SIZE = int(config['PARAMETERS']['image_size'])
NUM_CLASSES = int(config['PARAMETERS']['num_classes'])
NUM_CHANNELS = int(config['PARAMETERS']['num_channels'])
QKV_BIAS = bool(config['PARAMETERS']['qkv_bias'])
SEQ_LEN = int(config['PARAMETERS']['seq_len'])

def drop_path(x, drop_prob: float = 0., training: bool = False):
    # return x
    # if drop_prob == 0. or not training:
    if drop_prob == 0.:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# class Attention(nn.Module): #Basic attention
#     def __init__(
#         self,
#         dim:             int,
#         num_heads:       int = NUM_ATTENTION_HEADS,
#         qkv_bias:        bool = False,
#         qk_norm:         bool = False,
#         attn_drop:       float = 0.,
#         proj_drop:       float = 0.,
#         norm_layer:      nn.Module = nn.LayerNorm,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape #batch size, patch, features
#         qkv = self.qkv(x).reshape(B,N,3,self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0) #unbinding tensor into 3 separate tensors
#         q, k = self.q_norm(q), self.k_norm(k) #normalizing q and k
#         q = q * self.scale #scaling q
#         attn = q @ k.transpose(-2, -1) 
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = attn @ v
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn


class Attention(nn.Module): #Linformer transformer
    def __init__(
        self,
        dim:             int,
        num_heads:       int = NUM_ATTENTION_HEADS,
        qkv_bias:        bool = False,
        qk_norm:         bool = False,
        attn_drop:       float = 0.,
        proj_drop:       float = 0.,
        seq_len:         int = SEQ_LEN,
        norm_layer:      nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads #Number of attention Heads
        self.head_dim = dim // num_heads #input dimension divided by num heads
        self.scale = self.head_dim ** -0.5  

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Linformer configuration
        self.linformer_attn = utils.LinformerSelfAttention(dim=dim, seq_len=seq_len, heads=num_heads, dim_head=self.head_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape #batch, 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale

        # Use Linformer attention
        attn_output = self.linformer_attn(x)
        
        # Ensure the attn_output is consistent with the original attention output
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_output




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
        img_size: int = 224, 
        patch_size:int = 16,
        hidden_size: int = 768,
        in_channels:int = 3) -> None: 
        super().__init__()
        self.img_size = IMAGE_SIZE #(h,w)
        self.patch_size = PATCH_SIZE # 16
        self.in_channels = NUM_CHANNELS # 1 or 3
        self.hidden_size = HIDDEN_SIZE #usually 768
        
        self.num_patches = (self.img_size // self.patch_size) ** 2 # ((h * w)/p)^2
         # (C, Hid, kernel size, stride)
        self.proj = nn.Conv2d(in_channels= self.in_channels, #Number of color channels
                              out_channels=self.hidden_size, #the hidden layer
                              kernel_size=self.patch_size, #patch size of image
                              stride=self.patch_size) #stride with patches

    def forward(self, x):
        B, C, H, W = x.shape #not necessary but good for showing what's what
        #N = HW/P^2
        # x-> x_p = (B,C,H,W) -> (B,N, P^2 * C)
        #(B,C,H,W) -> (B,N, Hidden) ->
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=HIDDEN_SIZE,out_features=INTERMEDIATE_SIZE)

        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=INTERMEDIATE_SIZE,out_features=HIDDEN_SIZE)

        self.drop = nn.Dropout(p = HIDDEN_DROPOUT_PROB)
       

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
        dim = HIDDEN_SIZE
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP()
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
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_chans=NUM_CHANNELS, num_classes=NUM_CLASSES, embed_dim=INTERMEDIATE_SIZE, depth=12,
                 num_heads=NUM_ATTENTION_HEADS, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs): #TODO: FIX NORM LAYER BEING CALLED OUT EARLIER?
        super().__init__()
        self.img_size = IMAGE_SIZE
        self.hidden_size = embed_dim
        self.num_classes = NUM_CLASSES
        self.num_features = self.hidden_size
        # Create the embedding module
        self.patch_embed = PatchEmbeddings(img_size=img_size, patch_size=patch_size, in_channels=in_chans, hidden_size=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(embed_dim)
        # Create a linear layer to project the encoder's output to the number of classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity
        # Initialize the weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
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
        npatch = x.shape[1] - 1 # the 1 is to ignore the class token
        N = self.pos_embed.shape[1] - 1 
        if npatch == N and w == h: #sizes match
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0] #class positional embeddings
        patch_pos_embed = self.pos_embed[:, 1:] #token positional embeddings
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size 
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate( #interpolate to 2D
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



"""
This portion is dedicated to adding any features from DINO
"""


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x