import torch
import torch.nn as nn


### Main Components

class PatchEmbed(nn.Module):
    
    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 768) -> None:
        super.__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels= in_chans,
            out_channels= embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.proj(x)
            x.flatten(2).transpose(1,2)
            return x
        
        
        
class Attention(nn.Module):
    def __init__(self, dim: int, n_heads:int = 12, qkv_bias: bool = True, attn_p: float = 0., proj_p: float = 0.) -> None:
        super.__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads #makes it so concatenation will result in same dimension as input
        self.scale = self.head_dim ** -0.5 #prevents small gradients coming from large inputs to softmax
        #generate qkv/linear mapping of a token, doing each individually is an option 
        self.qkv = nn.Linear(in_features = dim, out_features = dim * 3, bias = qkv_bias) #apply linear transformation, here we create the size of the learnable parameter
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples, n_tokens, dim = x.shape
        #ensure that embedding dimension is the same as the constructor
        if dim!= self.dim:
            raise ValueError
        #queries, keys, values
        qkv = self.qkv(x)  #(n_samples, n_patches + 1, 3 * dim)
        #add a dimension for the dimension of the heads
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) # (n_samples, n_patches, 3, n_heads + 1, head_dim)
        #change the order of dimensions
        qkv = qkv.permute(2,0,3,1,4) #(3, n_samples, n_heads, n_patches + 1, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2] 
        #prepare for dot product
        k_T = k.transpose(-2, -1) #(n_samples, n_heads, head_dim, n_patches + 1)
        
        dp = (q @ k_T) * self.scale 
        attn = dp.softmax(dim = -1)  #only over last dimension
        attn = self.attn_drop(attn)
        
        weighted_avg = attn @ v 
        weighted_avg = weighted_avg.tranpose(1,2)
        weighted_avg = weighted_avg.flatten(2)
        
        x = self.proj (weighted_avg)
        x = self.proj_drop(x)
        
        return x
    
    
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features:int, out_features:int, p:float = 0.) -> None:
        super.__init__()
        self.fc1 = nn.Linear(in_features = in_features, out_features= hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features= hidden_features, out_features= out_features)
        self.drop = nn.Dropout(p) 
    
    def forward(self, x:torch.Tensor):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, dim:int, n_heads:int, mlp_ratio: float = 4.0, qkv_bias: bool = True, p:float = 0., attn_p: float = 0.) -> None:
        super.__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features= hidden_features,
            out_features= dim
        )
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    
    
    
class ViT(nn.Module):
    def __init__(self, 
                 img_size:int, 
                 patch_size:int, 
                 in_chans:int = 3, 
                 n_classes:int = 7, 
                 embed_dim:int = 768, 
                 depth:int = 12, 
                 n_heads:int = 12, 
                 mlp_ratio:float = 4., 
                 qkv_bias:bool = True,
                 p:float = 0.,
                 attn_p: float = 0.) -> None:
        super.__init__()
        
        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x