import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import math

from torch import nn, einsum
from einops import rearrange


MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SELEFF(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1)
        self.linear2 = nn.Linear(hidden_dim, dim)

        self.se = ChannelGate(dim,dim)

    def forward(self, x):

        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.se(x.permute(0,2,1).reshape(bs,c,hw,1)).squeeze(3).permute(0,2,1)

        x = F.gelu(self.linear1(x))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = F.gelu(self.dwconv(x))

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = F.gelu(self.linear2(x))

        return x



class LEFF(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1)
        self.linear2 = nn.Linear(hidden_dim, dim)


    def forward(self, x):

        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = F.gelu(self.linear1(x))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = F.gelu(self.dwconv(x))

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = F.gelu(self.linear2(x))

        return x



class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, output_channels, reduction_ratio=16, pool_types=['avg', 'max'], activation_type = 'sigmoid'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, output_channels)
            )
        self.pool_types = pool_types
        self.activation_type = activation_type

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.adaptive_avg_pool2d( x, 1)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.adaptive_max_pool2d( x, 1)
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw


        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 ):

        super().__init__()
       
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads


        x = rearrange(x, 'b (l w) n -> b n l w', l=int(math.sqrt(n)), w=int(math.sqrt(n)))
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out




class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,norm):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if norm:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                    Residual(PreNorm(dim, SELEFF(dim, mlp_dim, dropout = dropout)))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Residual( Attention(dim, heads = heads, dropout = dropout)),
                    Residual(SELEFF(dim, mlp_dim, dropout = dropout))
                ]))
    def forward(self, x, mask = None):
        # skip-connection for each 4 layers
        res = x
        for i in range(len(self.layers)):
            attn, ff = self.layers[i]
            x = attn(x, mask = mask)
            x = ff(x)

            if (i + 1) % 2 == 0:
                x = x + res
                res = x

        return x


class myViT(nn.Module):
    def __init__(self, *, hidden_dim=512, nheads=8, depth=6, output_channels=64, mlp_dim=2048, image_size=64, patch_size=1, channels = 64, dropout = 0.1, emb_dropout = 0.1,norm=True):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, 'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, hidden_dim)
        self.embedding_to_patch= nn.Linear(hidden_dim, output_channels * patch_size ** 2)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(hidden_dim, depth, nheads, mlp_dim, dropout,norm)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.nheads = nheads
        self.mlp_dim = mlp_dim
       
    def forward(self, img, mask = None):
        p = self.patch_size
        h = img.size(2)//p
        w = img.size(3)//p
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        if self.pos_embedding.size(1)!=x.size(1):
            self.pos_embedding.data = F.interpolate(self.pos_embedding.data.unsqueeze(1),[x.size(1),self.hidden_dim],mode='bilinear').squeeze(1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = self.embedding_to_patch(x)
        x = rearrange(x, ' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = h, w = w, p1 = p, p2 = p)
        return x

class DownViT(nn.Module):
    def __init__(self, *, hidden_dim=512, nheads=8, depth=6, output_channels=64, mlp_dim=2048, image_size=64, patch_size=1, channels = 64, dropout = 0.1, emb_dropout = 0.1,norm=True):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2 # we extract the overlapping patch here
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, 'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, hidden_dim)
        # self.embedding_to_patch = nn.Linear(hidden_dim, output_channels * patch_size ** 2)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(hidden_dim, depth, nheads, mlp_dim, dropout,norm)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.nheads = nheads
        self.mlp_dim = mlp_dim
       
    def forward(self, img, mask = None):
        p = self.patch_size
        h = img.size(2)//p
        w = img.size(3)//p

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) 
        # x = F.unfold(img, p).transpose(1,2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        if self.pos_embedding.size(1)!=x.size(1):
            self.pos_embedding.data = F.interpolate(self.pos_embedding.data.unsqueeze(1),[x.size(1),self.hidden_dim],mode='bilinear').squeeze(1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)

        # x = self.embedding_to_patch(x)
        x = rearrange(x, ' b (h w) c -> b c h w', h = h, w = w)

        print(x.size())

        return x



class xViT(nn.Module):
    def __init__(self, *, hidden_dim=512, nheads=8, depth=6, output_channels=64, mlp_dim=2048, image_size=64, patch_size=1, channels = 64, dropout = 0.1, emb_dropout = 0.1,norm=True):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2 # we extract the overlapping patch here
        patch_dim = channels * patch_size ** 2
        # assert num_patches > MIN_NUM_PATCHES, 'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, hidden_dim)
        self.embedding_to_patch = nn.Linear(hidden_dim, output_channels)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(hidden_dim, depth, nheads, mlp_dim, dropout,norm)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.nheads = nheads
        self.mlp_dim = mlp_dim
       
    def forward(self, img, mask = None):
        p = self.patch_size
        h = img.size(2)
        w = img.size(3)

        # p=4
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) 
        x = F.unfold(img, p, padding=1).transpose(1,2) # keeping the original size 
        new_c = x.size(1)

        # import pdb; pdb.set_trace()
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # if self.pos_embedding.size(1)!=x.size(1):
        #     self.pos_embedding.data = F.interpolate(self.pos_embedding.data.unsqueeze(1),[x.size(1),self.hidden_dim],mode='bilinear').squeeze(1)
        # x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = self.embedding_to_patch(x)
        x = rearrange(x, ' b (h w) c -> b c h w', h = h, w = w)

        # print(x.size())
        # import pdb; pdb.set_trace()

        return x