import torch.nn as nn
import torch
from functools import partial
from collections import OrderedDict

def drop_path(x,drop_prob:float=0,training:bool=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """2D->1D"""
    def __init__(self,img_size=224,patch_size=16,in_c=3,embed_dim=768,norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        B,C,H,W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],f"input size error({H},{W})"
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale else head_dim ** -0.5
        self.qkv = nn.Linear(dim,dim * 3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self,x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B,N,C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C // self.num_heads).permute(2,0,3,1,4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q,k,v = qkv[0],qkv[1],qkv[2]
        # [batch_size, num_heads, num_patches + 1,num_patches + 1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        out = (attn @ v).transpose(1,2).resize(B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Mlp(nn.Module):
    def __init__(self,in_channel,hidden_channel=None,out_channel=None,dropout_ratio=0.):
        super(Mlp, self).__init__()
        self.hidden_c = hidden_channel or in_channel
        self.out_c = out_channel or in_channel
        self.fc1 = nn.Linear(in_channel,self.hidden_c)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(self.hidden_c,self.out_c)

    def forward(self,x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 drop_prob,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 mlp_ratio=4,
                 dropout_ratio=0.,
                 norm_layer=nn.LayerNorm
                 ):
        """
        @param drop_prob: droppath中的ratio
        @param dim: 嵌入的维度
        @param num_heads: 注意力头个数
        @param qkv_bias: qkv是否使用偏置
        @param qk_scale: qk是否用规定的scale
        @param attn_drop_ratio: 注意力的drop
        @param dropout_ratio: MLP的drop(包括注意力尾部的mlp)
        @param norm_layer: Layer层
        """
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,num_heads=num_heads,qkv_bias=qkv_bias,
                              qk_scale=qk_scale,attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=dropout_ratio)
        self.drop1 = DropPath(drop_prob=drop_prob) if drop_prob > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_channel=dim,hidden_channel=int(dim * mlp_ratio),out_channel=dim,dropout_ratio=dropout_ratio)
        self.drop2 = DropPath(drop_prob=drop_prob)

    def forward(self,x):
        out = self.norm1(x)
        out = self.attn(out)
        out = self.drop1(out)
        out += x
        out2 = self.norm2(out)
        out2 = self.mlp(out2)
        out2 = self.drop2(out2)
        out2 += out
        return out2

class VisionTransformer(nn.Module):
    @staticmethod
    def _init_vit_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def __init__(self,img_size=224,patch_size=16,in_c=3,embed_dim=768,
                 num_classes=1000,depth=12,num_heads=12,mlp_ratio=4,
                 qkv_bias=True,qk_scale=None,representation_size=None,
                 distilled=False,drop_ratio=0.,attn_drop_ratio=0.,
                 drop_path_ratio=0.,embed_layer=PatchEmbed,norm_layer=None
                 ):
        """

        @param img_size:
        @param patch_size:
        @param in_c:
        @param embed_dim:
        @param num_classes:
        @param depth:Transformer Block的层数
        @param num_heads:
        @param mlp_ratio:
        @param qkv_bias:
        @param qk_scale:
        @param representation_size: 分类头的pre-logits全连接的输出节点数，详见霹雳吧啦vit博客 https://blog.csdn.net/qq_37541097/article/details/118242600
        @param distilled:没用
        @param drop_ratio:
        @param attn_drop_ratio:
        @param drop_path_ratio:
        @param embed_layer:
        @param norm_layer:
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm,eps=1e-6)

        self.patch_embed = embed_layer(img_size=img_size,patch_size=patch_size,
                                      in_c=in_c,embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1,self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.num_patches,self.dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        # Transformer Encoder
        dpr = [x.item() for x in torch.linspace(0,drop_path_ratio,depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                  attn_drop_ratio=attn_drop_ratio,mlp_ratio=mlp_ratio,drop_prob=dpr[i],
                  dropout_ratio=drop_ratio,norm_layer=norm_layer)
            for i in range(depth)
        ])


        self.norm = norm_layer(self.dim)

        # 分类头
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc",nn.Linear(self.dim,representation_size)),
                ("act",nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(self._init_vit_weights)



    def forward_feature(self,x):
        # [B,C,H,W] -> [B,num_pathce,embed]
        x = self.patch_embed(x) # [B,196,768]
        # cls:[1,1,768] -> [B,1,768]
        cls_token = self.cls_token.expand(x.shape[0],-1,-1) # 复制到其他batch维度
        # [B,197,768]
        x = torch.concat((cls_token,x),dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:,0])

    def forward(self,x):
        x =self.forward_feature(x)
        x = self.head(x)
        return x


def vit_base_patch16_224_in21k(num_classes:int = 21843,has_logits: bool = True):
    model = VisionTransformer(img_size=224,patch_size=16,embed_dim=768,
                              depth=12,num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model



