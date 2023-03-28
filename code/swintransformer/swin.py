import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DropPath(nn.Module):
    def __init__(self, drop_path_ratio=0., training=False):
        super(DropPath, self).__init__()
        self.drop_prob = drop_path_ratio
        self.training = training

    def drop_f(self, x):
        if self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_f(x)


def window_partition(x, window_size):
    """
    将传入的feature map按window_size划分出窗口区域
    @param x: feature map (B,H,W,C)
    @param window_size: 窗口的大小 (M)
    @return:(num_windows*B,M,M,C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将划分好的窗口还原成feature map
    @param windows: 划分好的窗口 (num_windows*B,M,M,C)
    @param window_size: 窗口的大小 (M)
    @param H: feature的高
    @param W: feature的宽
    @return: (B,H,W,C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channel=3, patch_size=4, embed_dim=96):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size

        mid_channel = in_channel * patch_size * patch_size
        self.patch_partition = nn.Conv2d(
            in_channel, mid_channel, kernel_size=patch_size, stride=patch_size)
        self.linearembed = nn.Linear(mid_channel, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        '''
        @return:[B,M*M,C]
        '''
        _, _, H, W = x.shape
        pad_need = (H % self.patch_size != 0) or (W % self.patch_size != 0)
        if pad_need:
            x = F.pad(x, [0, self.patch_size - W % self.patch_size,
                          0, self.patch_size - H % self.patch_size,
                          0, 0])
        # patch_partition: [B,C,H,W] -> [B,C*16,H/4,W/4]
        # linearembed: [B,C*16,H/4,W/4] -> [B,96,H/4,W/4]
        x = self.linearembed(self.patch_partition(x))
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.proj = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        """x:[B,H*W,C]"""
        B, L, C = x.shape
        assert L == H * W, "input feature size wrong"

        x = x.view(B, H, W, C)
        pad_need = (H % 2 != 0) or (W % 2 != 0)
        if pad_need:
            x = F.pad(x, [0, 0, 0, W % 2, 0, H % 2])

        x0 = x[:, 0::2, 0::2, 0]
        x1 = x[:, 1::2, 0::2, 0]
        x2 = x[:, 0::2, 1::2, 0]
        x3 = x[:, 1::2, 1::2, 0]

        x = torch.concat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_channel, hidden_channel=None, out_channel=None, dropout_ratio=0.):
        super(Mlp, self).__init__()
        self.hidden_c = hidden_channel or in_channel
        self.out_c = out_channel or in_channel
        self.fc1 = nn.Linear(in_channel, self.hidden_c)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(self.hidden_c, self.out_c)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置部分
        # relative_position_table :[(2M-1)*(2M-1),nH]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # 为每个点生成对应的相对位置索引relative_position_index
        idx_h = torch.arange(self.window_size)
        idx_w = torch.arange(self.window_size)
        idx = torch.stack(torch.meshgrid(
            [idx_h, idx_w], indexing='ij'))  # [2,M,M],其中2代表行、列
        idx_flatten = torch.flatten(idx, 1)  # [2,M*M]
        # [2,M*M,1] - [2,1,M*M]
        relative_idx = idx_flatten[:, :, None] - \
            idx_flatten[:, None, :]  # [2,M*M,M*M]
        relative_idx = relative_idx.permute(
            1, 2, 0).contiguous()  # [M*M,M*M,2]
        # 先行列标都加上M-1,再行标乘上2M-1,最后行列标相加
        relative_idx[:, :, 0] += self.window_size - 1
        relative_idx[:, :, 1] += self.window_size - 1
        relative_idx[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = relative_idx.sum(-1)  # [M*M,M*M]
        self.register_buffer('relative_position_index',
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
            @ return {*} [B*num_windows,M*M,total_embed_dim]
            @ param {*} self
            @ param {*} x : [num_windows*B,M*M,C]
            @ param {*} mask : 是否为SWMSA,shape=[num_windows,Wh*Ww,Wh*Ww] or None
        """
        # [batch_size*num_windows,M*M,total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, M*M, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, M*M, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, M*M, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)

        # q,k,v : [batch_size*num_windows, num_heads, M*M, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # [batch_size*num_windows, num_heads, M*M, M*M]
        attn = q @ k.transpose(-2, -1)

        # relative_position_bias_table :[M*M*M*M,nH]->[M*M,M*M,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # [nH,M*M,M*M]
        attn = attn + relative_position_bias.unsqueeze[0]

        if mask is not None:
            # mask:[nW,M*M,M*M]
            nW = mask.shape[0]
            # attn.view:[batch_size, num_windows, num_heads, M*M, M*M]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsquee(0)
            attn = attn.view(-1, self.num_heads, N, N)
            # attn:[batch_size *num_windows,num_heads,M*M,M*M]
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @ : [batch_size*num_windows, num_heads, M*M, embed_dim_per_head]
        # reshape : [batch_size*num_windows,M*M,total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.) -> None:
        super().__init__()
        self.window_size = window_size
        assert 0 <= shift_size < window_size, "shift size wrong"
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, dim, drop)

    def forward(self, x, attn_mask):
        '''
        @return:[B,M*M,C]
        '''
        H, W = self.H, self.W
        B, L, C = x.shape
        assert H * W == L, 'input feature size wrong in SwinTransformerBlock'

        short_cut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # 要将x送进windows_partition

        # 将feature map给pad到window_size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, 0, pad_l, pad_r, pad_t, pad_b])
        _, Hp, Wp, _ = x.shape

        # window shift操作
        if self.shift > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # 分割窗口
        x_windows = window_partition(
            shifted_x, self.window_size)  # [nW*B, M, M, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)  # [B*nW,M*M,C]

        # 窗口还原
        # [B*nW,M,M,C]
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # shift还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 如果有pad则去掉pad
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, L, C)

        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4, qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., downsample=False) -> None:
        super().__init__()

        self.shift_size = window_size // 2
        self.window_size = window_size

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        if downsample:
            self.downsample = PatchMerging(dim)
        else:
            self.downsample = nn.Identity()

    def create_mask(self, x, H, W):
        """
                为SW-MSA生成蒙板
        @ return {*}
        @ param {*} self
        @ param {*} x
        @ param {*} H
        @ param {*} W
        """
        # 保证Hp,Wp为window_size整数倍
        Hp = int(np.ceil(H / self.window_size) * self.window_size)
        Wp = int(np.ceil(W / self.window_size) * self.window_size)

        # 形状符合window_partition的输入
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size)  # [nW, M, M, 1]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        # 仅在连续区域（cnt一样的区域）进行有效的自注意力，非零表示不为同一区域
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W

class SwinTransformer(nn.Module):
    def __init__(self,patch_size=4,in_channels=3,num_classes=1000,
                 embed_dim=96,depth=(2,2,6,2),num_heads=(3,6,12,24),
                 window_size=96, mlp_ratio=4., qkv_bias=True,
                 drop_ratio=0., attn_drop_rate=0., drop_path_rate=0.1) -> None:
        super().__init__()

        self.num_layer =len(depth)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * (2 ** (self.num_layer - 1))) # stage4的输出channel

        self.patch_embed = PatchEmbed(in_channel=in_channels,patch_size=patch_size,
                                      embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(drop_ratio)

        # stochastic drop path
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,sum(depth))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layer):
            layers = BasicLayer(dim=int(embed_dim * (2 ** i_layer)),
                                depth=depth[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_ratio,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depth[:i_layer]):sum(depth[:i_layer+1])],
                                downsample=True if (i_layer < self.num_layer - 1) else False)
            self.layers.append(layers)
        
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x):
        x,H,W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x,H,W = layer(x,H,W)

        x = self.norm(x) # [B,M*M,C]
        x = self.avgpool(x.transpose(1,2)) # [B,C,1]
        x = torch.flatten(x,1)
        x = self.head(x)
        return x
