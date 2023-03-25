import torch
from torch import nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self,drop_path_ratio = 0.,training = False):
        super(DropPath, self).__init__()
        self.drop_prob = drop_path_ratio
        self.training = training

    def drop_f(self,x):
        if self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape,dtype=x.dtype,device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self,x):
        return self.drop_f(x)

def window_partition(x,window_size):
    """
    将传入的feature map按window_size划分出窗口区域
    @param x: feature map (B,H,W,C)
    @param window_size: 窗口的大小 (M)
    @return:(num_windows*B,M,M,C)
    """
    B,H,W,C = x.shape
    x = x.view(B,H // window_size,window_size,W // window_size,window_size,C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows

def window_reverse(windows,window_size,H,W):
    """
    将划分好的窗口还原成feature map
    @param windows: 划分好的窗口 (num_windows*B,M,M,C)
    @param window_size: 窗口的大小 (M)
    @param H: feature的高
    @param W: feature的宽
    @return: (B,H,W,C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,H // window_size,W // window_size,window_size,window_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x

class PatchEmbed(nn.Module):
    def __init__(self,in_channel=3,patch_size=4,embed_dim=96):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size

        mid_channel = in_channel * patch_size * patch_size
        self.patch_partition = nn.Conv2d(in_channel,mid_channel,kernel_size=patch_size,stride=patch_size)
        self.linearembed = nn.Linear(mid_channel,embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x):
        _,_,H,W = x.shape
        pad_need = (H % self.patch_size != 0) or (W % self.patch_size != 0)
        if pad_need:
            x = F.pad(x,[0,self.patch_size - W % self.patch_size,
                         0,self.patch_size - H % self.patch_size,
                         0,0])
        # patch_partition: [B,C,H,W] -> [B,C*16,H/4,W/4]
        # linearembed: [B,C*16,H/4,W/4] -> [B,96,H/4,W/4]
        x = self.linearembed(self.patch_partition(x))
        _,_,H,W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x,H,W




