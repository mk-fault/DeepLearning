import torch
from torch import nn
window_size = (2,2)
num_heads = 3
relative_position_bias_index = torch.randint(0,9,(window_size[0] * window_size[1],window_size[0] * window_size[1]))
# relative_position_bias_table = torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
relative_position_bias_table = torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),num_heads)
nn.init.trunc_normal_(relative_position_bias_table,std=0.2)
# print(relative_position_bias_table)
# print(relative_position_bias_index)

x = relative_position_bias_table[relative_position_bias_index.view(-1)]
print(x)
print(relative_position_bias_table.shape,relative_position_bias_index.shape,x.shape)