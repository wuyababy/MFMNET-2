import torch
from torch_geometric.nn import GCNConv

device = torch.device("xpu")

x = torch.randn(4, 78, device=device)
edge_index = torch.tensor(
    [[0,1,2,3,0,2],
     [1,0,3,2,2,0]],
    dtype=torch.long,
    device=device
)

conv = GCNConv(78, 128).to(device)
y = conv(x, edge_index)

print(y.shape)
print(y.device)