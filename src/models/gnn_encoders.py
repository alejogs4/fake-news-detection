import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool

class GNNEncoder(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels):
        super().__init__()
        
        if model_type == 'GCN':
            self.conv = GCNConv(in_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv = SAGEConv(in_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv = GATConv(in_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported GNN model type: {model_type}")

    def forward(self, x, edge_index, batch):
        h = self.conv(x, edge_index).relu()
        h = global_max_pool(h, batch)
        return h
