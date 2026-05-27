import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool

class GNNEncoder(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, concat=True):
        super().__init__()
        self.concat = concat
        if model_type == 'GCN':
            self.conv = GCNConv(in_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv = SAGEConv(in_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv = GATConv(in_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported GNN model type: {model_type}")

        if self.concat:
            self.lin0 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv(x, edge_index).relu()
        h = global_max_pool(h, batch)

        if self.concat:
            # Get the root node (news content) features of each graph
            root_indices = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root_indices = torch.cat([root_indices.new_zeros(1), root_indices + 1], dim=0)
            news = x[root_indices]
            news = self.lin0(news).relu()
            h = torch.cat([h, news], dim=1)
            h = self.lin1(h).relu()
            
        return h
