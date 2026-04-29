import torch
from torch.nn import Linear

class TextEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin = Linear(in_channels, hidden_channels)

    def forward(self, x, batch):
        # Get the root node (news content) features of each graph:
        # In UPFD, the first node of each graph in the batch is the root node.
        # However, to be robust across batches, we find the first occurrence of each batch index.
        
        # This logic identifies the indices of the first node for each graph in the batch
        root_indices = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root_indices = torch.cat([root_indices.new_zeros(1), root_indices + 1], dim=0)
        
        news_features = x[root_indices]
        return self.lin(news_features).relu()
