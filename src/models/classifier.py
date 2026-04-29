import torch.nn as nn
from torch.nn import Linear

class MLPClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A simple MLP head. This can be made more complex if needed.
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x):
        h = self.lin(x)
        return h.log_softmax(dim=-1)
