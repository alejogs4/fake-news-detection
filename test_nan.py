import torch
import torch.nn.functional as F

attention = -9e15 * torch.ones(1, 10, 5)
attention_node = F.softmax(attention.transpose(1, 2), dim=2)
print("Any NaNs?", torch.isnan(attention_node).any().item())
