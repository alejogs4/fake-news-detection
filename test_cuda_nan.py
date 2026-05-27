import torch
import torch.nn.functional as F

attention = -9e15 * torch.ones(1, 10, 5)
if torch.cuda.is_available():
    attention = attention.cuda()
    attention_node = F.softmax(attention, dim=2)
    print("CUDA NaNs?", torch.isnan(attention_node).any().item())
else:
    print("CUDA not available")
