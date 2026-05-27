import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from src.models.hgfnd import HGFND

def run():
    config = {
        'model': {
            'use_hgfnd': True,
            'gnn_type': 'SAGE',
            'hidden_channels': 16,
            'dropout': 0.3
        }
    }
    model = HGFND(config, 10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.randn(20, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    batch = torch.tensor([0]*10 + [1]*10)
    H = torch.randint(0, 2, (2, 5)).float()
    y = torch.tensor([0, 1])
    
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, batch, H)
        loss = F.nll_loss(out, y)
        loss.backward()
        
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}, Preds: {out.exp()}")
        optimizer.step()

if __name__ == '__main__':
    run()
