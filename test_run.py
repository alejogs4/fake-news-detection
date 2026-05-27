import torch
from src.models.hgfnd import HGFND
from src.models.gnn_encoders import GNNEncoder
from src.models.upfd_model import UPFDModel

def test_models():
    print("Testing GNNEncoder...")
    encoder = GNNEncoder('SAGE', 10, 16)
    x = torch.randn(20, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    batch = torch.tensor([0]*10 + [1]*10)
    out = encoder(x, edge_index, batch)
    print("GNNEncoder out shape:", out.shape)
    
    print("Testing UPFDModel with HGFND...")
    config = {
        'model': {
            'use_hgfnd': True,
            'gnn_type': 'SAGE',
            'hidden_channels': 16,
            'dropout': 0.3
        }
    }
    model = UPFDModel(config, 10, 2)
    H = torch.randint(0, 2, (2, 5)).float() # 2 graphs, 5 hyperedges
    out = model(x, edge_index, batch, H=H)
    print("HGFND out shape:", out.shape)
    print("Tests passed!")

if __name__ == '__main__':
    test_models()
