import torch
import torch.nn as nn
from src.models.gnn_encoders import GNNEncoder
from src.models.text_encoders import TextEncoder
from src.models.classifier import MLPClassifier

class UPFDModel(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        model_cfg = config['model']
        self.use_gnn = model_cfg['use_gnn']
        self.use_text = model_cfg['use_text']
        hidden_channels = model_cfg['hidden_channels']

        self.gnn_encoder = None
        self.text_encoder = None
        combined_channels = 0

        if self.use_gnn:
            self.gnn_encoder = GNNEncoder(
                model_cfg['gnn_type'], in_channels, hidden_channels
            )
            combined_channels += hidden_channels

        if self.use_text:
            self.text_encoder = TextEncoder(in_channels, hidden_channels)
            combined_channels += hidden_channels

        if combined_channels == 0:
            raise ValueError("At least one encoder (GNN or Text) must be enabled in the config.")

        self.classifier = MLPClassifier(combined_channels, out_channels)

    def forward(self, x, edge_index, batch):
        embeddings = []

        if self.use_gnn:
            embeddings.append(self.gnn_encoder(x, edge_index, batch))

        if self.use_text:
            embeddings.append(self.text_encoder(x, batch))

        # Concatenate embeddings from all active encoders
        if len(embeddings) > 1:
            h = torch.cat(embeddings, dim=-1)
        else:
            h = embeddings[0]

        return self.classifier(h)
