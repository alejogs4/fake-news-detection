import torch
import torch.nn as nn
from src.models.gnn_encoders import GNNEncoder
from src.models.text_encoders import TextEncoder
from src.models.classifier import MLPClassifier
from src.models.cmcg import MultiHeadCoAttentionLayer

class SentimentEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # assuming sentiment feature is provided separately (e.g., shape [batch_size, in_channels])
        self.lin = nn.Linear(in_channels, hidden_channels)

    def forward(self, sentiment_features):
        return self.lin(sentiment_features).relu()

class UPFDModel(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        model_cfg = config['model']
        self.use_gnn = model_cfg.get('use_gnn', True)
        self.use_text = model_cfg.get('use_text', True)
        self.use_sentiment = model_cfg.get('use_sentiment', False)
        self.use_cmcg = model_cfg.get('use_cmcg', False)
        
        hidden_channels = model_cfg['hidden_channels']

        self.gnn_encoder = None
        self.text_encoder = None
        self.sentiment_encoder = None
        combined_channels = 0

        if self.use_gnn:
            self.gnn_encoder = GNNEncoder(
                model_cfg['gnn_type'], in_channels, hidden_channels
            )
            combined_channels += hidden_channels

        if self.use_text:
            self.text_encoder = TextEncoder(in_channels, hidden_channels)
            combined_channels += hidden_channels

        if self.use_cmcg and self.use_gnn and self.use_text:
            self.co_attention = MultiHeadCoAttentionLayer(hidden_channels, hidden_channels, num_heads=2)
            # Channels remain the same after co-attention since we still concat them

        if self.use_sentiment:
            sentiment_dim = model_cfg.get('sentiment_dim', 1) # default to 1D sentiment score
            self.sentiment_encoder = SentimentEncoder(sentiment_dim, hidden_channels)
            combined_channels += hidden_channels

        if combined_channels == 0:
            raise ValueError("At least one encoder (GNN, Text, or Sentiment) must be enabled in the config.")

        self.classifier = MLPClassifier(combined_channels, out_channels)

    def forward(self, x, edge_index, batch, sentiment_features=None):
        embeddings = []
        h_gnn = None
        h_text = None

        if self.use_gnn:
            h_gnn = self.gnn_encoder(x, edge_index, batch)
            
        if self.use_text:
            h_text = self.text_encoder(x, batch)

        if self.use_cmcg and h_gnn is not None and h_text is not None:
            h_gnn, h_text = self.co_attention(h_gnn, h_text)
            
        if h_gnn is not None:
            embeddings.append(h_gnn)
        if h_text is not None:
            embeddings.append(h_text)

        if self.use_sentiment:
            if sentiment_features is None:
                # Fallback to zeros if not provided in the batch
                sentiment_features = torch.zeros(batch.max().item() + 1, self.sentiment_encoder.lin.in_features).to(x.device)
            h_sent = self.sentiment_encoder(sentiment_features)
            embeddings.append(h_sent)

        # Concatenate embeddings from all active encoders
        if len(embeddings) > 1:
            h = torch.cat(embeddings, dim=-1)
        else:
            h = embeddings[0]

        return self.classifier(h)
