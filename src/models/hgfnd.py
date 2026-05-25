import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter

class PropagationEncoder(nn.Module):
    r"""
    Encodes the propagation tree of each news piece.
    xi = ROOT(GNN(Pi))
    We concatenate the encoded root representation with the original news content features
    to generate an enhanced news representation v0 = f(sigma(xi_gnn_root \oplus xi_orig_root)).
    """
    def __init__(self, gnn_type: str, in_channels: int, hidden_channels: int):
        super().__init__()
        if gnn_type == 'SAGE':
            self.conv = SAGEConv(in_channels, hidden_channels)
        else:
            # Fallback/support for other types if needed, SAGE is default in HGFND
            from torch_geometric.nn import GCNConv, GATConv
            if gnn_type == 'GCN':
                self.conv = GCNConv(in_channels, hidden_channels)
            elif gnn_type == 'GAT':
                self.conv = GATConv(in_channels, hidden_channels)
            else:
                raise ValueError(f"Unsupported GNN type for PropagationEncoder: {gnn_type}")
                
        self.fc = nn.Linear(hidden_channels + in_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        # Run GNN over the propagation tree
        h_all = self.conv(x, edge_index).relu()
        
        # In UPFD, the first node of each graph in the batch is the root node.
        # Find the root node indices for the batch
        root_indices = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root_indices = torch.cat([root_indices.new_zeros(1), root_indices + 1], dim=0)
        
        h_gnn_root = h_all[root_indices] # [num_graphs, hidden_channels]
        h_orig_root = x[root_indices]     # [num_graphs, in_channels]
        
        # Concatenate GNN root representation and original root news content
        h_concat = torch.cat([h_gnn_root, h_orig_root], dim=-1) # [num_graphs, hidden_channels + in_channels]
        
        # Linear projection with activation
        v0 = self.fc(h_concat).relu() # [num_graphs, hidden_channels]
        return v0

class HyperGATLayer(nn.Module):
    """
    Implements a single layer of Hypergraph Attention Network with a dual-level attention mechanism.
    1. Node-level attention: aggregates node representations to form hyperedge representations.
    2. Hyperedge-level attention: aggregates hyperedge representations to form node representations.
    """
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.d = hidden_channels
        self.W1 = nn.Linear(self.d, self.d, bias=False)
        self.W2 = nn.Linear(self.d, self.d, bias=False)
        self.a1 = nn.Parameter(torch.randn(self.d, 1))
        self.a2 = nn.Parameter(torch.randn(2 * self.d, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.a1)
        nn.init.xavier_uniform_(self.a2)

    def forward(self, v, H):
        """
        v: Node representations of shape [N, d]
        H: Incidence matrix of shape [N, M] (where H_ij = 1 if node i belongs to hyperedge j)
        """
        N, M = H.shape
        
        # --- 1. Node-level Attention for Hyperedge Representation ---
        # Equation (3): el_j = \sigma( \sum_{vk \in ej} \alpha_{jk} W1 vl-1_k )
        W1_v = self.W1(v) # [N, d]
        h_node = self.leaky_relu(W1_v) # [N, d]
        
        # attn_scores: exp(a1^T hk) of shape [N, 1]
        attn_scores = torch.matmul(h_node, self.a1) # [N, 1]
        exp_attn = torch.exp(attn_scores - attn_scores.max()) # [N, 1]
        
        # Sum exp_attn for each hyperedge: denominator = H_T * exp_attn [M, 1]
        H_T = H.t() # [M, N]
        denom = torch.matmul(H_T, exp_attn) + 1e-9 # [M, 1]
        
        # alpha_jk = H_T[j, k] * exp_attn[k] / denom[j]
        # alpha is of shape [M, N]
        alpha = H_T * exp_attn.view(1, -1) / denom # [M, N]
        
        # el_j = \sigma( \sum_k \alpha_jk W1_v[k] )
        # e = \sigma( alpha * W1_v ) of shape [M, d]
        e = torch.matmul(alpha, W1_v).relu() # [M, d]
        
        # --- 2. Hyperedge-level Attention for Node Representation ---
        # Equation (5): vl_i = \sigma( \sum_{ej \in Ei} \beta_{ij} W2 el_j )
        # rj = LeakyReLU([W2 el_j \oplus W1 vl-1_i])
        W2_e = self.W2(e) # [M, d]
        
        # Find connection indices where H[i, j] == 1
        i_idx, j_idx = H.nonzero(as_tuple=True)
        
        # Connected pair representations
        feat_i = W1_v[i_idx] # [num_edges, d]
        feat_j = W2_e[j_idx] # [num_edges, d]
        
        # Concatenate and apply activation
        concat_feat = torch.cat([feat_j, feat_i], dim=-1) # [num_edges, 2d]
        r_val = self.leaky_relu(concat_feat) # [num_edges, 2d]
        
        # Calculate attention scores
        scores = torch.matmul(r_val, self.a2).squeeze(-1) # [num_edges]
        
        # Compute softmax over j for each node i
        max_score = scatter(scores, i_idx, dim=0, dim_size=N, reduce='max')
        scores_exp = torch.exp(scores - max_score[i_idx])
        sum_exp = scatter(scores_exp, i_idx, dim=0, dim_size=N, reduce='sum') + 1e-9
        beta = scores_exp / sum_exp[i_idx] # [num_edges]
        
        # Compute new node representation
        weighted_e = beta.view(-1, 1) * W2_e[j_idx] # [num_edges, d]
        v_next = scatter(weighted_e, i_idx, dim=0, dim_size=N, reduce='sum').relu() # [N, d]
        
        return v_next

class HGFND(nn.Module):
    """
    HGFND: Hypergraph Neural Network for Fake News Detection.
    Connects news pieces through hyperedges and processes their relations transductively.
    """
    def __init__(self, config: dict, in_channels: int, out_channels: int):
        super().__init__()
        model_cfg = config['model']
        self.gnn_type = model_cfg.get('gnn_type', 'SAGE')
        self.hidden_channels = model_cfg.get('hidden_channels', 128)
        self.num_layers = model_cfg.get('hgfnd_layers', 2)
        
        # 1. Propagation Encoder to initialize node representations v0
        self.prop_encoder = PropagationEncoder(self.gnn_type, in_channels, self.hidden_channels)
        
        # 2. HyperGAT Layers
        self.layers = nn.ModuleList([
            HyperGATLayer(self.hidden_channels) for _ in range(self.num_layers)
        ])
        
        # 3. Classifier Head
        self.classifier = nn.Linear(self.hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch, H):
        """
        x: All node features across all propagation trees merged together
        edge_index: Edge index of all trees
        batch: Graph assignment of all nodes
        H: Global incidence matrix [N, M]
        """
        # Step 1: Encode propagation trees to get initial node representations v0
        v = self.prop_encoder(x, edge_index, batch) # [N, hidden_channels]
        
        # Step 2: Pass through HyperGAT layers
        for layer in self.layers:
            v = layer(v, H)
            
        # Step 3: Classify
        logits = self.classifier(v)
        return logits.log_softmax(dim=-1)
