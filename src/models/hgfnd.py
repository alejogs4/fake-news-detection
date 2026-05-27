import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, global_max_pool

class HyperGraphAttentionLayerSparse(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        pair = adj.nonzero().t()

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)

        pair_h = torch.cat((q1, x1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=2)

        edge = torch.matmul(attention_edge, x)

        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)

        get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
        y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        pair_h = torch.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1, 2), dim=2)

        node = torch.matmul(attention_node, edge)

        if self.concat:
            node = F.elu(node)

        return node, edge  # edge_4att

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False, concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True, concat=False)

    def forward(self, x, H):
        x0 = x
        x, e = self.gat1(x, H)
        x = x + x0
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = x
        x, e = self.gat2(x, H)
        x = x + x1
        return x, e

class NewsHypergraph(nn.Module):
    def __init__(self, hidden_size, n_categories, dropout=0.3):
        super(NewsHypergraph, self).__init__()
        self.hidden_size = hidden_size
        self.n_categories = n_categories
        self.dropout = dropout
        self.hgnn = HGNN_ATT(self.hidden_size, self.hidden_size, self.hidden_size, dropout=self.dropout)

    def forward(self, nodes, HT):
        hypergraph, edge_att = self.hgnn(nodes, HT)
        return hypergraph, edge_att

class HGFND(nn.Module):
    def __init__(self, config: dict, in_channels: int, out_channels: int):
        super().__init__()
        model_cfg = config['model']
        self.gnn_type = model_cfg.get('gnn_type', 'SAGE')
        self.hidden_channels = model_cfg.get('hidden_channels', 128)
        self.dropout = model_cfg.get('dropout', 0.3)
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        if self.gnn_type == 'SAGE':
            self.conv1 = SAGEConv(self.in_channels, self.hidden_channels)
        elif self.gnn_type == 'GCN':
            self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
        elif self.gnn_type == 'GAT':
            self.conv1 = GATConv(self.in_channels, self.hidden_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
            
        self.lin0 = nn.Linear(self.in_channels, self.hidden_channels)
        self.lin1 = nn.Linear(2 * self.hidden_channels, self.hidden_channels)
        self.cls = nn.Linear(self.hidden_channels, self.out_channels, bias=True)
        
        self.hypergraph_model = NewsHypergraph(self.hidden_channels, self.out_channels, self.dropout)

    def forward(self, x, edge_index, batch, H):
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        news = x[root]
        news = self.lin0(news).relu()

        p = self.conv1(x, edge_index).relu()
        p = global_max_pool(p, batch)
        p = self.lin1(torch.cat([news, p], dim=-1)).relu()

        v = p.unsqueeze(0)
        
        # H is of shape (N, M). 
        # The layer expects adj of shape (batch, M, N), so we do H.t().unsqueeze(0)
        HT = H.t().unsqueeze(0)
        
        v, e = self.hypergraph_model(v, HT)
        result = v.squeeze(0)
        
        return self.cls(result).log_softmax(dim=-1)
