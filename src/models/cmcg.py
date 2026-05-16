import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCoAttentionLayer(nn.Module):
    def __init__(self, hidden_channels1, hidden_channels2, num_heads=2, dropout_rate=0.01):
        super(MultiHeadCoAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2

        self.query1 = nn.Linear(hidden_channels1, hidden_channels1 * num_heads, bias=False)
        self.key1 = nn.Linear(hidden_channels1, hidden_channels1 * num_heads, bias=False)
        self.value1 = nn.Linear(hidden_channels1, hidden_channels1 * num_heads, bias=False)

        self.query2 = nn.Linear(hidden_channels2, hidden_channels2 * num_heads, bias=False)
        self.key2 = nn.Linear(hidden_channels2, hidden_channels2 * num_heads, bias=False)
        self.value2 = nn.Linear(hidden_channels2, hidden_channels2 * num_heads, bias=False)

        self.out1 = nn.Linear(hidden_channels1 * num_heads, hidden_channels1)
        self.out2 = nn.Linear(hidden_channels2 * num_heads, hidden_channels2)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, x2):
        Q1 = self.query1(x1).view(-1, self.num_heads, self.hidden_channels1)
        K2 = self.key2(x2).view(-1, self.num_heads, self.hidden_channels2)
        V2 = self.value2(x2).view(-1, self.num_heads, self.hidden_channels2)

        Q2 = self.query2(x2).view(-1, self.num_heads, self.hidden_channels2)
        K1 = self.key1(x1).view(-1, self.num_heads, self.hidden_channels1)
        V1 = self.value1(x1).view(-1, self.num_heads, self.hidden_channels1)

        attn_scores1 = torch.matmul(Q1, K2.transpose(-2, -1)) / (self.hidden_channels2 ** 0.5)
        attn_weights1 = self.dropout(F.softmax(attn_scores1, dim=-1))
        attended_x1 = torch.matmul(attn_weights1, V2)

        attn_scores2 = torch.matmul(Q2, K1.transpose(-2, -1)) / (self.hidden_channels1 ** 0.5)
        attn_weights2 = self.dropout(F.softmax(attn_scores2, dim=-1))
        attended_x2 = torch.matmul(attn_weights2, V1)

        attended_x1 = attended_x1.view(-1, self.num_heads * self.hidden_channels1)
        attended_x2 = attended_x2.view(-1, self.num_heads * self.hidden_channels2)

        out_x1 = self.out1(attended_x1) + x1
        out_x2 = self.out2(attended_x2) + x2

        return out_x1, out_x2
