import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv,GCNConv,GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import DataLoader, Batch
from torch.nn import Sequential, Linear, ReLU, LogSoftmax
#préparation des données
from torch_geometric.datasets import UPFD
train_data1 = UPFD(root=".", name="gossipcop", feature="profile", split="train")
test_data1 = UPFD(root=".", name="gossipcop", feature="profile", split="test")
print("Train Samples: ", len(test_data1))
print("Test Samples: ", len(train_data1))
test_loader1 = DataLoader(train_data1, batch_size=128, shuffle=False)
#print(batch_idx)
train_loader1 = DataLoader(test_data1, batch_size=128, shuffle=False)
from torch_geometric.datasets import UPFD
train_data2= UPFD(root=".", name="gossipcop", feature="bert", split="train")
test_data2= UPFD(root=".", name="gossipcop", feature="bert", split="test")
print("Train Samples: ", len(test_data2))
print("Test Samples: ", len(train_data2))
test_loader2 = DataLoader(train_data2, batch_size=128, shuffle=False)
train_loader2= DataLoader(test_data2, batch_size=128, shuffle=False)
class GINLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINLayer, self).__init__()
        self.mlp = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )
        self.gin = GINConv(self.mlp)

    def forward(self, x, edge_index):
        return self.gin(x, edge_index)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_ch, out_channels):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        
        # Define the GIN layers
        self.conv1 = GINLayer(in_channels, hidden_ch)
        self.conv2 = GINLayer(hidden_ch, hidden_ch)
        self.conv3 = GINLayer(hidden_ch, hidden_channels)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        h = F.elu(self.conv3(h, edge_index))
        return h
class MultiHeadCoAttentionLayer(nn.Module):
    def __init__(self, hidden_channels1, hidden_channels2, num_heads,dropout_rate=0.01):
        super(MultiHeadCoAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2

        # Define separate attention weights for each head
        self.query1 = nn.Linear(hidden_channels1, hidden_channels1 * num_heads, bias=False)
        self.key1 = nn.Linear(hidden_channels1, hidden_channels1 * num_heads, bias=False)
        self.value1 = nn.Linear(hidden_channels1, hidden_channels1 * num_heads, bias=False)

        self.query2 = nn.Linear(hidden_channels2, hidden_channels2 * num_heads, bias=False)
        self.key2 = nn.Linear(hidden_channels2, hidden_channels2 * num_heads, bias=False)
        self.value2 = nn.Linear(hidden_channels2, hidden_channels2 * num_heads, bias=False)

        # Output linear layers
        self.out1 = nn.Linear(hidden_channels1 * num_heads, hidden_channels1)
        self.out2 = nn.Linear(hidden_channels2 * num_heads, hidden_channels2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, x2):
        """
        x1: Tensor of shape [batch_size, hidden_channels1]
        x2: Tensor of shape [batch_size, hidden_channels2]
        """
        # Multi-head queries, keys, and values
        Q1 = self.query1(x1).view(-1, self.num_heads, self.hidden_channels1)
        K2 = self.key2(x2).view(-1, self.num_heads, self.hidden_channels2)
        V2 = self.value2(x2).view(-1, self.num_heads, self.hidden_channels2)

        Q2 = self.query2(x2).view(-1, self.num_heads, self.hidden_channels2)
        K1 = self.key1(x1).view(-1, self.num_heads, self.hidden_channels1)
        V1 = self.value1(x1).view(-1, self.num_heads, self.hidden_channels1)

        # Scaled dot-product attention
        attn_scores1 = torch.matmul(Q1, K2.transpose(-2, -1)) / (self.hidden_channels2 ** 0.5)
        attn_weights1 = F.softmax(attn_scores1, dim=-1)
        # Apply dropout to attention weights
        attn_weights11 = self.dropout(attn_weights1)
        attended_x1 = torch.matmul(attn_weights11, V2)

        attn_scores2 = torch.matmul(Q2, K1.transpose(-2, -1)) / (self.hidden_channels1 ** 0.5)
        attn_weights2 = F.softmax(attn_scores2, dim=-1)
        # Apply dropout to attention weights
        attn_weights22 = self.dropout(attn_weights2)
        attended_x2 = torch.matmul(attn_weights2, V1)

        # Concatenate heads and project back to the original dimension
        attended_x1 = attended_x1.view(-1, self.num_heads * self.hidden_channels1)
        attended_x2 = attended_x2.view(-1, self.num_heads * self.hidden_channels2)

        out_x1 = self.out1(attended_x1) + x1  # Residual connection
        out_x2 = self.out2(attended_x2) + x2  # Residual connection

        return out_x1, out_x2
class CombinedSAGE(nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_channels1, hidden_channels2, out_channels):
        super(CombinedSAGE, self).__init__()
        self.model1 = GAT(in_channels1, hidden_channels, hidden_channels1, out_channels)
        self.model2 = GAT(in_channels2, hidden_channels, hidden_channels2, out_channels)
        
        
        # Multi-Head Co-Attention Layer
        self.co_attention = MultiHeadCoAttentionLayer(hidden_channels, hidden_channels, num_heads)

        #self.co_attention = CoAttentionLayer(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels + hidden_channels, out_channels)  # Fusion FC layer
        #self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x1, x2, edge_index1, edge_index2, batch_idx):
        # Pass inputs through individual models
        t1 = self.model1(x1, edge_index1)
        t2 = self.model2(x2, edge_index2)
        #print(t1.shape)
        #print(t2.shape)

        # Apply co-attention
        t1, t2 = self.co_attention(t1, t2)
        #print(t1.shape)
        #print(t2.shape)

        # Fuse features and pool
        fused_features = torch.cat((t1, t2), dim=-1)
        flatten = global_max_pool(fused_features, batch=batch_idx)
        out = self.fc(flatten)
        #out1 = self.logsoftmax(out)
        return torch.sigmoid(out)

in_channels1 = train_data1.num_features # Number of input features for dataset 1
hidden_channels1 = 26
hidden_channels2 = 200
hidden_channels = 100 # Number of hidden features for dataset 1
attention_dim=90
out_channels = 1 # Number of output features for dataset 1
in_channels2 = train_data2.num_features # Number of input features for dataset 2
num_heads=1
dropout2 = 0.6 # Dropout probability for dataset 2
negative_slope2 = 0.2 # Negative slope for LeakyReLU for dataset 2
#combined_model = CombinedGAT(model1, model2)
CombinedSAGE(in_channels1, in_channels2, hidden_channels1, hidden_channels2, out_channels)
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score, precision_score, recall_score
combined_model=CombinedSAGE(in_channels1, in_channels2, hidden_channels1, hidden_channels2, out_channels)
criterion = nn.CrossEntropyLoss()
loss_fnc = torch.nn.BCELoss()
#optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001, weight_decay=0.001)
#num_epochs = 20
def metrics(preds, gts):
    probs = torch.cat(preds).cpu().numpy()  # raw sigmoid outputs
    preds_bin = torch.round(torch.cat(preds)).cpu().numpy()
    gts = torch.cat(gts).cpu().numpy()

    acc = accuracy_score(gts, preds_bin)
    f1 = f1_score(gts, preds_bin, zero_division=0)
    prec = precision_score(gts, preds_bin, zero_division=0)
    rec = recall_score(gts, preds_bin, zero_division=0)

    if len(set(gts)) < 2:
        print("Cannot compute AUC: only one class present.")
        auc = None
    else:
        auc = roc_auc_score(gts, probs)  # Correct input: probs, not preds_bin

    return acc, f1, auc, prec, rec
#print(f'Test Loss: {running_loss2/len(train_loader1.dataset)}')
#for epoch in range(num_epochs):
def train(epoch):
    combined_model.train()
    running_loss = 0.0
    for data1, data2 in zip(train_loader1, train_loader2):
        #combined_batch = Batch.from_data_list([data1, data2])
        optimizer.zero_grad()
        x1, edge_index1, y1,batch1 = data1.x, data1.edge_index, data1.y,data1.batch
        x2, edge_index2, y2,batch2 = data2.x, data2.edge_index, data2.y,data2.batch
        out1 = combined_model(x1, x2, edge_index1, edge_index2,batch1)
        loss1 = loss_fnc(out1.squeeze(), y1.to(torch.float))
        loss1.backward()
        optimizer.step()
        running_loss += loss1.item()
    return running_loss/len(train_loader1.dataset)
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader1.dataset)}')
    
@torch.no_grad()
def test(epoch):
    combined_model.eval()
    running_loss2 = 0.0
    all_preds = []
    all_labels = []
    #with torch.no_grad():
    for data1, data2 in zip(test_loader1, test_loader2):
        x1, edge_index1, x2, edge_index2, batch11 = data1.x, data1.edge_index, data2.x,data1.edge_index, data1.batch
        
        xt1, edge_indext1, yt1,batcht1 = data1.x, data1.edge_index, data1.y,data1.batch
        xt2, edge_indext2, yt2,batcht2 = data2.x, data2.edge_index, data2.y,data2.batch
        outt1= combined_model(xt1, xt2, edge_indext1, edge_indext2, batcht1)
        losst1 = loss_fnc(outt1.squeeze(), yt1.to(torch.float))
        running_loss2 += losst1.item()
        all_preds.append(outt1.squeeze())
        #print(all_preds.shape)
        all_labels.append(yt1.to(torch.float))
        
    accuracy, f1,auc,prec,rec = metrics(all_preds, all_labels)
    return running_loss2 / len(test_loader1.dataset), accuracy, f1,auc,prec,rec
    
for epoch in range(58):
    train_loss = train(epoch)
    test_loss, test_acc, test_f1,test_auc,test_prec,test_rec = test(epoch)
    print(f'Epoch: {epoch:02d} |  TestAcc: {test_acc:.4f} |'
          f'TestF1: {test_f1:.4f} | Test_auc: {test_auc:.4f} |Test_prec: {test_prec:.4f}|Test_rec: {test_rec:.4f}')







