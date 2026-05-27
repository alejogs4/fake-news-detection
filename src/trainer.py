import torch
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device, train_loader, val_loader, test_loader):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _get_model_kwargs(self, data):
        kwargs = {
            'x': data.x,
            'edge_index': data.edge_index,
            'batch': data.batch
        }
        if hasattr(data, 'sentiment'):
            kwargs['sentiment_features'] = data.sentiment
        return kwargs

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(**self._get_model_kwargs(data))
            loss = F.nll_loss(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = total_examples = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(**self._get_model_kwargs(data))
            pred = out.argmax(dim=-1)
            total_correct += int((pred == data.y).sum())
            total_examples += data.num_graphs
        return total_correct / total_examples

    def fit(self, epochs):
        best_val_acc = 0
        best_test_acc = 0
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            train_acc = self.test(self.train_loader)
            val_acc = self.test(self.val_loader)
            test_acc = self.test(self.test_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        return best_test_acc

class TransductiveTrainer:
    def __init__(self, model, optimizer, device, giant_batch, H, train_idx, val_idx, test_idx, y):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.giant_batch = giant_batch.to(device)
        self.H = H.to(device)
        self.train_idx = train_idx.to(device)
        self.val_idx = val_idx.to(device)
        self.test_idx = test_idx.to(device)
        self.y = y.to(device)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        # Forward pass on the entire hypergraph
        out = self.model(
            x=self.giant_batch.x,
            edge_index=self.giant_batch.edge_index,
            batch=self.giant_batch.batch,
            H=self.H
        )
        # Compute loss on training nodes only
        loss = F.nll_loss(out[self.train_idx], self.y[self.train_idx])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, indices):
        self.model.eval()
        out = self.model(
            x=self.giant_batch.x,
            edge_index=self.giant_batch.edge_index,
            batch=self.giant_batch.batch,
            H=self.H
        )
        pred = out[indices].argmax(dim=-1)
        correct = int((pred == self.y[indices]).sum())
        return correct / len(indices)

    def fit(self, epochs):
        best_val_acc = 0
        best_test_acc = 0
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            train_acc = self.test(self.train_idx)
            val_acc = self.test(self.val_idx)
            test_acc = self.test(self.test_idx)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        return best_test_acc


