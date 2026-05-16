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
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            train_acc = self.test(self.train_loader)
            val_acc = self.test(self.val_loader)
            test_acc = self.test(self.test_loader)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

