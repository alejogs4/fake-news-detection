import os.path as osp
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

def get_loaders(config):
    """
    Creates train, val, and test data loaders based on the provided configuration.
    """
    data_cfg = config['data']
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', data_cfg['data_dir'])
    
    # Common transform to ensure graphs are undirected
    transform = ToUndirected()

    train_dataset = UPFD(path, data_cfg['dataset'], data_cfg['feature'], 'train', transform)
    val_dataset = UPFD(path, data_cfg['dataset'], data_cfg['feature'], 'val', transform)
    test_dataset = UPFD(path, data_cfg['dataset'], data_cfg['feature'], 'test', transform)

    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset
