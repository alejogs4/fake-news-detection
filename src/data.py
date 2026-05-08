import os
import os.path as osp
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

class LocalUPFD(UPFD):
    """
    A PyTorch Geometric Dataset class that forces local loading of the UPFD dataset.
    This prevents the dataset from attempting to download files if they are missing
    and instead relies on the unzipped files in the dataset/{name}/raw directory.
    """
    def download(self):
        # Prevent any download attempts and throw an explicit error if files are missing.
        raise RuntimeError(
            f"Dataset files not found in {self.raw_dir}. "
            "Please ensure the dataset zip file is unzipped into the raw directory."
        )

def get_loaders(config):
    """
    Creates train, val, and test data loaders based on the provided configuration.
    """
    data_cfg = config['data']
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', data_cfg['data_dir'])
    
    # Common transform to ensure graphs are undirected
    transform = ToUndirected()

    train_dataset = LocalUPFD(path, data_cfg['dataset'], data_cfg['feature'], 'train', transform)
    val_dataset = LocalUPFD(path, data_cfg['dataset'], data_cfg['feature'], 'val', transform)
    test_dataset = LocalUPFD(path, data_cfg['dataset'], data_cfg['feature'], 'test', transform)

    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset
