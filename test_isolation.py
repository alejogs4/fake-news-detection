import torch
import numpy as np
import os.path as osp
from torch_geometric.datasets import UPFD
from main import build_hypergraph

def test_hypergraph():
    config = {
        'data': {
            'dataset': 'politifact',
            'data_dir': 'dataset'
        },
        'model': {
            'time_decimals': 2,
            'entity_clusters': 50
        }
    }
    path = osp.join(osp.dirname(osp.realpath(__file__)), config['data']['data_dir'])
    train_dataset = UPFD(path, 'politifact', 'bert', 'train')
    val_dataset = UPFD(path, 'politifact', 'bert', 'val')
    test_dataset = UPFD(path, 'politifact', 'bert', 'test')
    
    H = build_hypergraph(train_dataset, val_dataset, test_dataset, config)
    node_degrees = H.sum(dim=1)
    isolated_nodes = (node_degrees == 0).sum().item()
    print(f"Total nodes: {H.shape[0]}")
    print(f"Isolated nodes: {isolated_nodes}")
    print(f"Non-isolated nodes: {H.shape[0] - isolated_nodes}")

if __name__ == '__main__':
    test_hypergraph()
