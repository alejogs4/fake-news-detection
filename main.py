import torch
import os.path as osp
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Batch
from src.utils import load_config, get_device
from src.data import get_loaders
from src.models.upfd_model import UPFDModel
from src.trainer import Trainer, TransductiveTrainer

def build_hypergraph(train_dataset, val_dataset, test_dataset, config):
    """
    Builds the global hypergraph incidence matrix H representing news relations.
    1. User Hyperedges: Shared user features across graphs.
    2. Time Hyperedges: Round relative creation time to proximal bins.
    3. Entity Hyperedges: K-Means clustering of news content.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    
    all_graphs = list(train_dataset) + list(val_dataset) + list(test_dataset)
    N = len(all_graphs)
    print(f"Building hypergraph with N={N} nodes (graphs)...")
    
    # 1. User Hyperedges
    user_to_graphs = {}
    for g_idx, data in enumerate(all_graphs):
        # Exclude the root node (news content node) from user matching
        user_features = data.x[1:]
        for i in range(user_features.size(0)):
            feat = tuple(np.round(user_features[i].numpy(), 6))
            if feat not in user_to_graphs:
                user_to_graphs[feat] = set()
            user_to_graphs[feat].add(g_idx)
            
    # Filter users to build hyperedges (keep users who shared at least TWO graphs)
    user_hyperedges = [list(graphs) for graphs in user_to_graphs.values() if len(graphs) >= 2]
    print(f"Constructed {len(user_hyperedges)} user-based hyperedges (size >= 2).")
    
    # 2. Time Hyperedges
    # Load profile features for temporal information to ensure feature-agnostic robust creation
    path = osp.join(osp.dirname(osp.realpath(__file__)), config['data']['data_dir'])
    p_train = UPFD(path, config['data']['dataset'], 'profile', 'train')
    p_val = UPFD(path, config['data']['dataset'], 'profile', 'val')
    p_test = UPFD(path, config['data']['dataset'], 'profile', 'test')
    all_p_graphs = list(p_train) + list(p_val) + list(p_test)
    
    time_decimals = config['model'].get('time_decimals', 2)
    time_to_graphs = {}
    for g_idx, p_data in enumerate(all_p_graphs):
        if p_data.x.size(1) > 9:
            time_vals = p_data.x[1:, 9].numpy()
            for t in time_vals:
                t_rounded = np.round(t, time_decimals)
                if t_rounded not in time_to_graphs:
                    time_to_graphs[t_rounded] = set()
                time_to_graphs[t_rounded].add(g_idx)
                
    time_hyperedges = [list(graphs) for graphs in time_to_graphs.values() if len(graphs) >= 2]
    print(f"Constructed {len(time_hyperedges)} time-based hyperedges (size >= 2).")
    
    # 3. Entity Hyperedges
    root_features = []
    for data in all_graphs:
        root_features.append(data.x[0].numpy())
    root_features = np.vstack(root_features)
    
    num_clusters = config['model'].get('entity_clusters', 50)
    kmeans = KMeans(n_clusters=min(num_clusters, N), random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(root_features)
    
    entity_to_graphs = {}
    for g_idx, label in enumerate(cluster_labels):
        if label not in entity_to_graphs:
            entity_to_graphs[label] = set()
        entity_to_graphs[label].add(g_idx)
        
    entity_hyperedges = [list(graphs) for graphs in entity_to_graphs.values() if len(graphs) >= 2]
    print(f"Constructed {len(entity_hyperedges)} entity-based hyperedges (size >= 2).")
    
    # Combine hyperedges
    all_hyperedges = user_hyperedges + time_hyperedges + entity_hyperedges
    M = len(all_hyperedges)
    
    H = torch.zeros((N, M), dtype=torch.float)
    for h_idx, graphs in enumerate(all_hyperedges):
        for g_idx in graphs:
            H[g_idx, h_idx] = 1.0
            
    print(f"Global incidence matrix shape: {H.shape}")
    return H

def run_experiment(config):
    # Get device
    device = get_device(config['training']['device'])
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, test_loader, train_dataset = get_loaders(config)
    
    model_cfg = config['model']
    use_hgfnd = model_cfg.get('use_hgfnd', False)
    
    if use_hgfnd:
        print("Initializing HGFND transductive experiment...")
        path = osp.join(osp.dirname(osp.realpath(__file__)), config['data']['data_dir'])
        val_dataset = UPFD(path, config['data']['dataset'], config['data']['feature'], 'val', ToUndirected())
        test_dataset = UPFD(path, config['data']['dataset'], config['data']['feature'], 'test', ToUndirected())
        
        all_graphs = list(train_dataset) + list(val_dataset) + list(test_dataset)
        
        # Build global batch & incidence matrix H
        H = build_hypergraph(train_dataset, val_dataset, test_dataset, config)
        giant_batch = Batch.from_data_list(all_graphs)
        
        # Prepare transductive split indices and labels
        N_train = len(train_dataset)
        N_val = len(val_dataset)
        N_all = len(all_graphs)
        
        train_idx = torch.arange(0, N_train, dtype=torch.long)
        val_idx = torch.arange(N_train, N_train + N_val, dtype=torch.long)
        test_idx = torch.arange(N_train + N_val, N_all, dtype=torch.long)
        
        y = torch.cat([data.y for data in all_graphs], dim=0)
        
        # Initialize model
        model = UPFDModel(
            config=config,
            in_channels=train_dataset.num_features,
            out_channels=train_dataset.num_classes
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['training']['lr'], 
            weight_decay=config['training']['weight_decay']
        )
        
        # Transductive Trainer
        trainer = TransductiveTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            giant_batch=giant_batch,
            H=H,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            y=y
        )
        
        # Start training
        trainer.fit(config['training']['epochs'])
        test_acc = trainer.test(trainer.test_idx)
        return test_acc
    else:
        # Initialize standard GNN/Text model
        model = UPFDModel(
            config=config,
            in_channels=train_dataset.num_features,
            out_channels=train_dataset.num_classes
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['training']['lr'], 
            weight_decay=config['training']['weight_decay']
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Start training
        trainer.fit(config['training']['epochs'])
        test_acc = trainer.test(trainer.test_loader)
        return test_acc

def main():
    # Load configuration
    config = load_config('config.yaml')
    run_experiment(config)

if __name__ == "__main__":
    main()

