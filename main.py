import torch
from src.utils import load_config, get_device
from src.data import get_loaders
from src.models.upfd_model import UPFDModel
from src.trainer import Trainer

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Get device
    device = get_device(config['training']['device'])
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, test_loader, dataset = get_loaders(config)
    
    # Initialize model
    model = UPFDModel(
        config=config,
        in_channels=dataset.num_features,
        out_channels=dataset.num_classes
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

if __name__ == "__main__":
    main()
