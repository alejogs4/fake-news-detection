import yaml
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)
