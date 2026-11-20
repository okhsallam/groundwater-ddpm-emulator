import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from scipy import fft
import yaml
import torch
from numpy import fft

from pathlib import Path
from types import SimpleNamespace


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert nested dictionaries to SimpleNamespace objects
        self.config = self._dict_to_namespace(config_dict)
        
    
    def _dict_to_namespace(self, d):
        """Recursively convert dictionary to SimpleNamespace"""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self._dict_to_namespace(value)
            elif isinstance(value, list):
                # Handle lists of dictionaries (e.g., train_cases)
                d[key] = [self._dict_to_namespace(item) if isinstance(item, dict) else item 
                         for item in value]
        return SimpleNamespace(**d)
    
    
    def __getattr__(self, name):
        return getattr(self.config, name)
    
def load_config(config_path):
    """Load configuration from YAML file"""
    return Config(config_path)




def normalize_channel(tensor):
    """Normalize tensor to [0, 1] and return normalization parameters."""
    normalized = tensor.clone()
    min_vals = torch.zeros(3, device=tensor.device)  # Store min values per channel
    max_vals = torch.zeros(3, device=tensor.device)  # Store max values per channel

    for c in range(3):
        min_vals[c] = tensor[c].min()
        max_vals[c] = tensor[c].max()
        normalized[c] = (tensor[c] - min_vals[c]) / (max_vals[c] - min_vals[c])  # Normalize

    return normalized, min_vals, max_vals


def denormalize_channel(normalized, min_vals, max_vals):
    """Reverses normalization using stored min/max values before saving."""
    device = normalized.device  # Ensure all tensors are on the same device
    min_vals = min_vals.to(device)
    max_vals = max_vals.to(device)

    denormalized = normalized.clone()
    for c in range(3):
        denormalized[c] = normalized[c] * (max_vals[c] - min_vals[c]) + min_vals[c]
    
    return denormalized