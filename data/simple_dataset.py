"""
Simple synthetic dataset for demonstration.
"""

import torch

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=100, input_dim=10):
        self.x = torch.randn(size, input_dim)
        self.y = torch.sum(self.x, dim=1, keepdim=True) + torch.randn(size, 1) * 0.1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
