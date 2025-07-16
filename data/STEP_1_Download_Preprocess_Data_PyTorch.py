"""
STEP 1: Download and preprocess MNIST data using PyTorch.
"""

import torch
from torchvision import datasets, transforms

def get_mnist_dataloaders(batch_size=64):
    """
      Downloads the MNIST dataset and returns data loaders for training and testing.
    """

    # Define the transformation for the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    # Download the MNIST dataset
    train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Download the test dataset
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # Create data loader for testing
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    # Return the data loaders
    return train_loader, test_loader

# get_mnist_dataloaders()