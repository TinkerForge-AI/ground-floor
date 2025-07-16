"""
STEP 3: PyTorch Lightning experiment for MNIST classification.
"""

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from data.STEP_1_Download_Preprocess_Data_PyTorch import get_mnist_dataloaders
from utils.STEP_5_Logging_Tracking_WandB import init_wandb

class MNISTLightningModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__() # Initialize the parent class
        # Define a simple MLP with two layers
        # 28 * 28 is the size of flattened MNIST images
        # 128 is the number of hidden units
        # hidden units are the neurons in the hidden layer of the MLP
        # hidden layers are the layers between the input and output layers in a neural network
        self.layer1 = torch.nn.Linear(28*28, 128) 
        # Define the output layer
        # 128 is the number of hidden units, this should match the output of the previous layer
        # 10 represents the number of classes in MNIST (digits 0-9)
        self.layer2 = torch.nn.Linear(128, 10)
        # Learning rate for the optimizer
        self.lr = lr

    # Forward pass of the model
    def forward(self, x):
        # Flatten the input images
        # x.size(0) is the batch size, -1 means flatten the rest of the dimensions
        # This reshapes the input tensor to have shape (batch_size, 28*28)
        # Flattening is necessary because the input to the MLP should be a 2D tensor
        # where each row is a flattened image
        # x.view() reshapes the tensor without changing its data
        # This is similar to numpy's reshape function
        x = x.view(x.size(0), -1)
        # Apply the first layer and ReLU activation
        # ReLU (Rectified Linear Unit) is an activation function that introduces non-linearity
        # activation functions are used to transform the output of a layer 
        # for the purpose of learning complex patterns
        # refer to this website for info on activation functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/
        x = torch.relu(self.layer1(x))
        # Apply the second layer to get logits
        # Logits are the raw outputs of the model before applying softmax or sigmoid in the form of a tensor
        return self.layer2(x)

    # a single training step for the PyTorch Lightning model
    def training_step(self, batch, batch_idx):
        x, y = batch # unpack the batch into inputs (x) and targets (y)
        logits = self(x) # forward pass through the model
        # a forward pass is the process of passing inputs through the model to get outputs
        loss = F.cross_entropy(logits, y) # measures how well the model's predictions match the targets
        self.log('train_loss', loss) # log the training loss to track performance
        return loss # returns the loss for optimization by the optimizer
        ### an optimizer is an algorithm that updates the model's parameters based on the loss

    def configure_optimizers(self):
        # Configure the optimizer for the model
        # Adam is a popular optimization algorithm that adapts the learning rate for each parameter
        # It combines the benefits of two other extensions of stochastic gradient descent
        # read more about optimization functions at https://pytorch.org/docs/stable/optim.html
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def run_experiment(config):
    # Initialize Weights & Biases for logging
    init_wandb(config)
    # Get the MNIST data loaders, the data loaders are used to load the data in batches
    # test_loader and train_loader are PyTorch DataLoader objects
    train_loader, test_loader = get_mnist_dataloaders(config['batch_size'])
    # Create an instance of the MNISTLightningModel with the specified learning rate
    model = MNISTLightningModel(lr=config['lr'])
    # Create a PyTorch Lightning Trainer
    # The Trainer is responsible for training the model, logging, and managing the training loop
    trainer = pl.Trainer(max_epochs=config['epochs'], logger=pl.loggers.WandbLogger())
    # Fit the model using the training data loader
    # The fit method trains the model for the specified number of epochs using the provided data loader
    # It handles the training loop, validation, and logging automatically
    # The model is trained on the training data and validated on the validation data
    # by the end of the training, the model should have learned to classify MNIST digits
    # the model is saved after training and can be used to make predictions on new data
    trainer.fit(model, train_loader)

import yaml

if __name__ == "__main__":
    with open("configs/STEP_4_Config_Management_YAML.yaml") as f:
        config = yaml.safe_load(f)
    run_experiment(config)
