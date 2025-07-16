"""
Minimal PyTorch Lightning experiment script with wandb logging.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from data.simple_dataset import SimpleDataset
from utils.wandb_utils import init_wandb

class SimpleLightningModel(pl.LightningModule):
    def __init__(self, input_dim, lr):
        super().__init__()
        self.layer = torch.nn.Linear(input_dim, 1)
        self.lr = lr

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    init_wandb(config)
    dataset = SimpleDataset(size=config["data"]["size"], input_dim=config["model"]["input_dim"])
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"])
    model = SimpleLightningModel(input_dim=config["model"]["input_dim"], lr=config["training"]["lr"])
    trainer = pl.Trainer(max_epochs=config["training"]["epochs"], logger=pl.loggers.WandbLogger())
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
