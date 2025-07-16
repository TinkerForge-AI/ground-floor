"""
Weights & Biases integration helpers.
"""

import wandb

def init_wandb(config):
    wandb.init(
        project=config.get("wandb_project", "ground-floor-ml"),
        config=config,
        reinit=True
    )
