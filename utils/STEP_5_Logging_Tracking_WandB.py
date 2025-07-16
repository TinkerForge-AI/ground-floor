"""
STEP 5: WandB logging utility for experiments.
"""

import wandb

def init_wandb(config):
    wandb.init(
        project=config.get("wandb_project", "mnist-tutorial"),
        config=config,
        reinit=True
    )
