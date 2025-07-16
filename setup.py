from setuptools import setup, find_packages

setup(
    name="ground_floor_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "flax",
        "torch",
        "pytorch-lightning",
        "wandb",
        "pyyaml"
    ],
    description="Modular ML research framework with JAX/Flax and PyTorch Lightning.",
    author="Your Name",
)
