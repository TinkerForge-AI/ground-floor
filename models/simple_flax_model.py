"""
Minimal JAX/Flax model example.
"""

import flax.linen as nn
import jax.numpy as jnp

class SimpleMLP(nn.Module):
    features: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Example usage:
# model = SimpleMLP(features=32)
# params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 10)))
