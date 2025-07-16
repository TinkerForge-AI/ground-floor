"""
STEP 2: Define a simple MLP for MNIST using JAX/Flax.

MLP - Multi-Layer Perceptron
This model will take flattened MNIST images as input and output logits for each class.

MNIST - Modified National Institute of Standards and Technology (i.e. the dataset creator)
"""

import flax.linen as nn
import jax.numpy as jnp

class MNISTMLP(nn.Module):
    # Define the number of features in the hidden layer
    @nn.compact
    # Define the forward pass
    def __call__(self, x):
        # Reshape the input to flatten the images#
        x = x.reshape((x.shape[0], -1))
        # Apply a dense layer with ReLU activation
        x = nn.Dense(128)(x)
        # Apply another dense layer to output logits
        x = nn.relu(x)
        # Apply the final dense layer to get class scores
        x = nn.Dense(10)(x)
        # Return the logits. A logit is the raw output of the model before applying softmax or sigmoid.
        # Logits are typically used for classification tasks—they represent scores for each class, 
        # which are then converted to probabilities.
        #   - For multi-class classification, logits are passed through a softmax to get class probabilities.
        #   - For binary classification, a single logit is passed through a sigmoid.
        return x
