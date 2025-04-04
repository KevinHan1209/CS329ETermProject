import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(FeedForwardNN, self).__init__()
        layers = []
        in_size = input_size

        # Dynamically create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size

        # Add the output layer (cardinality of 3)
        layers.append(nn.Linear(in_size, 3))

        # Register layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply activation for all but the last layer
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # No activation for the output layer
        return x

