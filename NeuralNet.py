import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_prob=0.5):
        super(FeedForwardNN, self).__init__()
        layers = []
        in_size = input_size

        # Dynamically create hidden layers with dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.Dropout(dropout_prob))  # Add dropout after each hidden layer
            in_size = hidden_size

        # Add the output layer (cardinality of 3)
        layers.append(nn.Linear(in_size, 3))

        # Register layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply activation and dropout for all but the last layer
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
        x = self.layers[-1](x)  # No activation or dropout for the output layer
        return x
