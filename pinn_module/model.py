import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, output_dim=2, num_layers=4):
        super().__init__()

        layers = []

        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers: (num_layers - 2) fully-connected hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Last layer: hidden → output (Cd, Cl)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
