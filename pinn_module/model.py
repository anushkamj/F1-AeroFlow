import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, output_dim=2, num_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))  # outputs: Cd, Cl
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
