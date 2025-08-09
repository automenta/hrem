import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron baseline."""
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The input for MLP is expected to be flattened
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
