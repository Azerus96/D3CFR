import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingNetwork(nn.Module):
    def __init__(self, input_size, max_actions, hidden_size=512):
        super(DuelingNetwork, self).__init__()
        self.max_actions = max_actions

        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.max_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        output = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return output
