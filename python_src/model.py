# mccfr_ofc-main/python_src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingNetwork(nn.Module):
    """
    Архитектура Dueling Network, адаптированная для предсказания сожалений.
    """
    def __init__(self, input_size, max_actions, hidden_size=512):
        super(DuelingNetwork, self).__init__()
        self.max_actions = max_actions

        # Общий "ствол" сети
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Голова для оценки ценности состояния (Value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Голова для оценки преимущества каждого действия (Advantage)
        # Выходной слой должен быть достаточно большим, чтобы покрыть максимальное число действий
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.max_actions)
        )

    def forward(self, x, num_actions):
        """
        Прямой проход. num_actions - реальное количество действий в текущем состоянии.
        """
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Мы используем только срез, соответствующий реальному количеству действий
        advantages_sliced = advantages[:, :num_actions]

        # Комбинируем Value и Advantage для получения Q-values (в нашем случае, сожалений)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        output = value + (advantages_sliced - advantages_sliced.mean(dim=1, keepdim=True))
        
        return output
