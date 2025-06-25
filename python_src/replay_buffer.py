# mccfr_ofc-main/python_src/replay_buffer.py

import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, infoset_vector, target_regrets, num_actions):
        # Убедимся, что вектор сожалений имеет правильную длину
        padded_regrets = np.zeros(200, dtype=np.float32) # 200 - ACTION_LIMIT
        padded_regrets[:num_actions] = target_regrets
        self.buffer.append((infoset_vector, padded_regrets, num_actions))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        infosets, targets, num_actions_list = zip(*samples)
        
        # Преобразуем в тензоры PyTorch
        infosets_tensor = torch.FloatTensor(np.array(infosets))
        targets_tensor = torch.FloatTensor(np.array(targets))
        
        return infosets_tensor, targets_tensor, list(num_actions_list)

    def __len__(self):
        return len(self.buffer)
