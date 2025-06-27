import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, action_limit):
        self.buffer = deque(maxlen=capacity)
        self.action_limit = action_limit

    def push(self, infoset_vector, target_regrets, num_actions):
        padded_regrets = np.zeros(self.action_limit, dtype=np.float32)
        if num_actions > 0:
            padded_regrets[:num_actions] = target_regrets
        self.buffer.append((infoset_vector, padded_regrets, num_actions))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None
        samples = random.sample(self.buffer, batch_size)
        infosets, targets, num_actions_list = zip(*samples)
        
        infosets_tensor = torch.FloatTensor(np.array(infosets))
        targets_tensor = torch.FloatTensor(np.array(targets))
        
        return infosets_tensor, targets_tensor, list(num_actions_list)

    def __len__(self):
        return len(self.buffer)
