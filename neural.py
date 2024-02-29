import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

class DDQN():
    def __init__(self, in_features, out_features, device):
        self.online = DQN(in_features, out_features).float().to(device)
        self.target = DQN(in_features, out_features).float().to(device)

        self.sync_target()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape  # (4, 84, 84)
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc_input_dim = self.feature_size()

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)

            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values

    def feature_size(self):
        dummy_input = torch.zeros(1, *self.input_shape)
        dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))
        return int(np.prod(dummy_output.size()[1:])) 

