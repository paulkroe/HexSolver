import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_height, board_width, action_size):
        super(AlphaZeroNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc_policy = nn.Linear(128 * board_height * board_width, action_size)
        self.fc_value = nn.Linear(128 * board_height * board_width, 1)

    def forward(self, x, mask):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        policy = self.fc_policy(x)
        policy = policy + mask
        value = torch.tanh(self.fc_value(x))

        return value, policy