import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_height, board_width, alpha=0.03, eps=0.25):
        super(AlphaZeroNetwork, self).__init__()
        
        self.alpha = alpha
        self.eps = eps
        
        self.board_height = board_height
        self.board_width = board_width
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc_policy = nn.Linear(128 * self.board_height * self.board_width, self.board_height * self.board_width)
        self.fc_value = nn.Linear(128 * self.board_height * self.board_width, 1)

    '''
    def forward(self, x, mask, add_noise=True):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        policy = self.fc_policy(x).view((-1, self.board_height, self.board_width))
        policy = policy + mask
        print(f"policy + mask: {policy.shape}")
        
        # add Dirichlett noise to make self play more interesting
        if add_noise:
            flat_policy = policy.view(-1, self.board_height*self.board_width)
            print(flat_policy.shape)
            dirichlet_distribution = distributions.Dirichlet(torch.full((flat_policy.shape[0],), self.alpha))
            noise = dirichlet_distribution.sample()
            print(f"noise.shape: {noise.shape}")
            flat_policy_with_noise = (1 - self.eps) * flat_policy + self.eps * noise
            policy = flat_policy_with_noise.view((-1, self.board_height, self.board_width))

        policy = F.softmax(policy.view(-1), dim=0).view((self.board_height, self.board_width))
        value = torch.tanh(self.fc_value(x))

        return value, policy
    '''
    
    def forward(self, x, mask, add_noise=True):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        policy = self.fc_policy(x).view(-1, self.board_height, self.board_width)
        if not mask is None: 
            policy = policy + mask # masks are not used during training to avoid -inf values in training TODO: fix this
        
        # add Dirichlett noise to make self-play more interesting
        if add_noise:
            dirichlet_distribution = Dirichlet(torch.full((self.board_height * self.board_width,), self.alpha).repeat(policy.size(0), 1))
            noise = dirichlet_distribution.sample()
            policy = (1 - self.eps) * policy + self.eps * noise.reshape(policy.shape)

        policy = F.softmax(policy.view(policy.size(0), -1), dim=1).view(-1, self.board_height, self.board_width)
        value = torch.tanh(self.fc_value(x))
        return value, policy