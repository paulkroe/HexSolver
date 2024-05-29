import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

'''
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
'''
    
class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_height, board_width, val_lin_layers=[], policy_lin_layers=[], alpha=0.03, eps=0.25):
        super(AlphaZeroNetwork, self).__init__()
        
        self.alpha = alpha
        self.eps = eps
        
        self.board_height = board_height
        self.board_width = board_width
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv1_ = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_ = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn1_ = nn.BatchNorm2d(64)
        self.bn2_ = nn.BatchNorm2d(128)
        self.bn3_ = nn.BatchNorm2d(128)

        
        val_lin_layers = [128 * self.board_height * self.board_width] + val_lin_layers + [1]
        layers = []
        for i in range(1, len(val_lin_layers)):
            layers.append(nn.Linear(val_lin_layers[i-1], val_lin_layers[i]))
            if i < len(val_lin_layers) - 1:
                layers.append(nn.ReLU())
        self.value_head = nn.Sequential(*layers)
        
        policy_lin_layers = [128 * self.board_height * self.board_width] + policy_lin_layers + [self.board_height * self.board_width]
        layers = []
        for i in range(1, len(policy_lin_layers)):
            layers.append(nn.Linear(policy_lin_layers[i-1], policy_lin_layers[i]))
            if i < len(policy_lin_layers) - 1:
                layers.append(nn.ReLU())
        self.policy_head = nn.Sequential(*layers)   

    def forward(self, input, mask, add_noise=True):
        
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        policy = self.policy_head(x).view(-1, self.board_height, self.board_width)
        
        if not mask is None: 
            policy = policy + mask # masks are not used during training to avoid -inf values in training TODO: fix this
        
        # add Dirichlett noise to make self-play more interesting
        if add_noise:
            dirichlet_distribution = Dirichlet(torch.full((self.board_height * self.board_width,), self.alpha).repeat(policy.size(0), 1))
            noise = dirichlet_distribution.sample()             
            policy = (1 - self.eps) * policy + self.eps * noise.reshape(policy.shape)

        policy = F.softmax(policy.view(policy.size(0), -1), dim=1).view(-1, self.board_height, self.board_width)
        
        x_ = F.relu(self.bn1_(self.conv1_(input)))
        x_ = F.relu(self.bn2_(self.conv2_(x_)))
        x_ = F.relu(self.bn3_(self.conv3_(x_)))
        x_ = x_.view(x_.size(0), -1)
        
        value = torch.tanh(self.value_head(x_))
        return value, policy