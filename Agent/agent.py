import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Agent.MCTS as mcts

import torch
from torch.utils.data import TensorDataset, DataLoader

import Game.Hex as hex

class Agent():
    def __init__(self, model):
        self.model = model
    
    def get_move(self, game, simulations=100):
        tree = mcts.MCTS(game, self.model)
        move = tree.run(simulations)
        probabilities = tree.probabilities
        return move, probabilities
    
    def train(self, data, epochs, batch_size):
        pass
    