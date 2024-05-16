import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import Game.Hex as hex

class RandModel():
    def __init__(self):
        self.size=hex.SIZE
    def __call__(self, x, mask):
        return random.random(), torch.softmax(torch.rand(self.size, self.size), dim=1)