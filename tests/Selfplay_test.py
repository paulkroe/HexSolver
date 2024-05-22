import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

import Game.Hex as hex
import Agent.Networks.CNN as cnn
import Agent.MCTS as mcts
import Agent.Networks.CNN as cnn
import Agent.agent as agent
import RandomModel as randmodel
import Training.SelfPlay as self_play
import Training.Match as match

model0 = randmodel.RandModel()
agent0 = agent.Agent(model0)

model1 = cnn.AlphaZeroNetwork(hex.SIZE, hex.SIZE)
agent1 = agent.Agent(model1)

winner = match.match(agent0, agent1)
