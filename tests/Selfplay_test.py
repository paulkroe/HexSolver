import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from tqdm import tqdm

import Game.Hex as hex
import Agent.Networks.CNN as cnn
import Agent.MCTS as mcts
import Agent.Networks.CNN as cnn
import Agent.agent as agent
import RandomModel as randmodel
import Training.SelfPlay as self_play
import Training.Match as match

NUM_GAMES = 10



def eval(agent0, agent1):

    # agent0 is player 1
    wins = [0, 0]
    for _ in tqdm(range(NUM_GAMES), desc="agent 0 playing agent 1: "):
        winner = match.match(agent0, agent1, verbose=False)
        wins[winner] += 1
    print(f"agent 0 vs. agent 1: {wins[0]}:{wins[1]}")

    # agent1 is player 1
    wins = [0, 0]
    for _ in tqdm(range(NUM_GAMES), desc="agent 1 playing agent 0: "):
        winner = match.match(agent1, agent0, verbose=False)
        wins[winner] += 1
    print(f"agent 1 vs. agent 0: {wins[0]}:{wins[1]}")
    
    return
    
model0 = randmodel.RandModel()
agent0 = agent.Agent(model0)

model1 = cnn.AlphaZeroNetwork(hex.SIZE, hex.SIZE)
agent1 = agent.Agent(model1)

eval(agent0, agent1)

SelfGame = self_play.SelfPlay(agent1)
data = SelfGame.generate_data(25, 1, shuffle=True)
print(f"len(data): {len(data)}")
agent1.train(data, 200, 0.001, 4)

print("After training agent 1: ")
eval(agent0, agent1)