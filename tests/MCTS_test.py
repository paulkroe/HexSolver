import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

import Game.Hex as hex
import Data.preprocess_data as preprocess_data
import Agent.Networks.CNN as cnn
import Agent.MCTS as mcts
import Agent.Networks.CNN as cnn
import RandomModel as randmodel

# random model
game = hex.HexGame()

for i in range(0, hex.SIZE-1):
    for j in range(1, 3):
        game.place_piece(i, j)   
        game.take_turn()    

model = randmodel.RandModel()
tree = mcts.MCTS(game, model)

(x, y) = tree.run(100)
game.place_piece(x, y)
win, path = game.check_winner()
assert not win is None
print("Random MCTS test passed")

# untrained
game = hex.HexGame()

for i in range(0, hex.SIZE-1):
    for j in range(1, 3):
        game.place_piece(i, j)   
        game.take_turn()    

model = cnn.AlphaZeroNetwork(hex.SIZE, hex.SIZE)
tree = mcts.MCTS(game, model)

(x, y) = tree.run(100)
game.place_piece(x, y)
win, path = game.check_winner()
assert not win is None
print("Untrained MCTS test passed")

print("MCTS test passed")