import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Game.Hex as hex

def embed_board(board, player):
    board = torch.tensor(board, dtype=torch.float32)
    X_mask = torch.zeros((3, hex.SIZE, hex.SIZE))
    X_mask[0] = (board == 1).float()
    X_mask[1] = (board == 0).float()
    if player == 1 or player == 'o' or player == 'O':
        X_mask[2] = torch.ones((hex.SIZE, hex.SIZE))
    return X_mask

def get_loader(boards, pros, wins, batch_size, shuffle=True):
    boards = [embed_board(board, player) for board, player in boards]
    labels = zip(pros, wins)
    data = TensorDataset(boards, labels)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)