import torch
from torch.utils.data import Dataset, DataLoader
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Game.Hex as hex


class BoardGameDataset(Dataset):
    def __init__(self, boards, masks, probs, wins):
        self.boards = boards
        self.masks = masks
        self.probs = probs
        self.wins = wins

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        mask = self.masks[idx]
        prob = self.probs[idx]
        win = self.wins[idx]
        return board, mask, prob, win

def embed_board(board, player):
    board = torch.tensor(board, dtype=torch.float32)
    X_mask = torch.zeros((3, hex.SIZE, hex.SIZE))
    X_mask[0] = (board == 1).float()
    X_mask[1] = (board == 0).float()
    if player == 1 or player == 'o' or player == 'O':
        X_mask[2] = torch.ones((hex.SIZE, hex.SIZE))
    return X_mask

def get_loader(boards, masks, probs, wins, batch_size, shuffle=True):
    dataset = BoardGameDataset(boards, masks, probs, wins)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_loaders(data, batch_size, shuffle=True):
        boards = [d[0] for d in data]
        masks = [d[1] for d in data]
        probabilities = [d[2] for d in data]
        wins = [d[3] for d in data]
         
        loader = get_loader(boards, masks, probabilities, wins, batch_size, shuffle)
        return loader