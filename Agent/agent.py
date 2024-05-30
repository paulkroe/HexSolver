import random
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Agent.MCTS as mcts
import Data.preprocess_data as preprocess_data
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import Game.Hex as hex
    
    
class Agent():
    def __init__(self, model, name=None, random=False):
        self.model = model
        self.elo = 1200
        self.name = name
        self.history = [] # to keep track of elo ratings
        self.random = random
        self.outcomes = []
        self.logger = None
               
    def get_move(self, game, simulations=100):
        tree = mcts.MCTS(game, self.model)
        move = tree.run(simulations)
        probabilities = tree.probabilities
        return move, probabilities
    
    def eval(self, val_loader):
        self.model.eval()
        avg_loss = 0
        avg_mse_loss = 0
        avg_ce_loss = 0
        mse_loss_fn = MSELoss()
        
        with torch.no_grad():
            for board, mask, y_probs, y_wins in val_loader:
                pred_wins, pred_probs = self.model(board, None, add_noise=False)
                
                # value loss
                pred_wins = pred_wins.squeeze(dim=1)
                loss_mse = mse_loss_fn(y_wins, pred_wins)
                
                # policy loss
                y_probs = y_probs.view((-1, hex.SIZE*hex.SIZE))
                pred_probs = torch.log(pred_probs.view((-1, hex.SIZE*hex.SIZE)))
                pred_probs[torch.isneginf(pred_probs)] = 0
                loss_ce = -torch.mean(torch.sum(y_probs * pred_probs, dim=1))
 
                loss = loss_mse + loss_ce
                avg_loss += loss.item()
                avg_mse_loss += loss_mse.item()
                avg_ce_loss += loss_ce.item()
        
        num_samples = len(val_loader)
        return {
            'total_loss': avg_loss / num_samples,
            'mse_loss': avg_mse_loss / num_samples,
            'ce_loss': avg_ce_loss / num_samples
        }


    def train(self, data, epochs, lr, batch_size, l2=1e-1, verbose=False):
        if self.random:
            # If the agent is random, there is no need to train
            return
        losses = []
        mse_losses = []
        ce_losses = []

        mse_loss_fn = MSELoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=l2)

        loader = preprocess_data.get_loaders(data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            avg_loss, avg_mse_loss, avg_ce_loss = 0, 0, 0
            for board, mask, y_probs, y_wins in loader:
                pred_wins, pred_probs = self.model(board, None, add_noise=False)
                pred_wins = pred_wins.squeeze(dim=1)
                # value loss
                loss_mse = mse_loss_fn(y_wins, pred_wins)
                # policy loss
                y_probs = y_probs.view((-1, hex.SIZE*hex.SIZE))
                pred_probs = torch.log(pred_probs.view((-1, hex.SIZE*hex.SIZE)))
                pred_probs[torch.isneginf(pred_probs)] = 0
                loss_ce = -torch.mean(torch.sum(y_probs * pred_probs, dim=1))
                loss = loss_mse + loss_ce
                avg_loss += loss.item()
                avg_mse_loss += loss_mse.item()
                avg_ce_loss += loss_ce.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self.logger:
                self.logger.log({"train/loss": avg_loss / len(loader), "train/mse-loss": avg_mse_loss / len(loader), "train/ce-loss": avg_ce_loss / len(loader)})
            losses.append(avg_loss / len(loader))
            mse_losses.append(avg_mse_loss / len(loader))
            ce_losses.append(avg_ce_loss / len(loader))
            
            if verbose:
                print(f"[{epoch+1}]/[{epochs}]: Training Loss: {losses[-1]} - MSE Loss: {mse_losses[-1]} - CE Loss: {ce_losses[-1]}")
            
    def log(self):
        self.history.append(self.elo)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)


            