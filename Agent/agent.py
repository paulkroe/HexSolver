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
    def __init__(self, model):
        self.model = model
    
    def get_move(self, game, simulations=100):
        tree = mcts.MCTS(game, self.model)
        move = tree.run(simulations)
        probabilities = tree.probabilities
        return move, probabilities
    
    def eval(self, val_loader):
        avg_loss = 0
        avg_mse_loss = 0
        avg_ce_loss = 0
        mse_loss_fn = MSELoss()
        
        with torch.no_grad():
            for board, _, y_probs, y_wins in val_loader:
                pred_wins, pred_probs = self.model(board, None)
                
                # value loss
                pred_wins = pred_wins.squeeze(dim=1)
                loss_mse = mse_loss_fn(y_wins, pred_wins)
                
                # policy loss
                y_probs = y_probs.view((-1, hex.SIZE*hex.SIZE))
                pred_probs = torch.log(pred_probs.view((-1, hex.SIZE*hex.SIZE)))
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


    def train(self, data, epochs, lr, batch_size, verbose=True):
        train_losses, val_losses = [], []
        train_mse_losses, train_ce_losses = [], []
        val_mse_losses, val_ce_losses = [], []

        mse_loss_fn = MSELoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        train_loader, val_loader = preprocess_data.get_loaders(data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            avg_loss, avg_mse_loss, avg_ce_loss = 0, 0, 0

            for board, _, y_probs, y_wins in train_loader:
                pred_wins, pred_probs = self.model(board, None)
                pred_wins = pred_wins.squeeze(dim=1)

                # value loss
                loss_mse = mse_loss_fn(y_wins, pred_wins)
                # policy loss
                y_probs = y_probs.view((-1, hex.SIZE*hex.SIZE))
                pred_probs = torch.log(pred_probs.view((-1, hex.SIZE*hex.SIZE)))
                loss_ce = -torch.mean(torch.sum(y_probs * pred_probs, dim=1))
                
                loss = loss_mse + loss_ce
                avg_loss += loss.item()
                avg_mse_loss += loss_mse.item()
                avg_ce_loss += loss_ce.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_losses.append(avg_loss / len(train_loader))
            train_mse_losses.append(avg_mse_loss / len(train_loader))
            train_ce_losses.append(avg_ce_loss / len(train_loader))

            if verbose:
                eval_results = self.eval(val_loader)
                val_losses.append(eval_results['total_loss'])
                val_mse_losses.append(eval_results['mse_loss'])
                val_ce_losses.append(eval_results['ce_loss'])

                print(f"[{epoch+1}]/[{epochs}]: Training Loss: {train_losses[-1]} - MSE Loss: {train_mse_losses[-1]} - CE Loss: {train_ce_losses[-1]}")
                print(f"[{epoch+1}]/[{epochs}]: Validation Loss: {val_losses[-1]} - Val MSE Loss: {val_mse_losses[-1]} - Val CE Loss: {val_ce_losses[-1]}")
        if verbose:
            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Total Train Loss")
            plt.plot(train_mse_losses, label="Train MSE Loss")
            plt.plot(train_ce_losses, label="Train CE Loss")
            plt.plot(val_losses, label="Total Validation Loss")
            plt.plot(val_mse_losses, label="Validation MSE Loss")
            plt.plot(val_ce_losses, label="Validation CE Loss")
            plt.legend()
            plt.title("Loss Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()


            