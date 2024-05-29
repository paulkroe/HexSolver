import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from tqdm import tqdm
import Game.Hex as hex
import Data.preprocess_data as preprocess_data



class SelfPlay:
    def __init__(self, agent):
        self.agent = agent

    def play_game(self):
        game = hex.HexGame()
        history = []
        p = game.current_player
        while True:
            board = preprocess_data.embed_board(game.board, game.current_player)
            mask = game.get_moves()
            move, probabilities = self.agent.get_move(game)
            probabilities = torch.tensor(probabilities, dtype=torch.float32).view((hex.SIZE, hex.SIZE))
            history.append((board, mask, probabilities, None))
            game.place_piece(*move)
            is_terminal, winner = game.is_terminal()
            if is_terminal:
                assert winner in [0,1]
                for i in range(len(history)):
                    board, mask, probabilities, _ = history[i]
                    if winner == 0:
                        history[i] = (board, mask, probabilities, torch.tensor(1-2*p, dtype=torch.float32))
                    elif winner == 1:
                        assert  winner == 1
                        history[i] = (board, mask, probabilities, torch.tensor(-1 + 2*p, dtype=torch.float32))
                    else: # draw
                        winner = 2
                        history[i] = (board, mask, probabilities, torch.tensor(0, dtype=torch.float32))
                    p = 1-p
                return history, winner
            game.take_turn()
        
    def generate_data(self, iterations, fraction, shuffle=True):
        data = []
        wins = [0,0,0]
        for _ in tqdm(range(iterations), desc="Games of Self Play: "):
            game_result, winner = self.play_game()
            wins[winner]+=1
            if shuffle:
                data.extend(random.choices(game_result, k=int(len(game_result)*fraction)))
            else:
                data.extend(game_result[:len(game_result)*fraction])
        print(f"Player 0 wins: {wins[0]}, Player 1 wins: {wins[1]}, Draws: {wins[2]}")
        return data 
    
        