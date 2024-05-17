import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Game.Hex as hex
import Data.preprocess_data as preprocess_data
import Agent.agent as agent

import torch
import torch.nn as nn


class SelfPlay:
    def __init__(self, agent):
        self.agent = agent

    def play_game(self):
        game = hex.HexGame()
        history = []

        while True:
            board = preprocess_data.embed_board(game.board, game.current_player)
            move, probabilities = self.agent.get_move(game)
            history.append((board, probabilities, None))
            game.place_piece(*move)
            is_terminal, winner = game.is_terminal()
            if is_terminal:
                print("Outcome: ", winner)
                for i in range(len(history)):
                    board, probabilities, _ = history[i]
                
                    history[i] = (board, probabilities, winner)
                print(game)
                return history
            game.take_turn()
        
    def generate_data(self, iterations):
        data = []
        for _ in range(iterations):
            game_result = self.play_game()
            data.extend(game_result)
        return data