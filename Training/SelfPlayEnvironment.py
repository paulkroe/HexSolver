import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tqdm import tqdm

import Game.Hex as hex
from Training.ELO import calculate_elo
import Data.preprocess_data as preprocess_data
import Agent.agent as Agent
import Agent.Networks.CNN as CNN

class EvaluationEnvironment:
    def __init__(self, agent_pool):
        self.agent_pool = agent_pool
        self.num_games = 5
    
    def play_game(self, agent1, agent2):
        agents = [agent1, agent2]
        game = hex.HexGame()
        p = 0
        while True:
            move, _ = agents[p].get_move(game)
            game.place_piece(*move)
            is_terminal, winner = game.is_terminal()
            if is_terminal:
                if winner == 0:
                    return 1
                return 0
            p = 1-p
            game.take_turn()
            # print("debugging:\n", game)
    
    def __call__(self):
        print("Evaluation:")
        for i, agent1 in enumerate(self.agent_pool):
            for agent2 in self.agent_pool[:i+1]:
                wins = 0
                for _ in range(self.num_games):
                    wins += self.play_game(agent1, agent2)
                print(f"{agent1.name} vs {agent2.name}: {wins}:{self.num_games-wins}")
                wins = 0
                if agent1.name != agent2.name:
                    for _ in range(self.num_games):
                        wins += self.play_game(agent2, agent1)
                    print(f"{agent2.name} vs {agent1.name}: {wins}:{self.num_games-wins}")

                
                

class SelfPlayEnvironment:
    def __init__(self, agent_pool):
        self.agent_pool = agent_pool
    
    def play_game(self, agent1, agent2):
        agents = [agent1, agent2]
        game = hex.HexGame()
        history_ = []
        start = random.choice([0,1]) # select starting player 
        p = start
        while True:
            board = preprocess_data.embed_board(game.board, game.current_player)
            mask = game.get_moves()
            move, probabilities = agents[p].get_move(game)
            probabilities = torch.tensor(probabilities, dtype=torch.float32).view((hex.SIZE, hex.SIZE))
            history_.append((board, mask, probabilities, None))
            game.place_piece(*move)
            is_terminal, winner = game.is_terminal()
            if is_terminal:
                history = []
                for i in range(start, len(history_), 2):
                    board, mask, probabilities, _ = history_[i]
                    if winner == start: # win
                        history.append((board, mask, probabilities, torch.tensor(1, dtype=torch.float32)))
                    elif winner == 1-start: # loss
                        history.append((board, mask, probabilities, torch.tensor(-1, dtype=torch.float32)))
                    else: # draw
                        history.append((board, mask, probabilities, torch.tensor(0, dtype=torch.float32)))
                
                self.update_elo(agents[0], agents[1], 1-start-winner)# 1 if agent1 wins, 0.5 if draw, 0 if agent1 loses
                agent1.log()
                if start == 0 and winner == 0:
                    agent1.outcomes.append(1)
                elif start == 1 and winner == 1:
                    agent1.outcomes.append(1)
                else:
                    agent1.outcomes.append(0)
                return history
            p = 1-p
            game.take_turn()
            # print("debugging:\n", game)

    def update_elo(self, agent1, agent2, result):
        if result == 1:
            agent1.elo = calculate_elo(agent1.elo, agent2.elo, 1)
            agent2.elo = calculate_elo(agent2.elo, agent1.elo, 0)
        elif result == 0.5:
            agent1.elo = calculate_elo(agent1.elo, agent2.elo, 0.5)
            agent2.elo = calculate_elo(agent2.elo, agent1.elo, 0.5)
        else:
            agent1.elo = calculate_elo(agent1.elo, agent2.elo, 0)
            agent2.elo = calculate_elo(agent2.elo, agent1.elo, 1)
    
    def get_opponent(self, agent):
        # Find an opponent with a similar Elo rating
        candidates = [a for a in self.agent_pool if a != agent]
        candidates.sort(key=lambda a: abs(a.elo - agent.elo))
        return candidates[0] if candidates else None

import pandas as pd
import matplotlib.pyplot as plt

def train_agents(agent_pool, num_games):
    environment = SelfPlayEnvironment(agent_pool)
    
    for game in range(num_games):
        for agent in agent_pool:
            opponent = environment.get_opponent(agent)
            if opponent:
                hist = environment.play_game(agent, opponent)
                agent.train(hist, epochs=1, lr=0.001, batch_size=1, verbose=False)  # Pass actual experience
                agent.history[-1] = (game, agent.history[-1])  # Update the game number
                
                # maybe want to train the opponent too
        
            # Print the Elo ratings of all agents
        print(f"After game {game + 1}")
        for agent in agent_pool:
            print(f"{agent.name}: Elo {agent.elo}")
    
    # Plot the progress
    plot_progress(agent_pool)
    # save networks
    for agent in agent_pool:
        agent.save(agent.name + ".pt")

def plot_progress(agent_pool):
    for agent in agent_pool:
        history = pd.DataFrame(agent.history, columns=["Game", "Elo"])
        plt.plot(history["Game"], history["Elo"], label=agent.name)
    plt.xlabel("Game Number")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.title("Agent Elo Ratings Over Time")
    plt.show()