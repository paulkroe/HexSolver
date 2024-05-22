import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Game.Hex as hex

def match(agent0, agent1, verbose=True):
    players = [agent0, agent1]
    player = 0
    game = hex.HexGame()
    is_terminal, winner = game.is_terminal()
    
    while not is_terminal:
        move, _ = players[player][0]
        game.place_piece(*move)
        if verbose:
            print(game)
        player = 1-player
        is_terminal, winner = game.is_terminal()
        
    return winner

    