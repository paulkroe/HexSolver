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
        move = players[player].get_move(game)[0]
        game.place_piece(*move)
        if verbose:
            print(game)
        is_terminal, winner = game.is_terminal()
        game.take_turn()
        player = 1-player
    return winner

