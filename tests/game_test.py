import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Game.Hex as hex

print("Testing Hex game")
wins = [0, 0, 0]
for _ in range(100):
    game = hex.HexGame()
    while True:
        mask = game.get_moves()
        valid_moves = game.legal_moves(mask)
        # convert mask into a list of valid moves
        move = random.choice(valid_moves)
        game.place_piece(*move)
        winner, path = game.check_winner()
        if winner:
            wins[winner] += 1
            assert len(path) >= game.size
            break
        if game.check_draw():
            wins[2] += 1
            assert len(path) == 0
            break
        game.take_turn()