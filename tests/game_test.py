import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Game.Hex as Hex

print("Testing Hex game")
wins = {'o': 0, 'x': 0, 'd': 0}
for _ in range(100):
    game = Hex.HexGame()
    while True:
        mask = game.get_moves()
        # convert mask into a list of valid moves
        valid_moves = [(row, col) for row in range(game.size) for col in range(game.size) if mask[row][col] == 1]
        move = random.choice(valid_moves)
        game.place_piece(*move)
        winner, path = game.check_winner()
        if winner:
            wins[winner] += 1
            assert len(path) >= game.size
            break
        if game.check_draw():
            wins['d'] += 1
            assert len(path) == 0
            break