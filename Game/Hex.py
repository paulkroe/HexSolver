import numpy as np
import torch
from copy import deepcopy
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, -1)] 
SIZE = 4



game_elements = {
    0: 'x',
    1: 'o',
    -1: '.'
}

class HexGame:
    def __init__(self):
        self.size = SIZE
        self.board = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = 0

    def copy(self):
        new_game = HexGame()
        new_game.board = deepcopy(self.board)
        new_game.current_player = self.current_player
        return new_game
    
    def __repr__(self):
        output = ""
        for i, row in enumerate(self.board):
            output += " " * i
            for cell in row:
                output += game_elements[cell] + " "
            output += "\n"
        return output                       

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == -1

    def is_terminal(self):
        winner, _ = self.check_winner()
        if winner is not None:
            return True, winner
        if self.check_draw():
            return True, 'd'
        return False, None

    def sim(self, row, col):
        new_game = self.copy()
        assert new_game.is_valid_move(row, col)
        assert (row, col) in new_game.legal_moves(new_game.get_moves())
        new_game.board[row][col] = new_game.current_player
        new_game.take_turn()
        return new_game 
    
    def place_piece(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            return True
        else:
            return False

    def take_turn(self):
        self.current_player = 1 - self.current_player

    def get_moves(self):
        mask = torch.zeros((self.size, self.size))
        for row in range(self.size):
            for col in range(self.size):
                mask[row][col] = 0 if self.board[row][col] == -1 else -np.inf
        return mask

    def legal_moves(self, mask):
        return [(row, col) for row in range(self.size) for col in range(self.size) if mask[row][col] == 0]
        
    
    def check_winner(self):
        player = self.current_player # The other player
        if player == 0:
            for col in range(self.size):
                if self.board[0][col] == 0:
                    path = self.dfs(0, col, set(), [])
                    if path:
                        return 0, path
        else:
            for row in range(self.size):
                if self.board[row][0] == 1:
                    path = self.dfs(row, 0, set(), [])
                    if path:
                        return 1, path
        return None, []

    def dfs(self, row, col, visited, path):
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return None
        if (row, col) in visited:
            return None
        if self.board[row][col] != self.current_player:
            return None
        path.append((row, col))
        if self.current_player == 0 and row == self.size - 1:
            return path
        if self.current_player == 1 and col == self.size - 1:
            return path
        visited.add((row, col))
        for dx, dy in DIRECTIONS:
            result = self.dfs(row + dx, col + dy, visited, path.copy())
            if result:
                return result
        path.pop()
        return None

    def play(self):
        while True:
            print(self)
            row = int(input("Player {} enter row: ".format(self.players[self.current_player])))
            col = int(input("Player {} enter column: ".format(self.players[self.current_player])))
            if self.place_piece(row, col):
                winner = self.check_winner() 
                if winner:
                    print(self)
                    print("Player {} wins!".format(winner))
                    break
                if self.check_draw():
                    print("It's a draw!")
                    break
                self.take_turn()
    
    def check_draw(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == -1:
                    return False
        return True