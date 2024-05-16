import torch
from torch.utils.data import TensorDataset, DataLoader

def embed_board(board, player):
    board_tensor = torch.tensor([[ord(cell) for cell in row] for row in board])
    print(board_tensor)
    X_mask = torch.zeros((3, 9, 9))

    X_mask[0] = ((board_tensor == ord('x')) | (board_tensor == ord('X'))).float()

    X_mask[1] = ((board_tensor == ord('o')) | (board_tensor == ord('O'))).float()

    if player == 1 or player == 'o' or player == 'O':
        X_mask[2] = torch.ones((9, 9))

    return X_mask

'''
def embed_board(board, player):
    board_tensor = torch.tensor(board)
    print(board_tensor)
    X_mask = torch.zeros((3, 9, 9))
    X_mask[0] = ((board_tensor == 'x') | (board_tensor == 'X')).float()
    X_mask[1] = ((board_tensor == 'o') | (board_tensor == 'O')).float()

    if player == 1 or player == 'o' or player == 'O':
        X_mask[2] = torch.ones((9, 9))

    return X_mask
'''

def get_loader(boards, pros, wins, batch_size, shuffle=True):
    boards = [embed_board(board, player) for board, player in boards]
    labels = zip(pros, wins)
    data = TensorDataset(boards, labels)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)