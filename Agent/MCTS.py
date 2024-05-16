import numpy as np
from graphviz import Digraph
from queue import Queue
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Data.preprocess_data as preprocess_data

C_PUT = 1.0

class Node():
    def __init__(self, parent, prior, state, move):
        self.move = move
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = []
        self.q = 0
        self.n = 0
        # TODO: fix this
        self.state.take_turn()
        self.is_terminal, self.outcome = self.state.is_terminal()
        self.state.take_turn()
    
    @property
    def u(self):
        if self.parent is None:
            return 0
        return C_PUT * self.prior * np.sqrt(self.parent.n) / (1 + self.n)
    
    @property
    def score(self):
        return self.q + self.u
    
    def update(self, value):
        self.q = (self.q * self.n + value) / (self.n + 1)
        self.n += 1
        if self.parent:
            self.parent.update(-value)
            
    def expand_node(self, policy, legal_moves):      
        for move in legal_moves:
            new_state = self.state.sim(*move)
            child_node = Node(state=new_state, prior=policy[move], parent=self, move=move)
            self.children.append(child_node)
    
    def select_child(self):
        best_score = max(child.score for child in self.children)
        best_children = [child for child in self.children if child.score == best_score]
        return random.choice(best_children)

    def select_move(self):
        best_n = max(child.n for child in self.children)
        best_children = [child for child in self.children if child.n == best_n]
        return random.choice(best_children).move
    
class MCTS():
    def __init__(self, board, net):
        self.net = net
        self.root = Node(parent=None, prior=1.0, state=board, move=None)
    
    def select_leaf(self, node):
        while len(node.children) != 0 and not node.is_terminal:
            node = node.select_child()
        return node
    
    def expand(self, node):
        if node.is_terminal:
            value = 0
            if node.outcome != self.root.state.current_player:
                value = -1
            elif node.outcome == self.root.state.current_player:
                value = 1
            node.update(value)
        else:
            mask = node.state.get_moves()
            legal_moves = node.state.legal_moves(mask)
            # todo think about player here
            board = preprocess_data.embed_board(node.state.board, node.state.current_player)
            value, policy = self.net(board.unsqueeze(dim=0), mask)
            node.update(value)
            node.expand_node(policy, legal_moves)
    
    def run(self, simulations):
        for _ in range(simulations):
            node = self.select_leaf(self.root)
            self.expand(node)
        return self.best_move
    
    @property
    def best_move(self):
        return self.root.select_move()

    def visualize(self):
        dot = Digraph()
        q = Queue()
        
        q.put(self.root)
        
        while not q.empty():
            node = q.get()
            node_label = node.state.__repr__()+f"prior: {node.prior}, score: {node.score}, n: {node.n}"
            shape = 'oval'
            color = 'black'
            if node.is_terminal:
                node_label += f"\noutcome: {node.outcome}"
                shape = 'octagon'
                color = 'red'
            dot.node(str(node), label=node_label, shape=shape, color=color)
            
            for child in node.children:
                q.put(child)
                dot.edge(str(node), str(child), label=str(child.move))
        
        dot.render('mcts', format='pdf', cleanup=True)