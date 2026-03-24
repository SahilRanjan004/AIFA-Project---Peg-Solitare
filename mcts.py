# mcts.py
import math
import random
import time

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_moves())

    def best_child(self, exploration_constant=1.4):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.wins / child.visits
            explore = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        tried_moves = {child.move for child in self.children}
        for move in self.state.get_moves():
            if move not in tried_moves:
                new_state = self.state.apply_move(move)
                child = MCTSNode(new_state, parent=self, move=move)
                self.children.append(child)
                return child
        return None

    def rollout(self):
        current_state = self.state
        while not current_state.is_terminal():
            moves = current_state.get_moves()
            if not moves:
                break
            move = random.choice(moves)
            current_state = current_state.apply_move(move)
        winner = 1 - current_state.player
        return 1 if winner == self.state.player else 0

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    def __init__(self, state, time_limit=180.0, simulation_limit=100000):
        self.root = MCTSNode(state)
        self.time_limit = time_limit
        self.simulation_limit = simulation_limit

    def search(self):
        start_time = time.time()
        sim_count = 0
        while (time.time() - start_time) < self.time_limit and sim_count < self.simulation_limit:
            node = self.select()
            if not node.state.is_terminal():
                if not node.is_fully_expanded():
                    node = node.expand()
                result = node.rollout()
                node.backpropagate(result)
            sim_count += 1
        best = max(self.root.children, key=lambda c: c.visits)
        return best.move

    def select(self):
        node = self.root
        while not node.state.is_terminal() and node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node