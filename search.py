# search.py
"""
IDA* search algorithm with symmetry pruning.
Used for solving to exactly one peg.
"""

import time
from heuristic import get_heuristic

class IDAStar:
    def __init__(self, start_state, heuristic=get_heuristic, max_time=180, on_expand=None):
        self.start = start_state
        self.heuristic = heuristic
        self.max_time = max_time
        self.on_expand = on_expand
        self.start_time = None
        self.nodes_expanded = 0
        self.solution = None
        self.found = False

    def search(self):
        self.start_time = time.time()
        bound = self.heuristic(self.start)
        while True:
            if time.time() - self.start_time > self.max_time:
                print("Time limit exceeded.")
                return None
            self.nodes_expanded = 0
            visited = set()
            t = self._search(self.start, 0, bound, visited)
            if t == "FOUND":
                return self.solution
            if t == float('inf'):
                return None
            bound = t

    def _search(self, node, g, bound, visited):
        f = g + self.heuristic(node)
        if f > bound:
            return f
        if node.is_goal():
            self.found = True
            self.solution = self._build_solution(node)
            return "FOUND"
        if time.time() - self.start_time > self.max_time:
            return float('inf')
        min_f = float('inf')
        for child in node.get_successors():
            canon = child.canonical_string()
            if canon in visited:
                continue
            visited.add(canon)
            self.nodes_expanded += 1
            if self.on_expand:
                self.on_expand()
            if self.on_expand:
                time.sleep(0)
            t = self._search(child, g + 1, bound, visited)
            if t == "FOUND":
                return "FOUND"
            if t < min_f:
                min_f = t
        return min_f

    def _build_solution(self, node):
        moves = []
        while node.parent:
            moves.append(node.move)
            node = node.parent
        moves.reverse()
        return moves