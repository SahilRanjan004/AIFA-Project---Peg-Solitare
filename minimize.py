# minimize.py
"""
Best‑first search to find a terminal state with the fewest pegs.
Uses heuristic = remaining pegs (lower is better) as priority.
The search is deterministic: children are sorted and tie‑breakers use canonical string.
"""

import time
import heapq
from board import PegSolitaireBoard
from state import PegState

class BestFirstMinimize:
    def __init__(self, start_state, max_time=60, on_expand=None):
        self.start = start_state
        self.max_time = max_time
        self.on_expand = on_expand
        self.nodes_expanded = 0
        self.best_state = None
        self.best_pegs = float('inf')
        self.solution_moves = None

    def search(self):
        self.start_time = time.time()
        visited = set()
        heap = []
        start_h = PegSolitaireBoard.count_pegs(self.start.grid) - 1
        counter = 0
        # Push (heuristic, tie‑breaker, state) to guarantee deterministic order
        heapq.heappush(heap, (start_h, self.start.canonical_string(), counter, self.start))
        visited.add(self.start.canonical_string())
        counter += 1

        while heap and (time.time() - self.start_time) < self.max_time:
            h, canon, _, node = heapq.heappop(heap)
            self.nodes_expanded += 1
            if self.on_expand:
                self.on_expand()

            if node.has_no_moves():
                pegs = PegSolitaireBoard.count_pegs(node.grid)
                if pegs < self.best_pegs:
                    self.best_pegs = pegs
                    self.best_state = node
                    if pegs == 1:
                        break
                continue

            for child in node.get_successors():
                canon_child = child.canonical_string()
                if canon_child in visited:
                    continue
                visited.add(canon_child)
                h_child = PegSolitaireBoard.count_pegs(child.grid) - 1
                heapq.heappush(heap, (h_child, canon_child, counter, child))
                counter += 1

        if self.best_state:
            self.solution_moves = self._build_solution(self.best_state)
        return self.best_pegs, self.solution_moves

    def _build_solution(self, node):
        moves = []
        while node.parent:
            moves.append(node.move)
            node = node.parent
        moves.reverse()
        return moves