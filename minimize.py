"""
Best-first search to find a terminal state with the fewest pegs.
Uses tiered heuristic: peg count (primary), mobility (secondary tiebreaker).
The search is deterministic: uses canonical strings for symmetry breaking.

This is useful for finding "almost solutions" when reducing to 1 peg is impossible
due to board modifications (custom valid masks).
"""

import time
import heapq
from board import PegSolitaireBoard
from state import PegState


def minimize_heuristic(state):
    """
    Tiered heuristic for best-first minimize search.
    
    Primary goal: Fewer pegs is ALWAYS better.
    Secondary goal: Among states with same peg count, prefer more mobility.
    
    Formula: (pegs * 1000) - min(moves, 100)
    - Multiply by 1000 ensures peg count dominates (max pegs = 33 → 33,000)
    - Subtract mobility (max 100) only affects tie-breaking
    - Clamped to 100 moves to prevent negative values
    
    Returns:
        int: Lower value = better state
    """
    pegs = PegSolitaireBoard.count_pegs(state.grid)
    
    # Perfect state - can't get better
    if pegs == 1:
        return 0
    
    # Count legal moves (mobility)
    moves = len(PegSolitaireBoard.get_moves(state.grid))
    
    # Peg count dominates, mobility is tiebreaker
    return (pegs * 1000) - min(moves, 100)


class BestFirstMinimize:
    """
    Best-first search to minimize remaining pegs.
    
    Uses a priority queue (heap) ordered by minimize_heuristic.
    Explores most promising states first, stops when time limit reached
    or when a state with 1 peg is found.
    
    Attributes:
        start: Starting PegState
        max_time: Maximum search time in seconds
        on_expand: Optional callback for progress updates
        best_state: Best terminal state found so far
        best_pegs: Peg count of best_state
        solution_moves: List of moves to reach best_state
    """
    
    def __init__(self, start_state, max_time=60, on_expand=None):
        """
        Initialize best-first minimizer.
        
        Args:
            start_state: Initial PegState to search from
            max_time: Maximum search time in seconds
            on_expand: Callback function called on each node expansion
        """
        self.start = start_state
        self.max_time = max_time
        self.on_expand = on_expand
        self.nodes_expanded = 0
        self.best_state = None
        self.best_pegs = float('inf')
        self.solution_moves = None

    def search(self):
        """
        Run the best-first search.
        
        Returns:
            tuple: (best_pegs, solution_moves) where solution_moves is list of moves
                   to reach the best terminal state, or (inf, None) if no solution.
        """
        self.start_time = time.time()
        visited = set()      # Track visited states via canonical string
        heap = []            # Priority queue: (heuristic, canon, counter, state)
        
        # Initialize heap with start state
        start_h = minimize_heuristic(self.start)
        counter = 0
        heapq.heappush(heap, (start_h, self.start.canonical_string(), counter, self.start))
        visited.add(self.start.canonical_string())
        counter += 1

        # Main search loop
        while heap and (time.time() - self.start_time) < self.max_time:
            # Pop best state from heap
            h, canon, _, node = heapq.heappop(heap)
            self.nodes_expanded += 1
            
            # Callback for progress updates (e.g., GUI refresh)
            if self.on_expand:
                self.on_expand()

            # Terminal state found (no legal moves)
            if node.has_no_moves():
                pegs = PegSolitaireBoard.count_pegs(node.grid)
                if pegs < self.best_pegs:
                    self.best_pegs = pegs
                    self.best_state = node
                    # Perfect solution - can't do better than 1 peg
                    if pegs == 1:
                        break
                continue

            # Expand children (all legal moves from this state)
            for child in node.get_successors():
                canon_child = child.canonical_string()
                if canon_child in visited:
                    continue
                    
                visited.add(canon_child)
                h_child = minimize_heuristic(child)
                heapq.heappush(heap, (h_child, canon_child, counter, child))
                counter += 1

        # Reconstruct solution path if a terminal state was found
        if self.best_state:
            self.solution_moves = self._build_solution(self.best_state)
            
        return self.best_pegs, self.solution_moves

    def _build_solution(self, node):
        """
        Reconstruct the move sequence from start state to given node.
        
        Args:
            node: Target PegState node
            
        Returns:
            list: List of moves ((r1,c1), (r2,c2)) from start to node
        """
        moves = []
        while node.parent:
            moves.append(node.move)
            node = node.parent
        moves.reverse()
        return moves