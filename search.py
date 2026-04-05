"""
IDA* (Iterative Deepening A*) search algorithm with symmetry pruning.
Used for solving Peg Solitaire to exactly one peg.

IDA* is ideal for this problem because:
- Memory efficient (uses depth-first search with a cost bound)
- Guarantees optimal solutions (admissible heuristic)
- Symmetry pruning eliminates 8x duplicate states
"""

import time
from heuristic import get_heuristic


class IDAStar:
    """
    IDA* solver for Peg Solitaire.
    
    Algorithm:
    1. Start with bound = heuristic(initial_state)
    2. Perform depth-first search, pruning branches where f = g + h > bound
    3. If no solution found, increase bound to the smallest f that exceeded it
    4. Repeat until solution found or time limit exceeded
    
    Attributes:
        start: Starting PegState
        heuristic: Heuristic function (must be admissible)
        max_time: Maximum search time in seconds
        on_expand: Optional callback for progress updates
        nodes_expanded: Counter for statistics
        solution: List of moves when found
    """
    
    def __init__(self, start_state, heuristic=get_heuristic, max_time=180, on_expand=None):
        """
        Initialize IDA* solver.
        
        Args:
            start_state: Initial PegState to solve from
            heuristic: Heuristic function (default: remaining_pegs - 1)
            max_time: Maximum search time in seconds
            on_expand: Callback function called on each node expansion
        """
        self.start = start_state
        self.heuristic = heuristic
        self.max_time = max_time
        self.on_expand = on_expand
        self.start_time = None
        self.nodes_expanded = 0
        self.solution = None
        self.found = False

    def search(self):
        """
        Run the IDA* search.
        
        Returns:
            List of moves if solution found, None otherwise
        """
        self.start_time = time.time()
        bound = self.heuristic(self.start)  # Initial cost bound
        
        while True:
            # Check time limit
            if time.time() - self.start_time > self.max_time:
                print("Time limit exceeded.")
                return None
            
            # Reset for this iteration
            self.nodes_expanded = 0
            visited = set()  # Tracks states in current DFS path (prevents loops)
            
            # Run DFS with current bound
            t = self._search(self.start, 0, bound, visited)
            
            # Solution found
            if t == "FOUND":
                return self.solution
            
            # No solution exists (search exhausted)
            if t == float('inf'):
                return None
            
            # Increase bound to minimum f that exceeded previous bound
            bound = t

    def _search(self, node, g, bound, visited):
        """
        Recursive depth-first search with pruning.
        
        Args:
            node: Current state being explored
            g: Cost from start to current node (number of moves)
            bound: Current cost limit (prune if f = g + h > bound)
            visited: Set of canonical strings in current path (prevents cycles)
        
        Returns:
            - "FOUND" if solution found
            - Next bound value (minimum f that exceeded current bound)
            - float('inf') if search exhausted
        """
        f = g + self.heuristic(node)
        
        # Prune: this path exceeds the cost bound
        if f > bound:
            return f
        
        # Goal reached (exactly 1 peg remaining)
        if node.is_goal():
            self.found = True
            self.solution = self._build_solution(node)
            return "FOUND"
        
        # Time limit check
        if time.time() - self.start_time > self.max_time:
            return float('inf')
        
        min_f = float('inf')
        
        # Explore children in the order they're generated
        # (move ordering could be added here for better performance)
        for child in node.get_successors():
            canon = child.canonical_string()
            
            # Skip if this state already visited in current path (cycle detection)
            if canon in visited:
                continue
            
            visited.add(canon)
            self.nodes_expanded += 1
            
            # Progress callback (e.g., for GUI updates)
            if self.on_expand:
                self.on_expand()
            
            # Recursive search
            t = self._search(child, g + 1, bound, visited)
            
            # Solution found deeper in the tree
            if t == "FOUND":
                return "FOUND"
            
            # Track minimum f that exceeded bound (for next iteration)
            if t < min_f:
                min_f = t
        
        return min_f

    def _build_solution(self, node):
        """
        Reconstruct the move sequence from start to goal state.
        
        Args:
            node: Goal state node
            
        Returns:
            List of moves from start to goal (oldest first)
        """
        moves = []
        while node.parent:
            moves.append(node.move)
            node = node.parent
        moves.reverse()
        return moves