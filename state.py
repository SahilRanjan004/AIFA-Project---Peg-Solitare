"""
Immutable state representation for Peg Solitaire with symmetry pruning.

Key features:
- Immutable: grid stored as tuple of tuples (hashable)
- Symmetry pruning: canonical string accounts for 8 board symmetries (rotations/reflections)
- Supports both single-player (player always 0) and two-player (player 0 or 1)
- Uses __slots__ for memory efficiency
"""

from board import PegSolitaireBoard


class PegState:
    """
    Immutable game state for Peg Solitaire.
    
    Uses __slots__ to reduce memory overhead (important for storing many states in search).
    
    Attributes:
        grid: Tuple of tuples (7x7) with values -1 (invalid), 0 (empty), 1 (peg)
        depth: Number of moves from initial state
        parent: Parent PegState (for move reconstruction)
        move: Move that led to this state ((from_pos, to_pos))
        player: Current player to move (0 or 1, always 0 for single-player)
        _canonical: Cached canonical string (for symmetry pruning)
    """
    
    __slots__ = ('grid', 'depth', 'parent', 'move', '_canonical', 'player')
    
    # Board symmetries: 4 rotations + 4 reflections = 8 total
    # The canonical string is the lexicographically smallest among all 8

    def __init__(self, grid, depth=0, parent=None, move=None, player=0):
        """
        Initialize a PegState.
        
        Args:
            grid: 7x7 board (list of lists or tuple of tuples)
            depth: Number of moves from initial state
            parent: Parent state (None for root)
            move: Move that led to this state
            player: Current player to move (0 or 1)
        """
        # Convert to tuple of tuples for immutability (makes state hashable)
        self.grid = tuple(tuple(row) for row in grid)
        self.depth = depth
        self.parent = parent
        self.move = move
        self.player = player
        self._canonical = None  # Lazy-cached canonical string

    def is_goal(self):
        """Check if this state is a goal (exactly 1 peg remaining)."""
        return PegSolitaireBoard.count_pegs(self.grid) == 1

    def has_no_moves(self):
        """Check if current player has any legal moves."""
        return len(PegSolitaireBoard.get_moves(self.grid)) == 0

    def is_terminal(self):
        """
        Check if game has ended (no moves for current player).
        For two-player, terminal means current player loses.
        """
        return self.has_no_moves()

    def get_moves(self):
        """Return all legal moves from this state."""
        return PegSolitaireBoard.get_moves(self.grid)

    def get_successors(self):
        """
        Generate child states for single-player search.
        Player remains the same (always 0).
        
        Returns:
            List of child PegState objects
        """
        children = []
        for move in self.get_moves():
            new_grid = PegSolitaireBoard.apply_move(self.grid, move)
            child = PegState(new_grid, self.depth + 1, self, move, self.player)
            children.append(child)
        return children

    def apply_move(self, move):
        """
        Apply a move and return a new state for two-player game.
        Switches player to the opponent (1 - current player).
        
        Args:
            move: ((r1,c1), (r2,c2)) move to apply
            
        Returns:
            New PegState with opponent to move
        """
        new_grid = PegSolitaireBoard.apply_move(self.grid, move)
        return PegState(new_grid, self.depth + 1, self, move, 1 - self.player)

    def canonical_string(self):
        """
        Generate a canonical string representation that is invariant under
        board symmetries (rotations and reflections).
        
        This allows symmetric states to be recognized as identical, reducing
        the search space by a factor of up to 8.
        
        Returns:
            String representing the board in a canonical (minimal) form
        """
        # Return cached value if already computed
        if self._canonical is not None:
            return self._canonical
        
        # Convert to mutable list for transformations
        grid = [list(row) for row in self.grid]
        symmetries = []

        def to_fixed_string(g):
            """Convert grid to compact string: #=invalid, .=empty, *=peg"""
            return ''.join('#' if v == -1 else '.' if v == 0 else '*' 
                          for row in g for v in row)

        # Generate all 8 symmetric variations
        # Original
        symmetries.append(to_fixed_string(grid))
        
        # 90°, 180°, 270° rotations
        rot90 = self._rotate90(grid)
        symmetries.append(to_fixed_string(rot90))
        
        rot180 = self._rotate90(rot90)
        symmetries.append(to_fixed_string(rot180))
        
        rot270 = self._rotate90(rot180)
        symmetries.append(to_fixed_string(rot270))
        
        # Horizontal reflection
        reflect_h = self._reflect_h(grid)
        symmetries.append(to_fixed_string(reflect_h))
        
        # Reflection + rotations
        reflect_rot90 = self._rotate90(reflect_h)
        symmetries.append(to_fixed_string(reflect_rot90))
        
        reflect_rot180 = self._rotate90(reflect_rot90)
        symmetries.append(to_fixed_string(reflect_rot180))
        
        reflect_rot270 = self._rotate90(reflect_rot180)
        symmetries.append(to_fixed_string(reflect_rot270))
        
        # Canonical = lexicographically smallest string
        self._canonical = min(symmetries)
        return self._canonical

    @staticmethod
    def _rotate90(grid):
        """Rotate a 7x7 grid 90 degrees clockwise."""
        # zip(*grid[::-1]) is the classic Python 2D rotation idiom
        return [list(row) for row in zip(*grid[::-1])]

    @staticmethod
    def _reflect_h(grid):
        """Reflect a 7x7 grid horizontally (left-right mirror)."""
        return [row[::-1] for row in grid]

    def __hash__(self):
        """Hash based on canonical state AND player (for two-player)."""
        return hash((self.canonical_string(), self.player))

    def __eq__(self, other):
        """Equality based on canonical state AND player."""
        if other is None:
            return False
        return (self.canonical_string() == other.canonical_string() and 
                self.player == other.player)

    def __lt__(self, other):
        """
        Comparison for priority queues.
        Primary: fewer pegs is better
        Secondary: lexicographic order of (canonical_string, player) for determinism
        """
        pegs_self = PegSolitaireBoard.count_pegs(self.grid)
        pegs_other = PegSolitaireBoard.count_pegs(other.grid)
        
        if pegs_self != pegs_other:
            return pegs_self < pegs_other
        
        return (self.canonical_string(), self.player) < (other.canonical_string(), other.player)

    def __repr__(self):
        """Developer-friendly string representation."""
        return (f"PegState(pegs={PegSolitaireBoard.count_pegs(self.grid)}, "
                f"player={self.player}, depth={self.depth})")