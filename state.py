# state.py
"""
Immutable state representation with symmetry pruning.
For single‑player, player is always 0.
For two‑player, player indicates whose turn it is (0 or 1).
"""

from board import PegSolitaireBoard

class PegState:
    __slots__ = ('grid', 'depth', 'parent', 'move', '_canonical', 'player')

    def __init__(self, grid, depth=0, parent=None, move=None, player=0):
        self.grid = tuple(tuple(row) for row in grid)
        self.depth = depth
        self.parent = parent
        self.move = move
        self.player = player
        self._canonical = None

    def is_goal(self):
        return PegSolitaireBoard.count_pegs(self.grid) == 1

    def has_no_moves(self):
        return len(PegSolitaireBoard.get_moves(self.grid)) == 0

    def is_terminal(self):
        # Two‑player terminal: current player has no moves
        return self.has_no_moves()

    def get_moves(self):
        """Return list of moves (as tuples of ((r1,c1),(r2,c2))) for the current player."""
        return PegSolitaireBoard.get_moves(self.grid)

    def get_successors(self):
        """Return list of child states (for single‑player search)."""
        children = []
        for move in self.get_moves():
            new_grid = PegSolitaireBoard.apply_move(self.grid, move)
            child = PegState(new_grid, self.depth+1, self, move, self.player)
            children.append(child)
        return children

    def apply_move(self, move):
        """Apply a move and return a new state with the opposite player to move (two‑player)."""
        new_grid = PegSolitaireBoard.apply_move(self.grid, move)
        return PegState(new_grid, self.depth+1, self, move, 1 - self.player)

    def canonical_string(self):
        if self._canonical is not None:
            return self._canonical
        grid = [list(row) for row in self.grid]
        syms = []

        def to_fixed_string(g):
            return ''.join('#' if v == -1 else '.' if v == 0 else '*' for row in g for v in row)

        syms.append(to_fixed_string(grid))
        rot90 = self._rotate90(grid)
        syms.append(to_fixed_string(rot90))
        rot180 = self._rotate90(rot90)
        syms.append(to_fixed_string(rot180))
        rot270 = self._rotate90(rot180)
        syms.append(to_fixed_string(rot270))
        reflect_h = self._reflect_h(grid)
        syms.append(to_fixed_string(reflect_h))
        reflect_rot90 = self._rotate90(reflect_h)
        syms.append(to_fixed_string(reflect_rot90))
        reflect_rot180 = self._rotate90(reflect_rot90)
        syms.append(to_fixed_string(reflect_rot180))
        reflect_rot270 = self._rotate90(reflect_rot180)
        syms.append(to_fixed_string(reflect_rot270))

        self._canonical = min(syms)
        return self._canonical

    @staticmethod
    def _rotate90(grid):
        return [list(row) for row in zip(*grid[::-1])]

    @staticmethod
    def _reflect_h(grid):
        return [row[::-1] for row in grid]

    def __hash__(self):
        return hash((self.canonical_string(), self.player))

    def __eq__(self, other):
        return self.canonical_string() == other.canonical_string() and self.player == other.player

    def __lt__(self, other):
        pegs_self = PegSolitaireBoard.count_pegs(self.grid)
        pegs_other = PegSolitaireBoard.count_pegs(other.grid)
        if pegs_self != pegs_other:
            return pegs_self < pegs_other
        return (self.canonical_string(), self.player) < (other.canonical_string(), other.player)

    def __repr__(self):
        return f"PegState(pegs={PegSolitaireBoard.count_pegs(self.grid)}, player={self.player}, depth={self.depth})"