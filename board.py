# board.py
"""
Board representation for English cross peg solitaire.
Grid values: -1 = invalid cell, 1 = peg, 0 = empty hole.
The valid mask can be changed at runtime.
"""

class PegSolitaireBoard:
    # Original mask (class variable, mutable)
    _valid_mask = [
        [0,0,1,1,1,0,0],
        [0,0,1,1,1,0,0],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [0,0,1,1,1,0,0],
        [0,0,1,1,1,0,0],
    ]

    @classmethod
    def is_valid(cls, r, c):
        """Return True if (r,c) is a valid hole (i.e., part of the board)."""
        return 0 <= r < 7 and 0 <= c < 7 and cls._valid_mask[r][c] == 1

    @classmethod
    def toggle_valid(cls, r, c):
        """Toggle the validity of a cell. Returns True if it became valid, False if invalid."""
        if not (0 <= r < 7 and 0 <= c < 7):
            return None
        current = cls._valid_mask[r][c]
        if current == 1:
            cls._valid_mask[r][c] = 0   # make invalid
            return False
        else:
            cls._valid_mask[r][c] = 1   # make valid
            return True

    @classmethod
    def reset_mask(cls):
        """Restore the original English cross shape."""
        cls._valid_mask = [
            [0,0,1,1,1,0,0],
            [0,0,1,1,1,0,0],
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,1,1,1,0,0],
        ]

    @classmethod
    def initial_grid(cls):
        """
        Create the starting grid:
        All valid holes contain a peg (1) except the centre which is empty (0).
        Invalid cells are -1.
        """
        grid = [[-1 for _ in range(7)] for _ in range(7)]
        for r in range(7):
            for c in range(7):
                if cls.is_valid(r, c):
                    grid[r][c] = 1
        # Centre hole (3,3) is empty (if it's valid)
        if cls.is_valid(3,3):
            grid[3][3] = 0
        return grid

    @classmethod
    def get_moves(cls, grid):
        """
        Generate all legal moves from the given grid.
        Returns a list of moves, each as ((r1,c1), (r2,c2)).
        """
        moves = []
        for r in range(7):
            for c in range(7):
                if grid[r][c] != 1:
                    continue
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    r_mid = r + dr
                    c_mid = c + dc
                    r_to = r + 2*dr
                    c_to = c + 2*dc
                    if (cls.is_valid(r_mid, c_mid) and grid[r_mid][c_mid] == 1 and
                        cls.is_valid(r_to, c_to) and grid[r_to][c_to] == 0):
                        moves.append(((r,c), (r_to,c_to)))
        return moves

    @classmethod
    def apply_move(cls, grid, move):
        """
        Return a new grid after applying a move.
        The original grid is not modified.
        """
        # Convert to list of lists if needed
        if isinstance(grid[0], tuple):
            new_grid = [list(row) for row in grid]
        else:
            new_grid = [row[:] for row in grid]
        (r1,c1), (r2,c2) = move
        r_mid = (r1 + r2) // 2
        c_mid = (c1 + c2) // 2
        new_grid[r1][c1] = 0
        new_grid[r_mid][c_mid] = 0
        new_grid[r2][c2] = 1
        return new_grid

    @classmethod
    def count_pegs(cls, grid):
        """Return the number of pegs (1's) on the board."""
        return sum(row.count(1) for row in grid)

    @classmethod
    def grid_to_string(cls, grid):
        """Human‑readable representation: X = peg, . = empty, space = invalid."""
        lines = []
        for r in range(7):
            line = []
            for c in range(7):
                if grid[r][c] == 1:
                    line.append('X')
                elif grid[r][c] == 0:
                    line.append('.')
                else:   # -1
                    line.append(' ')
            lines.append(''.join(line))
        return '\n'.join(lines)