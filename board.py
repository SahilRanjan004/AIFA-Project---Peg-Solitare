"""
Board representation for English cross peg solitaire.
Grid values: -1 = invalid cell, 1 = peg, 0 = empty hole.
The valid mask can be changed at runtime (useful for custom puzzles).
"""

class PegSolitaireBoard:
    # Standard English cross board mask (7x7)
    # 1 = valid cell, 0 = invalid (hole doesn't exist)
    _valid_mask = [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ]

    @classmethod
    def is_valid(cls, r, c):
        """
        Check if a cell (r,c) is a valid hole on the board.
        Returns True if the cell exists and is part of the playable area.
        """
        return 0 <= r < 7 and 0 <= c < 7 and cls._valid_mask[r][c] == 1

    @classmethod
    def toggle_valid(cls, r, c):
        """
        Toggle a cell's validity (for custom board shapes).
        Returns True if cell became valid, False if became invalid, None if out of bounds.
        Note: This modifies the shared class mask - affects all game modes.
        """
        if not (0 <= r < 7 and 0 <= c < 7):
            return None
        
        if cls._valid_mask[r][c] == 1:
            cls._valid_mask[r][c] = 0
            return False
        else:
            cls._valid_mask[r][c] = 1
            return True

    @classmethod
    def reset_mask(cls):
        """Restore the original English cross board shape."""
        cls._valid_mask = [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]

    @classmethod
    def initial_grid(cls):
        """
        Create the standard starting position:
        - All valid cells contain a peg (1)
        - The center cell (3,3) is empty (0)
        - Invalid cells are -1
        """
        grid = [[-1 for _ in range(7)] for _ in range(7)]
        
        # Fill all valid cells with pegs
        for r in range(7):
            for c in range(7):
                if cls.is_valid(r, c):
                    grid[r][c] = 1
        
        # Remove the center peg (standard starting position)
        if cls.is_valid(3, 3):
            grid[3][3] = 0
            
        return grid

    @classmethod
    def get_moves(cls, grid):
        """
        Generate all legal moves from the given board position.
        
        A move is valid if:
        1. There's a peg at (r,c)
        2. There's a peg adjacent (up/down/left/right) at (r_mid, c_mid)
        3. The landing cell two steps away is empty and valid
        
        Returns: List of moves, each as ((r1,c1), (r2,c2))
        """
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for r in range(7):
            for c in range(7):
                if grid[r][c] != 1:  # Only check cells with pegs
                    continue
                    
                for dr, dc in directions:
                    r_mid = r + dr
                    c_mid = c + dc
                    r_to = r + 2 * dr
                    c_to = c + 2 * dc
                    
                    # Check: middle cell has peg, landing cell is empty and valid
                    if (cls.is_valid(r_mid, c_mid) and grid[r_mid][c_mid] == 1 and
                        cls.is_valid(r_to, c_to) and grid[r_to][c_to] == 0):
                        moves.append(((r, c), (r_to, c_to)))
        
        return moves

    @classmethod
    def apply_move(cls, grid, move):
        """
        Apply a move to a board and return a NEW board (original unchanged).
        
        Args:
            grid: Current board state
            move: Tuple of ((r1,c1), (r2,c2)) - from position to landing position
        
        Returns: New grid after the move
        """
        # Convert to list of lists for mutability (handles both tuple and list inputs)
        if isinstance(grid[0], tuple):
            new_grid = [list(row) for row in grid]
        else:
            new_grid = [row[:] for row in grid]
        
        (r1, c1), (r2, c2) = move
        r_mid = (r1 + r2) // 2
        c_mid = (c1 + c2) // 2
        
        # Apply the move: source peg removed, jumped peg removed, landing cell gets peg
        new_grid[r1][c1] = 0
        new_grid[r_mid][c_mid] = 0
        new_grid[r2][c2] = 1
        
        return new_grid

    @classmethod
    def count_pegs(cls, grid):
        """Return the number of pegs (value 1) on the board."""
        return sum(row.count(1) for row in grid)

    @classmethod
    def grid_to_string(cls, grid):
        """
        Convert board to human-readable string.
        X = peg
        . = empty hole
        (space) = invalid cell
        """
        lines = []
        for r in range(7):
            line_chars = []
            for c in range(7):
                if grid[r][c] == 1:
                    line_chars.append('X')
                elif grid[r][c] == 0:
                    line_chars.append('.')
                else:  # -1
                    line_chars.append(' ')
            lines.append(''.join(line_chars))
        return '\n'.join(lines)