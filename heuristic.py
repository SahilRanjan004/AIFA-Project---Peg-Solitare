"""
Heuristic functions for Peg Solitaire.

IMPORTANT: For IDA* (optimal solving), heuristics MUST be admissible 
(never overestimate remaining moves). The simple heuristic is the only 
truly admissible one. Other heuristics are for best-first search and MCTS.
"""

from board import PegSolitaireBoard

# ============================================================================
# CONFIGURATION
# ============================================================================

# Select which heuristic to use:
#   'simple'   - Peg count only (admissible, best for IDA*)
#   'center'   - Manhattan distance to center (overestimates, NOT for IDA*)
#   'weighted' - Penalizes isolated pegs (for best-first search)
#   'improved' - Corner penalty + distance (for MCTS/best-first)
HEURISTIC_TYPE = 'simple'  # Changed from 'improved' - simple is best for IDA*

# ============================================================================
# HEURISTIC FUNCTIONS
# ============================================================================

def remaining_pegs_heuristic(state):
    """
    Simple admissible heuristic for IDA*.
    Each move removes exactly 1 peg, so remaining_pegs - 1 is the exact lower bound.
    This is the ONLY admissible heuristic for Peg Solitaire.
    """
    return PegSolitaireBoard.count_pegs(state.grid) - 1


def center_distance_heuristic(state):
    """
    Manhattan distance to center heuristic (NOT admissible for IDA*).
    Sums distances of all pegs to center. Each move can reduce distance by at most 2.
    Use this for best-first search or MCTS only.
    """
    pegs = PegSolitaireBoard.count_pegs(state.grid)
    if pegs == 1:
        return 0
    
    total_dist = 0
    for r in range(7):
        for c in range(7):
            if state.grid[r][c] == 1:
                total_dist += abs(r - 3) + abs(c - 3)
    
    # Each move reduces total distance by at most 2
    return max(pegs - 1, total_dist // 2)


def weighted_pegs_heuristic(state):
    """
    Weighted peg count for best-first minimize search.
    Penalizes isolated pegs (harder to remove) to guide search.
    Lower value = better state.
    """
    pegs = PegSolitaireBoard.count_pegs(state.grid)
    
    # Calculate isolation penalty
    isolation = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(7):
        for c in range(7):
            if state.grid[r][c] != 1:
                continue
            
            adjacent = 0
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if PegSolitaireBoard.is_valid(nr, nc) and state.grid[nr][nc] == 1:
                    adjacent += 1
            
            if adjacent == 0:
                isolation += 2   # Fully isolated peg (very bad)
            elif adjacent == 1:
                isolation += 1   # Semi-isolated peg (somewhat bad)
    
    return pegs + isolation


def improved_heuristic(state):
    """
    Enhanced heuristic for best-first search and MCTS (NOT for IDA*).
    Combines peg count, Manhattan distance, and corner penalties.
    Lower value = better state.
    """
    pegs = PegSolitaireBoard.count_pegs(state.grid)
    if pegs == 1:
        return 0
    
    # Quick win detection: if 2 pegs can jump each other
    if pegs == 2:
        moves = PegSolitaireBoard.get_moves(state.grid)
        for move in moves:
            temp_grid = PegSolitaireBoard.apply_move(state.grid, move)
            if PegSolitaireBoard.count_pegs(temp_grid) == 1:
                return 0  # Can win in 1 move
    
    total_dist = 0
    corner_penalty = 0
    
    # Hard-to-remove positions (corners and edge cells)
    bad_positions = [
        (0, 2), (0, 3), (0, 4),  # Top row
        (2, 0), (3, 0), (4, 0),  # Left column
        (2, 6), (3, 6), (4, 6),  # Right column
        (6, 2), (6, 3), (6, 4)   # Bottom row
    ]
    
    for r in range(7):
        for c in range(7):
            if state.grid[r][c] == 1:
                total_dist += abs(r - 3) + abs(c - 3)
                if (r, c) in bad_positions:
                    corner_penalty += 1
    
    # Lower bound: each move removes 1 peg but may not reduce distance
    return max(pegs - 1 + corner_penalty, total_dist // 3)


def get_heuristic(state, heuristic_type=None):
    """
    Get heuristic value for a state using the configured or specified type.
    
    Args:
        state: PegState object
        heuristic_type: One of 'simple', 'center', 'weighted', 'improved'
                       If None, uses global HEURISTIC_TYPE.
    
    Returns:
        int: Heuristic value (lower is better for search)
    """
    if heuristic_type is None:
        heuristic_type = HEURISTIC_TYPE
    
    if heuristic_type == 'simple':
        return remaining_pegs_heuristic(state)
    elif heuristic_type == 'center':
        return center_distance_heuristic(state)
    elif heuristic_type == 'weighted':
        return weighted_pegs_heuristic(state)
    elif heuristic_type == 'improved':
        return improved_heuristic(state)
    else:
        # Fallback to simple heuristic for unknown types
        return remaining_pegs_heuristic(state)