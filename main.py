"""
Command-line interface for the Peg Solitaire solver.
Uses IDA* to find the optimal solution to reduce to a single peg.
"""

from board import PegSolitaireBoard
from state import PegState
from search import IDAStar


def print_board(state):
    """Display the current board state in a human-readable format."""
    print(PegSolitaireBoard.grid_to_string(state.grid))


def print_moves(moves):
    """
    Print a list of moves in a readable format.
    
    Args:
        moves: List of ((r1,c1), (r2,c2)) move tuples
    """
    for i, move in enumerate(moves):
        (r1, c1), (r2, c2) = move
        print(f"{i+1}: ({r1},{c1}) -> ({r2},{c2})")


def main():
    """Run the Peg Solitaire solver from the command line."""
    
    # Create the standard starting position (all pegs except center empty)
    grid = PegSolitaireBoard.initial_grid()
    initial_state = PegState(grid)

    # Display initial board
    print("Initial board:")
    print_board(initial_state)
    print(f"Peg count: {PegSolitaireBoard.count_pegs(initial_state.grid)}")
    print("\nSolving...")

    # Run IDA* search
    solver = IDAStar(initial_state)
    solution = solver.search()

    if solution:
        # Solution found - display results
        print(f"Solution found! {len(solution)} moves.")
        print_moves(solution)
        
        # Apply all moves to show final board state
        final_grid = grid
        for move in solution:
            final_grid = PegSolitaireBoard.apply_move(final_grid, move)
        final_state = PegState(final_grid)
        
        print("\nFinal board:")
        print_board(final_state)
        print(f"Peg count: {PegSolitaireBoard.count_pegs(final_state.grid)}")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()