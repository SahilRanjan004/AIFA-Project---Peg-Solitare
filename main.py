# main.py
"""
Command‑line interface for the Peg Solitaire solver.
"""

from board import PegSolitaireBoard
from state import PegState
from search import IDAStar

def print_board(state):
    print(PegSolitaireBoard.grid_to_string(state.grid))

def print_moves(moves):
    for i, move in enumerate(moves):
        (r1,c1), (r2,c2) = move
        print(f"{i+1}: ({r1},{c1}) -> ({r2},{c2})")

def main():
    grid = PegSolitaireBoard.initial_grid()
    initial_state = PegState(grid)

    print("Initial board:")
    print_board(initial_state)
    print(f"Peg count: {PegSolitaireBoard.count_pegs(initial_state.grid)}")
    print("\nSolving...")

    solver = IDAStar(initial_state)
    solution = solver.search()

    if solution:
        print(f"Solution found! {len(solution)} moves.")
        print_moves(solution)
        # Show final board
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