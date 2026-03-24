# heuristic.py
"""
Heuristic functions.
"""

from board import PegSolitaireBoard

def remaining_pegs_heuristic(state):
    return PegSolitaireBoard.count_pegs(state.grid) - 1

def get_heuristic(state):
    return remaining_pegs_heuristic(state)