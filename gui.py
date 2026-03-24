# gui.py
"""
Pygame GUI for Peg Solitaire with two‑player variants.
- Single‑player: Solve (IDA* to one peg), Minimize (best‑first), Random Board.
- Two‑player: Human vs AI (MCTS, 5s per move, 0.5s delay) and Human vs Human (hints/solution).
- Click cells to toggle peg/empty (left) or validity (right/Ctrl+click).
- Space steps through single‑player solutions.
- Continuous evaluation bar (two‑player only) shows Player 1's win probability, updated live.
"""

import pygame
import sys
import threading
import time
import random
from board import PegSolitaireBoard
from state import PegState
from search import IDAStar
from minimize import BestFirstMinimize
from mcts import MCTS

# Constants
WIDTH, HEIGHT = 1000, 600
CELL_SIZE = 75
BOARD_SIZE = 7 * CELL_SIZE
OFFSET_X = (WIDTH - BOARD_SIZE) // 2
OFFSET_Y = (HEIGHT - BOARD_SIZE) // 2
BAR_WIDTH = 40
BAR_HEIGHT = 400
BAR_X = OFFSET_X + BOARD_SIZE + 20
BAR_Y = (HEIGHT - BAR_HEIGHT) // 2

COLORS = {
    'background': (240, 240, 240),
    'hole': (200, 200, 200),
    'peg': (50, 50, 150),
    'selected_peg': (255, 255, 100),
    'hint_peg': (255, 255, 100),
    'hint_landing': (100, 255, 100),
    'text': (0, 0, 0),
    'button': (100, 200, 100),
    'button_hover': (150, 250, 150),
    'button_active': (70, 130, 180),
    'button_computing': (200, 150, 100),
    'edge': (0, 0, 0),
    'error': (200, 50, 50),
    'input_bg': (255, 255, 255),
    'input_border': (0, 0, 0),
    'ai_turn': (200, 200, 100),
    'winner': (0, 200, 0),
    'eval_bar_p1': (0, 200, 0),
    'eval_bar_p2': (200, 0, 0)
}

class Gui:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Peg Solitaire")
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

        # Game mode
        self.mode = 'single'
        self.two_player_submode = 'vs_ai'
        self.two_player_turn = 0
        self.selected_cell = None
        self.winner = None
        self.win_probability = None
        self.ai_thinking = False

        # Hint / solution
        self.hint_from = None
        self.hint_to = None
        self.best_move_cache = None
        self.last_state_hash = None
        self.computing_hint = False
        self.computing_solution = False

        # Continuous evaluation
        self.win_prob_p1 = 0.5
        self.eval_thread_running = False
        self.eval_lock = threading.Lock()
        self.current_eval_state = None   # the state we are evaluating (for thread safety)

        # Game state
        self.grid = PegSolitaireBoard.initial_grid()
        self.state = PegState(self.grid, player=0)
        self.solution = None
        self.solution_index = 0
        self.solution_start_grid = None
        self.show_start = False
        self.searching = False
        self.search_thread = None
        self.nodes_expanded = 0
        self.search_time = 0
        self.timeout_msg = False
        self.best_pegs = None

        # Buttons
        self.button_twoplayer = pygame.Rect(WIDTH-150, 50, 120, 40)
        self.button_text_twoplayer = "Two-Player"
        self.button_vs_ai = pygame.Rect(WIDTH-150, 100, 120, 30)
        self.button_vs_human = pygame.Rect(WIDTH-150, 135, 120, 30)
        self.button_text_vs_ai = "Vs AI"
        self.button_text_vs_human = "Vs Human"
        self.button_solve = pygame.Rect(30, HEIGHT-70, 120, 40)
        self.button_text_solve = "Solve"
        self.button_minimize = pygame.Rect(WIDTH-150, HEIGHT-110, 120, 40)
        self.button_random = pygame.Rect(WIDTH-150, HEIGHT-160, 120, 40)
        self.button_text_minimize = "Minimize"
        self.button_text_random = "Random Board"
        self.button_hint = pygame.Rect(WIDTH-150, HEIGHT-110, 120, 40)
        self.button_solution = pygame.Rect(WIDTH-150, HEIGHT-160, 120, 40)
        self.button_text_hint = "Hint"
        self.button_text_solution = "Solution"

        # Input field
        self.input_rect = pygame.Rect(WIDTH-150, HEIGHT-40, 120, 30)
        self.input_active = False
        self.input_text = ""
        self.input_label = self.font.render("Pegs:", True, COLORS['text'])
        self.input_label_rect = self.input_label.get_rect(midright=(self.input_rect.left-10, self.input_rect.centery))

        # Search parameters
        self.max_search_time = 60
        self.max_minimize_time = 60
        self.mcts_time_limit = 5            # seconds per AI move
        self.eval_simulations_per_cycle = 50  # batch size for live evaluation
        self.eval_update_interval = 0.2       # seconds between bar updates
        self.ai_delay = 0.5

        # Start continuous evaluation thread (will only run when in two-player mode)
        self.start_evaluation_thread()

    # ----------------------------------------------------------------------
    # Continuous evaluation (live bar)
    # ----------------------------------------------------------------------
    def start_evaluation_thread(self):
        """Start the background MCTS evaluation thread."""
        if self.eval_thread_running:
            return
        self.eval_thread_running = True
        self.eval_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.eval_thread.start()

    def stop_evaluation_thread(self):
        """Stop the evaluation thread (called when leaving two‑player mode)."""
        self.eval_thread_running = False
        # Wait a bit for thread to finish
        if hasattr(self, 'eval_thread') and self.eval_thread.is_alive():
            time.sleep(0.1)

    def _evaluation_loop(self):
        """Background loop: run MCTS in small batches, update win probability."""
        while self.eval_thread_running:
            # Only evaluate when in two‑player mode and not searching for a solution
            if self.mode != 'two' or self.searching:
                time.sleep(0.1)
                continue

            # Get the current state snapshot (thread‑safe)
            with self.eval_lock:
                state = self.state
                # Avoid evaluating if the game is over (terminal)
                if state.is_terminal():
                    winner = 1 - state.player
                    self.win_prob_p1 = 1.0 if winner == 0 else 0.0
                    time.sleep(0.2)
                    continue

                # We need to run MCTS on this state. Since MCTS modifies its internal tree,
                # we'll create a fresh MCTS each time? That would lose progress. Better to keep a persistent MCTS?
                # But the state changes on each move, so we'd need to reset anyway.
                # We'll use a fresh MCTS each cycle for simplicity, but we can also persist across cycles
                # if the state hasn't changed. We'll store a persistent MCTS object per state.

                # For simplicity, we'll create a new MCTS each cycle and run a small number of simulations.
                # This is not optimal but keeps code simple.
                # We'll use a class variable to keep the MCTS across cycles if the state is the same.
                # Let's implement a simple cache: if the state hash hasn't changed, we continue the same MCTS.
                # But that's more complex. We'll just run a fixed number of simulations each cycle.

            # Run a batch of MCTS simulations
            mcts = MCTS(state, time_limit=999)  # time limit not used; we'll manually limit simulations
            for _ in range(self.eval_simulations_per_cycle):
                if not self.eval_thread_running or self.mode != 'two':
                    break
                node = mcts.select()
                if not node.state.is_terminal():
                    if not node.is_fully_expanded():
                        node = node.expand()
                    result = node.rollout()
                    node.backpropagate(result)

            # Compute win probability for Player 1 from the root
            if mcts.root.visits > 0:
                # The root's player is state.player.
                # win rate from the perspective of the root player.
                # We'll take the average of the children's win rates weighted by visits.
                # Simpler: the root's win rate is (wins/visits) of the child that is actually the best move? Not straightforward.
                # The root itself doesn't have a win rate; we need to compute the value for the player to move.
                # In MCTS, the root's value is the win rate of the best move (or the average of all children).
                # We'll use the win rate of the child with the most visits (the chosen move).
                best_child = max(mcts.root.children, key=lambda c: c.visits, default=None)
                if best_child and best_child.visits > 0:
                    win_rate = best_child.wins / best_child.visits
                else:
                    win_rate = 0.5
            else:
                win_rate = 0.5

            # Convert to Player 1's perspective
            with self.eval_lock:
                if state.player == 0:
                    p1_win = win_rate
                else:
                    p1_win = 1 - win_rate
                self.win_prob_p1 = p1_win

            # Wait before next batch
            time.sleep(self.eval_update_interval)

    def draw_evaluation_bar(self):
        """Draw the bar only in two‑player mode."""
        if self.mode != 'two':
            return
        # Background (red for P2)
        pygame.draw.rect(self.screen, COLORS['eval_bar_p2'], (BAR_X, BAR_Y, BAR_WIDTH, BAR_HEIGHT))
        split_height = int(self.win_prob_p1 * BAR_HEIGHT)
        if split_height > 0:
            green_rect = pygame.Rect(BAR_X, BAR_Y + BAR_HEIGHT - split_height, BAR_WIDTH, split_height)
            pygame.draw.rect(self.screen, COLORS['eval_bar_p1'], green_rect)
        pygame.draw.rect(self.screen, COLORS['edge'], (BAR_X, BAR_Y, BAR_WIDTH, BAR_HEIGHT), 2)
        percent = int(self.win_prob_p1 * 100)
        text = self.font.render(f"P1 {percent}%", True, COLORS['text'])
        text_rect = text.get_rect(center=(BAR_X + BAR_WIDTH//2, BAR_Y - 20))
        self.screen.blit(text, text_rect)

    # ----------------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------------
    def update_state(self, new_grid, player=None):
        if player is None:
            player = self.two_player_turn if self.mode == 'two' else 0
        self.grid = new_grid
        self.state = PegState(self.grid, player=player)
        if self.mode == 'two':
            self.selected_cell = None
            self.hint_from = None
            self.hint_to = None
            self.best_move_cache = None
            self.last_state_hash = None
            # No need to restart the evaluation thread; it will pick up the new state on next cycle

    def clear_solution_data(self):
        self.solution = None
        self.solution_index = 0
        self.solution_start_grid = None
        self.show_start = False
        self.best_pegs = None
        self.timeout_msg = False

    # ----------------------------------------------------------------------
    # Single‑player methods
    # ----------------------------------------------------------------------
    def generate_random_board(self, peg_count=None):
        grid = [[-1 for _ in range(7)] for _ in range(7)]
        valid_cells = [(r,c) for r in range(7) for c in range(7) if PegSolitaireBoard.is_valid(r,c)]
        total = len(valid_cells)
        if peg_count is not None and 1 <= peg_count <= total:
            chosen = random.sample(valid_cells, peg_count)
            for r,c in valid_cells:
                grid[r][c] = 1 if (r,c) in chosen else 0
        else:
            for r,c in valid_cells:
                grid[r][c] = 1 if random.random() < 0.5 else 0
        if PegSolitaireBoard.count_pegs(grid) == 0:
            r,c = random.choice(valid_cells)
            grid[r][c] = 1
        if PegSolitaireBoard.count_pegs(grid) == total:
            r,c = random.choice(valid_cells)
            grid[r][c] = 0
        return grid

    def toggle_cell(self, r, c):
        if self.searching:
            return
        if not PegSolitaireBoard.is_valid(r, c):
            return
        new_grid = [list(row) for row in self.state.grid]
        new_grid[r][c] = 0 if new_grid[r][c] == 1 else 1
        self.update_state(new_grid)
        self.clear_solution_data()

    def toggle_validity(self, r, c):
        if self.searching:
            return
        if not (0 <= r < 7 and 0 <= c < 7):
            return
        new_valid = PegSolitaireBoard.toggle_valid(r, c)
        new_grid = [list(row) for row in self.state.grid]
        new_grid[r][c] = 0 if new_valid else -1
        self.update_state(new_grid)
        self.clear_solution_data()

    # ----------------------------------------------------------------------
    # Two‑player (vs AI)
    # ----------------------------------------------------------------------
    def try_human_move(self, from_pos, to_pos):
        moves = self.state.get_moves()
        legal = any(move[0] == from_pos and move[1] == to_pos for move in moves)
        if legal:
            new_grid = PegSolitaireBoard.apply_move(self.state.grid, ((from_pos, to_pos)))
            self.two_player_turn = 1 - self.two_player_turn
            self.update_state(new_grid, player=self.two_player_turn)
            if self.state.is_terminal():
                self.winner = 1 - self.two_player_turn
            else:
                if self.two_player_submode == 'vs_ai' and self.two_player_turn == 1:
                    self.ai_thinking = True
                    threading.Thread(target=self.ai_delayed_move, daemon=True).start()
            return True
        return False

    def ai_delayed_move(self):
        time.sleep(self.ai_delay)
        self.ai_move_thread()

    def ai_move_thread(self):
        mcts = MCTS(self.state, time_limit=self.mcts_time_limit)
        best_move = mcts.search()
        for child in mcts.root.children:
            if child.move == best_move:
                self.win_probability = child.wins / child.visits if child.visits > 0 else 0.5
                break
        if best_move:
            new_grid = PegSolitaireBoard.apply_move(self.state.grid, best_move)
            self.two_player_turn = 1 - self.two_player_turn
            self.update_state(new_grid, player=self.two_player_turn)
            if self.state.is_terminal():
                self.winner = 1 - self.two_player_turn
        self.ai_thinking = False

    # ----------------------------------------------------------------------
    # Two‑player (vs human hints)
    # ----------------------------------------------------------------------
    def compute_best_move(self):
        state_hash = hash(self.state)
        if self.best_move_cache is not None and self.last_state_hash == state_hash:
            return self.best_move_cache
        mcts = MCTS(self.state, time_limit=self.mcts_time_limit)
        best_move = mcts.search()
        self.best_move_cache = best_move
        self.last_state_hash = state_hash
        return best_move

    def show_hint(self):
        if self.mode != 'two' or self.two_player_submode != 'vs_human' or self.winner is not None:
            return
        def hint_thread():
            self.computing_hint = True
            best_move = self.compute_best_move()
            if best_move:
                self.hint_from = best_move[0]
                self.hint_to = None
            self.computing_hint = False
        threading.Thread(target=hint_thread, daemon=True).start()

    def show_solution(self):
        if self.mode != 'two' or self.two_player_submode != 'vs_human' or self.winner is not None:
            return
        def solution_thread():
            self.computing_solution = True
            best_move = self.compute_best_move()
            if best_move:
                self.hint_from = best_move[0]
                self.hint_to = best_move[1]
            self.computing_solution = False
        threading.Thread(target=solution_thread, daemon=True).start()

    # ----------------------------------------------------------------------
    # Single‑player search threads
    # ----------------------------------------------------------------------
    def solve_classic_in_thread(self):
        start_state = PegState(self.grid, player=0)
        self.solution_start_grid = [row[:] for row in self.grid]
        last = [0]
        def on_expand():
            self.nodes_expanded = solver.nodes_expanded
            if self.nodes_expanded - last[0] >= 10000:
                last[0] = self.nodes_expanded
                print(f"Classic solve: {self.nodes_expanded} nodes...")
        solver = IDAStar(start_state, max_time=self.max_search_time, on_expand=on_expand)
        start = time.time()
        solution = solver.search()
        self.search_time = time.time() - start
        self.nodes_expanded = solver.nodes_expanded
        if solution:
            self.solution = solution
            self.solution_index = 0
            self.show_start = True
            self.best_pegs = None
            self.timeout_msg = False
            print(f"Solution found! {len(solution)} moves.")
        else:
            self.solution = None
            self.solution_start_grid = None
            self.show_start = False
            self.timeout_msg = True
        self.searching = False
        self.button_text_solve = "Solve"
        self.button_text_minimize = "Minimize"
        self.button_text_twoplayer = "Two-Player"
        self.button_text_random = "Random Board"

    def solve_minimize_in_thread(self):
        start_state = PegState(self.grid, player=0)
        self.solution_start_grid = [row[:] for row in self.grid]
        last = [0]
        def on_expand():
            self.nodes_expanded = solver.nodes_expanded
            if self.nodes_expanded - last[0] >= 10000:
                last[0] = self.nodes_expanded
                print(f"Minimize: {self.nodes_expanded} nodes...")
        solver = BestFirstMinimize(start_state, max_time=self.max_minimize_time, on_expand=on_expand)
        start = time.time()
        best_pegs, moves = solver.search()
        self.search_time = time.time() - start
        self.nodes_expanded = solver.nodes_expanded
        self.best_pegs = best_pegs
        if moves:
            self.solution = moves
            self.solution_index = 0
            self.show_start = True
            final = self.grid
            for move in moves:
                final = PegSolitaireBoard.apply_move(final, move)
            self.grid = final
            self.state = PegState(final, player=0)
            print(f"Minimize: found {best_pegs} pegs, {len(moves)} moves.")
        else:
            self.solution = None
            self.solution_start_grid = None
            self.show_start = False
        self.searching = False
        self.button_text_solve = "Solve"
        self.button_text_minimize = "Minimize"
        self.button_text_twoplayer = "Two-Player"
        self.button_text_random = "Random Board"

    def random_board_in_thread(self):
        peg_count = None
        if self.input_text.strip().isdigit():
            peg_count = int(self.input_text.strip())
            if not (1 <= peg_count <= 33):
                peg_count = None
        new_grid = self.generate_random_board(peg_count)
        self.update_state(new_grid, player=0)
        self.clear_solution_data()
        self.searching = False
        self.button_text_solve = "Solve"
        self.button_text_minimize = "Minimize"
        self.button_text_twoplayer = "Two-Player"
        self.button_text_random = "Random Board"

    # ----------------------------------------------------------------------
    # Drawing
    # ----------------------------------------------------------------------
    def draw_board(self):
        self.screen.fill(COLORS['background'])
        for r in range(7):
            for c in range(7):
                if PegSolitaireBoard.is_valid(r, c):
                    x = OFFSET_X + c * CELL_SIZE
                    y = OFFSET_Y + r * CELL_SIZE
                    pygame.draw.rect(self.screen, COLORS['hole'], (x, y, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(self.screen, COLORS['edge'], (x, y, CELL_SIZE, CELL_SIZE), 2)
                    if self.state.grid[r][c] == 1:
                        color = COLORS['peg']
                        if self.mode == 'two' and self.selected_cell == (r, c):
                            color = COLORS['selected_peg']
                        elif self.mode == 'two' and self.hint_from == (r, c):
                            color = COLORS['hint_peg']
                        pygame.draw.circle(self.screen, color, (x+CELL_SIZE//2, y+CELL_SIZE//2), CELL_SIZE//3)
                    if self.mode == 'two' and self.hint_to == (r, c):
                        pygame.draw.rect(self.screen, COLORS['hint_landing'], (x, y, CELL_SIZE, CELL_SIZE), 3)

    def draw_stats(self):
        y = 20
        text = self.font.render(f"Mode: {'Two-Player' if self.mode=='two' else 'Single'}", True, COLORS['text'])
        self.screen.blit(text, (20, y)); y += 30
        text = self.font.render(f"Pegs: {PegSolitaireBoard.count_pegs(self.state.grid)}", True, COLORS['text'])
        self.screen.blit(text, (20, y)); y += 30
        if self.mode == 'two':
            if self.winner is not None:
                text = self.font.render(f"Winner: {'Human' if self.winner==0 else 'AI'}!", True, COLORS['winner'])
                self.screen.blit(text, (20, y)); y += 30
            else:
                if self.two_player_submode == 'vs_ai':
                    turn_text = "Player 1 (Human)" if self.two_player_turn == 0 else "Player 2 (AI)"
                    text = self.font.render(f"Turn: {turn_text}", True, COLORS['ai_turn'] if self.two_player_turn==1 else COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 30
                else:
                    turn_text = "Player 1" if self.two_player_turn == 0 else "Player 2"
                    text = self.font.render(f"Turn: {turn_text}", True, COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 30
                    text = self.font.render("Use Hint/Solution buttons", True, COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 30
        else:
            text = self.font.render(f"Nodes expanded: {self.nodes_expanded}", True, COLORS['text'])
            self.screen.blit(text, (20, y)); y += 30
            text = self.font.render(f"Search time: {self.search_time:.2f}s", True, COLORS['text'])
            self.screen.blit(text, (20, y)); y += 30
            if self.solution:
                text = self.font.render(f"Solution length: {len(self.solution)}", True, COLORS['text'])
                self.screen.blit(text, (20, y)); y += 30
                if self.solution_index < len(self.solution) or self.show_start:
                    step_text = "Step: 0 (press SPACE to start)" if self.show_start else f"Step: {self.solution_index} / {len(self.solution)}"
                    text = self.font.render(step_text, True, COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 30
            if self.best_pegs is not None:
                text = self.font.render(f"Best pegs: {self.best_pegs}", True, COLORS['text'])
                self.screen.blit(text, (20, y)); y += 30
            if self.timeout_msg:
                text = self.font.render("Search timed out", True, COLORS['error'])
                self.screen.blit(text, (20, y)); y += 30

    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        # Two-Player
        color = COLORS['button_hover'] if self.button_twoplayer.collidepoint(mouse_pos) else COLORS['button']
        pygame.draw.rect(self.screen, color, self.button_twoplayer)
        surf = self.font.render(self.button_text_twoplayer, True, COLORS['text'])
        self.screen.blit(surf, surf.get_rect(center=self.button_twoplayer.center))
        if self.mode == 'two':
            # Vs AI
            base = COLORS['button_active'] if self.two_player_submode == 'vs_ai' else COLORS['button']
            color = COLORS['button_hover'] if self.button_vs_ai.collidepoint(mouse_pos) else base
            pygame.draw.rect(self.screen, color, self.button_vs_ai)
            surf = self.font.render(self.button_text_vs_ai, True, COLORS['text'])
            self.screen.blit(surf, surf.get_rect(center=self.button_vs_ai.center))
            # Vs Human
            base = COLORS['button_active'] if self.two_player_submode == 'vs_human' else COLORS['button']
            color = COLORS['button_hover'] if self.button_vs_human.collidepoint(mouse_pos) else base
            pygame.draw.rect(self.screen, color, self.button_vs_human)
            surf = self.font.render(self.button_text_vs_human, True, COLORS['text'])
            self.screen.blit(surf, surf.get_rect(center=self.button_vs_human.center))
        # Solve
        color = COLORS['button_hover'] if self.button_solve.collidepoint(mouse_pos) else COLORS['button']
        pygame.draw.rect(self.screen, color, self.button_solve)
        surf = self.font.render(self.button_text_solve, True, COLORS['text'])
        self.screen.blit(surf, surf.get_rect(center=self.button_solve.center))
        # Minimize (only in single)
        if self.mode == 'single':
            color = COLORS['button_hover'] if self.button_minimize.collidepoint(mouse_pos) else COLORS['button']
            pygame.draw.rect(self.screen, color, self.button_minimize)
            surf = self.font.render(self.button_text_minimize, True, COLORS['text'])
            self.screen.blit(surf, surf.get_rect(center=self.button_minimize.center))
        # Random (only in single)
        if self.mode == 'single':
            color = COLORS['button_hover'] if self.button_random.collidepoint(mouse_pos) else COLORS['button']
            pygame.draw.rect(self.screen, color, self.button_random)
            surf = self.font.render(self.button_text_random, True, COLORS['text'])
            self.screen.blit(surf, surf.get_rect(center=self.button_random.center))
        # Hint/Solution (only in vs_human)
        if self.mode == 'two' and self.two_player_submode == 'vs_human':
            # Hint
            base = COLORS['button_computing'] if self.computing_hint else COLORS['button']
            color = COLORS['button_hover'] if self.button_hint.collidepoint(mouse_pos) else base
            pygame.draw.rect(self.screen, color, self.button_hint)
            surf = self.font.render(self.button_text_hint, True, COLORS['text'])
            self.screen.blit(surf, surf.get_rect(center=self.button_hint.center))
            # Solution
            base = COLORS['button_computing'] if self.computing_solution else COLORS['button']
            color = COLORS['button_hover'] if self.button_solution.collidepoint(mouse_pos) else base
            pygame.draw.rect(self.screen, color, self.button_solution)
            surf = self.font.render(self.button_text_solution, True, COLORS['text'])
            self.screen.blit(surf, surf.get_rect(center=self.button_solution.center))
        # Input field
        pygame.draw.rect(self.screen, COLORS['input_bg'], self.input_rect)
        pygame.draw.rect(self.screen, COLORS['input_border'], self.input_rect, 2)
        input_surf = self.font.render(self.input_text, True, COLORS['text'])
        self.screen.blit(input_surf, (self.input_rect.x+5, self.input_rect.y+7))
        self.screen.blit(self.input_label, self.input_label_rect)

    # ----------------------------------------------------------------------
    # Event handling
    # ----------------------------------------------------------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Input field
                if self.input_rect.collidepoint(event.pos):
                    self.input_active = True
                else:
                    self.input_active = False
                # Two-Player toggle
                if self.button_twoplayer.collidepoint(event.pos) and not self.searching:
                    self.mode = 'two'
                    PegSolitaireBoard.reset_mask()
                    new_grid = PegSolitaireBoard.initial_grid()
                    self.two_player_turn = 0
                    self.winner = None
                    self.win_probability = None
                    self.selected_cell = None
                    self.hint_from = None
                    self.hint_to = None
                    self.update_state(new_grid, player=self.two_player_turn)
                    self.clear_solution_data()
                    self.two_player_submode = 'vs_ai'
                    if not self.ai_thinking and self.two_player_turn == 1 and not self.state.is_terminal():
                        self.ai_thinking = True
                        threading.Thread(target=self.ai_delayed_move, daemon=True).start()
                    continue
                # Submode buttons
                if self.mode == 'two':
                    if self.button_vs_ai.collidepoint(event.pos):
                        self.two_player_submode = 'vs_ai'
                        self.selected_cell = None
                        self.hint_from = None
                        self.hint_to = None
                        self.winner = None
                        self.two_player_turn = 0
                        self.update_state(self.grid, player=self.two_player_turn)
                        if not self.ai_thinking and self.two_player_turn == 1 and not self.state.is_terminal():
                            self.ai_thinking = True
                            threading.Thread(target=self.ai_delayed_move, daemon=True).start()
                        continue
                    if self.button_vs_human.collidepoint(event.pos):
                        self.two_player_submode = 'vs_human'
                        self.selected_cell = None
                        self.hint_from = None
                        self.hint_to = None
                        self.winner = None
                        self.two_player_turn = 0
                        self.update_state(self.grid, player=0)
                        continue
                # Solve
                if self.button_solve.collidepoint(event.pos) and not self.searching:
                    self.mode = 'single'
                    PegSolitaireBoard.reset_mask()
                    new_grid = PegSolitaireBoard.initial_grid()
                    self.update_state(new_grid, player=0)
                    self.clear_solution_data()
                    self.searching = True
                    self.button_text_solve = "Searching..."
                    self.button_text_minimize = "Minimize"
                    self.button_text_twoplayer = "Two-Player"
                    self.button_text_random = "Random Board"
                    self.search_thread = threading.Thread(target=self.solve_classic_in_thread)
                    self.search_thread.daemon = True
                    self.search_thread.start()
                    continue
                # Minimize
                if self.mode == 'single' and self.button_minimize.collidepoint(event.pos) and not self.searching:
                    self.searching = True
                    self.button_text_solve = "Solve"
                    self.button_text_minimize = "Searching..."
                    self.button_text_twoplayer = "Two-Player"
                    self.button_text_random = "Random Board"
                    self.search_thread = threading.Thread(target=self.solve_minimize_in_thread)
                    self.search_thread.daemon = True
                    self.search_thread.start()
                    continue
                # Random
                if self.mode == 'single' and self.button_random.collidepoint(event.pos) and not self.searching:
                    self.searching = True
                    self.button_text_solve = "Solve"
                    self.button_text_minimize = "Minimize"
                    self.button_text_twoplayer = "Two-Player"
                    self.button_text_random = "Generating..."
                    self.search_thread = threading.Thread(target=self.random_board_in_thread)
                    self.search_thread.daemon = True
                    self.search_thread.start()
                    continue
                # Hint / Solution
                if self.mode == 'two' and self.two_player_submode == 'vs_human':
                    if self.button_hint.collidepoint(event.pos) and not self.searching and self.winner is None:
                        self.show_hint()
                        continue
                    if self.button_solution.collidepoint(event.pos) and not self.searching and self.winner is None:
                        self.show_solution()
                        continue
                # Cell clicks
                if self.mode == 'two' and not self.searching:
                    if self.winner is not None:
                        pass
                    elif self.two_player_submode == 'vs_ai' and not self.ai_thinking:
                        if self.two_player_turn == 0:
                            cell = None
                            for r in range(7):
                                for c in range(7):
                                    x = OFFSET_X + c * CELL_SIZE
                                    y = OFFSET_Y + r * CELL_SIZE
                                    if x <= event.pos[0] <= x+CELL_SIZE and y <= event.pos[1] <= y+CELL_SIZE:
                                        cell = (r,c)
                                        break
                                if cell:
                                    break
                            if cell:
                                if self.selected_cell is None:
                                    if self.state.grid[cell[0]][cell[1]] == 1:
                                        self.selected_cell = cell
                                else:
                                    if self.try_human_move(self.selected_cell, cell):
                                        self.selected_cell = None
                                    else:
                                        self.selected_cell = None
                    elif self.two_player_submode == 'vs_human':
                        cell = None
                        for r in range(7):
                            for c in range(7):
                                x = OFFSET_X + c * CELL_SIZE
                                y = OFFSET_Y + r * CELL_SIZE
                                if x <= event.pos[0] <= x+CELL_SIZE and y <= event.pos[1] <= y+CELL_SIZE:
                                    cell = (r,c)
                                    break
                            if cell:
                                break
                        if cell:
                            if self.selected_cell is None:
                                if self.state.grid[cell[0]][cell[1]] == 1:
                                    self.selected_cell = cell
                            else:
                                moves = self.state.get_moves()
                                legal = any(move[0] == self.selected_cell and move[1] == cell for move in moves)
                                if legal:
                                    new_grid = PegSolitaireBoard.apply_move(self.state.grid, ((self.selected_cell, cell)))
                                    self.two_player_turn = 1 - self.two_player_turn
                                    self.update_state(new_grid, player=self.two_player_turn)
                                    if self.state.is_terminal():
                                        self.winner = 1 - self.two_player_turn
                                    self.selected_cell = None
                                    self.hint_from = None
                                    self.hint_to = None
                                else:
                                    self.selected_cell = None
                else:
                    # Single‑player: left‑click toggles peg/empty, right‑click toggles validity
                    cell = None
                    for r in range(7):
                        for c in range(7):
                            x = OFFSET_X + c * CELL_SIZE
                            y = OFFSET_Y + r * CELL_SIZE
                            if x <= event.pos[0] <= x+CELL_SIZE and y <= event.pos[1] <= y+CELL_SIZE:
                                cell = (r,c)
                                break
                        if cell:
                            break
                    if cell and not self.searching:
                        if event.button == 3 or (event.button == 1 and pygame.key.get_mods() & pygame.KMOD_CTRL):
                            self.toggle_validity(*cell)
                        elif event.button == 1:
                            self.toggle_cell(*cell)
            if event.type == pygame.KEYDOWN:
                if self.input_active:
                    if event.key == pygame.K_RETURN:
                        self.input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    else:
                        if event.unicode.isdigit():
                            self.input_text += event.unicode
                elif self.mode == 'single' and event.key == pygame.K_SPACE and self.solution:
                    if self.show_start:
                        if self.solution_start_grid is not None:
                            new_grid = [row[:] for row in self.solution_start_grid]
                            self.update_state(new_grid, player=0)
                        self.show_start = False
                    else:
                        if self.solution_index < len(self.solution):
                            move = self.solution[self.solution_index]
                            new_grid = PegSolitaireBoard.apply_move(self.state.grid, move)
                            self.update_state(new_grid, player=0)
                            self.solution_index += 1
        return True

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw_board()
            self.draw_stats()
            self.draw_buttons()
            self.draw_evaluation_bar()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = Gui()
    gui.run()