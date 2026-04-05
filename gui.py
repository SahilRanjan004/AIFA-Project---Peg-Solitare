"""
Pygame GUI for Peg Solitaire with two-player variants.

Features:
- Single-player: Solve (IDA* to one peg), Minimize (best-first), Random Board
- Two-player: Human vs AI (MCTS) and Human vs Human
- Fullscreen mode with MCTS tree visualization
- Move heatmap (single-player only)
- Dark/Light theme switcher
- Player name editing, win streaks, move history
- Legal moves highlighting, tooltips, animations
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
from tree_viz import MCTSTreeVisualizer

# ============================================================================
# CONSTANTS
# ============================================================================

# Screen dimensions (will be updated dynamically in fullscreen)
WIDTH, HEIGHT = 1000, 600
CELL_SIZE = 75
BOARD_SIZE = 7 * CELL_SIZE
OFFSET_X = (WIDTH - BOARD_SIZE) // 2
OFFSET_Y = (HEIGHT - BOARD_SIZE) // 2
BAR_WIDTH = 40
BAR_HEIGHT = 400
BAR_X = OFFSET_X + BOARD_SIZE + 20
BAR_Y = (HEIGHT - BAR_HEIGHT) // 2
ANIMATION_SPEED = 10  # frames for peg movement animation

# Theme colors
THEMES = {
    'light': {
        'background': (240, 240, 240),
        'hole': (200, 200, 200),
        'peg': (50, 50, 150),
        'selected_peg': (255, 255, 100),
        'hint_peg': (255, 255, 100),
        'hint_landing': (100, 255, 100),
        'legal_move': (100, 200, 100),
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
        'eval_bar_p2': (200, 0, 0),
        'tooltip_bg': (40, 40, 40),
        'streak': (255, 200, 0)
    },
    'dark': {
        'background': (30, 30, 30),
        'hole': (60, 60, 60),
        'peg': (100, 150, 200),
        'selected_peg': (255, 200, 100),
        'hint_peg': (255, 200, 100),
        'hint_landing': (100, 200, 100),
        'legal_move': (80, 180, 80),
        'text': (220, 220, 220),
        'button': (70, 130, 180),
        'button_hover': (100, 160, 210),
        'button_active': (50, 100, 150),
        'button_computing': (180, 130, 80),
        'edge': (150, 150, 150),
        'error': (200, 80, 80),
        'input_bg': (80, 80, 80),
        'input_border': (150, 150, 150),
        'ai_turn': (180, 180, 100),
        'winner': (0, 200, 0),
        'eval_bar_p1': (0, 200, 0),
        'eval_bar_p2': (200, 0, 0),
        'tooltip_bg': (80, 80, 80),
        'streak': (255, 200, 0)
    }
}

# Start with light theme
COLORS = THEMES['light'].copy()

# Tooltip texts for buttons
TOOLTIPS = {
    "One Player": "Switch to single-player mode",
    "Two Player": "Switch to two-player mode",
    "Vs AI": "Play against AI opponent\nUses MCTS algorithm",
    "Vs Human": "Two players take turns\nUse Hint/Solution for help",
    "Solve": "Find optimal solution to reach 1 peg\nUses IDA* algorithm",
    "Minimize": "Find best terminal state with fewest pegs\nUses best-first search",
    "Random Board": "Generate random board\nEnter peg count or leave empty for 50% fill",
    "Undo": "Undo last move\nWorks for both players",
    "Restart": "Restart current game\nResets win streaks",
    "New Game": "Start a new game\nPreserves win streaks",
    "Hint": "Highlight best move peg\nShows where to start",
    "Solution": "Show the full best move\nHighlights peg and destination",
    "Theme": "Switch between light and dark theme",
    "Fullscreen": "Toggle fullscreen mode\nShows MCTS tree visualization",
    "Heatmap": "Toggle move heatmap overlay (Single-player only)\nShows position usage frequency"
}


# ============================================================================
# MAIN GUI CLASS
# ============================================================================

class Gui:
    """Main GUI application for Peg Solitaire."""
    
    def __init__(self):
        """Initialize pygame, screen, fonts, game state, and UI components."""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Peg Solitaire")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()
        
        # Window management
        self.windowed_size = (WIDTH, HEIGHT)
        self.fullscreen = False
        self.screen_width = WIDTH
        self.screen_height = HEIGHT

        # Game mode
        self.mode = 'single'           # 'single' or 'two'
        self.two_player_submode = 'vs_ai'  # 'vs_ai' or 'vs_human'
        self.two_player_turn = 0       # 0 = Player 1, 1 = Player 2/AI
        self.selected_cell = None
        self.winner = None
        self.win_probability = None
        self.ai_thinking = False
        
        # Animation
        self.animating = False
        self.solution_animating = False
        self.anim_from = None
        self.anim_to = None
        self.anim_progress = 0
        self.anim_start_pos = None
        self.anim_end_pos = None
        self.anim_peg_grid = None
        self.pending_move = None
        
        # Legal moves
        self.legal_moves = []
        self.show_legal_moves = True
        
        # Player names
        self.player1_name = "Player 1"
        self.player2_name = "Player 2"
        self.ai_name = "AI"
        
        self.button_text_edit_p1 = "Edit P1"
        self.button_text_edit_p2 = "Edit P2"
        self.name_input_rect = None
        self.name_input_active = False
        self.name_input_text = ""
        self.editing_player = None
        
        # Win/loss streaks
        self.player1_streak = 0
        self.player2_streak = 0
        
        # Tooltip
        self.tooltip_text = None
        self.tooltip_rect = None
        self.tooltip_timer = 0

        # Hint/solution
        self.hint_from = None
        self.hint_to = None
        self.best_move_cache = None
        self.last_state_hash = None
        self.computing_hint = False
        self.computing_solution = False

        # Continuous evaluation (two-player win probability)
        self.win_prob_p1 = 0.5
        self.eval_thread_running = False
        self.eval_lock = threading.Lock()
        self.persistent_mcts = None 
        self.current_eval_state_hash = None

        # Move history
        self.move_history = []
        self.history_states = []
        self.history_count = 0
        self.max_history = 30
        self.last_move = None

        # Statistics
        self.move_count = 0
        self.game_start_time = None
        self.game_end_time = None
        self.ai_thinking_time = 0
        self.simulations = 0
        
        # Theme
        self.current_theme = 'light'
        
        # MCTS tree visualization
        self.tree_visualizer = None
        self.show_tree_viz = False
        self.current_mcts_root = None
        self.highlighted_node = None
        self.waiting_for_tree_click = False
        
        # Heatmap (single-player only)
        self.heatmap_enabled = False
        self.move_heatmap = {}
        
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

        # UI Components
        self.create_buttons()
        
        # Input field for random board peg count
        self.input_rect = pygame.Rect(WIDTH-150, HEIGHT-220, 100, 30)
        self.input_active = False
        self.input_text = ""
        self.input_label = self.font.render("Pegs:", True, COLORS['text'])
        self.input_label_rect = self.input_label.get_rect(midright=(self.input_rect.left-5, self.input_rect.centery))

        # Search parameters
        self.max_search_time = 60
        self.max_minimize_time = 60
        self.mcts_time_limit = 5
        self.eval_update_interval = 0.5
        self.ai_delay = 0.5

        # Start background evaluation thread
        self.start_evaluation_thread()
        self.update_legal_moves()

    # ------------------------------------------------------------------------
    # UI Setup - Button Creation and Positioning
    # ------------------------------------------------------------------------
    
    def create_buttons(self):
        """Create all UI button rectangles."""
        # Mode buttons (top right)
        self.button_oneplayer = pygame.Rect(self.screen_width-150, 50, 120, 40)
        self.button_twoplayer = pygame.Rect(self.screen_width-150, 100, 120, 40)
        self.button_text_oneplayer = "One Player"
        self.button_text_twoplayer = "Two Player"
        
        # Top left buttons
        self.button_fullscreen = pygame.Rect(30, 20, 120, 40)
        self.button_text_fullscreen = "Fullscreen"
        self.button_heatmap = pygame.Rect(160, 20, 120, 40)
        self.button_text_heatmap = "Heatmap OFF"
        self.button_theme = pygame.Rect(30, 70, 120, 40)
        self.button_text_theme = "Dark Theme"
        
        # Two-player submode toggles
        self.button_vs_ai = pygame.Rect(self.screen_width-150, 150, 120, 30)
        self.button_vs_human = pygame.Rect(self.screen_width-150, 185, 120, 30)
        self.button_text_vs_ai = "Vs AI"
        self.button_text_vs_human = "Vs Human"
        
        # Single-player action buttons (bottom left)
        self.button_solve = pygame.Rect(30, self.screen_height-70, 120, 40)
        self.button_text_solve = "Solve"
        self.button_minimize = pygame.Rect(30, self.screen_height-120, 120, 40)
        self.button_text_minimize = "Minimize"
        self.button_random = pygame.Rect(30, self.screen_height-170, 120, 40)
        self.button_text_random = "Random Board"
        
        # Common buttons (bottom right)
        self.button_undo = pygame.Rect(self.screen_width-150, self.screen_height-70, 120, 40)
        self.button_restart = pygame.Rect(self.screen_width-150, self.screen_height-120, 120, 40)
        self.button_newgame = pygame.Rect(self.screen_width-150, self.screen_height-170, 120, 40)
        self.button_text_undo = "Undo"
        self.button_text_restart = "Restart"
        self.button_text_newgame = "New Game"
        
        # Hint/Solution buttons
        self.button_hint = pygame.Rect(self.screen_width-150, self.screen_height-270, 120, 40)
        self.button_solution = pygame.Rect(self.screen_width-150, self.screen_height-320, 120, 40)
        self.button_text_hint = "Hint"
        self.button_text_solution = "Solution"
        self.button_ai_hint = pygame.Rect(self.screen_width-150, self.screen_height-270, 120, 40)
        self.button_ai_solution = pygame.Rect(self.screen_width-150, self.screen_height-320, 120, 40)
        self.button_text_ai_hint = "Hint"
        self.button_text_ai_solution = "Solution"

    def update_offsets(self):
        """Update board positioning and button positions after screen resize."""
        self.screen_width, self.screen_height = self.screen.get_size()
        global OFFSET_X, OFFSET_Y, BAR_X, BAR_Y, WIDTH, HEIGHT
        WIDTH, HEIGHT = self.screen_width, self.screen_height
        OFFSET_X = (self.screen_width - BOARD_SIZE) // 2
        OFFSET_Y = (self.screen_height - BOARD_SIZE) // 2
        BAR_X = OFFSET_X + BOARD_SIZE + 20
        BAR_Y = (self.screen_height - BAR_HEIGHT) // 2
        
        # Update all button positions
        self.button_oneplayer = pygame.Rect(self.screen_width-150, 50, 120, 40)
        self.button_twoplayer = pygame.Rect(self.screen_width-150, 100, 120, 40)
        self.button_fullscreen = pygame.Rect(30, 20, 120, 40)
        self.button_heatmap = pygame.Rect(160, 20, 120, 40)
        self.button_theme = pygame.Rect(30, 70, 120, 40)
        self.button_vs_ai = pygame.Rect(self.screen_width-150, 150, 120, 30)
        self.button_vs_human = pygame.Rect(self.screen_width-150, 185, 120, 30)
        self.button_solve = pygame.Rect(30, self.screen_height-70, 120, 40)
        self.button_minimize = pygame.Rect(30, self.screen_height-120, 120, 40)
        self.button_random = pygame.Rect(30, self.screen_height-170, 120, 40)
        self.button_undo = pygame.Rect(self.screen_width-150, self.screen_height-70, 120, 40)
        self.button_restart = pygame.Rect(self.screen_width-150, self.screen_height-120, 120, 40)
        self.button_newgame = pygame.Rect(self.screen_width-150, self.screen_height-170, 120, 40)
        self.button_hint = pygame.Rect(self.screen_width-150, self.screen_height-270, 120, 40)
        self.button_solution = pygame.Rect(self.screen_width-150, self.screen_height-320, 120, 40)
        self.button_ai_hint = pygame.Rect(self.screen_width-150, self.screen_height-270, 120, 40)
        self.button_ai_solution = pygame.Rect(self.screen_width-150, self.screen_height-320, 120, 40)

    # ------------------------------------------------------------------------
    # Thread Management
    # ------------------------------------------------------------------------
    
    def start_evaluation_thread(self):
        """Start background MCTS evaluation for win probability bar."""
        if self.eval_thread_running:
            return
        self.eval_thread_running = True
        self.eval_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.eval_thread.start()

    def stop_evaluation_thread(self):
        """Stop background evaluation thread."""
        self.eval_thread_running = False
        if hasattr(self, 'eval_thread') and self.eval_thread.is_alive():
            self.eval_thread.join(timeout=0.5)

    # ------------------------------------------------------------------------
    # Display Modes
    # ------------------------------------------------------------------------
    
    def toggle_fullscreen(self):
        """Toggle between windowed and fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            display_info = pygame.display.Info()
            self.screen = pygame.display.set_mode((display_info.current_w, display_info.current_h), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size)
        self.update_offsets()
        self.tree_visualizer = MCTSTreeVisualizer(self.screen, self.font, self.small_font)
        self.clear_tree_viz()

    def switch_theme(self):
        """Switch between light and dark color themes."""
        if self.current_theme == 'light':
            self.current_theme = 'dark'
            self.button_text_theme = "Light Theme"
        else:
            self.current_theme = 'light'
            self.button_text_theme = "Dark Theme"
        
        for key, value in THEMES[self.current_theme].items():
            COLORS[key] = value
        self.input_label = self.font.render("Pegs:", True, COLORS['text'])

    # ------------------------------------------------------------------------
    # MCTS Tree Visualization
    # ------------------------------------------------------------------------
    
    def capture_mcts_tree(self, mcts):
        """Capture MCTS tree for visualization in fullscreen mode."""
        if self.fullscreen and mcts and mcts.root:
            self.current_mcts_root = mcts.root
            self.show_tree_viz = True
            if self.tree_visualizer:
                self.tree_visualizer.build_tree(mcts.root)

    def clear_tree_viz(self):
        """Clear the tree visualization."""
        self.show_tree_viz = False
        self.current_mcts_root = None
        self.waiting_for_tree_click = False

    # ------------------------------------------------------------------------
    # Game State Management
    # ------------------------------------------------------------------------
    
    def update_legal_moves(self):
        """Update the list of legal moves for the current player."""
        self.legal_moves = self.state.get_moves()

    def get_legal_landing_cells(self, from_pos):
        """Get all legal landing cells for a given peg."""
        return [move[1] for move in self.legal_moves if move[0] == from_pos]

    def update_state(self, new_grid, player=None):
        """
        Update the game state with a new board.
        
        Args:
            new_grid: New board configuration
            player: Player to move (None = auto-detect from mode)
        """
        if player is None:
            player = self.two_player_turn if self.mode == 'two' else 0
        
        # Record move in history
        if self.mode == 'single' and self.last_move is not None:
            self.move_history.append((self.last_move[0], self.last_move[1], 0))
            self.history_states.append([row[:] for row in self.grid])
            if len(self.move_history) > self.max_history:
                self.move_history.pop(0)
                self.history_states.pop(0)
            self.move_count += 1
            self.last_move = None
        elif self.mode == 'two' and self.last_move is not None:
            self.move_history.append((self.last_move[0], self.last_move[1], 1 - player))
            self.history_states.append([row[:] for row in self.grid])
            if len(self.move_history) > self.max_history:
                self.move_history.pop(0)
                self.history_states.pop(0)
            self.move_count += 1
            self.last_move = None
        
        self.grid = new_grid
        self.state = PegState(self.grid, player=player)
        
        if self.mode == 'two' and self.state.is_terminal() and self.game_end_time is None:
            self.game_end_time = time.time()
        
        if self.mode == 'two':
            self.selected_cell = None
            self.hint_from = None
            self.hint_to = None
            self.best_move_cache = None
            self.last_state_hash = None
            with self.eval_lock:
                self.persistent_mcts = None
                self.current_eval_state_hash = None
        
        self.clear_tree_viz()
        self.update_legal_moves()

    def clear_solution_data(self):
        """Clear solution and heatmap data."""
        self.solution = None
        self.solution_index = 0
        self.solution_start_grid = None
        self.show_start = False
        self.best_pegs = None
        self.timeout_msg = False
        
        if self.mode == 'single':
            self.move_heatmap = {}
            self.heatmap_enabled = False
            self.button_text_heatmap = "Heatmap OFF"

    # ------------------------------------------------------------------------
    # Heatmap (Single-Player Only)
    # ------------------------------------------------------------------------
    
    def track_move_for_heatmap(self, move):
        """Record a move for heatmap visualization."""
        from_pos, to_pos = move
        r_mid = (from_pos[0] + to_pos[0]) // 2
        c_mid = (from_pos[1] + to_pos[1]) // 2
        jumped_pos = (r_mid, c_mid)
        
        self.move_heatmap[from_pos] = self.move_heatmap.get(from_pos, 0) + 1
        self.move_heatmap[to_pos] = self.move_heatmap.get(to_pos, 0) + 1
        self.move_heatmap[jumped_pos] = self.move_heatmap.get(jumped_pos, 0) + 1

    # ------------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------------
    
    def start_animation(self, from_pos, to_pos):
        """Start peg movement animation."""
        self.animating = True
        self.anim_from = from_pos
        self.anim_to = to_pos
        self.anim_progress = 0
        self.anim_start_pos = (OFFSET_X + from_pos[1] * CELL_SIZE + CELL_SIZE//2,
                                OFFSET_Y + from_pos[0] * CELL_SIZE + CELL_SIZE//2)
        self.anim_end_pos = (OFFSET_X + to_pos[1] * CELL_SIZE + CELL_SIZE//2,
                              OFFSET_Y + to_pos[0] * CELL_SIZE + CELL_SIZE//2)

    def update_animation(self):
        """Update animation progress and apply move when complete."""
        if not self.animating:
            return
        
        self.anim_progress += 1
        if self.anim_progress >= ANIMATION_SPEED:
            self.animating = False
            if self.anim_from and self.anim_to:
                self.apply_move_after_animation(self.anim_from, self.anim_to)
                self.anim_from = None
                self.anim_to = None

    def apply_move_after_animation(self, from_pos, to_pos):
        """Apply the move after animation completes."""
        moves = self.state.get_moves()
        legal = any(move[0] == from_pos and move[1] == to_pos for move in moves)
        
        if legal:
            is_solution_step = (self.solution is not None and self.solution_index > 0)
            self.last_move = (from_pos, to_pos)
            new_grid = PegSolitaireBoard.apply_move(self.state.grid, ((from_pos, to_pos)))
            
            if self.mode == 'two':
                self.two_player_turn = 1 - self.two_player_turn
                self.update_state(new_grid, player=self.two_player_turn)
                if self.state.is_terminal():
                    self.winner = 1 - self.two_player_turn
                    if self.winner == 0:
                        self.player1_streak += 1
                        self.player2_streak = 0
                    else:
                        self.player2_streak += 1
                        self.player1_streak = 0
                    self.game_end_time = time.time()
                else:
                    if self.two_player_submode == 'vs_ai' and self.two_player_turn == 1 and not is_solution_step:
                        self.ai_thinking = True
                        self.ai_thinking_time = 0
                        threading.Thread(target=self.ai_delayed_move, daemon=True).start()
            else:
                self.update_state(new_grid, player=0)
                if not is_solution_step and self.solution:
                    self.solution = None
                    self.solution_index = 0
                    self.show_start = False
            
            self.update_legal_moves()

    def try_human_move_animated(self, from_pos, to_pos):
        """Attempt a human move with animation."""
        if not self.animating and any(move[0] == from_pos and move[1] == to_pos 
                                       for move in self.state.get_moves()):
            self.start_animation(from_pos, to_pos)
            return True
        return False

    # ------------------------------------------------------------------------
    # Two-Player AI
    # ------------------------------------------------------------------------
    
    def ai_delayed_move(self):
        """Delay AI move for better visibility."""
        time.sleep(self.ai_delay)
        self.ai_move_thread()

    def ai_move_thread(self):
        """Execute AI move using MCTS."""
        start_think_time = time.time()
        mcts_container = [None]
        
        def on_progress():
            if mcts_container[0]:
                self.capture_mcts_tree(mcts_container[0])
        
        mcts = MCTS(self.state, time_limit=self.mcts_time_limit, use_heuristic=True)
        mcts_container[0] = mcts
        best_move = mcts.search(progress_callback=on_progress)
        self.capture_mcts_tree(mcts)
        
        self.ai_thinking_time = time.time() - start_think_time
        self.simulations = mcts.simulations
        
        # Highlight chosen move in tree visualization
        if best_move and self.fullscreen and self.tree_visualizer and mcts.root:
            chosen_node = next((child for child in mcts.root.children if child.move == best_move), None)
            if chosen_node:
                if not self.tree_visualizer.root_viz or self.tree_visualizer.root_viz.node != mcts.root:
                    self.tree_visualizer.build_tree(mcts.root)
                self.tree_visualizer.set_highlighted_node(chosen_node)
                self.tree_visualizer.set_waiting_for_click(True)
                self.waiting_for_tree_click = True
                while self.waiting_for_tree_click and self.fullscreen:
                    time.sleep(0.1)
                self.tree_visualizer.set_waiting_for_click(False)
        
        # Apply move
        if best_move:
            self.last_move = best_move
            new_grid = PegSolitaireBoard.apply_move(self.state.grid, best_move)
            self.two_player_turn = 1 - self.two_player_turn
            self.update_state(new_grid, player=self.two_player_turn)
            if self.state.is_terminal():
                self.winner = 1 - self.two_player_turn
                if self.winner == 0:
                    self.player1_streak += 1
                    self.player2_streak = 0
                else:
                    self.player2_streak += 1
                    self.player1_streak = 0
                self.game_end_time = time.time()
        
        self.ai_thinking = False
        if self.tree_visualizer:
            self.tree_visualizer.set_highlighted_node(None)

    # ------------------------------------------------------------------------
    # Single-Player Search Threads
    # ------------------------------------------------------------------------
    
    def solve_classic_in_thread(self):
        """Run IDA* solver in background thread."""
        start_state = PegState(self.grid, player=0)
        self.solution_start_grid = [row[:] for row in self.grid]
        
        def on_expand():
            self.nodes_expanded = solver.nodes_expanded
        
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
            self.move_heatmap = {}
            for move in solution:
                self.track_move_for_heatmap(move)
        else:
            self.solution = None
            self.solution_start_grid = None
            self.show_start = False
            self.timeout_msg = True
        
        self.searching = False
        self.button_text_solve = "Solve"
        self.button_text_minimize = "Minimize"
        self.button_text_twoplayer = "Two Player"
        self.button_text_oneplayer = "One Player"
        self.button_text_random = "Random Board"

    def solve_minimize_in_thread(self):
        """Run best-first minimizer in background thread."""
        start_state = PegState(self.grid, player=0)
        self.solution_start_grid = [row[:] for row in self.grid]
        
        def on_expand():
            self.nodes_expanded = solver.nodes_expanded
        
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
            self.move_heatmap = {}
            for move in moves:
                self.track_move_for_heatmap(move)
            final = self.grid
            for move in moves:
                final = PegSolitaireBoard.apply_move(final, move)
            self.grid = final
            self.state = PegState(final, player=0)
        else:
            self.solution = None
            self.solution_start_grid = None
            self.show_start = False
        
        self.searching = False
        self.button_text_solve = "Solve"
        self.button_text_minimize = "Minimize"
        self.button_text_twoplayer = "Two Player"
        self.button_text_oneplayer = "One Player"
        self.button_text_random = "Random Board"

    def random_board_in_thread(self):
        """Generate random board in background thread."""
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
        self.button_text_twoplayer = "Two Player"
        self.button_text_oneplayer = "One Player"
        self.button_text_random = "Random Board"

    def generate_random_board(self, peg_count=None):
        """Generate a random board with optional peg count."""
        grid = [[-1 for _ in range(7)] for _ in range(7)]
        valid_cells = [(r, c) for r in range(7) for c in range(7) if PegSolitaireBoard.is_valid(r, c)]
        total = len(valid_cells)
        
        if peg_count is not None and 1 <= peg_count <= total:
            chosen = set(random.sample(valid_cells, peg_count))
            for r, c in valid_cells:
                grid[r][c] = 1 if (r, c) in chosen else 0
        else:
            for r, c in valid_cells:
                grid[r][c] = 1 if random.random() < 0.5 else 0
        
        # Ensure board is not empty or full
        if PegSolitaireBoard.count_pegs(grid) == 0:
            r, c = random.choice(valid_cells)
            grid[r][c] = 1
        if PegSolitaireBoard.count_pegs(grid) == total:
            r, c = random.choice(valid_cells)
            grid[r][c] = 0
        return grid

    # ------------------------------------------------------------------------
    # Drawing Methods
    # ------------------------------------------------------------------------
    
    def draw_board(self):
        """Draw the game board, pegs, and legal move highlights."""
        self.screen.fill(COLORS['background'])
        
        for r in range(7):
            for c in range(7):
                if not PegSolitaireBoard.is_valid(r, c):
                    continue
                    
                x = OFFSET_X + c * CELL_SIZE
                y = OFFSET_Y + r * CELL_SIZE
                
                # Draw hole
                pygame.draw.rect(self.screen, COLORS['hole'], (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, COLORS['edge'], (x, y, CELL_SIZE, CELL_SIZE), 2)
                
                # Draw peg if present
                if self.state.grid[r][c] == 1 and not (self.animating and 
                   (self.anim_from == (r, c) or self.anim_to == (r, c))):
                    color = COLORS['peg']
                    if self.mode == 'two' and self.selected_cell == (r, c):
                        color = COLORS['selected_peg']
                    elif self.mode == 'two' and self.hint_from == (r, c):
                        color = COLORS['hint_peg']
                    pygame.draw.circle(self.screen, color, (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3)
                
                # Draw hint landing
                if self.mode == 'two' and self.hint_to == (r, c):
                    pygame.draw.rect(self.screen, COLORS['hint_landing'], (x, y, CELL_SIZE, CELL_SIZE), 3)
        
        self.draw_legal_moves()
        self.draw_animated_peg()
        self.draw_heatmap()

    def draw_legal_moves(self):
        """Draw highlights for legal moves from selected peg."""
        if not (self.show_legal_moves and self.mode == 'two' and self.winner is None and self.selected_cell):
            return
        
        for to_pos in self.get_legal_landing_cells(self.selected_cell):
            x = OFFSET_X + to_pos[1] * CELL_SIZE
            y = OFFSET_Y + to_pos[0] * CELL_SIZE
            pygame.draw.circle(self.screen, COLORS['legal_move'], 
                             (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3, 3)
            alpha_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            alpha_surface.fill((100, 200, 100, 80))
            self.screen.blit(alpha_surface, (x, y))

    def draw_animated_peg(self):
        """Draw the peg during animation."""
        if not self.animating:
            return
        t = self.anim_progress / ANIMATION_SPEED
        x = self.anim_start_pos[0] + (self.anim_end_pos[0] - self.anim_start_pos[0]) * t
        y = self.anim_start_pos[1] + (self.anim_end_pos[1] - self.anim_start_pos[1]) * t
        pygame.draw.circle(self.screen, COLORS['peg'], (int(x), int(y)), CELL_SIZE//3)

    def draw_heatmap(self):
        """Draw heatmap overlay (single-player only)."""
        if not (self.heatmap_enabled and self.move_heatmap and self.mode == 'single'):
            return
        
        max_usage = max(self.move_heatmap.values()) if self.move_heatmap else 1
        
        for r in range(7):
            for c in range(7):
                if not PegSolitaireBoard.is_valid(r, c):
                    continue
                pos = (r, c)
                if pos not in self.move_heatmap:
                    continue
                
                intensity = self.move_heatmap[pos] / max_usage
                
                # Color gradient: Blue -> Cyan -> Yellow -> Red
                if intensity < 0.33:
                    r_color, g_color, b_color = 0, int(100 + 155 * (intensity / 0.33)), 200
                elif intensity < 0.66:
                    t = (intensity - 0.33) / 0.33
                    r_color, g_color, b_color = int(0 + 255 * t), 255, int(200 - 200 * t)
                else:
                    t = (intensity - 0.66) / 0.34
                    r_color, g_color, b_color = 255, int(255 - 255 * t), 0
                
                x, y = OFFSET_X + c * CELL_SIZE, OFFSET_Y + r * CELL_SIZE
                alpha = int(100 + 100 * intensity)
                overlay_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(overlay_surf, (r_color, g_color, b_color, alpha), 
                                 (CELL_SIZE//2, CELL_SIZE//2), CELL_SIZE//3)
                if intensity > 0.7:
                    pygame.draw.circle(overlay_surf, (255, 255, 255, 200),
                                     (CELL_SIZE//2, CELL_SIZE//2), CELL_SIZE//3 + 2, 2)
                self.screen.blit(overlay_surf, (x, y))
                
                if intensity > 0.8:
                    count_text = self.small_font.render(str(self.move_heatmap[pos]), True, (255, 255, 255))
                    count_rect = count_text.get_rect(center=(x + CELL_SIZE//2, y + CELL_SIZE//2))
                    pygame.draw.rect(self.screen, (0, 0, 0, 180), count_rect.inflate(6, 4))
                    self.screen.blit(count_text, count_rect)

    def draw_heatmap_legend(self):
        """Draw legend for heatmap colors (single-player only)."""
        if not (self.heatmap_enabled and self.move_heatmap and self.mode == 'single'):
            return
        
        legend_x, legend_y = 30, self.screen_height - 280
        legend_panel = pygame.Surface((180, 100), pygame.SRCALPHA)
        legend_panel.fill((0, 0, 0, 200))
        self.screen.blit(legend_panel, (legend_x, legend_y))
        
        title = self.small_font.render("Heatmap Legend", True, (255, 255, 255))
        self.screen.blit(title, (legend_x + 10, legend_y + 5))
        
        colors = [((0, 100, 200), "Low"), ((100, 255, 100), "Med"), 
                  ((255, 255, 0), "High"), ((255, 0, 0), "V High")]
        y_offset = legend_y + 30
        for color, label in colors:
            pygame.draw.circle(self.screen, color, (legend_x + 20, y_offset + 8), 8)
            text = self.small_font.render(label, True, (200, 200, 200))
            self.screen.blit(text, (legend_x + 35, y_offset + 2))
            y_offset += 20

    def draw_stats(self):
        """Draw game statistics panel."""
        y = 120
        text = self.font.render(f"Mode: {'Two-Player' if self.mode=='two' else 'Single-Player'}", True, COLORS['text'])
        self.screen.blit(text, (20, y)); y += 30
        text = self.font.render(f"Pegs: {PegSolitaireBoard.count_pegs(self.state.grid)}", True, COLORS['text'])
        self.screen.blit(text, (20, y)); y += 30
        
        if self.mode == 'two':
            # Two-player stats
            text = self.font.render(f"Moves: {self.move_count}", True, COLORS['text'])
            self.screen.blit(text, (20, y)); y += 30
            
            if self.game_start_time:
                elapsed = (self.game_end_time or time.time()) - self.game_start_time
                minutes, seconds = int(elapsed // 60), int(elapsed % 60)
                text = self.font.render(f"Time: {minutes:02d}:{seconds:02d}", True, COLORS['text'])
                self.screen.blit(text, (20, y)); y += 30
            
            if self.two_player_submode == 'vs_ai' and self.ai_thinking_time > 0:
                text = self.font.render(f"AI think: {self.ai_thinking_time:.2f}s", True, COLORS['text'])
                self.screen.blit(text, (20, y)); y += 30
                if self.simulations > 0:
                    text = self.font.render(f"Simulations: {self.simulations}", True, COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 30
            
            text = self.font.render(f"Legal moves: {len(self.legal_moves)}", True, COLORS['text'])
            self.screen.blit(text, (20, y)); y += 30
            
            # Player names
            text = self.font.render("Player Names:", True, COLORS['text'])
            self.screen.blit(text, (20, y)); y += 25
            
            p1_text = self.small_font.render(f"P1: {self.player1_name}", True, COLORS['text'])
            self.screen.blit(p1_text, (20, y))
            self.edit_p1_rect = pygame.Rect(140, y, 50, 22)
            if not self.ai_thinking and self.winner is None:
                color = COLORS['button_hover'] if self.edit_p1_rect.collidepoint(pygame.mouse.get_pos()) else COLORS['button']
                pygame.draw.rect(self.screen, color, self.edit_p1_rect)
                surf = self.small_font.render("Edit", True, COLORS['text'])
                self.screen.blit(surf, (self.edit_p1_rect.x+8, self.edit_p1_rect.y+3))
            y += 25
            
            p2_label = "P2" if self.two_player_submode == 'vs_human' else "AI"
            p2_text = self.small_font.render(f"{p2_label}: {self.player2_name if self.two_player_submode == 'vs_human' else self.ai_name}", True, COLORS['text'])
            self.screen.blit(p2_text, (20, y))
            self.edit_p2_rect = pygame.Rect(140, y, 50, 22)
            if not self.ai_thinking and self.winner is None:
                color = COLORS['button_hover'] if self.edit_p2_rect.collidepoint(pygame.mouse.get_pos()) else COLORS['button']
                pygame.draw.rect(self.screen, color, self.edit_p2_rect)
                surf = self.small_font.render("Edit", True, COLORS['text'])
                self.screen.blit(surf, (self.edit_p2_rect.x+8, self.edit_p2_rect.y+3))
            y += 35
            
            # Name input panel
            if self.name_input_active and self.editing_player:
                panel_rect = pygame.Rect(15, y, 210, 85)
                pygame.draw.rect(self.screen, (230, 230, 230), panel_rect)
                pygame.draw.rect(self.screen, COLORS['edge'], panel_rect, 2)
                
                player_names = {'p1': "Player 1", 'p2': "Player 2", 'ai': "AI"}
                inst_text = self.small_font.render(f"Edit {player_names.get(self.editing_player, '')} name:", True, COLORS['text'])
                self.screen.blit(inst_text, (20, panel_rect.y + 8))
                
                self.name_input_rect = pygame.Rect(20, panel_rect.y + 32, 180, 28)
                pygame.draw.rect(self.screen, COLORS['input_bg'], self.name_input_rect)
                pygame.draw.rect(self.screen, COLORS['input_border'], self.name_input_rect, 2)
                input_surf = self.small_font.render(self.name_input_text, True, COLORS['text'])
                self.screen.blit(input_surf, (self.name_input_rect.x+5, self.name_input_rect.y+5))
                
                hint_text = self.small_font.render("ENTER to save, ESC to cancel", True, COLORS['text'])
                self.screen.blit(hint_text, (20, panel_rect.y + 65))
                y += 90
            
            # Win streaks
            if self.player1_streak > 0 or self.player2_streak > 0:
                text = self.font.render("Win Streaks:", True, COLORS['streak'])
                self.screen.blit(text, (20, y)); y += 25
                if self.player1_streak > 0:
                    text = self.font.render(f"P1: {self.player1_streak}", True, COLORS['streak'])
                    self.screen.blit(text, (20, y)); y += 25
                if self.player2_streak > 0:
                    text = self.font.render(f"P2: {self.player2_streak}", True, COLORS['streak'])
                    self.screen.blit(text, (20, y)); y += 25

            if self.fullscreen and self.waiting_for_tree_click and self.show_tree_viz:
                text = self.font.render("Click on tree to continue", True, COLORS['streak'])
                self.screen.blit(text, (20, y)); y += 30
        
        # Winner / Turn display
        if self.mode == 'two':
            if self.winner is not None:
                winner_text = "Human" if self.winner == 0 else ("AI" if self.two_player_submode == 'vs_ai' else "Player 2")
                text = self.font.render(f"Winner: {winner_text}!", True, COLORS['winner'])
                self.screen.blit(text, (20, y)); y += 30
            else:
                turn_text = f"{self.player1_name} (Human)" if self.two_player_turn == 0 else (self.ai_name if self.two_player_submode == 'vs_ai' else self.player2_name)
                text = self.font.render(f"Turn: {turn_text}", True, COLORS['ai_turn'] if self.two_player_turn == 1 else COLORS['text'])
                self.screen.blit(text, (20, y)); y += 30
            
            # Recent moves
            if self.move_history:
                text = self.font.render("Recent moves:", True, COLORS['text'])
                self.screen.blit(text, (20, y)); y += 25
                start_idx = max(0, len(self.move_history) - 5)
                for i in range(start_idx, len(self.move_history)):
                    move = self.move_history[i]
                    from_pos, to_pos, player = move
                    player_name = self.player1_name if player == 0 else (self.ai_name if self.two_player_submode == 'vs_ai' else self.player2_name)
                    move_text = f"{player_name}: ({from_pos[0]},{from_pos[1]}) -> ({to_pos[0]},{to_pos[1]})"
                    text = self.font.render(move_text, True, COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 22
        else:
            # Single-player stats
            text = self.font.render(f"Nodes expanded: {self.nodes_expanded:,}", True, COLORS['text'])
            self.screen.blit(text, (20, y)); y += 30
            text = self.font.render(f"Search time: {self.search_time:.2f}s", True, COLORS['text'])
            self.screen.blit(text, (20, y)); y += 30
            
            if self.solution:
                text = self.font.render(f"Solution length: {len(self.solution)}", True, COLORS['text'])
                self.screen.blit(text, (20, y)); y += 30
                
                if self.search_time > 0 and len(self.solution) > 0:
                    pygame.draw.line(self.screen, COLORS['edge'], (20, y), (250, y), 1)
                    y += 10
                    text = self.small_font.render("▶ Move Statistics:", True, COLORS['streak'])
                    self.screen.blit(text, (20, y)); y += 22
                    
                    avg_time = self.search_time / len(self.solution)
                    text = self.small_font.render(f"  Avg time/move: {avg_time:.4f}s", True, COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 20
                    
                    if self.nodes_expanded > 0:
                        branching = self.nodes_expanded ** (1 / len(self.solution))
                        branch_color = (100, 255, 100) if branching < 2.0 else (255, 255, 100) if branching < 3.0 else (255, 100, 100)
                        text = self.small_font.render(f"  Effective branching: {branching:.2f}", True, branch_color)
                        self.screen.blit(text, (20, y)); y += 20
                    
                    efficiency = self.nodes_expanded / len(self.solution)
                    eff_color = (100, 255, 100) if efficiency < 50000 else (255, 255, 100) if efficiency < 200000 else (255, 100, 100)
                    text = self.small_font.render(f"  Efficiency: {efficiency:,.0f} nodes/move", True, eff_color)
                    self.screen.blit(text, (20, y)); y += 20
                    
                    if self.search_time > 0:
                        nodes_per_sec = self.nodes_expanded / self.search_time
                        text = self.small_font.render(f"  Search speed: {nodes_per_sec:,.0f} nodes/s", True, COLORS['text'])
                        self.screen.blit(text, (20, y)); y += 20
                    y += 5
                
                if self.move_heatmap and self.mode == 'single':
                    heatmap_text = "Heatmap: " + ("ON" if self.heatmap_enabled else "OFF")
                    text = self.small_font.render(heatmap_text, True, COLORS['text'])
                    self.screen.blit(text, (20, y)); y += 22
                
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
        """Draw all UI buttons."""
        mouse_pos = pygame.mouse.get_pos()
        self.check_tooltip(mouse_pos)
        
        # Top left buttons
        self._draw_button(self.button_fullscreen, self.button_text_fullscreen, mouse_pos)
        if self.mode == 'single':
            self._draw_button(self.button_heatmap, self.button_text_heatmap, mouse_pos, enabled=self.move_heatmap)
        self._draw_button(self.button_theme, self.button_text_theme, mouse_pos)
        
        # Mode buttons
        self._draw_button(self.button_oneplayer, self.button_text_oneplayer, mouse_pos, active=self.mode == 'single')
        self._draw_button(self.button_twoplayer, self.button_text_twoplayer, mouse_pos, active=self.mode == 'two')
        
        # Two-player submode buttons
        if self.mode == 'two':
            self._draw_button(self.button_vs_ai, self.button_text_vs_ai, mouse_pos, active=self.two_player_submode == 'vs_ai')
            self._draw_button(self.button_vs_human, self.button_text_vs_human, mouse_pos, active=self.two_player_submode == 'vs_human')
        
        # Single-player action buttons
        if self.mode == 'single':
            self._draw_button(self.button_solve, self.button_text_solve, mouse_pos)
            self._draw_button(self.button_minimize, self.button_text_minimize, mouse_pos)
            self._draw_button(self.button_random, self.button_text_random, mouse_pos)
            pygame.draw.rect(self.screen, COLORS['input_bg'], self.input_rect)
            pygame.draw.rect(self.screen, COLORS['input_border'], self.input_rect, 2)
            input_surf = self.font.render(self.input_text, True, COLORS['text'])
            self.screen.blit(input_surf, (self.input_rect.x+5, self.input_rect.y+7))
            self.screen.blit(self.input_label, self.input_label_rect)
        
        # Common buttons
        self._draw_button(self.button_undo, self.button_text_undo, mouse_pos)
        self._draw_button(self.button_restart, self.button_text_restart, mouse_pos)
        if self.mode == 'two':
            self._draw_button(self.button_newgame, self.button_text_newgame, mouse_pos)
        
        # Hint/Solution buttons
        if self.mode == 'two' and self.winner is None and not self.ai_thinking:
            if self.two_player_submode == 'vs_human':
                self._draw_button(self.button_hint, self.button_text_hint, mouse_pos, computing=self.computing_hint)
                self._draw_button(self.button_solution, self.button_text_solution, mouse_pos, computing=self.computing_solution)
            elif self.two_player_submode == 'vs_ai' and self.two_player_turn == 0:
                self._draw_button(self.button_ai_hint, self.button_text_ai_hint, mouse_pos, computing=self.computing_hint)
                self._draw_button(self.button_ai_solution, self.button_text_ai_solution, mouse_pos, computing=self.computing_solution)
        
        self.draw_tooltips()

    def _draw_button(self, rect, text, mouse_pos, active=False, enabled=True, computing=False):
        """Helper to draw a single button."""
        if computing:
            base_color = COLORS['button_computing']
        elif active:
            base_color = COLORS['button_active']
        else:
            base_color = COLORS['button']
        
        color = COLORS['button_hover'] if rect.collidepoint(mouse_pos) and enabled else base_color
        pygame.draw.rect(self.screen, color, rect)
        surf = self.font.render(text, True, COLORS['text'])
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw_tooltips(self):
        """Draw tooltip for button under mouse."""
        if self.tooltip_timer <= 0 or not self.tooltip_text:
            return
        
        self.tooltip_timer -= 1
        lines = self.tooltip_text.split('\n')
        line_surfaces = [self.small_font.render(line, True, (255, 255, 255)) for line in lines]
        max_width = max(s.get_width() for s in line_surfaces)
        
        padding, line_height = 10, self.small_font.get_height()
        tooltip_width = max_width + padding * 2
        tooltip_height = len(lines) * line_height + padding * 2
        
        x = self.tooltip_rect.x + self.tooltip_rect.width // 2 - tooltip_width // 2
        y = self.tooltip_rect.y - tooltip_height - 5
        
        # Clamp to screen edges
        x = max(5, min(x, self.screen_width - tooltip_width - 5))
        y = max(5, y) if y > 5 else self.tooltip_rect.y + self.tooltip_rect.height + 5
        
        tooltip_rect = pygame.Rect(x, y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, COLORS['tooltip_bg'], tooltip_rect)
        pygame.draw.rect(self.screen, COLORS['edge'], tooltip_rect, 2)
        
        text_y = y + padding
        for surf in line_surfaces:
            self.screen.blit(surf, (x + padding, text_y))
            text_y += line_height

    def check_tooltip(self, mouse_pos):
        """Check if mouse is over any button and set tooltip."""
        button_tooltips = [
            (self.button_fullscreen, "Fullscreen"),
            (self.button_heatmap, "Heatmap"),
            (self.button_theme, "Theme"),
            (self.button_oneplayer, "One Player"),
            (self.button_twoplayer, "Two Player"),
            (self.button_undo, "Undo"),
            (self.button_restart, "Restart"),
        ]
        
        if self.mode == 'single':
            button_tooltips.extend([
                (self.button_solve, "Solve"),
                (self.button_minimize, "Minimize"),
                (self.button_random, "Random Board"),
            ])
        elif self.mode == 'two':
            button_tooltips.extend([
                (self.button_vs_ai, "Vs AI"),
                (self.button_vs_human, "Vs Human"),
                (self.button_newgame, "New Game"),
            ])
            if self.winner is None and not self.ai_thinking:
                if self.two_player_submode == 'vs_human':
                    button_tooltips.extend([
                        (self.button_hint, "Hint"),
                        (self.button_solution, "Solution"),
                    ])
                elif self.two_player_submode == 'vs_ai' and self.two_player_turn == 0:
                    button_tooltips.extend([
                        (self.button_ai_hint, "Hint"),
                        (self.button_ai_solution, "Solution"),
                    ])
        
        for btn, tip_key in button_tooltips:
            if btn.collidepoint(mouse_pos):
                self.tooltip_text = TOOLTIPS.get(tip_key, "")
                self.tooltip_rect = btn
                self.tooltip_timer = 60
                return

    # ------------------------------------------------------------------------
    # Event Handling
    # ------------------------------------------------------------------------
    
    def handle_events(self):
        """Process pygame events (mouse clicks, keyboard input)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if self.fullscreen and self.tree_visualizer and self.show_tree_viz:
                self.tree_visualizer.handle_event(event)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event)
            
            if event.type == pygame.KEYDOWN:
                self._handle_keyboard(event)
        
        self.update_animation()
        return True

    def _handle_mouse_click(self, event):
        """Handle mouse click events."""
        # Input field activation
        if self.mode == 'single' and self.input_rect.collidepoint(event.pos):
            self.input_active = True
        else:
            self.input_active = False
        
        # Clear tree visualization on click
        if self.fullscreen and self.waiting_for_tree_click and self.show_tree_viz:
            self.waiting_for_tree_click = False
            self.clear_tree_viz()
        
        # Mode buttons
        if self.button_oneplayer.collidepoint(event.pos) and not self.searching:
            self._switch_to_single_player()
            return
        if self.button_twoplayer.collidepoint(event.pos) and not self.searching:
            self._switch_to_two_player()
            return
        
        # Two-player submode buttons
        if self.mode == 'two':
            if self.button_vs_ai.collidepoint(event.pos):
                self._switch_to_vs_ai()
                return
            if self.button_vs_human.collidepoint(event.pos):
                self._switch_to_vs_human()
                return
        
        # Name edit buttons
        if self.mode == 'two' and not self.searching and self.winner is None and not self.ai_thinking:
            if hasattr(self, 'edit_p1_rect') and self.edit_p1_rect.collidepoint(event.pos):
                self.start_name_edit('p1')
                return
            if hasattr(self, 'edit_p2_rect') and self.edit_p2_rect.collidepoint(event.pos):
                self.start_name_edit('p2' if self.two_player_submode == 'vs_human' else 'ai')
                return
        
        # Single-player action buttons
        if self.mode == 'single':
            if self.button_solve.collidepoint(event.pos) and not self.searching:
                self._start_solve()
                return
            if self.button_minimize.collidepoint(event.pos) and not self.searching:
                self._start_minimize()
                return
            if self.button_random.collidepoint(event.pos) and not self.searching:
                self._start_random_board()
                return
        
        # Theme and fullscreen
        if self.button_theme.collidepoint(event.pos) and not self.searching:
            self.switch_theme()
            return
        if self.button_fullscreen.collidepoint(event.pos) and not self.searching:
            self.toggle_fullscreen()
            return
        
        # Heatmap button (single-player only)
        if self.mode == 'single' and self.button_heatmap.collidepoint(event.pos) and not self.searching and self.move_heatmap:
            self.heatmap_enabled = not self.heatmap_enabled
            self.button_text_heatmap = "Heatmap ON" if self.heatmap_enabled else "Heatmap OFF"
            return
        
        # Common buttons
        if self.button_undo.collidepoint(event.pos) and not self.searching:
            self.undo_single_player() if self.mode == 'single' else self.undo_last_move()
            return
        if self.button_restart.collidepoint(event.pos) and not self.searching:
            self.restart_single_player() if self.mode == 'single' else self.restart_game()
            return
        if self.mode == 'two' and self.button_newgame.collidepoint(event.pos) and not self.searching:
            self.new_game()
            return
        
        # Hint/Solution buttons
        if self.mode == 'two' and self.winner is None and not self.ai_thinking:
            if self.two_player_submode == 'vs_human':
                if self.button_hint.collidepoint(event.pos):
                    self.show_hint()
                    return
                if self.button_solution.collidepoint(event.pos):
                    self.show_solution()
                    return
            elif self.two_player_submode == 'vs_ai' and self.two_player_turn == 0:
                if self.button_ai_hint.collidepoint(event.pos):
                    self.show_hint()
                    return
                if self.button_ai_solution.collidepoint(event.pos):
                    self.show_solution()
                    return
        
        # Cell clicks (board interaction)
        self._handle_cell_click(event)

    def _handle_cell_click(self, event):
        """Handle clicks on board cells."""
        if self.searching or self.animating:
            return
        
        # Find clicked cell
        cell = None
        for r in range(7):
            for c in range(7):
                x = OFFSET_X + c * CELL_SIZE
                y = OFFSET_Y + r * CELL_SIZE
                if x <= event.pos[0] <= x + CELL_SIZE and y <= event.pos[1] <= y + CELL_SIZE:
                    cell = (r, c)
                    break
            if cell:
                break
        
        if not cell:
            return
        
        # Two-player mode
        if self.mode == 'two':
            if self.winner is not None:
                return
            if self.two_player_submode == 'vs_ai' and (self.ai_thinking or self.two_player_turn != 0):
                return
            
            if self.selected_cell is None:
                if self.state.grid[cell[0]][cell[1]] == 1:
                    self.selected_cell = cell
            else:
                if self.try_human_move_animated(self.selected_cell, cell):
                    self.selected_cell = None
                else:
                    self.selected_cell = None
        else:
            # Single-player mode
            if event.button == 3 or (event.button == 1 and pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.toggle_validity(*cell)
            elif event.button == 1:
                self.toggle_cell(*cell)

    def _handle_keyboard(self, event):
        """Handle keyboard input."""
        # Single-player input field
        if self.mode == 'single' and self.input_active:
            if event.key == pygame.K_RETURN:
                self.input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.unicode.isdigit():
                self.input_text += event.unicode
        
        # Two-player name editing
        elif self.mode == 'two' and self.name_input_active:
            if event.key == pygame.K_RETURN:
                self.save_name_edit()
            elif event.key == pygame.K_ESCAPE:
                self.name_input_active = False
                self.editing_player = None
                self.name_input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.name_input_text = self.name_input_text[:-1]
            elif event.unicode.isprintable() and len(event.unicode) == 1:
                self.name_input_text += event.unicode
        
        # Space to step through solution
        elif self.mode == 'single' and event.key == pygame.K_SPACE and self.solution:
            if self.show_start:
                if self.solution_start_grid is not None:
                    self.update_state([row[:] for row in self.solution_start_grid], player=0)
                self.show_start = False
            else:
                if self.solution_index < len(self.solution) and not self.animating:
                    move = self.solution[self.solution_index]
                    self.last_move = move
                    new_grid = PegSolitaireBoard.apply_move(self.state.grid, move)
                    self.update_state(new_grid, player=0)
                    self.solution_index += 1

    # ------------------------------------------------------------------------
    # Mode Switching Helpers
    # ------------------------------------------------------------------------
    
    def _switch_to_single_player(self):
        """Switch to single-player mode."""
        self.mode = 'single'
        PegSolitaireBoard.reset_mask()
        self.update_state(PegSolitaireBoard.initial_grid(), player=0)
        self.clear_solution_data()
        self.winner = None
        self.ai_thinking = False
        self.move_history = []
        self.history_states = []
        self.move_count = 0
        self.last_move = None
        self.player1_streak = 0
        self.player2_streak = 0

    def _switch_to_two_player(self):
        """Switch to two-player mode."""
        self.mode = 'two'
        PegSolitaireBoard.reset_mask()
        new_grid = PegSolitaireBoard.initial_grid()
        self.two_player_turn = 0
        self.winner = None
        self.win_probability = None
        self.selected_cell = None
        self.hint_from = None
        self.hint_to = None
        self.move_count = 0
        self.move_history = []
        self.history_states = []
        self.last_move = None
        self.game_start_time = time.time()
        self.game_end_time = None
        self.player1_streak = 0
        self.player2_streak = 0
        self.update_state(new_grid, player=self.two_player_turn)
        self.clear_solution_data()
        self.two_player_submode = 'vs_ai'
        
        if self.eval_thread_running:
            self.stop_evaluation_thread()
            time.sleep(0.1)
        self.start_evaluation_thread()
        
        if not self.ai_thinking and self.two_player_turn == 1 and not self.state.is_terminal():
            self.ai_thinking = True
            threading.Thread(target=self.ai_delayed_move, daemon=True).start()

    def _switch_to_vs_ai(self):
        """Switch to Human vs AI mode."""
        self.two_player_submode = 'vs_ai'
        self.selected_cell = None
        self.hint_from = None
        self.hint_to = None
        self.winner = None
        self.two_player_turn = 0
        self.move_count = 0
        self.move_history = []
        self.history_states = []
        self.last_move = None
        self.game_start_time = time.time()
        self.game_end_time = None
        self.player1_streak = 0
        self.player2_streak = 0
        self.update_state(self.grid, player=self.two_player_turn)
        
        if self.eval_thread_running:
            self.stop_evaluation_thread()
            time.sleep(0.1)
        self.start_evaluation_thread()
        
        if not self.ai_thinking and self.two_player_turn == 1 and not self.state.is_terminal():
            self.ai_thinking = True
            threading.Thread(target=self.ai_delayed_move, daemon=True).start()

    def _switch_to_vs_human(self):
        """Switch to Human vs Human mode."""
        self.two_player_submode = 'vs_human'
        self.selected_cell = None
        self.hint_from = None
        self.hint_to = None
        self.winner = None
        self.two_player_turn = 0
        self.move_count = 0
        self.move_history = []
        self.history_states = []
        self.last_move = None
        self.game_start_time = time.time()
        self.game_end_time = None
        self.player1_streak = 0
        self.player2_streak = 0
        self.update_state(self.grid, player=0)
        
        if self.eval_thread_running:
            self.stop_evaluation_thread()
            time.sleep(0.1)
        self.start_evaluation_thread()

    def _start_solve(self):
        """Start IDA* solver."""
        PegSolitaireBoard.reset_mask()
        self.update_state(PegSolitaireBoard.initial_grid(), player=0)
        self.clear_solution_data()
        self.searching = True
        self.button_text_solve = "Searching..."
        self.button_text_minimize = "Minimize"
        self.button_text_twoplayer = "Two Player"
        self.button_text_oneplayer = "One Player"
        self.button_text_random = "Random Board"
        self.search_thread = threading.Thread(target=self.solve_classic_in_thread, daemon=True)
        self.search_thread.start()

    def _start_minimize(self):
        """Start best-first minimizer."""
        self.searching = True
        self.button_text_solve = "Solve"
        self.button_text_minimize = "Searching..."
        self.button_text_twoplayer = "Two Player"
        self.button_text_oneplayer = "One Player"
        self.button_text_random = "Random Board"
        self.search_thread = threading.Thread(target=self.solve_minimize_in_thread, daemon=True)
        self.search_thread.start()

    def _start_random_board(self):
        """Start random board generation."""
        self.searching = True
        self.button_text_solve = "Solve"
        self.button_text_minimize = "Minimize"
        self.button_text_twoplayer = "Two Player"
        self.button_text_oneplayer = "One Player"
        self.button_text_random = "Generating..."
        self.search_thread = threading.Thread(target=self.random_board_in_thread, daemon=True)
        self.search_thread.start()

    # ------------------------------------------------------------------------
    # Name Editing
    # ------------------------------------------------------------------------
    
    def start_name_edit(self, player):
        """Start editing a player's name."""
        self.editing_player = player
        self.name_input_text = getattr(self, f"{player}_name", "")
        self.name_input_active = True
    
    def save_name_edit(self):
        """Save the edited name."""
        if self.editing_player == 'p1':
            self.player1_name = self.name_input_text[:20]
        elif self.editing_player == 'p2':
            self.player2_name = self.name_input_text[:20]
        elif self.editing_player == 'ai':
            self.ai_name = self.name_input_text[:20]
        self.name_input_active = False
        self.editing_player = None
        self.name_input_text = ""

    # ------------------------------------------------------------------------
    # Game Actions
    # ------------------------------------------------------------------------
    
    def toggle_cell(self, r, c):
        """Toggle peg on/off at a cell (single-player editing)."""
        if self.searching or self.animating or not PegSolitaireBoard.is_valid(r, c):
            return
        new_grid = [list(row) for row in self.state.grid]
        new_grid[r][c] = 0 if new_grid[r][c] == 1 else 1
        self.update_state(new_grid)
        self.clear_solution_data()

    def toggle_validity(self, r, c):
        """Toggle cell validity (for custom board shapes)."""
        if self.searching or not (0 <= r < 7 and 0 <= c < 7):
            return
        new_valid = PegSolitaireBoard.toggle_valid(r, c)
        new_grid = [list(row) for row in self.state.grid]
        new_grid[r][c] = 0 if new_valid else -1
        self.update_state(new_grid)
        self.clear_solution_data()

    def undo_single_player(self):
        """Undo last move in single-player mode."""
        if self.mode != 'single' or self.searching or not self.history_states:
            return False
        
        prev_grid = self.history_states.pop()
        if self.move_history:
            self.move_history.pop()
        self.grid = prev_grid
        self.state = PegState(self.grid, player=0)
        self.move_count = max(0, self.move_count - 1)
        self.hint_from = None
        self.hint_to = None
        self.selected_cell = None
        self.solution = None
        self.solution_index = 0
        self.show_start = False
        self.update_legal_moves()
        return True

    def restart_single_player(self):
        """Restart single-player game."""
        if self.mode != 'single' or self.searching:
            return
        PegSolitaireBoard.reset_mask()
        self.update_state(PegSolitaireBoard.initial_grid(), player=0)
        self.clear_solution_data()
        self.nodes_expanded = 0
        self.search_time = 0
        self.move_history = []
        self.history_states = []
        self.move_count = 0
        self.last_move = None

    def try_human_move(self, from_pos, to_pos):
        """Attempt a human move in two-player mode."""
        if not any(move[0] == from_pos and move[1] == to_pos for move in self.state.get_moves()):
            return False
        
        self.last_move = (from_pos, to_pos)
        new_grid = PegSolitaireBoard.apply_move(self.state.grid, ((from_pos, to_pos)))
        self.two_player_turn = 1 - self.two_player_turn
        self.update_state(new_grid, player=self.two_player_turn)
        
        if self.state.is_terminal():
            self.winner = 1 - self.two_player_turn
            if self.winner == 0:
                self.player1_streak += 1
                self.player2_streak = 0
            else:
                self.player2_streak += 1
                self.player1_streak = 0
            self.game_end_time = time.time()
        else:
            if self.two_player_submode == 'vs_ai' and self.two_player_turn == 1:
                self.ai_thinking = True
                self.ai_thinking_time = 0
                threading.Thread(target=self.ai_delayed_move, daemon=True).start()
        
        self.update_legal_moves()
        return True

    def undo_last_move(self):
        """Undo last move in two-player mode."""
        if self.mode != 'two' or self.ai_thinking or self.winner is not None or not self.history_states:
            return False
        
        prev_grid = self.history_states.pop()
        prev_move = self.move_history.pop() if self.move_history else None
        
        self.grid = prev_grid
        self.two_player_turn = prev_move[2] if prev_move else 0
        self.state = PegState(self.grid, player=self.two_player_turn)
        self.move_count = max(0, self.move_count - 1)
        
        self.ai_thinking = False
        self.hint_from = None
        self.hint_to = None
        self.selected_cell = None
        self.winner = None
        self.game_end_time = None
        
        if self.game_start_time is None:
            self.game_start_time = time.time()
        
        with self.eval_lock:
            self.persistent_mcts = None
            self.current_eval_state_hash = None
        
        self.update_legal_moves()
        return True

    def restart_game(self):
        """Restart two-player game (resets streaks)."""
        if self.mode != 'two' or self.ai_thinking:
            return
        
        PegSolitaireBoard.reset_mask()
        new_grid = PegSolitaireBoard.initial_grid()
        
        self.two_player_turn = 0
        self.winner = None
        self.win_probability = None
        self.selected_cell = None
        self.hint_from = None
        self.hint_to = None
        self.move_count = 0
        self.move_history = []
        self.history_states = []
        self.last_move = None
        self.game_start_time = time.time()
        self.game_end_time = None
        self.ai_thinking = False
        self.player1_streak = 0
        self.player2_streak = 0
        
        self.grid = new_grid
        self.state = PegState(self.grid, player=self.two_player_turn)
        self.clear_solution_data()
        self.update_legal_moves()
        
        with self.eval_lock:
            self.persistent_mcts = None
            self.current_eval_state_hash = None
        
        if self.two_player_submode == 'vs_ai' and self.two_player_turn == 1 and not self.state.is_terminal():
            self.ai_thinking = True
            threading.Thread(target=self.ai_delayed_move, daemon=True).start()

    def new_game(self):
        """Start a new two-player game (preserves streaks)."""
        if self.mode != 'two' or self.ai_thinking:
            return
        
        p1_streak, p2_streak = self.player1_streak, self.player2_streak
        
        PegSolitaireBoard.reset_mask()
        new_grid = PegSolitaireBoard.initial_grid()
        
        self.two_player_turn = 0
        self.winner = None
        self.win_probability = None
        self.selected_cell = None
        self.hint_from = None
        self.hint_to = None
        self.move_count = 0
        self.move_history = []
        self.history_states = []
        self.last_move = None
        self.game_start_time = time.time()
        self.game_end_time = None
        self.ai_thinking = False
        
        self.player1_streak = p1_streak
        self.player2_streak = p2_streak
        
        self.grid = new_grid
        self.state = PegState(self.grid, player=self.two_player_turn)
        self.clear_solution_data()
        self.update_legal_moves()
        
        with self.eval_lock:
            self.persistent_mcts = None
            self.current_eval_state_hash = None
        
        if self.two_player_submode == 'vs_ai' and self.two_player_turn == 1 and not self.state.is_terminal():
            self.ai_thinking = True
            threading.Thread(target=self.ai_delayed_move, daemon=True).start()

    def show_hint(self):
        """Show hint for current player (highlight best move peg)."""
        if self.mode != 'two' or self.winner is not None:
            return
        if self.two_player_submode == 'vs_ai' and self.two_player_turn != 0:
            return
        
        def hint_thread():
            self.computing_hint = True
            
            mcts = MCTS(self.state, time_limit=self.mcts_time_limit, use_heuristic=True)
            
            def on_progress():
                """Capture tree during MCTS search for visualization."""
                self.capture_mcts_tree(mcts)
            
            self.capture_mcts_tree(mcts)
            best_move = mcts.search(progress_callback=on_progress)
            self.capture_mcts_tree(mcts)
            
            if best_move:
                self.hint_from = best_move[0]
                self.hint_to = None
                
                # Optional: Highlight chosen node in tree
                if self.fullscreen and self.tree_visualizer and mcts.root:
                    chosen_node = None
                    for child in mcts.root.children:
                        if child.move == best_move:
                            chosen_node = child
                            break
                    
                    if chosen_node:
                        if not self.tree_visualizer.root_viz or self.tree_visualizer.root_viz.node != mcts.root:
                            self.tree_visualizer.build_tree(mcts.root)
                        self.tree_visualizer.set_highlighted_node(chosen_node)
                        self.tree_visualizer.set_waiting_for_click(True)
                        self.waiting_for_tree_click = True
                        while self.waiting_for_tree_click and self.fullscreen:
                            time.sleep(0.1)
                        self.tree_visualizer.set_waiting_for_click(False)
                        self.clear_tree_viz()
            
            self.computing_hint = False
        
        threading.Thread(target=hint_thread, daemon=True).start()

    def show_solution(self):
        """Show full solution (highlight both peg and destination)."""
        if self.mode != 'two' or self.winner is not None:
            return
        if self.two_player_submode == 'vs_ai' and self.two_player_turn != 0:
            return
        
        def solution_thread():
            self.computing_solution = True
            
            mcts = MCTS(self.state, time_limit=self.mcts_time_limit, use_heuristic=True)
            
            def on_progress():
                """Capture tree during MCTS search for visualization."""
                self.capture_mcts_tree(mcts)
            
            self.capture_mcts_tree(mcts)
            best_move = mcts.search(progress_callback=on_progress)
            self.capture_mcts_tree(mcts)
            
            if best_move:
                self.hint_from = best_move[0]
                self.hint_to = best_move[1]
                
                # Optional: Highlight chosen node in tree
                if self.fullscreen and self.tree_visualizer and mcts.root:
                    chosen_node = None
                    for child in mcts.root.children:
                        if child.move == best_move:
                            chosen_node = child
                            break
                    
                    if chosen_node:
                        if not self.tree_visualizer.root_viz or self.tree_visualizer.root_viz.node != mcts.root:
                            self.tree_visualizer.build_tree(mcts.root)
                        self.tree_visualizer.set_highlighted_node(chosen_node)
                        self.tree_visualizer.set_waiting_for_click(True)
                        self.waiting_for_tree_click = True
                        while self.waiting_for_tree_click and self.fullscreen:
                            time.sleep(0.1)
                        self.tree_visualizer.set_waiting_for_click(False)
                        self.clear_tree_viz()
            
            self.computing_solution = False
        
        threading.Thread(target=solution_thread, daemon=True).start()

    # ------------------------------------------------------------------------
    # Evaluation Bar & Win Probability
    # ------------------------------------------------------------------------
    
    def _evaluation_loop(self):
        """Background evaluation loop for win probability bar."""
        while self.eval_thread_running:
            try:
                if self.mode != 'two' or self.searching:
                    time.sleep(0.1)
                    continue

                with self.eval_lock:
                    try:
                        current_state = self.state
                        current_hash = hash(current_state)
                    except:
                        time.sleep(0.1)
                        continue
                    
                    if self.current_eval_state_hash != current_hash:
                        self.persistent_mcts = MCTS(current_state, time_limit=999, use_heuristic=True)
                        self.current_eval_state_hash = current_hash
                    
                    if current_state.is_terminal():
                        winner = 1 - current_state.player
                        self.win_prob_p1 = 1.0 if winner == 0 else 0.0
                        time.sleep(0.2)
                        continue
                
                start_time = time.time()
                while time.time() - start_time < self.eval_update_interval:
                    if not self.eval_thread_running:
                        break
                    with self.eval_lock:
                        if self.persistent_mcts is None:
                            break
                        try:
                            node = self.persistent_mcts.select()
                            if node and not node.state.is_terminal():
                                if not node.is_fully_expanded():
                                    expanded = node.expand()
                                    if expanded:
                                        node = expanded
                                if node and not node.state.is_terminal():
                                    result = node.rollout(use_heuristic=self.persistent_mcts.use_heuristic)
                                    node.backpropagate(result)
                        except:
                            pass
                
                with self.eval_lock:
                    if self.persistent_mcts and self.persistent_mcts.root and self.persistent_mcts.root.visits > 0:
                        root = self.persistent_mcts.root
                        win_rate = 0.5
                        if root.children:
                            try:
                                best_child = max(root.children, key=lambda c: c.visits if c else 0)
                                if best_child and best_child.visits > 0:
                                    win_rate = best_child.wins / best_child.visits
                            except:
                                pass
                        p1_win = win_rate if current_state.player == 0 else 1 - win_rate
                        self.win_prob_p1 = p1_win
            except:
                time.sleep(0.1)
        
        print("Evaluation thread exited")

    def has_winning_move(self):
        """Check if current player has an immediate winning move."""
        for move in self.state.get_moves():
            temp_state = self.state.apply_move(move)
            if temp_state.is_terminal():
                return True
        return False

    def draw_evaluation_bar(self):
        """Draw the win probability bar (two-player only)."""
        if self.mode != 'two':
            return
        
        # Determine win probability
        if self.has_winning_move():
            current_win_prob = 1.0 if self.state.player == 0 else 0.0
        elif self.state.is_terminal():
            current_win_prob = 0.0 if self.state.player == 0 else 1.0
        else:
            current_win_prob = self.win_prob_p1
        
        # Draw bar
        pygame.draw.rect(self.screen, COLORS['eval_bar_p2'], (BAR_X, BAR_Y, BAR_WIDTH, BAR_HEIGHT))
        split_height = int(current_win_prob * BAR_HEIGHT)
        if split_height > 0:
            green_rect = pygame.Rect(BAR_X, BAR_Y + BAR_HEIGHT - split_height, BAR_WIDTH, split_height)
            pygame.draw.rect(self.screen, COLORS['eval_bar_p1'], green_rect)
        pygame.draw.rect(self.screen, COLORS['edge'], (BAR_X, BAR_Y, BAR_WIDTH, BAR_HEIGHT), 2)
        
        # Label
        if self.state.is_terminal():
            label = "P1 WIN" if current_win_prob == 1.0 else "P1 LOSS"
            color = COLORS['eval_bar_p1'] if current_win_prob == 1.0 else COLORS['eval_bar_p2']
            text = self.font.render(label, True, color)
        else:
            percent = int(current_win_prob * 100)
            text = self.font.render(f"P1 {percent}%", True, COLORS['text'])
        text_rect = text.get_rect(center=(BAR_X + BAR_WIDTH//2, BAR_Y - 20))
        self.screen.blit(text, text_rect)
        
        # MCTS tree visualization
        if self.fullscreen and self.show_tree_viz and self.current_mcts_root:
            if not self.tree_visualizer:
                self.tree_visualizer = MCTSTreeVisualizer(self.screen, self.font, self.small_font)
                self.tree_visualizer.build_tree(self.current_mcts_root)
            else:
                if self.tree_visualizer.root_viz and self.tree_visualizer.root_viz.node != self.current_mcts_root:
                    self.tree_visualizer.build_tree(self.current_mcts_root)
            self.tree_visualizer.draw()

    # ------------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------------
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            self.draw_board()
            self.draw_stats()
            self.draw_buttons()
            self.draw_evaluation_bar()
            self.draw_heatmap_legend()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    gui = Gui()
    gui.run()