"""
Monte Carlo Tree Search for two-player Peg Solitaire.
Features:
- UCB1 node selection with heuristic guidance
- Winning move detection
- Heuristic-guided rollouts (70% heuristic, 30% random)
- Symmetry-aware state evaluation
"""

import math
import random
import time
from board import PegSolitaireBoard


class MCTSNode:
    """Node in the MCTS search tree."""
    
    # Predefined positions for performance (avoid recreating lists)
    _DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Corner positions (hard to remove, penalize heavily)
    _CORNERS = [
        (0, 2), (0, 3), (0, 4),
        (2, 0), (3, 0), (4, 0),
        (2, 6), (3, 6), (4, 6),
        (6, 2), (6, 3), (6, 4)
    ]
    
    # Center and near-center positions (reward)
    _CENTER_POSITIONS = [
        (3, 3), (2, 3), (3, 2), (3, 4), (4, 3),
        (2, 2), (2, 4), (4, 2), (4, 4)
    ]

    def __init__(self, state, parent=None, move=None):
        self.state = state          # PegState object
        self.parent = parent        # Parent node (None for root)
        self.move = move            # Move that led to this state
        self.children = []          # List of child nodes
        self.visits = 0             # Number of times this node was visited
        self.wins = 0.0             # Number of wins (can be fractional)

    def __eq__(self, other):
        """Two nodes are equal if they represent the same state and move."""
        if not isinstance(other, MCTSNode):
            return False
        if self.move != other.move:
            return False
        return self.state.grid == other.state.grid

    def __hash__(self):
        """Hash based on state grid and move (for set/dict lookups)."""
        return hash((self.state.grid, self.move))

    def is_fully_expanded(self):
        """Check if all possible moves from this state have been explored."""
        return len(self.children) == len(self.state.get_moves())

    def has_winning_move(self):
        """
        Check if current player has an immediate winning move.
        
        Returns:
            (bool, move): True and the winning move if exists, else (False, None)
        """
        for move in self.state.get_moves():
            temp_state = self.state.apply_move(move)
            if temp_state.is_terminal():
                return True, move
        return False, None

    def best_child(self, exploration_constant=1.4, use_heuristic=True):
        """
        Select best child using UCB1 formula with optional heuristic bonus.
        
        UCB1 formula: exploit + explore + heuristic_bonus
        - exploit = win_rate (wins/visits)
        - explore = C * sqrt(log(parent_visits) / child_visits)
        - heuristic_bonus = evaluation_value * 0.1
        
        Args:
            exploration_constant: Controls exploration vs exploitation (C)
            use_heuristic: Whether to add heuristic bonus to score
        
        Returns:
            Best child node according to UCB1
        """
        # Immediate win takes priority
        for child in self.children:
            if child.state.is_terminal():
                return child
        
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            # Unexplored nodes get infinite score (explore first)
            if child.visits == 0:
                return child
            
            # UCB1 calculation
            exploit = child.wins / child.visits
            explore = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            
            # Heuristic bonus (optional)
            heuristic_bonus = 0
            if use_heuristic:
                heuristic_value = self._evaluate_state(child.state)
                heuristic_bonus = heuristic_value * 0.1
            
            score = exploit + explore + heuristic_bonus
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def _evaluate_state(self, state):
        """
        Heuristic evaluation of a board state from the current player's perspective.
        Positive score = good for the player who just moved.
        
        Scoring components:
        - Peg count: Fewer pegs = better (exponential scaling)
        - Mobility: More legal moves = better
        - Center control: Center pegs are valuable
        - Corner penalty: Corner pegs are bad
        - Adjacent pairs: Moves that can be made = good
        - Isolation penalty: Pegs with no neighbors = bad
        
        Returns:
            int: Score clamped to [-100, 100]
        """
        grid = state.grid
        
        # ===== PEG COUNT (exponential scaling) =====
        pegs = PegSolitaireBoard.count_pegs(grid)
        if pegs == 1:
            return 1000  # Winning state (highest priority)
        elif pegs <= 3:
            peg_score = (33 - pegs) * 15  # Near-win: higher weight
        else:
            peg_score = (33 - pegs) * 5   # Normal weight
        
        # ===== MOBILITY =====
        moves = PegSolitaireBoard.get_moves(grid)
        mobility_score = len(moves) * 4
        
        # ===== CENTER CONTROL =====
        center_score = 0
        for r, c in self._CENTER_POSITIONS:
            if PegSolitaireBoard.is_valid(r, c) and grid[r][c] == 1:
                # Center cell (3,3) is more valuable
                if (r, c) == (3, 3):
                    center_score += 5
                else:
                    center_score += 2
        
        # ===== CORNER PENALTY =====
        corner_penalty = 0
        for r, c in self._CORNERS:
            if PegSolitaireBoard.is_valid(r, c) and grid[r][c] == 1:
                corner_penalty += 8
        
        # ===== ADJACENT PAIR BONUS =====
        # Check for pegs that can jump each other (potential moves)
        pair_bonus = 0
        for r in range(7):
            for c in range(7):
                if grid[r][c] != 1:
                    continue
                for dr, dc in self._DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if (PegSolitaireBoard.is_valid(nr, nc) and grid[nr][nc] == 1 and
                        PegSolitaireBoard.is_valid(r + 2*dr, c + 2*dc) and 
                        grid[r + 2*dr][c + 2*dc] == 0):
                        pair_bonus += 3
                        break  # Count each peg once
        
        # ===== ISOLATION PENALTY =====
        isolation_penalty = 0
        for r in range(7):
            for c in range(7):
                if grid[r][c] != 1:
                    continue
                adjacent = 0
                for dr, dc in self._DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if PegSolitaireBoard.is_valid(nr, nc) and grid[nr][nc] == 1:
                        adjacent += 1
                if adjacent == 0:
                    isolation_penalty += 4   # Completely isolated
                elif adjacent == 1:
                    isolation_penalty += 2   # Semi-isolated
        
        # ===== TOTAL SCORE =====
        total = (peg_score + mobility_score + center_score + pair_bonus - 
                 corner_penalty - isolation_penalty)
        
        # Clamp to [-100, 100] for consistent rollout returns
        return max(-100, min(100, total))

    def expand(self):
        """
        Expand the node by adding one unexplored child.
        Prioritizes winning moves if available.
        
        Returns:
            New child node, or None if node is fully expanded
        """
        # Priority 1: Immediate winning move
        has_win, winning_move = self.has_winning_move()
        if has_win:
            # Check if already explored
            for child in self.children:
                if child.move == winning_move:
                    return None
            # Add winning move as child
            new_state = self.state.apply_move(winning_move)
            child = MCTSNode(new_state, parent=self, move=winning_move)
            self.children.append(child)
            return child
        
        # Priority 2: Add any unexplored move
        tried_moves = {child.move for child in self.children}
        for move in self.state.get_moves():
            if move not in tried_moves:
                new_state = self.state.apply_move(move)
                child = MCTSNode(new_state, parent=self, move=move)
                self.children.append(child)
                return child
        
        # Fully expanded - return random existing child as fallback
        return random.choice(self.children) if self.children else None

    def rollout(self, use_heuristic=True):
        """
        Simulate a game from this node until termination.
        
        Move selection strategy:
        - 70% heuristic-guided (choose best move by evaluation)
        - 30% random (exploration)
        - Always take winning moves when available
        
        Args:
            use_heuristic: Whether to use heuristic guidance (vs pure random)
        
        Returns:
            float: 1.0 if current player wins, 0.0 if loses, fractional for non-terminal
        """
        current_state = self.state
        
        while not current_state.is_terminal():
            moves = current_state.get_moves()
            if not moves:
                break
            
            # Check for immediate winning move
            winning_move = None
            for move in moves:
                temp_state = current_state.apply_move(move)
                if temp_state.is_terminal():
                    winning_move = move
                    break
            
            if winning_move is not None:
                current_state = current_state.apply_move(winning_move)
                break
            
            # Heuristic-guided move selection (70% heuristic, 30% random)
            if use_heuristic and random.random() < 0.7:
                best_move = None
                best_score = -float('inf')
                for move in moves:
                    temp_state = current_state.apply_move(move)
                    score = self._evaluate_state(temp_state)
                    if score > best_score:
                        best_score = score
                        best_move = move
                move = best_move
            else:
                move = random.choice(moves)
                
            current_state = current_state.apply_move(move)
        
        # Winner is the player who DIDN'T make the last move
        winner = 1 - current_state.player
        
        if current_state.is_terminal():
            return 1.0 if winner == self.state.player else 0.0
        else:
            # Non-terminal: use heuristic evaluation as fractional result
            eval_score = self._evaluate_state(current_state)
            return max(0.0, min(1.0, 0.5 + (eval_score / 200)))

    def backpropagate(self, result):
        """
        Propagate the simulation result back up the tree.
        
        Args:
            result: Win probability (0.0 to 1.0) from current player's perspective
        """
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    """
    Monte Carlo Tree Search for two-player Peg Solitaire.
    
    Usage:
        mcts = MCTS(state, time_limit=5.0)
        best_move = mcts.search()
    """
    
    def __init__(self, state, time_limit=5.0, use_heuristic=True):
        """
        Initialize MCTS.
        
        Args:
            state: Starting PegState
            time_limit: Maximum search time in seconds
            use_heuristic: Whether to use heuristic guidance
        """
        self.root = MCTSNode(state)
        self.time_limit = time_limit
        self.use_heuristic = use_heuristic
        self.simulations = 0

    def search(self, progress_callback=None):
        """
        Run MCTS and return the best move.
        
        Args:
            progress_callback: Optional function called periodically for UI updates
        
        Returns:
            Best move as ((r1,c1), (r2,c2)) or None if no moves available
        """
        start_time = time.time()
        self.simulations = 0
        last_callback_time = start_time
        
        while (time.time() - start_time) < self.time_limit:
            # Selection phase
            node = self.select()
            if node is None:
                continue
                
            # Expansion and simulation phases
            if not node.state.is_terminal():
                if not node.is_fully_expanded():
                    expanded_node = node.expand()
                    if expanded_node is not None:
                        node = expanded_node
                
                # Rollout and backpropagation
                if node is not None and not node.state.is_terminal():
                    result = node.rollout(use_heuristic=self.use_heuristic)
                    node.backpropagate(result)
                    self.simulations += 1
            
            # Callback for UI updates (tree visualization)
            if progress_callback and time.time() - last_callback_time > 0.2:
                progress_callback()
                last_callback_time = time.time()
        
        # Select best move from root
        if not self.root.children:
            return None
        
        # Priority 1: Immediate winning move
        for child in self.root.children:
            if child.state.is_terminal():
                return child.move
        
        # Priority 2: Most visited child (robust)
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.move

    def select(self):
        """
        Traverse the tree to select a node for expansion.
        Uses UCB1 at each step until reaching an unexpanded or terminal node.
        
        Returns:
            Node to expand (or root if no selection possible)
        """
        node = self.root
        while not node.state.is_terminal() and node.is_fully_expanded() and node.children:
            next_node = node.best_child(use_heuristic=self.use_heuristic)
            if next_node is None:
                break
            node = next_node
        return node