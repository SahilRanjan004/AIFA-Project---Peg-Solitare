"""
MCTS Tree Visualization for Peg Solitaire.
Displays the growing search tree structure covering the entire screen in fullscreen mode.

Features:
- Node colors: Green (high win rate) to Red (low win rate)
- Node size scales with visit count (log scale)
- Glowing highlight for chosen AI move
- Performance optimizations: font caching, background caching, limited draw depth
"""

import pygame
import math


class MCTSNodeViz:
    """Visual representation of an MCTS node with screen coordinates."""
    
    def __init__(self, node, depth=0, x=0, y=0):
        """
        Initialize a visual tree node.
        
        Args:
            node: The MCTSNode being visualized
            depth: Tree depth (0 = root)
            x, y: Screen coordinates for drawing
        """
        self.node = node
        self.depth = depth
        self.x = x
        self.y = y
        self.children = []  # List of child MCTSNodeViz objects
        
    def layout(self, x, y, level_height=80):
        """
        Recursively compute positions for all nodes in the tree.
        
        Uses a simple tree layout algorithm where:
        - Parent is centered above children
        - Children are evenly spaced horizontally
        - Spacing decreases with depth to prevent excessive width
        
        Args:
            x: X-coordinate for this node
            y: Y-coordinate for this node
            level_height: Vertical spacing between levels (pixels)
        """
        self.x = x
        self.y = y
        
        if not self.node.children:
            return
        
        num_children = len(self.node.children)
        spacing = 100  # Base horizontal spacing
        
        # Reduce spacing for deeper levels (prevents tree from getting too wide)
        if self.depth > 0:
            spacing = 80 / (self.depth + 1)
        
        # Calculate total width and starting position
        total_width = num_children * spacing
        start_x = x - total_width / 2 + spacing / 2
        
        # Layout each child recursively
        for i, child in enumerate(self.node.children):
            child_viz = MCTSNodeViz(child, self.depth + 1)
            child_x = start_x + i * spacing
            child_viz.layout(child_x, y + level_height, level_height)
            self.children.append(child_viz)


class MCTSTreeVisualizer:
    """
    Full-screen tree visualizer for MCTS search.
    
    Performance optimizations:
    - Font objects cached (created once, reused)
    - Background surface cached (regenerated only on resize)
    - Pulse animation updates every 3 frames
    - Skips drawing nodes beyond depth 10
    """
    
    # Node color constants
    COLOR_BRIGHT_GREEN = (100, 255, 100)
    COLOR_YELLOW = (255, 255, 100)
    COLOR_RED = (255, 100, 100)
    COLOR_GREY = (80, 80, 80)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    
    def __init__(self, screen, font, small_font):
        """
        Initialize the tree visualizer.
        
        Args:
            screen: Pygame display surface
            font: Regular font for text (24px)
            small_font: Small font for stats/legends (18px)
        """
        self.screen = screen
        self.font = font
        self.small_font = small_font
        self.root_viz = None
        self.highlighted_node = None  # Node to highlight (chosen AI move)
        self.waiting_for_click = False
        self.pulse_offset = 0  # For glowing animation
        self.root_y = 80  # Y-position of root node
        
        # Cache fonts for performance (created once, reused per node)
        self.visit_font = pygame.font.Font(None, 28)   # Visit count font
        self.percent_font = pygame.font.Font(None, 26) # Win percentage font
        self.move_font = pygame.font.Font(None, 24)    # Move text font
        
        # Cache background surface (regenerate only on window resize)
        self.background_cache = None
        self.last_screen_size = None
        
        # Frame counter for pulse animation (updates every 3 frames)
        self.frame_counter = 0
        
    def build_tree(self, root_node):
        """
        Build the visual tree from an MCTS root node.
        
        Args:
            root_node: The root MCTSNode to visualize
        """
        if not root_node:
            return
        
        screen_width = self.screen.get_width()
        
        # Create root visualization node
        self.root_viz = MCTSNodeViz(root_node)
        
        # Layout the tree
        root_x = screen_width // 2
        self.root_viz.layout(root_x, self.root_y, level_height=80)
        
        # Adjust positions to prevent overlapping
        self.adjust_positions(self.root_viz)
        
        # Center the entire tree horizontally
        self.center_tree()
        
    def adjust_positions(self, node_viz):
        """
        Recursively adjust node positions to prevent overlapping.
        
        This is a simple force-directed adjustment:
        - If children overlap horizontally, push them apart
        - Scale children if they exceed parent width
        
        Args:
            node_viz: Current node to adjust (and its children)
        """
        if not node_viz.children:
            return
        
        # First adjust children recursively
        for child in node_viz.children:
            self.adjust_positions(child)
        
        # Check for overlapping children
        for i in range(len(node_viz.children)):
            for j in range(i + 1, len(node_viz.children)):
                child1 = node_viz.children[i]
                child2 = node_viz.children[j]
                
                # If they overlap, push them apart
                if abs(child1.x - child2.x) < 50:
                    separation = 80 - abs(child1.x - child2.x)
                    child1.x -= separation / 2
                    child2.x += separation / 2
        
        # Scale children to fit within parent bounds
        if node_viz.children:
            min_x = min(c.x for c in node_viz.children)
            max_x = max(c.x for c in node_viz.children)
            width = max_x - min_x
            parent_width = 120
            
            if width > parent_width:
                scale = parent_width / width
                center_x = (min_x + max_x) / 2
                for child in node_viz.children:
                    child.x = center_x + (child.x - center_x) * scale
    
    def center_tree(self):
        """Center the entire tree horizontally on the screen."""
        if not self.root_viz:
            return
        
        # Find min and max x positions
        min_x = float('inf')
        max_x = -float('inf')
        nodes = [self.root_viz]
        while nodes:
            node = nodes.pop()
            min_x = min(min_x, node.x)
            max_x = max(max_x, node.x)
            nodes.extend(node.children)
        
        # Calculate shift needed
        screen_width = self.screen.get_width()
        current_center = (min_x + max_x) / 2
        shift_x = screen_width / 2 - current_center
        
        # Apply shift to all nodes
        nodes = [self.root_viz]
        while nodes:
            node = nodes.pop()
            node.x += shift_x
            nodes.extend(node.children)
    
    def handle_event(self, event):
        """Handle mouse events (currently does nothing - for future zoom/pan)."""
        pass
        
    def set_highlighted_node(self, node):
        """Set the node to highlight (usually the chosen AI move)."""
        self.highlighted_node = node
        
    def set_waiting_for_click(self, waiting):
        """Set whether we're waiting for a click to clear the tree."""
        self.waiting_for_click = waiting
    
    def draw_background(self):
        """
        Draw the background with gradient and grid.
        Cached for performance - regenerated only on window resize.
        """
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        # Use cached background if screen size hasn't changed
        if (self.background_cache is not None and 
            self.last_screen_size == (screen_width, screen_height)):
            self.screen.blit(self.background_cache, (0, 0))
            return
        
        # Create new background surface
        bg_surface = pygame.Surface((screen_width, screen_height))
        
        # Draw dark gradient (top to bottom)
        for i in range(screen_height):
            intensity = 20 + int(15 * (i / screen_height))
            color = (intensity, intensity, intensity + 10)
            pygame.draw.line(bg_surface, color, (0, i), (screen_width, i))
        
        # Draw subtle grid lines (sparse for performance)
        grid_color = (50, 50, 70)
        step = 150  # Increased from 100 for fewer lines
        for x in range(0, screen_width, step):
            pygame.draw.line(bg_surface, grid_color, (x, 0), (x, screen_height), 1)
        for y in range(0, screen_height, step):
            pygame.draw.line(bg_surface, grid_color, (0, y), (screen_width, y), 1)
        
        # Cache the background
        self.background_cache = bg_surface
        self.last_screen_size = (screen_width, screen_height)
        
        # Draw to screen
        self.screen.blit(bg_surface, (0, 0))
        
    def draw_node(self, node_viz, depth=0):
        """
        Draw a single node and all its descendants recursively.
        
        Args:
            node_viz: Current node to draw
            depth: Current depth (used to limit recursion)
        """
        # Performance: skip very deep nodes (not visible clearly anyway)
        if depth > 10:
            return
            
        draw_x = node_viz.x
        draw_y = node_viz.y
        
        # Culling: skip nodes outside visible area
        screen_height = self.screen.get_height()
        screen_width = self.screen.get_width()
        node_radius = 30
        if draw_y + node_radius < 0 or draw_y - node_radius > screen_height:
            return
        if draw_x + node_radius < 0 or draw_x - node_radius > screen_width:
            return
        
        # Draw connections to children first (so they appear behind nodes)
        for child in node_viz.children:
            child_draw_x = child.x
            child_draw_y = child.y
            if -50 < child_draw_y < screen_height + 50:
                # Color line based on parent's win rate
                win_rate = (node_viz.node.wins / node_viz.node.visits 
                           if node_viz.node.visits > 0 else 0.5)
                line_color = (100, 150, 100) if win_rate > 0.5 else (150, 100, 100)
                pygame.draw.line(self.screen, line_color, 
                               (draw_x, draw_y), 
                               (child_draw_x, child_draw_y), 2)
            self.draw_node(child, depth + 1)
        
        # Check if this node should be highlighted
        is_highlighted = (self.highlighted_node is not None and 
                          node_viz.node == self.highlighted_node)
        
        # Update pulse animation (every 3 frames for performance)
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            self.pulse_offset = (self.pulse_offset + 0.1) % (math.pi * 2)
        
        # Node size scales with visit count (log scale)
        visits = node_viz.node.visits
        radius = min(30, max(20, 18 + int(math.log(visits + 1) * 2)))
        
        # Determine node color based on win rate (from AI's perspective)
        if visits > 0:
            win_rate = node_viz.node.wins / visits
            
            # Convert to AI's perspective (player 1 = AI)
            if node_viz.node.state.player == 0:  # Human's turn
                win_rate_for_ai = 1 - win_rate
            else:  # AI's turn
                win_rate_for_ai = win_rate
            
            # Color: Green (high win) to Red (low win)
            red = int(255 * (1 - win_rate_for_ai))
            green = int(255 * win_rate_for_ai)
            blue = 80
            color = (red, green, blue)
        else:
            color = self.COLOR_GREY  # Unexplored node
        
        # Draw glowing rings for highlighted node
        if is_highlighted:
            pulse = 0.7 + 0.3 * (math.sin(self.pulse_offset) + 1) / 2
            
            # Multiple expanding rings for glow effect
            for i in range(5, 0, -1):
                glow_radius = radius + i * 5
                alpha = int(120 * pulse * (1 - (i / 6)))
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (255, 220, 80, alpha), 
                                 (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (draw_x - glow_radius, draw_y - glow_radius))
            
            # Extra bright inner ring
            inner_radius = radius + 2
            pygame.draw.circle(self.screen, (255, 255, 150), 
                            (int(draw_x), int(draw_y)), inner_radius, 4)
        
        # Draw the main node circle
        pygame.draw.circle(self.screen, color, (int(draw_x), int(draw_y)), radius)
        
        # Draw border
        if is_highlighted:
            pygame.draw.circle(self.screen, (255, 255, 100), (int(draw_x), int(draw_y)), radius + 2, 3)
            pygame.draw.circle(self.screen, (255, 200, 50), (int(draw_x), int(draw_y)), radius, 2)
        else:
            pygame.draw.circle(self.screen, (200, 200, 200), (int(draw_x), int(draw_y)), radius, 2)
        
        # Draw visit count (inside node)
        if visits > 0:
            text = self.visit_font.render(str(visits), True, (255, 255, 200))
            text_rect = text.get_rect(center=(draw_x, draw_y - 8))
            # Dark background for contrast
            bg_rect = text_rect.inflate(10, 6)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
            pygame.draw.rect(self.screen, (80, 80, 80), bg_rect, 1)
            self.screen.blit(text, text_rect)
            
            # Draw win percentage (below node)
            win_rate = node_viz.node.wins / visits
            if node_viz.node.state.player == 0:
                win_percent = int((1 - win_rate) * 100)
            else:
                win_percent = int(win_rate * 100)
            
            # Color code based on percentage
            if win_percent >= 70:
                percent_color = self.COLOR_BRIGHT_GREEN
            elif win_percent >= 40:
                percent_color = self.COLOR_YELLOW
            else:
                percent_color = self.COLOR_RED
            
            text = self.percent_font.render(f"{win_percent}%", True, percent_color)
            text_rect = text.get_rect(center=(draw_x, draw_y + 18))
            # Dark background for contrast
            bg_rect = text_rect.inflate(8, 5)
            pygame.draw.rect(self.screen, (0, 0, 0, 200), bg_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), bg_rect, 1)
            self.screen.blit(text, text_rect)
        
        # Draw move info ONLY for the highlighted (chosen) node
        if node_viz.node.move and node_viz.depth > 0 and is_highlighted:
            move = node_viz.node.move
            
            # Move text above node
            move_text = f"{move[0][0]},{move[0][1]} -> {move[1][0]},{move[1][1]}"
            text = self.move_font.render(move_text, True, (255, 255, 150))
            text_rect = text.get_rect(center=(draw_x, draw_y - radius - 15))
            bg_rect = text_rect.inflate(20, 8)
            pygame.draw.rect(self.screen, (0, 0, 0, 220), bg_rect)
            pygame.draw.rect(self.screen, (255, 255, 100), bg_rect, 2)
            self.screen.blit(text, text_rect)
            
            # Move text below node (compact version)
            move_text2 = f"Move: ({move[0][0]},{move[0][1]})->({move[1][0]},{move[1][1]})"
            text2 = self.small_font.render(move_text2, True, (200, 200, 200))
            text2_rect = text2.get_rect(center=(draw_x, draw_y + radius + 12))
            bg2_rect = text2_rect.inflate(12, 6)
            pygame.draw.rect(self.screen, (0, 0, 0, 200), bg2_rect)
            self.screen.blit(text2, text2_rect)
        
        # Draw checkmark on highlighted node
        if is_highlighted:
            check_text = self.small_font.render("✓", True, (255, 255, 100))
            check_rect = check_text.get_rect(center=(draw_x + radius - 6, draw_y - radius + 6))
            self.screen.blit(check_text, check_rect)
    
    def draw(self):
        """Draw the entire tree covering the full screen."""
        if not self.root_viz or not self.root_viz.node:
            return
        
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        # Draw cached background
        self.draw_background()
        
        # Draw semi-transparent overlay for better readability
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))
        
        # Draw title with glow effect
        title_font = pygame.font.Font(None, 48)
        title = title_font.render("MCTS Search Tree", True, self.COLOR_WHITE)
        title_rect = title.get_rect(center=(screen_width // 2, 40))
        # Shadow for depth
        shadow = title_font.render("MCTS Search Tree", True, self.COLOR_BLACK)
        shadow_rect = shadow.get_rect(center=(screen_width // 2 + 2, 42))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(title, title_rect)
        
        # Draw stats panel (top left)
        root = self.root_viz.node
        if root.visits > 0:
            stats_panel = pygame.Surface((280, 100), pygame.SRCALPHA)
            stats_panel.fill((0, 0, 0, 180))
            self.screen.blit(stats_panel, (20, 20))
            
            stats_y = 30
            stats_texts = [
                f"Total Simulations: {root.visits}",
                f"Nodes Explored: {self.count_nodes(self.root_viz)}",
                f"Tree Depth: {self.get_tree_depth(self.root_viz)}",
                f"Avg Branching: {self.get_avg_branching(self.root_viz):.1f}"
            ]
            for text_str in stats_texts:
                text = self.small_font.render(text_str, True, (200, 200, 200))
                self.screen.blit(text, (30, stats_y))
                stats_y += 22
        
        # Draw legend panel (bottom left)
        legend_panel = pygame.Surface((280, 140), pygame.SRCALPHA)
        legend_panel.fill((0, 0, 0, 180))
        self.screen.blit(legend_panel, (20, screen_height - 160))
        
        legend_y = screen_height - 140
        legend_texts = [
            ("Node color: Green/Yellow/Red based on win %", (180, 180, 180)),
            ("Number inside: Visit count", (180, 180, 180)),
            ("Percentage below: Win %", (180, 180, 180)),
            ("Node size: Log(visits)", (180, 180, 180)),
            ("Glowing yellow rings: Chosen AI move", (255, 255, 100)),
        ]
        for text_str, color in legend_texts:
            text = self.small_font.render(text_str, True, color)
            self.screen.blit(text, (30, legend_y))
            legend_y += 20
        
        # Draw instruction panel (bottom center) when waiting for click
        if self.waiting_for_click:
            instruction_panel = pygame.Surface((350, 50), pygame.SRCALPHA)
            instruction_panel.fill((255, 200, 0, 200))
            self.screen.blit(instruction_panel, (screen_width // 2 - 175, screen_height - 70))
            
            inst_text = self.font.render("Click anywhere to continue", True, self.COLOR_BLACK)
            inst_rect = inst_text.get_rect(center=(screen_width // 2, screen_height - 45))
            self.screen.blit(inst_text, inst_rect)
        
        # Draw the tree
        self.draw_node(self.root_viz)
        
        # Draw border
        pygame.draw.rect(self.screen, (100, 100, 150, 50), (0, 0, screen_width, screen_height), 3)
    
    def count_nodes(self, node_viz):
        """Count total number of nodes in the tree."""
        count = 1
        for child in node_viz.children:
            count += self.count_nodes(child)
        return count
    
    def get_tree_depth(self, node_viz):
        """Get maximum depth of the tree."""
        if not node_viz.children:
            return 1
        return 1 + max(self.get_tree_depth(child) for child in node_viz.children)
    
    def get_avg_branching(self, node_viz):
        """Calculate average branching factor (average number of children per node)."""
        if not node_viz.children:
            return 0
        total_branches = len(node_viz.children)
        child_branches = sum(self.get_avg_branching(child) for child in node_viz.children)
        return (total_branches + child_branches) / (1 + len(node_viz.children))