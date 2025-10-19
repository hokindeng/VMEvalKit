"""
Tic-Tac-Toe Reasoning Task for VMEvalKit

Tic-tac-toe game state generation system for video model evaluation.
Follows the same data format as other tasks with first/final frames and prompts.

The task evaluates:
- Strategic thinking and game theory
- Pattern recognition for winning conditions
- Logical reasoning for optimal moves
- Visual understanding of game states

Author: VMEvalKit Team
"""

import json
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import prompts from centralized location
from .PROMPTS import PROMPTS


@dataclass
class TicTacToeTaskPair:
    """
    Data structure for tic-tac-toe video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The initial game state
    - final_image: The winning/optimal move state
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The initial game state image
    final_image_path: str           # The final game state image 
    task_category: str              # "TicTacToe"
    tictactoe_data: Dict[str, Any] = None  # Metadata
    difficulty: str = ""            # "easy", "medium", "hard"
    initial_board: List[List[str]] = None  # The initial 3x3 board state
    final_board: List[List[str]] = None    # The final 3x3 board state
    player_to_move: str = "X"       # "X" or "O" - who should make the move
    game_outcome: str = ""          # "win", "block", "optimal"
    winning_line: List[Tuple[int, int]] = None  # Coordinates of winning line
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class TicTacToeDataset:
    """Collection of TicTacToeTaskPair instances."""
    pairs: List[TicTacToeTaskPair]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "pairs": [asdict(pair) for pair in self.pairs]
        }


class TicTacToeTaskGenerator:
    """Generates tic-tac-toe reasoning tasks for video model evaluation."""
    
    def __init__(self):
        self.empty_board = [["" for _ in range(3)] for _ in range(3)]
        
    def generate_tasks(self, num_samples: int) -> List[TicTacToeTaskPair]:
        """Generate a collection of tic-tac-toe reasoning tasks."""
        tasks = []
        
        # Generate different types of scenarios
        scenarios = [
            ("winning_move", 0.4),      # 40% - Find winning move
            ("blocking_move", 0.3),      # 30% - Block opponent's win
            ("optimal_move", 0.2),       # 20% - Make optimal strategic move
            ("fork_setup", 0.1)         # 10% - Set up a fork (advanced)
        ]
        
        for i in range(num_samples):
            scenario_type = self._weighted_choice(scenarios)
            task = self._generate_scenario(scenario_type, i)
            tasks.append(task)
            
        return tasks
    
    def _weighted_choice(self, scenarios: List[Tuple[str, float]]) -> str:
        """Choose scenario type based on weights."""
        rand = random.random()
        cumulative = 0
        for scenario, weight in scenarios:
            cumulative += weight
            if rand <= cumulative:
                return scenario
        return scenarios[-1][0]  # Fallback
    
    def _generate_scenario(self, scenario_type: str, task_id: int) -> TicTacToeTaskPair:
        """Generate a specific type of tic-tac-toe scenario."""
        
        if scenario_type == "winning_move":
            return self._generate_winning_move_scenario(task_id)
        elif scenario_type == "blocking_move":
            return self._generate_blocking_move_scenario(task_id)
        elif scenario_type == "optimal_move":
            return self._generate_optimal_move_scenario(task_id)
        elif scenario_type == "fork_setup":
            return self._generate_fork_setup_scenario(task_id)
        else:
            return self._generate_winning_move_scenario(task_id)
    
    def _generate_winning_move_scenario(self, task_id: int) -> TicTacToeTaskPair:
        """Generate a scenario where player can win in one move."""
        board = [["" for _ in range(3)] for _ in range(3)]
        
        # Randomly choose who's about to win (X or O)
        player = random.choice(["X", "O"])
        opponent = "O" if player == "X" else "X"
        
        # Place some random moves first
        num_moves = random.randint(3, 6)
        positions = [(i, j) for i in range(3) for j in range(3)]
        random.shuffle(positions)
        
        current_player = "X"
        for i in range(num_moves):
            if i < len(positions):
                row, col = positions[i]
                board[row][col] = current_player
                current_player = "O" if current_player == "X" else "X"
        
        # Now find a winning move for the target player
        winning_move = self._find_winning_move(board, player)
        
        if winning_move:
            row, col = winning_move
            final_board = [row[:] for row in board]  # Deep copy
            final_board[row][col] = player
            
            difficulty = self._assess_difficulty(board, "winning_move")
            winning_line = self._get_winning_line(final_board, player)
            
            return TicTacToeTaskPair(
                id=f"tictactoe_{task_id:04d}",
                prompt=random.choice(PROMPTS),
                first_image_path="",  # Will be set during image generation
                final_image_path="",  # Will be set during image generation
                task_category="TicTacToe",
                difficulty=difficulty,
                initial_board=board,
                final_board=final_board,
                player_to_move=player,
                game_outcome="win",
                winning_line=winning_line,
                tictactoe_data={
                    "scenario_type": "winning_move",
                    "winning_move": winning_move,
                    "num_moves": num_moves
                }
            )
        else:
            # Fallback to blocking move if no winning move found
            return self._generate_blocking_move_scenario(task_id)
    
    def _generate_blocking_move_scenario(self, task_id: int) -> TicTacToeTaskPair:
        """Generate a scenario where player must block opponent's win."""
        board = [["" for _ in range(3)] for _ in range(3)]
        
        # Randomly choose who needs to block
        player = random.choice(["X", "O"])
        opponent = "O" if player == "X" else "X"
        
        # Place moves to create a situation where opponent is about to win
        num_moves = random.randint(4, 7)
        positions = [(i, j) for i in range(3) for j in range(3)]
        random.shuffle(positions)
        
        current_player = "X"
        for i in range(num_moves):
            if i < len(positions):
                row, col = positions[i]
                board[row][col] = current_player
                current_player = "O" if current_player == "X" else "X"
        
        # Find a blocking move
        blocking_move = self._find_winning_move(board, opponent)
        
        if blocking_move:
            row, col = blocking_move
            final_board = [row[:] for row in board]  # Deep copy
            final_board[row][col] = player  # Player blocks
            
            difficulty = self._assess_difficulty(board, "blocking_move")
            
            return TicTacToeTaskPair(
                id=f"tictactoe_{task_id:04d}",
                prompt=random.choice(PROMPTS),
                first_image_path="",  # Will be set during image generation
                final_image_path="",  # Will be set during image generation
                task_category="TicTacToe",
                difficulty=difficulty,
                initial_board=board,
                final_board=final_board,
                player_to_move=player,
                game_outcome="block",
                tictactoe_data={
                    "scenario_type": "blocking_move",
                    "blocking_move": blocking_move,
                    "num_moves": num_moves
                }
            )
        else:
            # Fallback to optimal move
            return self._generate_optimal_move_scenario(task_id)
    
    def _generate_optimal_move_scenario(self, task_id: int) -> TicTacToeTaskPair:
        """Generate a scenario requiring optimal strategic play."""
        board = [["" for _ in range(3)] for _ in range(3)]
        
        player = random.choice(["X", "O"])
        
        # Create a strategic position (e.g., center control, corner strategy)
        strategic_positions = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)]  # Center and corners
        random.shuffle(strategic_positions)
        
        # Place 2-4 strategic moves
        num_moves = random.randint(2, 4)
        current_player = "X"
        
        for i in range(min(num_moves, len(strategic_positions))):
            row, col = strategic_positions[i]
            board[row][col] = current_player
            current_player = "O" if current_player == "X" else "X"
        
        # Find optimal move
        optimal_move = self._find_optimal_move(board, player)
        
        if optimal_move:
            row, col = optimal_move
            final_board = [row[:] for row in board]  # Deep copy
            final_board[row][col] = player
            
            difficulty = self._assess_difficulty(board, "optimal_move")
            
            return TicTacToeTaskPair(
                id=f"tictactoe_{task_id:04d}",
                prompt=random.choice(PROMPTS),
                first_image_path="",  # Will be set during image generation
                final_image_path="",  # Will be set during image generation
                task_category="TicTacToe",
                difficulty=difficulty,
                initial_board=board,
                final_board=final_board,
                player_to_move=player,
                game_outcome="optimal",
                tictactoe_data={
                    "scenario_type": "optimal_move",
                    "optimal_move": optimal_move,
                    "num_moves": num_moves
                }
            )
        else:
            # Fallback to winning move
            return self._generate_winning_move_scenario(task_id)
    
    def _generate_fork_setup_scenario(self, task_id: int) -> TicTacToeTaskPair:
        """Generate a scenario where player sets up a fork (advanced)."""
        board = [["" for _ in range(3)] for _ in range(3)]
        
        player = random.choice(["X", "O"])
        
        # Create a fork setup position
        # This is more complex - for now, generate a strategic position
        board[1][1] = player  # Center
        board[0][0] = "O" if player == "X" else "X"  # Opponent in corner
        
        # Find a move that creates multiple threats
        fork_move = self._find_fork_move(board, player)
        
        if fork_move:
            row, col = fork_move
            final_board = [row[:] for row in board]  # Deep copy
            final_board[row][col] = player
            
            difficulty = "hard"  # Forks are advanced
            
            return TicTacToeTaskPair(
                id=f"tictactoe_{task_id:04d}",
                prompt=random.choice(PROMPTS),
                first_image_path="",  # Will be set during image generation
                final_image_path="",  # Will be set during image generation
                task_category="TicTacToe",
                difficulty=difficulty,
                initial_board=board,
                final_board=final_board,
                player_to_move=player,
                game_outcome="optimal",
                tictactoe_data={
                    "scenario_type": "fork_setup",
                    "fork_move": fork_move,
                    "num_moves": 2
                }
            )
        else:
            # Fallback to optimal move
            return self._generate_optimal_move_scenario(task_id)
    
    def _find_winning_move(self, board: List[List[str]], player: str) -> Optional[Tuple[int, int]]:
        """Find a winning move for the given player."""
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    # Try placing the player's mark
                    board[i][j] = player
                    if self._check_winner(board) == player:
                        board[i][j] = ""  # Reset
                        return (i, j)
                    board[i][j] = ""  # Reset
        return None
    
    def _find_optimal_move(self, board: List[List[str]], player: str) -> Optional[Tuple[int, int]]:
        """Find an optimal strategic move."""
        # Priority: center, corners, then edges
        priority_positions = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]
        
        for pos in priority_positions:
            row, col = pos
            if board[row][col] == "":
                return (row, col)
        return None
    
    def _find_fork_move(self, board: List[List[str]], player: str) -> Optional[Tuple[int, int]]:
        """Find a move that creates a fork (multiple winning threats)."""
        # Simplified fork detection - look for moves that create multiple threats
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = player
                    threats = self._count_threats(board, player)
                    board[i][j] = ""  # Reset
                    if threats >= 2:
                        return (i, j)
        return None
    
    def _count_threats(self, board: List[List[str]], player: str) -> int:
        """Count the number of winning threats for a player."""
        threats = 0
        
        # Check rows
        for i in range(3):
            if board[i].count(player) == 2 and board[i].count("") == 1:
                threats += 1
        
        # Check columns
        for j in range(3):
            col = [board[i][j] for i in range(3)]
            if col.count(player) == 2 and col.count("") == 1:
                threats += 1
        
        # Check diagonals
        diag1 = [board[i][i] for i in range(3)]
        if diag1.count(player) == 2 and diag1.count("") == 1:
            threats += 1
        
        diag2 = [board[i][2-i] for i in range(3)]
        if diag2.count(player) == 2 and diag2.count("") == 1:
            threats += 1
        
        return threats
    
    def _check_winner(self, board: List[List[str]]) -> Optional[str]:
        """Check if there's a winner on the board."""
        # Check rows
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != "":
                return board[i][0]
        
        # Check columns
        for j in range(3):
            if board[0][j] == board[1][j] == board[2][j] != "":
                return board[0][j]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != "":
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != "":
            return board[0][2]
        
        return None
    
    def _get_winning_line(self, board: List[List[str]], player: str) -> List[Tuple[int, int]]:
        """Get the coordinates of the winning line."""
        # Check rows
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == player:
                return [(i, 0), (i, 1), (i, 2)]
        
        # Check columns
        for j in range(3):
            if board[0][j] == board[1][j] == board[2][j] == player:
                return [(0, j), (1, j), (2, j)]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] == player:
            return [(0, 0), (1, 1), (2, 2)]
        if board[0][2] == board[1][1] == board[2][0] == player:
            return [(0, 2), (1, 1), (2, 0)]
        
        return []
    
    def _assess_difficulty(self, board: List[List[str]], scenario_type: str) -> str:
        """Assess the difficulty of the current board state."""
        empty_count = sum(row.count("") for row in board)
        
        if scenario_type == "winning_move":
            if empty_count <= 2:
                return "easy"
            elif empty_count <= 4:
                return "medium"
            else:
                return "hard"
        elif scenario_type == "blocking_move":
            if empty_count <= 3:
                return "easy"
            elif empty_count <= 5:
                return "medium"
            else:
                return "hard"
        else:  # optimal_move, fork_setup
            if empty_count <= 4:
                return "medium"
            else:
                return "hard"


def generate_tictactoe_board_image(board: List[List[str]], highlight_line: List[Tuple[int, int]] = None, 
                                 title: str = "Tic-Tac-Toe", output_path: str = None) -> str:
    """
    Generate a visual representation of a tic-tac-toe board.
    
    Args:
        board: 3x3 list representing the board state
        highlight_line: List of (row, col) tuples to highlight (for winning line)
        title: Title for the plot
        output_path: Path to save the image
        
    Returns:
        Path to the saved image
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Draw the grid
    for i in range(4):  # 4 lines for 3x3 grid
        ax.axhline(y=i, color='black', linewidth=2)
        ax.axvline(x=i, color='black', linewidth=2)
    
    # Draw the pieces
    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                # Draw X
                ax.plot([j+0.2, j+0.8], [2-i+0.2, 2-i+0.8], 'r-', linewidth=4)
                ax.plot([j+0.8, j+0.2], [2-i+0.2, 2-i+0.8], 'r-', linewidth=4)
            elif board[i][j] == "O":
                # Draw O
                circle = plt.Circle((j+0.5, 2-i+0.5), 0.3, fill=False, color='blue', linewidth=4)
                ax.add_patch(circle)
    
    # Highlight winning line if provided
    if highlight_line:
        for row, col in highlight_line:
            # Add a background highlight
            rect = patches.Rectangle((col+0.1, 2-row+0.1), 0.8, 0.8, 
                                   linewidth=0, facecolor='yellow', alpha=0.3)
            ax.add_patch(rect)
    
    # Set up the plot
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove the spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    # Save the image
    if output_path is None:
        output_path = f"/tmp/tictactoe_{random.randint(1000, 9999)}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """
    Create a dataset of tic-tac-toe reasoning tasks.
    
    Args:
        num_samples: Number of task pairs to generate
        
    Returns:
        Dictionary containing the dataset with task pairs
    """
    
    print(f"🎮 Generating {num_samples} Tic-Tac-Toe reasoning tasks...")
    
    # Generate tasks
    generator = TicTacToeTaskGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    # Create task pairs with images
    pairs = []
    base_dir = Path(__file__).parent.parent.parent.parent / "data" / "questions"
    
    for i, task in enumerate(tasks):
        task_id = f"tictactoe_{i:04d}"
        
        # Create per-question folder
        question_dir = base_dir / "tictactoe_task" / task_id
        question_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images
        first_image_path = question_dir / "first_frame.png"
        final_image_path = question_dir / "final_frame.png"
        
        # Generate first frame (initial state)
        generate_tictactoe_board_image(
            task.initial_board, 
            title=f"Tic-Tac-Toe - {task.player_to_move}'s Turn",
            output_path=str(first_image_path)
        )
        
        # Generate final frame (solution state)
        highlight_line = task.winning_line if task.game_outcome == "win" else None
        generate_tictactoe_board_image(
            task.final_board,
            highlight_line=highlight_line,
            title=f"Tic-Tac-Toe - {task.player_to_move}'s Move",
            output_path=str(final_image_path)
        )
        
        # Create task pair metadata
        pair = {
            "id": task_id,
            "prompt": task.prompt,
            "first_image_path": f"tictactoe_task/{task_id}/first_frame.png",
            "final_image_path": f"tictactoe_task/{task_id}/final_frame.png",
            "domain": "tictactoe",
            "task_category": task.task_category,
            "difficulty": task.difficulty,
            "tictactoe_data": task.tictactoe_data,
            "created_at": task.created_at
        }
        
        # Save metadata
        metadata_path = question_dir / "question_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(pair, f, indent=2)
        
        pairs.append(pair)
    
    print(f"   ✅ Generated {len(pairs)} tic-tac-toe task pairs")
    
    return {
        "name": "tictactoe_tasks",
        "description": f"Tic-Tac-Toe reasoning tasks ({len(pairs)} pairs)",
        "pairs": pairs
    }


if __name__ == "__main__":
    # Test the module
    dataset = create_dataset(5)
    print(f"Generated dataset with {len(dataset['pairs'])} pairs")
