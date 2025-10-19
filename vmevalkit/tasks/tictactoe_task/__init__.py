"""
Tic-Tac-Toe Task Module for VMEvalKit

This module provides tic-tac-toe reasoning tasks for evaluating video models' ability to:
- Understand tic-tac-toe game states from visual input
- Apply strategic thinking and game theory principles
- Demonstrate optimal moves through generated video
- Show step-by-step reasoning in game strategy

The tic-tac-toe tasks test strategic thinking, pattern recognition, game theory,
and logical reasoning capabilities in video models.
"""

from .tictactoe_reasoning import (
    TicTacToeTaskPair,
    TicTacToeDataset,
    TicTacToeTaskGenerator,
    create_dataset,
    generate_tictactoe_board_image
)

__all__ = [
    'TicTacToeTaskPair',
    'TicTacToeDataset', 
    'TicTacToeTaskGenerator',
    'create_dataset',
    'generate_tictactoe_board_image'
]

__version__ = "1.0.0"
