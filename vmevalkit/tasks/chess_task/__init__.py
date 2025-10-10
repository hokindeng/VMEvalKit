"""
Chess Task Module for VMEvalKit

This module provides chess mate-in-1 reasoning tasks for evaluating video models' ability to:
- Understand chess positions from visual input
- Identify winning tactical patterns
- Demonstrate solutions through generated video
- Execute precise piece movements accurately

The chess tasks test spatial reasoning, pattern recognition, strategic thinking,
and action demonstration capabilities in video models.
"""

from .chess_reasoning import (
    ChessTaskPair,
    ChessDataset, 
    create_chess_dataset,
    generate_chess_board_image,
    ChessMateValidator
)

from .chess_mate_in_1 import (
    MateIn1Puzzle,
    MateIn1Generator,
    MateIn1Validator
)

__all__ = [
    'ChessTaskPair',
    'ChessDataset',
    'create_chess_dataset', 
    'generate_chess_board_image',
    'ChessMateValidator',
    'MateIn1Puzzle',
    'MateIn1Generator', 
    'MateIn1Validator'
]

__version__ = "1.0.0"
