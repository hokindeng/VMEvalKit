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
    SelfContainedMateGenerator,
    create_dataset,
    create_chess_task_pair,
    generate_chess_board_png
)

__all__ = [
    'SelfContainedMateGenerator',
    'create_dataset',
    'create_chess_task_pair',
    'generate_chess_board_png'
]

__version__ = "1.0.0"
