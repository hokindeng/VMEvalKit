"""Maze task generators and data structures."""

from .maze_reasoning import (
    MazeTaskPair,
    MazeDataset,
    MazeTaskGenerator,
    create_dataset,
)

__all__ = [
    "MazeTaskPair",
    "MazeDataset",
    "MazeTaskGenerator",
    "create_dataset",
]
