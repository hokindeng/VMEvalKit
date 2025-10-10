"""
RAVEN Progressive Matrix Task Module for VMEvalKit

Implements Progressive Matrix reasoning tasks based on the CVPR 2019 RAVEN dataset.
Tests video models' ability to demonstrate abstract visual reasoning through pattern completion.

Key capabilities:
- Generate Progressive Matrix puzzles with 7 different configurations
- Create incomplete→complete image pairs for video reasoning evaluation
- Support multiple rule types: Constant, Progression, Arithmetic, Distribute_Three
- Classify tasks by difficulty and pattern complexity
- Evaluate abstract reasoning and analogical thinking in video models
"""

from .raven_reasoning import (
    RavenGenerator,
    create_dataset,
    create_task_pair,
    generate_task_images,
    generate_prompt
)

__all__ = [
    'RavenGenerator',
    'create_dataset', 
    'create_task_pair',
    'generate_task_images',
    'generate_prompt'
]

__version__ = "1.0.0"
