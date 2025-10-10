"""
3D Mental Rotation Task Module for VMEvalKit

This module provides functionality for generating and evaluating 3D mental rotation tasks.
Tasks involve 3D voxel structures (snake-like configurations) shown from different viewpoints,
requiring models to demonstrate spatial reasoning by generating videos showing rotation transitions.

Capabilities:
- Generate diverse 3D voxel structures with varying complexity
- Create mental rotation tasks with different difficulty levels
- Render high-quality images from multiple viewpoints
- Assess spatial reasoning through rotation demonstrations
"""

from .rotation_reasoning import (
    RotationGenerator,
    create_dataset,
    generate_task_images,
    create_task_pair,
    generate_prompt,
)

__all__ = [
    'RotationGenerator',
    'create_dataset', 
    'generate_task_images',
    'create_task_pair',
    'generate_prompt',
]

__version__ = "1.0.0"
