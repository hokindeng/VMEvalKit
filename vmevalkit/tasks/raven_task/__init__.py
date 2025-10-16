"""
Raven Progressive Matrices (RPM) task module for VMEvalKit.

This module provides functionality to generate RPM-style puzzles
for evaluating video model reasoning capabilities.
"""

from .rpm_generator import RPMPuzzleGenerator
from .raven_reasoning import create_dataset, visualize_solution_process, generate_raven_tasks

__all__ = [
    'RPMPuzzleGenerator',
    'create_dataset',  # Required for VMEvalKit integration
    'visualize_solution_process',
    'generate_raven_tasks'
]

# Task metadata
TASK_INFO = {
    'name': 'Raven Progressive Matrices',
    'description': 'Generate and solve RPM-style visual reasoning puzzles',
    'version': '2.0.0',
    'capabilities': [
        'pattern_recognition',
        'logical_reasoning',
        'spatial_reasoning',
        'abstract_thinking'
    ]
}