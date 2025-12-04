"""
Traffic Light Task Module for VMEvalKit

This module provides traffic light reasoning tasks for evaluating video models' ability to:
- Understand temporal concepts (countdown timers, time progression)
- Apply relative rules (opposite states between two traffic lights)
- Generate videos with changing numbers (countdown decrement)
- Demonstrate temporal reasoning and coordination understanding

The traffic light tasks test temporal perception, rule application, and coordination reasoning
capabilities in video models.
"""

from .traffic_light_reasoning import (
    TrafficLightTaskPair,
    TrafficLightDataset,
    TrafficLightTaskGenerator,
    TrafficLightGenerator,
    create_dataset,
    render_traffic_light
)

__all__ = [
    'TrafficLightTaskPair',
    'TrafficLightDataset',
    'TrafficLightTaskGenerator',
    'TrafficLightGenerator',
    'create_dataset',
    'render_traffic_light'
]

__version__ = "1.0.0"

