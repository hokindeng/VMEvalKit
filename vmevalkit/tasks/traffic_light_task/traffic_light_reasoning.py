"""
Traffic Light Reasoning Task for VMEvalKit

Traffic light reasoning system for video model evaluation.
The task shows two traffic lights with countdown timers and tests the model's ability to:
- Understand temporal concepts (countdown timers, time progression)
- Apply relative rules (opposite states between two traffic lights)
- Generate videos with changing numbers (countdown decrement)
- Demonstrate temporal reasoning and coordination understanding

Follows the same data format as other tasks with first/final frames and prompts.

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
from .PROMPTS import TYPE_PROMPTS, DEFAULT_PROMPT_INDEX


@dataclass
class TrafficLightTaskPair:
    """
    Data structure for traffic light reasoning video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The traffic lights in initial state
    - final_image: The traffic lights in target state
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The initial traffic light image
    final_image_path: str           # The target traffic light image
    task_category: str              # "Traffic Light"
    traffic_light_data: Dict[str, Any] = None  # Metadata
    light_a_state: str = ""          # "red" or "green"
    light_a_countdown: int = 0       # Countdown timer for light A
    light_b_state: str = ""          # "red" or "green"
    light_b_countdown: int = 0       # Countdown timer for light B
    task_type: int = 0              # Type 1-4
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class TrafficLightDataset:
    """Collection of TrafficLightTaskPair instances."""
    name: str
    description: str
    pairs: List[TrafficLightTaskPair]
    metadata: Dict[str, Any]
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> TrafficLightTaskPair:
        return self.pairs[idx]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TrafficLightDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert dictionaries back to TrafficLightTaskPair objects
        pairs = []
        for pair_data in data['pairs']:
            pairs.append(TrafficLightTaskPair(**pair_data))
        
        data['pairs'] = pairs
        return cls(**data)


class TrafficLightGenerator:
    """Traffic light image generator for temporal reasoning tasks."""
    
    def render_traffic_light(self, light_a_state: str, light_a_countdown: int,
                            light_b_state: str, light_b_countdown: int,
                            output_path: str, figsize: Tuple[int, int] = (6, 6)):
        """
        Create a traffic light scene image.
        
        Args:
            light_a_state: "red" or "green" for traffic light A
            light_a_countdown: Countdown timer for light A (0 means no countdown or countdown finished)
            light_b_state: "red" or "green" for traffic light B
            light_b_countdown: Countdown timer for light B (0 means no countdown or countdown finished)
            output_path: Path to save the image
            figsize: Figure size for matplotlib (default: 6x6 inches for ~600x600px at 150 DPI)
        """
        # TODO: Implement traffic light rendering
        # - Draw two traffic lights (left and right or top and bottom)
        # - Show red/green state with colored circles
        # - Display countdown numbers below or inside the lights
        # - Create a simple crossroad scene background
        pass


class TrafficLightTaskGenerator:
    """Main class for generating traffic light reasoning tasks."""
    
    def __init__(self):
        self.traffic_light_gen = TrafficLightGenerator()
    
    def _calculate_final_state(self, light_a_state: str, light_a_countdown: int,
                              light_b_state: str, light_b_countdown: int,
                              time_elapsed: int) -> Tuple[str, int, str, int]:
        """
        Calculate the final state based on countdown timers and relative rules.
        
        Args:
            light_a_state: Initial state of light A ("red" or "green")
            light_a_countdown: Initial countdown for light A
            light_b_state: Initial state of light B ("red" or "green")
            light_b_countdown: Initial countdown for light B
            time_elapsed: Time to advance (seconds)
        
        Returns:
            Tuple of (final_light_a_state, final_light_a_countdown,
                     final_light_b_state, final_light_b_countdown)
        """
        # TODO: Implement state calculation logic
        # - Track countdown decrements
        # - Determine which countdown reaches 0 first
        # - Apply relative rule when countdown reaches 0
        # - Handle multiple state switches if time_elapsed is large
        pass
    
    def _format_prompt(self, task_type: int, light_a_state: str, light_a_countdown: int,
                      light_b_state: str, light_b_countdown: int, time_elapsed: Optional[int] = None) -> str:
        """
        Format the prompt based on task type and parameters.
        
        Args:
            task_type: Type 1-4
            light_a_state: State of light A
            light_a_countdown: Countdown for light A
            light_b_state: State of light B
            light_b_countdown: Countdown for light B
            time_elapsed: Optional time elapsed for type 4
        
        Returns:
            Formatted prompt string
        """
        prompt_template = TYPE_PROMPTS[task_type][DEFAULT_PROMPT_INDEX[task_type]]
        
        if task_type in [1, 2]:
            return prompt_template.format(countdown_a=light_a_countdown)
        elif task_type == 3:
            return prompt_template.format(countdown_a=light_a_countdown, countdown_b=light_b_countdown)
        elif task_type == 4:
            return prompt_template.format(
                countdown_a=light_a_countdown,
                countdown_b=light_b_countdown,
                time_elapsed=time_elapsed
            )
        
        return prompt_template
    
    def generate_single_task(self, task_id: str, task_type: int,
                            light_a_state: str, light_a_countdown: int,
                            light_b_state: str, light_b_countdown: int,
                            time_elapsed: Optional[int] = None,
                            output_dir: Path = None) -> TrafficLightTaskPair:
        """
        Generate a single traffic light task.
        
        Args:
            task_id: Unique task ID
            task_type: Type 1-4
            light_a_state: Initial state of light A
            light_a_countdown: Initial countdown for light A
            light_b_state: Initial state of light B
            light_b_countdown: Initial countdown for light B
            time_elapsed: Optional time elapsed for type 4
            output_dir: Directory to save images
        
        Returns:
            TrafficLightTaskPair instance
        """
        # TODO: Implement task generation
        # - Generate initial state image
        # - Calculate final state
        # - Generate final state image
        # - Format prompt
        # - Create and return TrafficLightTaskPair
        pass
    
    def generate_dataset(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a dataset of traffic light reasoning tasks.
        
        Args:
            num_samples: If None, generate all possible tasks. Otherwise, generate only the specified number.
        
        Returns:
            Dataset dictionary
        """
        # TODO: Implement dataset generation
        # - Generate all possible task combinations
        # - Select samples if num_samples is specified
        # - Generate tasks
        # - Return dataset dictionary
        pass


def render_traffic_light(light_a_state: str, light_a_countdown: int,
                        light_b_state: str, light_b_countdown: int,
                        output_path: str) -> str:
    """Utility function to generate a traffic light image."""
    generator = TrafficLightGenerator()
    generator.render_traffic_light(light_a_state, light_a_countdown,
                                   light_b_state, light_b_countdown,
                                   output_path)
    return output_path


def create_dataset(num_samples: Optional[int] = None) -> Dict[str, Any]:
    """Main function to create traffic light reasoning dataset - matches API of other tasks."""
    
    generator = TrafficLightTaskGenerator()
    return generator.generate_dataset(num_samples)

