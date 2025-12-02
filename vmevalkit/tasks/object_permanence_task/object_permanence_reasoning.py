"""
Object Permanence Reasoning Task for VMEvalKit

This task tests whether video generation models understand object permanence:
- Objects continue to exist when occluded
- Objects remain unchanged (position, color, shape) when occluder moves
- Objects are revealed when occluder moves away

Design:
- First Frame: Objects fully visible, occluder on left side (not occluding)
- Final Frame: Occluder moved out of frame (right side), objects remain unchanged
- Occluder: Opaque gray panel that moves from left to right
"""

import json
import random
import numpy as np
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, RegularPolygon

# Import prompts from centralized location
from .PROMPTS import (
    PROMPTS,
    DEFAULT_PROMPT_INDEX
)


@dataclass
class ObjectPermanenceTaskPair:
    """
    Data structure for object permanence video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: Objects fully visible, occluder on left (not occluding)
    - final_image: Occluder moved out of frame, objects unchanged
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # Objects visible, occluder on left
    final_image_path: str           # Occluder out of frame, objects unchanged
    task_category: str = "ObjectPermanence"
    object_permanence_data: Dict[str, Any] = None  # Metadata
    difficulty: str = "easy"        # "easy", "medium", "hard"
    num_objects: int = 1            # Number of objects
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ObjectGenerator:
    """Generate objects with colors, shapes, and positions."""
    
    COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]
    SHAPES = ["square", "circle", "triangle", "trapezoid"]  # 2D shapes only
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), 
                 min_size: int = 20, max_size: int = 40,
                 min_spacing: int = 25):
        """
        Initialize object generator.
        
        Args:
            canvas_size: (width, height) of the canvas
            min_size: Minimum object size in pixels
            max_size: Maximum object size in pixels
            min_spacing: Minimum spacing between objects
        """
        self.canvas_size = canvas_size
        self.min_size = min_size
        self.max_size = max_size
        self.min_spacing = min_spacing
        self.rng = random.Random()
    
    def generate_objects(self, num_objects: int, seed: Optional[int] = None,
                        ensure_occluder_path: bool = True) -> List[Dict[str, Any]]:
        """
        Generate objects with random colors, shapes, and positions.
        Ensures objects don't overlap and are positioned so occluder can pass over them.
        
        Args:
            num_objects: Number of objects to generate
            seed: Random seed for reproducibility
            ensure_occluder_path: If True, ensure objects are positioned so occluder can pass over them
            
        Returns:
            List of object dictionaries with id, color, shape, x, y, size, area
        """
        if seed is not None:
            self.rng.seed(seed)
        
        objects = []
        canvas_w, canvas_h = self.canvas_size
        
        # For occluder path: objects should be in the middle/right area so occluder can pass over them
        # Occluder moves from left (x=0) to right (x=canvas_w)
        # Objects should be positioned so occluder path covers them
        
        for i in range(num_objects):
            # Try to place object (with collision detection)
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                # Fixed size for all objects
                size = 30  # Fixed size for consistency
                
                # Position: ensure objects are in area where occluder can pass over them
                if ensure_occluder_path:
                    # Objects should be in middle-right area, ensuring they don't overlap with occluder
                    # Occluder is on left side with width 50, starting at x=1 (with edge_margin)
                    # Occluder right edge is at: 1 + 50 = 51
                    # Objects must have sufficient distance from occluder right edge
                    margin = size // 2 + self.min_spacing
                    occluder_width = 50
                    occluder_x_start = 1  # With edge_margin
                    occluder_right_edge = occluder_x_start + occluder_width  # Right edge at x=51
                    min_distance_from_occluder = 40  # Minimum distance from occluder right edge (increased for better spacing)
                    x_min = max(margin, occluder_right_edge + min_distance_from_occluder)  # Ensure no overlap and sufficient distance
                    x_max = min(canvas_w - margin, 3 * canvas_w // 4)
                    x = self.rng.randint(x_min, x_max)
                    # Keep objects in middle vertical region (avoid too high or too low)
                    # Use middle 60% of canvas height (20% margin on top and bottom)
                    vertical_margin_ratio = 0.2  # 20% margin on top and bottom
                    y_min = int(canvas_h * vertical_margin_ratio) + margin
                    y_max = int(canvas_h * (1 - vertical_margin_ratio)) - margin
                    y = self.rng.randint(y_min, y_max)
                else:
                    # Random position (with margin for object size)
                    margin = size // 2 + self.min_spacing
                    x = self.rng.randint(margin, canvas_w - margin)
                    y = self.rng.randint(margin, canvas_h - margin)
                
                # Check collision with existing objects
                collision = False
                for obj in objects:
                    dx = x - obj["x"]
                    dy = y - obj["y"]
                    distance = np.sqrt(dx*dx + dy*dy)
                    # Increase minimum distance between objects for better spacing
                    min_distance = (size + obj["size"]) // 2 + self.min_spacing
                    
                    if distance < min_distance:
                        collision = True
                        break
                
                if not collision:
                    # Random color and shape
                    color = self.rng.choice(self.COLORS)
                    shape = self.rng.choice(self.SHAPES)
                    
                    # Calculate area (approximate)
                    if shape == "square":
                        area = size * size
                    elif shape == "circle":
                        area = int(np.pi * (size // 2) ** 2)
                    elif shape == "triangle":
                        area = int(size * size * 0.433)  # Equilateral triangle
                    else:  # trapezoid
                        area = int(size * size * 0.75)  # Approximate trapezoid area
                    
                    obj = {
                        "id": i,
                        "color": color,
                        "shape": shape,
                        "x": x,
                        "y": y,
                        "size": size,
                        "area": area
                    }
                    objects.append(obj)
                    placed = True
                    break
            
            if not placed:
                # If we can't place, use a grid-based fallback
                # Ensure grid-based placement also respects occluder position
                occluder_width = 50
                occluder_x_start = 1  # With edge_margin
                occluder_right_edge = occluder_x_start + occluder_width
                min_distance_from_occluder = 30  # Minimum distance from occluder right edge (increased for better spacing)
                safe_x_start = occluder_right_edge + min_distance_from_occluder
                
                grid_size = int(np.ceil(np.sqrt(num_objects)))
                # Adjust grid to start after occluder
                available_width = canvas_w - safe_x_start
                cell_w = available_width // (grid_size + 1)
                # Keep objects in middle vertical region (avoid too high or too low)
                vertical_margin_ratio = 0.2  # 20% margin on top and bottom
                available_height = int(canvas_h * (1 - 2 * vertical_margin_ratio))
                cell_h = available_height // (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = safe_x_start + cell_w * (col + 1)
                y_base = int(canvas_h * vertical_margin_ratio)
                y = y_base + cell_h * (row + 1)
                
                size = 30  # Fixed size for consistency
                color = self.rng.choice(self.COLORS)
                shape = self.rng.choice(self.SHAPES)
                
                if shape == "square":
                    area = size * size
                elif shape == "circle":
                    area = int(np.pi * (size // 2) ** 2)
                elif shape == "triangle":
                    area = int(size * size * 0.433)
                else:  # trapezoid
                    area = int(size * size * 0.75)  # Approximate trapezoid area
                
                obj = {
                    "id": i,
                    "color": color,
                    "shape": shape,
                    "x": x,
                    "y": y,
                    "size": size,
                    "area": area
                }
                objects.append(obj)
        
        return objects


class SceneRenderer:
    """Render scene images with objects and occluder."""
    
    COLOR_MAP = {
        "red": "#FF0000",
        "green": "#00FF00",
        "blue": "#0000FF",
        "yellow": "#FFFF00",
        "orange": "#FFA500",
        "purple": "#800080"
    }
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), dpi: int = 100):
        self.canvas_size = canvas_size
        self.dpi = dpi
        self.occluder_width = 50  # Width of occluder panel
    
    def render_first_frame(self, objects: List[Dict[str, Any]], 
                          occluder_x: float,
                          output_path: Union[str, Path]) -> None:
        """
        Render first frame: objects fully visible, occluder on left (not occluding).
        
        Args:
            objects: List of object dictionaries
            occluder_x: X position of occluder (should be on left, not occluding objects)
            output_path: Path to save the image
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.canvas_size[0]/self.dpi, 
                                               self.canvas_size[1]/self.dpi), 
                              dpi=self.dpi)
        
        # Draw background
        ax.set_xlim(0, self.canvas_size[0])
        ax.set_ylim(0, self.canvas_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.invert_yaxis()
        
        # Draw each object (fully visible)
        for obj in objects:
            color = self.COLOR_MAP.get(obj["color"], "#000000")
            x = obj["x"]
            y = obj["y"]
            size = obj["size"]
            shape = obj["shape"]
            
            self._draw_object(ax, x, y, size, shape, color)
        
        # Draw occluder on left (not occluding objects)
        # Occluder is an opaque gray rectangle that will pass over objects
        # Ensure occluder is tall enough and positioned to cover all objects during horizontal movement
        occluder_width = 50  # Increased width for better visibility and coverage
        
        # Occluder should be full height to ensure it passes over all objects regardless of their y position
        # This ensures that when moving horizontally from left to right, it will definitely occlude objects
        # Position occluder with small margin to ensure all edges are fully visible and consistent
        # Edge linewidth is 2, so we need margin of at least 1 to ensure edges aren't clipped
        edge_margin = 1
        occluder_x_adjusted = occluder_x + edge_margin  # Ensure left edge is fully visible
        occluder_y = edge_margin  # Ensure bottom edge is fully visible
        occluder_height = self.canvas_size[1] - 2 * edge_margin  # Adjust height to keep top edge visible
        
        # Ensure occluder is positioned so it doesn't occlude objects in first frame
        # Objects are positioned starting from x > 50 (occluder width), so occluder won't overlap
        
        # Use darker gray for better opacity visibility
        # All edges (left, right, top, bottom) will have the same style and be fully visible
        occluder = Rectangle((occluder_x_adjusted, occluder_y), occluder_width, occluder_height,
                           facecolor="#606060", edgecolor="#303030", linewidth=2, 
                           alpha=1.0, joinstyle='miter', capstyle='butt')  # Consistent edge style for all sides
        ax.add_patch(occluder)
        
        # Save figure
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
    
    def render_final_frame(self, objects: List[Dict[str, Any]], 
                           output_path: Union[str, Path]) -> None:
        """
        Render final frame: occluder moved out of frame, objects unchanged.
        
        Args:
            objects: List of object dictionaries (same as first frame)
            output_path: Path to save the image
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.canvas_size[0]/self.dpi, 
                                               self.canvas_size[1]/self.dpi), 
                              dpi=self.dpi)
        
        # Draw background
        ax.set_xlim(0, self.canvas_size[0])
        ax.set_ylim(0, self.canvas_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.invert_yaxis()
        
        # Draw each object (exactly the same as first frame)
        for obj in objects:
            color = self.COLOR_MAP.get(obj["color"], "#000000")
            x = obj["x"]
            y = obj["y"]
            size = obj["size"]
            shape = obj["shape"]
            
            self._draw_object(ax, x, y, size, shape, color)
        
        # No occluder in final frame (moved out of frame)
        
        # Save figure
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
    
    def _draw_object(self, ax, x: float, y: float, size: int, shape: str, color: str):
        """Draw a single 2D object."""
        if shape == "square":
            # Draw square (2D flat shape)
            rect = Rectangle((x - size//2, y - size//2), size, size,
                           facecolor=color, edgecolor="black", linewidth=2)
            ax.add_patch(rect)
        elif shape == "circle":
            # Draw circle (2D flat shape, not sphere)
            circle = Circle((x, y), size//2, facecolor=color, 
                          edgecolor="black", linewidth=2)
            ax.add_patch(circle)
        elif shape == "triangle":
            # Draw triangle (2D flat shape, not pyramid)
            triangle = RegularPolygon((x, y), 3, radius=size//2,
                                    orientation=np.pi/6,
                                    facecolor=color, edgecolor="black", linewidth=2)
            ax.add_patch(triangle)
        elif shape == "trapezoid":
            # Draw trapezoid (2D flat shape, not cone)
            half_size = size // 2
            points = [
                (x - half_size, y - half_size),
                (x + half_size, y - half_size),
                (x + half_size//2, y + half_size),
                (x - half_size//2, y + half_size)
            ]
            polygon = patches.Polygon(points, facecolor=color, 
                                    edgecolor="black", linewidth=2)
            ax.add_patch(polygon)


class ObjectPermanenceGenerator:
    """Generator for object permanence tasks."""
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), 
                 temp_dir: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            canvas_size: Size of the canvas in pixels
            temp_dir: Temporary directory for image generation (if None, uses system temp)
        """
        self.canvas_size = canvas_size
        if temp_dir:
            self.temp_dir = temp_dir
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
            self._cleanup_temp = False
        else:
            # Use system temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="object_permanence_")
            self._cleanup_temp = True
        self.rng = random.Random()
        
        self.object_generator = ObjectGenerator(canvas_size=canvas_size)
        self.renderer = SceneRenderer(canvas_size=canvas_size)
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory if we created it."""
        if self._cleanup_temp and Path(self.temp_dir).exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    def __del__(self):
        """Clean up temporary directory if we created it.
        
        Note: We don't clean up here to avoid deleting files before dataset.py copies them.
        The temp directory will be cleaned up by the system or manually.
        """
        # Don't auto-cleanup to ensure files are available for dataset.py to copy
        pass
    
    def generate_single_task(self, task_id: str, difficulty: str = "easy", 
                            seed: Optional[int] = None) -> ObjectPermanenceTaskPair:
        """
        Generate a single object permanence task.
        
        Args:
            task_id: Unique identifier for the task
            difficulty: "easy" (1 object), "medium" (2 objects), "hard" (3 objects)
            seed: Random seed for reproducibility
            
        Returns:
            ObjectPermanenceTaskPair instance
        """
        if seed is not None:
            self.rng.seed(seed)
        
        # Determine number of objects based on difficulty
        if difficulty == "easy":
            num_objects = 1
        elif difficulty == "medium":
            num_objects = 2
        elif difficulty == "hard":
            num_objects = 3
        else:
            num_objects = 1
        
        # Generate objects
        objects = self.object_generator.generate_objects(
            num_objects=num_objects, 
            seed=seed,
            ensure_occluder_path=True
        )
        
        # Generate prompt using unified template
        prompt_template = PROMPTS[DEFAULT_PROMPT_INDEX]
        prompt = self._format_prompt(prompt_template, objects)
        
        # Occluder initial position (on left, not occluding objects)
        # Objects are positioned starting from x > 50 (after occluder width) to ensure no overlap
        # Occluder starts at x=0 (left edge) with width 50, so objects at x > 50 won't overlap
        occluder_x_start = 0  # Start at left edge of canvas for consistent edges
        
        # Create image paths
        first_image_path = Path(self.temp_dir) / f"{task_id}_first.png"
        final_image_path = Path(self.temp_dir) / f"{task_id}_final.png"
        
        # Render first frame: objects visible, occluder on left
        self.renderer.render_first_frame(objects, occluder_x_start, first_image_path)
        
        # Render final frame: occluder out of frame, objects unchanged
        self.renderer.render_final_frame(objects, final_image_path)
        
        # Create metadata
        object_permanence_data = {
            "objects": objects,
            "occluder_x_start": occluder_x_start,
            "canvas_size": self.canvas_size,
        }
        
        # Create task pair
        task_pair = ObjectPermanenceTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_image_path),
            final_image_path=str(final_image_path),
            task_category="ObjectPermanence",
            object_permanence_data=object_permanence_data,
            difficulty=difficulty,
            num_objects=num_objects,
            created_at=datetime.now().isoformat()
        )
        
        return task_pair
    
    def _format_prompt(self, template: str, objects: List[Dict[str, Any]]) -> str:
        """Format prompt template with object descriptions. Adapts to any number of objects."""
        num_objects = len(objects)
        
        # Helper function to get article (a/an) based on first letter
        def get_article(word: str) -> str:
            """Return 'a' or 'an' based on whether word starts with vowel sound."""
            vowels = ['a', 'e', 'i', 'o', 'u']
            return "an" if word[0].lower() in vowels else "a"
        
        # Helper function to convert number to word
        def number_to_word(n: int) -> str:
            """Convert number to word (1-10)."""
            words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
            if 1 <= n <= 10:
                return words[n - 1]
            else:
                return str(n)
        
        # Build object descriptions
        object_descriptions = []
        for obj in objects:
            article = get_article(obj['color'])
            object_descriptions.append(f"{article} {obj['color']} {obj['shape']}")
        
        # Format objects description based on count
        objects_count_description = number_to_word(num_objects)
        
        if num_objects == 1:
            # Single object: "a red square"
            object_word = "object"
            objects_description = object_descriptions[0]
            objects_reference = "object"
            objects_pronoun = "it"
        elif num_objects == 2:
            # Two objects: "a red square and a blue circle"
            object_word = "objects"
            objects_description = f"{object_descriptions[0]} and {object_descriptions[1]}"
            objects_reference = "objects"
            objects_pronoun = "them"
        else:
            # Multiple objects (3+): "a red square, a blue circle, and a green triangle"
            object_word = "objects"
            objects_description = ", ".join(object_descriptions[:-1]) + f", and {object_descriptions[-1]}"
            objects_reference = "objects"
            objects_pronoun = "them"
        
        return template.format(
            objects_count_description=objects_count_description,
            object_word=object_word,
            objects_description=objects_description,
            objects_reference=objects_reference,
            objects_pronoun=objects_pronoun
        )


def create_dataset(num_samples: int = 50, 
                  difficulty_distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create object permanence dataset.
    
    Args:
        num_samples: Total number of task pairs to generate
        difficulty_distribution: Optional dict like {
            "easy": 0.33,   # 1 object
            "medium": 0.33, # 2 objects
            "hard": 0.34    # 3 objects
        }
        
    Returns:
        Dictionary with 'pairs' key containing list of ObjectPermanenceTaskPair
    """
    if difficulty_distribution is None:
        # Default distribution - equal distribution
        difficulty_distribution = {
            "easy": 1/3,   # 1 object
            "medium": 1/3, # 2 objects
            "hard": 1/3   # 3 objects
        }
    
    print(f"🎯 Creating Object Permanence Dataset")
    print(f"   Total samples: {num_samples}")
    
    # Use system temporary directory
    generator = ObjectPermanenceGenerator(temp_dir=None)
    pairs = []
    
    # Calculate number of samples per difficulty
    easy_count = int(num_samples * difficulty_distribution.get("easy", 1/3))
    medium_count = int(num_samples * difficulty_distribution.get("medium", 1/3))
    hard_count = num_samples - easy_count - medium_count
    
    print(f"   Easy (1 object): {easy_count}")
    print(f"   Medium (2 objects): {medium_count}")
    print(f"   Hard (3 objects): {hard_count}")
    
    # Generate tasks
    task_idx = 0
    for difficulty, count in [("easy", easy_count), ("medium", medium_count), ("hard", hard_count)]:
        for i in range(count):
            task_id = f"object_permanence_{difficulty}_{task_idx:04d}"
            # Generate deterministic seed
            seed = task_idx * 1000 + hash(difficulty) % 1000
            
            try:
                task_pair = generator.generate_single_task(
                    task_id=task_id,
                    difficulty=difficulty,
                    seed=seed
                )
                pairs.append(task_pair)
                task_idx += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{count} {difficulty} tasks...")
            except Exception as e:
                print(f"❌ Error generating task {task_id}: {e}")
                continue
    
    # Convert to dictionary format
    pairs_dict = []
    for pair in pairs:
        pair_dict = {
            "id": pair.id,
            "prompt": pair.prompt,
            "first_image_path": pair.first_image_path,
            "final_image_path": pair.final_image_path,
            "task_category": pair.task_category,
            "object_permanence_data": pair.object_permanence_data,
            "difficulty": pair.difficulty,
            "num_objects": pair.num_objects,
            "created_at": pair.created_at
        }
        pairs_dict.append(pair_dict)
    
    # Create dataset dictionary
    dataset = {
        "name": "object_permanence_tasks",
        "description": f"Object permanence reasoning tasks for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs_dict,
        "metadata": {
            "total_tasks": len(pairs),
            "canvas_size": generator.canvas_size,
            "generation_date": datetime.now().isoformat()
        },
        "created_at": datetime.now().isoformat()
    }
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(pairs)}")
    
    # Note: temp_dir cleanup will happen when generator is garbage collected
    # or can be explicitly called: generator.cleanup_temp_dir()
    
    return dataset

