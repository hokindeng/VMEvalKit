import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from PIL import Image


IMAGE_SIZE = (400, 400)  # VMEvalKit standard size
FIGURE_SIZE = (8, 8)     # Matplotlib figure size for rendering

# Object properties
SHAPES = ['cube', 'sphere', 'cylinder', 'cone']
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
SPATIAL_RELATIONS = {
    'horizontal': ['left', 'right', 'next to', 'beside'],
    'vertical': ['on the top of','above', 'below', 'under'],
    'general': ['between']
}

# Prompt templates
PROMPT_TEMPLATES = {
    'basic': "Move the {color1} {shape1} to the {relation} of the {color2} {shape2}.",
    'place': "Place the {color1} {shape1} {relation} the {color2} {shape2}.",
    'put': "Put the {color1} {shape1} {relation} the {color2} {shape2}.",
    'multi_step': "First, move the {color1} {shape1} to {position1}, then place the {color2} {shape2} {relation} to it.",
    'stack': "Stack the {color1} {shape1} on top of the {color2} {shape2}, then move the {color3} {shape3} next to them."
}


@dataclass
class ObjectSpec:
    """Object specification for scene generation."""
    shape: str
    color: str
    position: Tuple[float, float]
    size: float = 0.1
    id: str = ""


class ObjectRearrGenerator:
    """Generator for object rearrangement tasks."""
    
    def __init__(self):
        self.generated_scenes = []
        random.seed(42)
        np.random.seed(42)
    
    def _get_grid_positions(self, num_objects: int) -> List[Tuple[float, float]]:
        """Generate grid positions for (num_objects+1) x (num_objects+1) grid."""
        grid_size = num_objects + 1
        positions = []
        
        # Create grid with margin (0.1 to 0.9 to leave border space)
        margin = 0.1
        available_space = 0.8
        
        if grid_size <= 1:
            # Fallback for edge case
            positions.append((0.5, 0.5))
            return positions
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = margin + (i / (grid_size - 1)) * available_space
                y = margin + (j / (grid_size - 1)) * available_space
                positions.append((x, y))
        
        return positions
    
    def _snap_to_grid(self, position: Tuple[float, float], num_objects: int) -> Tuple[float, float]:
        """Snap a position to the nearest grid point."""
        grid_positions = self._get_grid_positions(num_objects)
        x, y = position
        
        # Find nearest grid point
        min_dist = float('inf')
        nearest_pos = grid_positions[0]
        
        for grid_pos in grid_positions:
            dist = np.sqrt((x - grid_pos[0])**2 + (y - grid_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_pos = grid_pos
        
        return nearest_pos
    
    def _determine_difficulty(self, num_objects: int, num_steps: int, relation_type: str) -> str:
        """Determine difficulty based on task complexity."""
        if num_objects <= 3 and num_steps <= 2 and relation_type in ['horizontal', 'vertical']:
            return "easy"
        elif num_objects <= 4 and num_steps <= 3:
            return "medium"
        else:
            return "hard"
    
    def _generate_prompt(self, objects: List[ObjectSpec], target_relation: Dict[str, Any]) -> str:
        """Generate natural language prompt for rearrangement."""
        relation_type = target_relation.get('type', 'horizontal')
        relation = random.choice(SPATIAL_RELATIONS.get(relation_type, SPATIAL_RELATIONS['horizontal']))
        
        if len(objects) >= 2:
            obj1 = objects[0]
            obj2 = objects[1]
            template = random.choice(['basic', 'place', 'put'])
            base_prompt = PROMPT_TEMPLATES[template].format(
                color1=obj1.color,
                shape1=obj1.shape,
                relation=relation,
                color2=obj2.color,
                shape2=obj2.shape
            )
            return f"{base_prompt}, Keep all objects on grid intersection points."
        else:
            raise ValueError("At least 2 objects are required for spatial relations")
    
    def _render_scene(self, objects: List[ObjectSpec], save_path: Path, is_final: bool = False, num_objects: int = None) -> None:
        
        # Use higher DPI for better text rendering
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Draw grid if num_objects is provided
        if num_objects is not None:
            grid_size = num_objects + 1
            margin = 0.1
            available_space = 0.8
            
            if grid_size > 1:
                # Draw vertical grid lines
                for i in range(grid_size):
                    x = margin + (i / (grid_size - 1)) * available_space
                    ax.axvline(x, color='black', linewidth=0.5, zorder=0)
                
                # Draw horizontal grid lines
                for j in range(grid_size):
                    y = margin + (j / (grid_size - 1)) * available_space
                    ax.axhline(y, color='black', linewidth=0.5, zorder=0)
        
        # Color mapping
        color_map = {
            'red': '#ef4444',
            'blue': '#3b82f6',
            'green': '#22c55e',
            'yellow': '#eab308',
            'orange': '#f97316',
            'purple': '#a855f7'
        }
        
        # Render each object
        for obj in objects:
            color = color_map.get(obj.color, '#000000')
            x, y = obj.position
            
            if obj.shape == 'cube':
                rect = Rectangle((x - obj.size/2, y - obj.size/2), obj.size, obj.size,
                               facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
            elif obj.shape == 'sphere':
                circle = Circle((x, y), obj.size/2, facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(circle)
            elif obj.shape == 'cylinder':
                rect = FancyBboxPatch((x - obj.size/2, y - obj.size/3), obj.size, obj.size*2/3,
                                    boxstyle="round,pad=0.01", facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
            elif obj.shape == 'cone':
                # Simplified triangle for cone
                triangle = plt.Polygon([(x, y + obj.size/2), (x - obj.size/2, y - obj.size/2), 
                                       (x + obj.size/2, y - obj.size/2)],
                                      facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(triangle)
        
        # Add instruction text with antialiasing
        if is_final:
            ax.text(0.5, 0.95, "Target State", ha='center', va='top', 
                   fontsize=18, weight='bold', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.text(0.5, 0.95, "Initial State", ha='center', va='top',
                   fontsize=18, weight='bold', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        # Use higher DPI for better text rendering, then resize to target size
        temp_path = save_path.with_suffix('.temp.png')
        fig.savefig(temp_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Resize to target image size
        img = Image.open(temp_path)
        img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_resized.save(save_path, 'PNG')
        temp_path.unlink()
    
    def _apply_spatial_relation(self, objects: List[ObjectSpec], relation: Dict[str, Any], num_objects: int) -> List[ObjectSpec]:
        """Apply spatial relation to rearrange objects on grid."""
        # Create a copy of objects
        rearranged = [ObjectSpec(
            shape=obj.shape,
            color=obj.color,
            position=obj.position,
            size=obj.size,
            id=obj.id
        ) for obj in objects]
        
        relation_type = relation.get('type', 'horizontal')
        relation_value = relation.get('value', 'left')
        grid_positions = self._get_grid_positions(num_objects)
        occupied_positions = {obj.position for obj in rearranged[1:]}  # Exclude obj1
        
        if len(rearranged) >= 2:
            obj1 = rearranged[0]
            obj2 = rearranged[1]
            
            # Get grid step size
            grid_size = num_objects + 1
            margin = 0.1
            available_space = 0.8
            step_size = available_space / (grid_size - 1) if grid_size > 1 else 0
            
            if relation_type == 'horizontal':
                if relation_value in ['left', 'next to', 'beside']:
                    # Place obj1 to the left of obj2 on grid
                    target_x = obj2.position[0] - step_size
                    target_pos = (target_x, obj2.position[1])
                    obj1.position = self._snap_to_grid(target_pos, num_objects)
                elif relation_value == 'right':
                    target_x = obj2.position[0] + step_size
                    target_pos = (target_x, obj2.position[1])
                    obj1.position = self._snap_to_grid(target_pos, num_objects)
            elif relation_type == 'vertical':
                if relation_value in ['on top of', 'above']:
                    target_y = obj2.position[1] + step_size
                    target_pos = (obj2.position[0], target_y)
                    obj1.position = self._snap_to_grid(target_pos, num_objects)
                elif relation_value in ['below', 'under']:
                    target_y = obj2.position[1] - step_size
                    target_pos = (obj2.position[0], target_y)
                    obj1.position = self._snap_to_grid(target_pos, num_objects)
            elif relation_type == 'general':
                # For general relations like 'between'
                if relation_value == 'between':
                    # Place between obj2 and obj3 if available
                    if len(rearranged) >= 3:
                        obj3 = rearranged[2]
                        mid_x = (obj2.position[0] + obj3.position[0]) / 2
                        mid_y = (obj2.position[1] + obj3.position[1]) / 2
                        target_pos = (mid_x, mid_y)
                        obj1.position = self._snap_to_grid(target_pos, num_objects)
                    else:
                        target_x = obj2.position[0] + step_size
                        target_pos = (target_x, obj2.position[1])
                        obj1.position = self._snap_to_grid(target_pos, num_objects)
            
            # Ensure obj1 doesn't overlap with other objects
            if obj1.position in occupied_positions:
                # Find nearest available grid position
                available_positions = [pos for pos in grid_positions if pos not in occupied_positions]
                if available_positions:
                    min_dist = float('inf')
                    best_pos = available_positions[0]
                    for pos in available_positions:
                        dist = np.sqrt((obj1.position[0] - pos[0])**2 + (obj1.position[1] - pos[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_pos = pos
                    obj1.position = best_pos
        
        return rearranged
    
    def generate_task(self, num_objects: int = None, difficulty: str = None) -> Dict[str, Any]:

        
        # Generate initial objects
        initial_objects = []
        used_colors = set()
        used_shapes = set()
        if difficulty == 'easy':
            num_objects = 2
        elif difficulty == 'medium':
            num_objects = 3
        else:
            num_objects = 4
        
        # Get grid positions and randomly select unique positions
        grid_positions = self._get_grid_positions(num_objects)
        available_positions = grid_positions.copy()
        random.shuffle(available_positions)
        
        for i in range(num_objects):
            # Ensure unique color-shape combinations
            available_colors = [c for c in COLORS if c not in used_colors]
            available_shapes = [s for s in SHAPES if s not in used_shapes]
            
            if not available_colors:
                available_colors = COLORS
            if not available_shapes:
                available_shapes = SHAPES
            
            color = random.choice(available_colors)
            shape = random.choice(available_shapes)
            used_colors.add(color)
            used_shapes.add(shape)
            
            # Select position from grid
            if available_positions:
                position = available_positions.pop()
            else:
                # Fallback: use random grid position if somehow we run out
                position = random.choice(grid_positions)
            
            obj = ObjectSpec(
                shape=shape,
                color=color,
                position=position,
                size=0.1,
                id=f"obj_{i}"
            )
            initial_objects.append(obj)
        
        # Generate target relation
        relation_type = random.choice(list(SPATIAL_RELATIONS.keys()))
        relation_value = random.choice(SPATIAL_RELATIONS[relation_type])
        target_relation = {'type': relation_type, 'value': relation_value}
        
        # Apply relation to get final objects
        final_objects = self._apply_spatial_relation(initial_objects, target_relation, num_objects)
        
        # Generate prompt
        prompt = self._generate_prompt(initial_objects, target_relation)
        
        # Determine number of steps
        num_steps = 1 if difficulty == 'easy' else (2 if difficulty == 'medium' else 3)
        
        return {
            'initial_objects': initial_objects,
            'final_objects': final_objects,
            'prompt': prompt,
            'difficulty': difficulty,
            'num_objects': num_objects,
            'num_steps': num_steps,
            'target_relation': target_relation,
            'spatial_relations': [f"{relation_value}"]
        }
    
    def generate_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """Generate multiple object rearrangement tasks."""
        tasks = []
        for i in range(num_tasks):
            difficulty = random.choice(['easy', 'medium', 'hard'])
            task = self.generate_task(difficulty=difficulty)
            tasks.append(task)
        return tasks


def create_task_pair(task_data: Dict[str, Any], task_id: str, base_dir: Path = None) -> Dict[str, Any]:
    """Create a task pair in VMEvalKit format."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent.parent
    
    # Create temp directory for images
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    # Generate images
    first_image_path = temp_dir / f"{task_id}_first.png"
    final_image_path = temp_dir / f"{task_id}_final.png"
    
    generator = ObjectRearrGenerator()
    
    # Convert ObjectSpec to list for rendering
    initial_objects = task_data['initial_objects']
    final_objects = task_data['final_objects']
    num_objects = task_data['num_objects']
    
    generator._render_scene(initial_objects, first_image_path, is_final=False, num_objects=num_objects)
    generator._render_scene(final_objects, final_image_path, is_final=True, num_objects=num_objects)
    
    # Create task pair
    return {
        "id": task_id,
        "prompt": task_data['prompt'],
        "first_image_path": str(first_image_path),
        "final_image_path": str(final_image_path),
        "task_category": "ObjectRearr",
        "scene_data": {
            "generation_method": "2d_simplified_visualization",
            "num_objects": task_data['num_objects'],
            "num_steps": task_data['num_steps'],
            "spatial_relations": task_data['spatial_relations'],
            "target_relation": task_data['target_relation']
        },
        "difficulty": task_data['difficulty'],
        "num_objects": task_data['num_objects'],
        "num_steps": task_data['num_steps'],
        "objects": [
            {
                "shape": obj.shape,
                "color": obj.color,
                "position": obj.position,
                "size": obj.size,
                "id": obj.id
            }
            for obj in initial_objects
        ],
        "spatial_relations": task_data['spatial_relations'],
        "manipulation_type": "telekinetic",  # Simplified for 2D visualization
        "created_at": datetime.now().isoformat()
    }


def create_dataset(num_samples: int = 30, difficulty_distribution: Dict[str, float] = None) -> Dict[str, Any]:
    """Create object rearrangement dataset - main entry point matching other domains."""
    
    print(f"🎯 Creating Object Rearrangement Dataset")
    print(f"   Total samples: {num_samples}")
    print(f"   Difficulty distribution: {difficulty_distribution}")
    
    start_time = datetime.now()
    
    # Generate tasks
    generator = ObjectRearrGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    # Create task pairs
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"obj_rearrange_{i:04d}"
        pair = create_task_pair(task_data, task_id)
        pairs.append(pair)
        print(f"✅ Created task {task_id} ({pair['difficulty']})")
    
    # Create dataset
    dataset = {
        "name": "object_rearrangement_tasks",
        "description": f"Object rearrangement tasks with spatial manipulation instructions ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(pairs)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    
    return dataset

