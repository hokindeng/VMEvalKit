import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
    from PIL import Image
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    HAS_DEPENDENCIES = False

# Image resolution constants for VMEvalKit
IMAGE_SIZE = (400, 400)  # VMEvalKit standard size
FIGURE_SIZE = (8, 8)     # Matplotlib figure size for rendering

# Object properties
SHAPES = ['cube', 'sphere', 'cylinder', 'cone']
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
SPATIAL_RELATIONS = {
    'horizontal': ['left', 'right', 'next to', 'beside'],
    'vertical': ['on top of', 'above', 'below', 'under'],
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
            return PROMPT_TEMPLATES[template].format(
                color1=obj1.color,
                shape1=obj1.shape,
                relation=relation,
                color2=obj2.color,
                shape2=obj2.shape
            )
        else:
            return f"Rearrange the objects according to the instruction."
    
    def _render_scene(self, objects: List[ObjectSpec], save_path: Path, is_final: bool = False) -> None:
        """Render a 2D scene with objects (simplified visualization)."""
        if not HAS_DEPENDENCIES:
            raise ImportError("Matplotlib and PIL are required for object rearrangement tasks")
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=IMAGE_SIZE[0] // FIGURE_SIZE[0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
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
        
        # Add instruction text
        if is_final:
            ax.text(0.5, 0.95, "Target State", ha='center', va='top', 
                   fontsize=14, weight='bold', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.95, "Initial State", ha='center', va='top',
                   fontsize=14, weight='bold', transform=ax.transAxes)
        
        plt.tight_layout()
        fig.savefig(save_path, dpi=IMAGE_SIZE[0] // FIGURE_SIZE[0], bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _apply_spatial_relation(self, objects: List[ObjectSpec], relation: Dict[str, Any]) -> List[ObjectSpec]:
        """Apply spatial relation to rearrange objects."""
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
        
        if len(rearranged) >= 2:
            obj1 = rearranged[0]
            obj2 = rearranged[1]
            
            if relation_type == 'horizontal':
                if relation_value in ['left', 'next to', 'beside']:
                    # Place obj1 to the left of obj2
                    obj1.position = (obj2.position[0] - 0.2, obj2.position[1])
                elif relation_value == 'right':
                    obj1.position = (obj2.position[0] + 0.2, obj2.position[1])
            elif relation_type == 'vertical':
                if relation_value in ['on top of', 'above']:
                    obj1.position = (obj2.position[0], obj2.position[1] + 0.2)
                elif relation_value in ['below', 'under']:
                    obj1.position = (obj2.position[0], obj2.position[1] - 0.2)
            elif relation_type == 'general':
                # For general relations like 'between'
                if relation_value == 'between':
                    # Place between obj2 and obj3 if available
                    if len(rearranged) >= 3:
                        obj3 = rearranged[2]
                        obj1.position = (
                            (obj2.position[0] + obj3.position[0]) / 2,
                            (obj2.position[1] + obj3.position[1]) / 2
                        )
                    else:
                        obj1.position = (obj2.position[0] + 0.15, obj2.position[1])
        
        return rearranged
    
    def generate_task(self, num_objects: int = None, difficulty: str = None) -> Dict[str, Any]:
        """Generate a single object rearrangement task."""
        if not HAS_DEPENDENCIES:
            raise ImportError("Matplotlib and PIL are required for object rearrangement tasks")
        
        # Generate initial objects
        initial_objects = []
        used_colors = set()
        used_shapes = set()
        
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
            
            # Random initial position
            position = (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8))
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
        final_objects = self._apply_spatial_relation(initial_objects, target_relation)
        
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
    
    generator._render_scene(initial_objects, first_image_path, is_final=False)
    generator._render_scene(final_objects, final_image_path, is_final=True)
    
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
    
    if not HAS_DEPENDENCIES:
        raise ImportError("Matplotlib and PIL are required for object rearrangement tasks")
    
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

