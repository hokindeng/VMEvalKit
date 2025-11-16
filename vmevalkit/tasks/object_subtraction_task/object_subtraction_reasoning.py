"""
Object Subtraction Reasoning Task for VMEvalKit

Multi-level cognitive reasoning benchmark where video generation models must
remove specific objects while keeping others stationary.

Level 1: Explicit Specificity - Remove objects by explicit visual attributes (color, shape)

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
from matplotlib.patches import Rectangle, Circle, RegularPolygon, FancyBboxPatch

# Import prompts from centralized location
from .PROMPTS import PROMPTS_L1, PROMPTS_L2, PROMPTS_L3, PROMPTS_L4, DEFAULT_PROMPT_INDEX_L1, DEFAULT_PROMPT_INDEX_L2, DEFAULT_PROMPT_INDEX_L3, DEFAULT_PROMPT_INDEX_L4


@dataclass
class ObjectSubtractionTaskPair:
    """
    Data structure for object subtraction video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The initial scene with all objects
    - final_image: The scene with specified objects removed
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The initial scene image (all objects)
    final_image_path: str           # The final scene image (objects removed)
    task_category: str = "ObjectSubtraction"
    level: str = "L1"               # "L1", "L2", "L3", "L4"
    object_subtraction_data: Dict[str, Any] = None  # Metadata (rules, objects, etc.)
    difficulty: str = "easy"        # "easy", "medium", "hard"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ObjectGenerator:
    """Generate objects with colors, shapes, and positions."""
    
    COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]
    SHAPES = ["cube", "sphere", "pyramid", "cone"]
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), 
                 min_size: int = 20, max_size: int = 40,
                 min_spacing: int = 10):
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
    
    def generate_objects(self, num_objects: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate objects with random colors, shapes, and positions.
        Ensures objects don't overlap.
        
        Args:
            num_objects: Number of objects to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of object dictionaries with id, color, shape, x, y, size, area
        """
        if seed is not None:
            self.rng.seed(seed)
        
        objects = []
        canvas_w, canvas_h = self.canvas_size
        
        for i in range(num_objects):
            # Try to place object (with collision detection)
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                # Random size
                size = self.rng.randint(self.min_size, self.max_size)
                
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
                    min_distance = (size + obj["size"]) // 2 + self.min_spacing
                    
                    if distance < min_distance:
                        collision = True
                        break
                
                if not collision:
                    # Random color and shape
                    color = self.rng.choice(self.COLORS)
                    shape = self.rng.choice(self.SHAPES)
                    
                    # Calculate area (approximate)
                    if shape == "cube":
                        area = size * size
                    elif shape == "sphere":
                        area = int(np.pi * (size // 2) ** 2)
                    elif shape == "pyramid":
                        area = int(size * size * 0.433)  # Equilateral triangle
                    else:  # cone
                        area = int(np.pi * (size // 2) ** 2)
                    
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
                grid_size = int(np.ceil(np.sqrt(num_objects)))
                cell_w = canvas_w // (grid_size + 1)
                cell_h = canvas_h // (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = cell_w * (col + 1)
                y = cell_h * (row + 1)
                
                size = self.rng.randint(self.min_size, self.max_size)
                color = self.rng.choice(self.COLORS)
                shape = self.rng.choice(self.SHAPES)
                
                if shape == "cube":
                    area = size * size
                elif shape == "sphere":
                    area = int(np.pi * (size // 2) ** 2)
                elif shape == "pyramid":
                    area = int(size * size * 0.433)
                else:
                    area = int(np.pi * (size // 2) ** 2)
                
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
    """Render scene images with objects."""
    
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
    
    def render_scene(self, objects: List[Dict[str, Any]], 
                    output_path: Union[str, Path],
                    show_object_ids: Optional[List[int]] = None) -> None:
        """
        Render a scene with objects.
        
        Args:
            objects: List of object dictionaries
            show_object_ids: If provided, only show objects with these IDs
            output_path: Path to save the image
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.canvas_size[0]/self.dpi, 
                                               self.canvas_size[1]/self.dpi), 
                              dpi=self.dpi)
        
        # Filter objects if needed
        if show_object_ids is not None:
            objects_to_render = [obj for obj in objects if obj["id"] in show_object_ids]
        else:
            objects_to_render = objects
        
        # Draw each object
        for obj in objects_to_render:
            color = self.COLOR_MAP.get(obj["color"], "#000000")
            x = obj["x"]
            y = obj["y"]
            size = obj["size"]
            shape = obj["shape"]
            
            if shape == "cube":
                # Draw square
                rect = Rectangle((x - size//2, y - size//2), size, size,
                               facecolor=color, edgecolor="black", linewidth=2)
                ax.add_patch(rect)
            elif shape == "sphere":
                # Draw circle
                circle = Circle((x, y), size//2, facecolor=color, 
                              edgecolor="black", linewidth=2)
                ax.add_patch(circle)
            elif shape == "pyramid":
                # Draw triangle (equilateral)
                triangle = RegularPolygon((x, y), 3, radius=size//2,
                                        orientation=np.pi/6,
                                        facecolor=color, edgecolor="black", linewidth=2)
                ax.add_patch(triangle)
            elif shape == "cone":
                # Draw trapezoid (cone-like)
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
        
        # Set up axes
        ax.set_xlim(0, self.canvas_size[0])
        ax.set_ylim(0, self.canvas_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        # Note: Matplotlib uses bottom-left origin, but we want top-left origin for image coordinates
        # We'll invert y-axis to match image coordinate system
        ax.invert_yaxis()
        
        # Save figure
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)


class RuleGenerator:
    """Generate rules for different cognitive levels."""
    
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()
    
    def generate_l1_rule(self, objects: List[Dict[str, Any]], 
                        prompt_index: int = 0) -> Tuple[Dict[str, Any], str]:
        """
        Generate Level 1 rule: Remove objects by explicit visual attributes.
        
        Args:
            objects: List of objects
            prompt_index: Index into PROMPTS_L1 list
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Choose a removal criterion (color or shape)
        criterion_type = self.rng.choice(["color", "shape"])
        
        if criterion_type == "color":
            # Find all unique colors
            colors = list(set(obj["color"] for obj in objects))
            if len(colors) < 2:
                # Fallback to shape if not enough colors
                criterion_type = "shape"
        
        if criterion_type == "color":
            # Select a color to remove
            colors = list(set(obj["color"] for obj in objects))
            remove_color = self.rng.choice(colors)
            
            # Find all objects with this color
            target_object_ids = [obj["id"] for obj in objects if obj["color"] == remove_color]
            
            rule = {
                "level": "L1",
                "rule_type": "color",
                "remove_color": remove_color,
                "target_object_ids": target_object_ids
            }
            
            # Generate prompt
            prompt = PROMPTS_L1[prompt_index % len(PROMPTS_L1)]
            # Replace color in prompt if needed
            if "red" in prompt.lower():
                prompt = prompt.replace("red", remove_color)
            elif "blue" in prompt.lower():
                prompt = prompt.replace("blue", remove_color)
            elif "green" in prompt.lower():
                prompt = prompt.replace("green", remove_color)
            elif "yellow" in prompt.lower():
                prompt = prompt.replace("yellow", remove_color)
            else:
                # Use generic prompt
                prompt = f"Remove all {remove_color} objects from the scene. Keep all other objects in their exact positions."
        
        else:  # shape
            # Find all unique shapes
            shapes = list(set(obj["shape"] for obj in objects))
            remove_shape = self.rng.choice(shapes)
            
            # Find all objects with this shape
            target_object_ids = [obj["id"] for obj in objects if obj["shape"] == remove_shape]
            
            rule = {
                "level": "L1",
                "rule_type": "shape",
                "remove_shape": remove_shape,
                "target_object_ids": target_object_ids
            }
            
            # Generate prompt
            shape_name = remove_shape.capitalize()
            prompt = f"Remove all {shape_name.lower()} objects from the scene. Keep all other objects in their exact positions."
        
        return rule, prompt
    
    def _calculate_spatial_relations(self, objects: List[Dict[str, Any]], 
                                   canvas_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate spatial relations for all objects.
        Uses stricter boundaries to ensure spatial relations are visually clear.
        
        Args:
            objects: List of object dictionaries
            canvas_size: (width, height) of canvas
            
        Returns:
            Dictionary with spatial relation information
        """
        canvas_w, canvas_h = canvas_size
        center_x, center_y = canvas_w // 2, canvas_h // 2
        
        # Very strict boundaries to ensure maximum visual clarity and accuracy
        # For 256x256 canvas: center = 128
        # Use much tighter thresholds to ensure objects are clearly in their regions
        # Left: x < 64 (left 25%), Right: x > 192 (right 25%)
        # Top: y < 64 (top 25%), Bottom: y > 192 (bottom 25%)
        # This ensures objects are clearly positioned, not ambiguous
        margin_ratio = 0.25  # Use 25% margin from center for very clear distinction
        left_threshold = int(center_x * (1 - margin_ratio))  # 96 for 256 canvas
        right_threshold = int(center_x * (1 + margin_ratio))  # 160 for 256 canvas
        top_threshold = int(center_y * (1 - margin_ratio))  # 96 for 256 canvas
        bottom_threshold = int(center_y * (1 + margin_ratio))  # 160 for 256 canvas
        quadrant_margin = int(center_x * 0.3)  # 30% margin = 77px for 256 canvas, very strict
        
        relations = {
            "center": (center_x, center_y),
            "left_objects": [],  # x < center_x (loose, for general classification)
            "right_objects": [],  # x >= center_x (loose)
            "top_objects": [],  # y < center_y (loose)
            "bottom_objects": [],  # y >= center_y (loose)
            "clear_left_objects": [],  # x < left_threshold (strict, for selection)
            "clear_right_objects": [],  # x > right_threshold (strict)
            "clear_top_objects": [],  # y < top_threshold (strict)
            "clear_bottom_objects": [],  # y > bottom_threshold (strict)
            "distances_from_center": {},
            "quadrants": {},
            "clear_quadrants": {},  # Objects clearly in quadrants (with margin)
            "x_sorted_objects": [],  # Objects sorted by x coordinate (ascending)
            "y_sorted_objects": [],  # Objects sorted by y coordinate (ascending)
            "corner_distances": {}  # Distance to nearest corner for each object
        }
        
        for obj in objects:
            x, y = obj["x"], obj["y"]
            
            # Loose Left/right classification (for general reference)
            if x < center_x:
                relations["left_objects"].append(obj["id"])
            else:
                relations["right_objects"].append(obj["id"])
            
            # Strict Left/right classification (for clear visual distinction)
            if x < left_threshold:
                relations["clear_left_objects"].append(obj["id"])
            elif x > right_threshold:
                relations["clear_right_objects"].append(obj["id"])
            
            # Loose Top/bottom classification
            if y < center_y:
                relations["top_objects"].append(obj["id"])
            else:
                relations["bottom_objects"].append(obj["id"])
            
            # Strict Top/bottom classification
            if y < top_threshold:
                relations["clear_top_objects"].append(obj["id"])
            elif y > bottom_threshold:
                relations["clear_bottom_objects"].append(obj["id"])
            
            # Distance from center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            relations["distances_from_center"][obj["id"]] = distance
            
            # Loose Quadrant classification
            if x < center_x and y < center_y:
                relations["quadrants"][obj["id"]] = "top_left"
            elif x >= center_x and y < center_y:
                relations["quadrants"][obj["id"]] = "top_right"
            elif x < center_x and y >= center_y:
                relations["quadrants"][obj["id"]] = "bottom_left"
            else:
                relations["quadrants"][obj["id"]] = "bottom_right"
            
            # Strict Quadrant classification (with margin from center lines)
            if x < center_x - quadrant_margin and y < center_y - quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "top_left"
            elif x > center_x + quadrant_margin and y < center_y - quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "top_right"
            elif x < center_x - quadrant_margin and y > center_y + quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "bottom_left"
            elif x > center_x + quadrant_margin and y > center_y + quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "bottom_right"
            
            # Calculate distance to nearest corner
            corners = [
                (0, 0),  # top-left
                (canvas_w, 0),  # top-right
                (0, canvas_h),  # bottom-left
                (canvas_w, canvas_h)  # bottom-right
            ]
            min_corner_dist = min([np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in corners])
            relations["corner_distances"][obj["id"]] = min_corner_dist
        
        # Sort objects by x and y coordinates for leftmost/rightmost/topmost/bottommost selection
        relations["x_sorted_objects"] = sorted(objects, key=lambda o: o["x"])
        relations["y_sorted_objects"] = sorted(objects, key=lambda o: o["y"])
        
        return relations
    
    def generate_l3_rule(self, objects: List[Dict[str, Any]], 
                         canvas_size: Tuple[int, int],
                         prompt_index: int = 0) -> Tuple[Dict[str, Any], str]:
        """
        Generate Level 3 rule: Remove objects using spatial relations.
        
        Args:
            objects: List of objects
            canvas_size: (width, height) of canvas
            prompt_index: Index into PROMPTS_L3 list
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Ensure we have enough objects (at least 3 for Level 3)
        if len(objects) < 3:
            return self.generate_l1_rule(objects, prompt_index)
        
        # Calculate spatial relations
        relations = self._calculate_spatial_relations(objects, canvas_size)
        
        # Define available spatial relation types (using strict boundaries + sorting)
        relation_types = []
        
        # Check which relation types are available
        # Use leftmost/rightmost/topmost/bottommost for more accurate selection
        if len(relations["clear_left_objects"]) > 0:
            relation_types.append("leftmost")  # More accurate: use sorting
            relation_types.append("left_side")  # Keep for backward compatibility
        if len(relations["clear_right_objects"]) > 0:
            relation_types.append("rightmost")  # More accurate: use sorting
            relation_types.append("right_side")  # Keep for backward compatibility
        if len(relations["clear_top_objects"]) > 0:
            relation_types.append("topmost")  # More accurate: use sorting
            relation_types.append("top_half")  # Keep for backward compatibility
        if len(relations["clear_bottom_objects"]) > 0:
            relation_types.append("bottommost")  # More accurate: use sorting
            relation_types.append("bottom_half")  # Keep for backward compatibility
        
        # Distance-based relations (check for sufficient distance difference)
        if len(objects) >= 2:
            distances = list(relations["distances_from_center"].values())
            if len(distances) >= 2:
                max_dist = max(distances)
                min_dist = min(distances)
                # Only include if distance difference is significant (at least 50px)
                if max_dist - min_dist >= 50:
                    relation_types.append("farthest_from_center")
                    if len(objects) >= 3:
                        relation_types.append("nearest_to_center")
        
        # Corner-based relations (new: objects closest to corners)
        if len(objects) >= 2:
            corner_dists = list(relations["corner_distances"].values())
            if len(corner_dists) >= 2:
                min_corner_dist = min(corner_dists)
                max_corner_dist = max(corner_dists)
                # Only include if there's significant variation in corner distances
                if max_corner_dist - min_corner_dist >= 30:
                    relation_types.append("corner_closest")
        
        # Quadrant-based relations (using clear_quadrants for strict boundaries)
        for quadrant in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            count = sum(1 for q in relations["clear_quadrants"].values() if q == quadrant)
            if count > 0:
                relation_types.append(f"{quadrant}_quadrant")
        
        if not relation_types:
            return self.generate_l1_rule(objects, prompt_index)
        
        # Randomly select a relation type
        relation_type = self.rng.choice(relation_types)
        
        # Select target objects based on relation type (using strict boundaries + sorting)
        target_object_ids = []
        num_to_remove = 0
        
        if relation_type == "leftmost":
            # Use sorting: select the N leftmost objects that are clearly on the left
            candidates = [obj for obj in relations["x_sorted_objects"] 
                         if obj["id"] in relations["clear_left_objects"]]
            if not candidates:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the leftmost N objects (already sorted by x)
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "rightmost":
            # Use sorting: select the N rightmost objects that are clearly on the right
            candidates = [obj for obj in relations["x_sorted_objects"] 
                         if obj["id"] in relations["clear_right_objects"]]
            if not candidates:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the rightmost N objects (reverse sorted by x)
            candidates.reverse()
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "topmost":
            # Use sorting: select the N topmost objects that are clearly on the top
            candidates = [obj for obj in relations["y_sorted_objects"] 
                         if obj["id"] in relations["clear_top_objects"]]
            if not candidates:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the topmost N objects (already sorted by y)
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "bottommost":
            # Use sorting: select the N bottommost objects that are clearly on the bottom
            candidates = [obj for obj in relations["y_sorted_objects"] 
                         if obj["id"] in relations["clear_bottom_objects"]]
            if not candidates:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the bottommost N objects (reverse sorted by y)
            candidates.reverse()
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "left_side":
            # Use clear_left_objects for strict boundary, then sort for accuracy
            candidates = [obj for obj in relations["x_sorted_objects"] 
                         if obj["id"] in relations["clear_left_objects"]]
            if not candidates:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the leftmost N from candidates (already sorted)
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "right_side":
            # Use clear_right_objects for strict boundary, then sort for accuracy
            candidates = [obj for obj in relations["x_sorted_objects"] 
                         if obj["id"] in relations["clear_right_objects"]]
            if not candidates:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the rightmost N from candidates (reverse sorted)
            candidates.reverse()
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "top_half":
            # Use VERY strict clear_top_objects for maximum accuracy
            # Only objects clearly in top 25% of canvas
            target_object_ids = relations["clear_top_objects"]
            if not target_object_ids:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = len(target_object_ids)
            # Ensure at least 1 object remains
            if num_to_remove >= len(objects):
                target_object_ids = self.rng.sample(target_object_ids, len(objects) - 1)
                num_to_remove = len(target_object_ids)
        
        elif relation_type == "bottom_half":
            # Use VERY strict clear_bottom_objects for maximum accuracy
            # Only objects clearly in bottom 25% of canvas
            target_object_ids = relations["clear_bottom_objects"]
            if not target_object_ids:
                return self.generate_l1_rule(objects, prompt_index)
            num_to_remove = len(target_object_ids)
            # Ensure at least 1 object remains
            if num_to_remove >= len(objects):
                target_object_ids = self.rng.sample(target_object_ids, len(objects) - 1)
                num_to_remove = len(target_object_ids)
        
        elif relation_type == "corner_closest":
            # Select objects closest to corners
            sorted_by_corner = sorted(
                objects,
                key=lambda o: relations["corner_distances"][o["id"]]
            )
            num_to_remove = min(self.rng.randint(1, 2), len(objects) - 1)
            target_object_ids = [o["id"] for o in sorted_by_corner[:num_to_remove]]
        
        elif relation_type == "farthest_from_center":
            # Sort objects by distance from center (farthest first)
            sorted_objects = sorted(
                objects,
                key=lambda o: relations["distances_from_center"][o["id"]],
                reverse=True
            )
            num_to_remove = min(self.rng.randint(1, 3), len(objects) - 1)
            target_object_ids = [o["id"] for o in sorted_objects[:num_to_remove]]
        
        elif relation_type == "nearest_to_center":
            # Sort objects by distance from center (nearest first)
            sorted_objects = sorted(
                objects,
                key=lambda o: relations["distances_from_center"][o["id"]]
            )
            num_to_remove = min(self.rng.randint(1, 3), len(objects) - 1)
            target_object_ids = [o["id"] for o in sorted_objects[:num_to_remove]]
        
        elif relation_type.endswith("_quadrant"):
            # Extract quadrant name and use STRICT clear_quadrants for maximum accuracy
            # This ensures objects are clearly in the quadrant, not ambiguous
            quadrant_name = relation_type.replace("_quadrant", "")
            candidates = [obj["id"] for obj in objects 
                        if relations["clear_quadrants"].get(obj["id"]) == quadrant_name]
            if not candidates:
                return self.generate_l1_rule(objects, prompt_index)
            # Remove all objects in this quadrant, or sample if too many
            if len(candidates) >= len(objects):
                target_object_ids = self.rng.sample(candidates, len(objects) - 1)
            else:
                target_object_ids = candidates
            num_to_remove = len(target_object_ids)
        
        # Generate prompt based on relation type
        prompt = self._generate_l3_prompt(relation_type, num_to_remove)
        
        # Create rule
        rule = {
            "level": "L3",
            "relation_type": relation_type,
            "target_object_ids": target_object_ids,
            "num_objects": num_to_remove
        }
        
        return rule, prompt
    
    def _calculate_conceptual_attributes(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate conceptual/semantic attributes for Level 4 reasoning.
        
        Args:
            objects: List of objects
            
        Returns:
            Dictionary with conceptual attributes:
            - size_categories: {"large": [ids], "small": [ids]}
            - shape_groups: {shape: [ids]}
            - color_groups: {color: [ids]}
            - outlier_objects: [ids] (objects that differ from majority)
        """
        attributes = {
            "size_categories": {"large": [], "small": []},
            "shape_groups": {},
            "color_groups": {},
            "outlier_objects": [],
            "size_threshold": None
        }
        
        if len(objects) < 2:
            return attributes
        
        # Calculate size categories (large vs small)
        # Use extreme percentiles (10th and 90th) to ensure VERY clear size distinction
        # This makes large vs small visually obvious and reduces ambiguity
        sizes = [obj["size"] for obj in objects]
        areas = [obj["area"] for obj in objects]
        
        sorted_sizes = sorted(sizes)
        num_objects = len(sorted_sizes)
        
        # Use 10th and 90th percentiles for extreme size distinction
        # Small: bottom 10%, Large: top 10%
        # This ensures maximum visual difference between large and small objects
        p10_index = max(0, num_objects // 10 - 1)  # 10th percentile
        p90_index = min(num_objects - 1, (9 * num_objects) // 10)  # 90th percentile
        
        if p10_index < len(sorted_sizes) and p90_index < len(sorted_sizes) and p10_index < p90_index:
            size_threshold_small = sorted_sizes[p10_index]
            size_threshold_large = sorted_sizes[p90_index]
            
            # Additional check: ensure minimum size gap between large and small
            # If gap is too small (< 15 pixels), use even more extreme thresholds
            min_gap = 15
            if size_threshold_large - size_threshold_small < min_gap:
                # Use 5th and 95th percentiles for even more extreme distinction
                p5_index = max(0, num_objects // 20 - 1)
                p95_index = min(num_objects - 1, (19 * num_objects) // 20)
                if p5_index < len(sorted_sizes) and p95_index < len(sorted_sizes):
                    size_threshold_small = sorted_sizes[p5_index]
                    size_threshold_large = sorted_sizes[p95_index]
        else:
            # Fallback: use median with very large margin
            size_median = np.median(sizes)
            size_threshold_small = size_median - 15  # Very large margin
            size_threshold_large = size_median + 15
        
        attributes["size_threshold"] = (size_threshold_small, size_threshold_large)
        
        for obj in objects:
            # Size classification using extreme percentiles
            # Small: size <= 10th percentile (or 5th if gap too small)
            # Large: size >= 90th percentile (or 95th if gap too small)
            if obj["size"] <= size_threshold_small:
                attributes["size_categories"]["small"].append(obj["id"])
            elif obj["size"] >= size_threshold_large:
                attributes["size_categories"]["large"].append(obj["id"])
            
            # Shape grouping
            shape = obj["shape"]
            if shape not in attributes["shape_groups"]:
                attributes["shape_groups"][shape] = []
            attributes["shape_groups"][shape].append(obj["id"])
            
            # Color grouping
            color = obj["color"]
            if color not in attributes["color_groups"]:
                attributes["color_groups"][color] = []
            attributes["color_groups"][color].append(obj["id"])
        
        # Find outlier objects (different from majority)
        # Outlier: objects that are clearly different from the majority
        # Strategy: Find the majority group (color+shape combination that appears most)
        # Then mark all objects NOT in this majority group as outliers
        # This ensures outliers are visually distinct and minority
        
        # Count all color+shape combinations
        combo_counts = {}
        for obj in objects:
            combo = (obj["color"], obj["shape"])
            if combo not in combo_counts:
                combo_counts[combo] = []
            combo_counts[combo].append(obj["id"])
        
        # Find the majority combination (must be > 33% to be considered majority)
        # This ensures clear distinction between majority and outliers
        majority_threshold = len(objects) * 0.33
        majority_combo = None
        max_count = 0
        
        for combo, ids in combo_counts.items():
            if len(ids) > max_count:
                max_count = len(ids)
                majority_combo = combo
        
        # For Level 4, we want to find THE ONE outlier (singular)
        # So we only mark outliers if there's a VERY clear majority (>=60%) and exactly ONE outlier
        # This will be checked again in generate_l4_rule, but we prepare the list here
        if majority_combo and max_count >= len(objects) * 0.6:
            # All objects NOT in the majority group are potential outliers
            for obj in objects:
                combo = (obj["color"], obj["shape"])
                if combo != majority_combo:
                    attributes["outlier_objects"].append(obj["id"])
        else:
            # If no clear majority, mark unique combinations as potential outliers
            for obj in objects:
                combo = (obj["color"], obj["shape"])
                same_combo_count = combo_counts[combo]
                # Only mark as outlier if truly unique (appears only once)
                if same_combo_count == 1:
                    attributes["outlier_objects"].append(obj["id"])
        
        # Remove duplicates from outlier_objects
        attributes["outlier_objects"] = list(dict.fromkeys(attributes["outlier_objects"]))
        
        return attributes
    
    def generate_l4_rule(self, objects: List[Dict[str, Any]], 
                         prompt_index: int = 0) -> Tuple[Dict[str, Any], str]:
        """
        Generate Level 4 rule: Remove objects based on conceptual/semantic properties.
        
        Args:
            objects: List of objects
            prompt_index: Index into PROMPTS_L4 list
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Ensure we have enough objects (at least 3 for Level 4)
        if len(objects) < 3:
            return self.generate_l1_rule(objects, prompt_index)
        
        # Calculate conceptual attributes
        attributes = self._calculate_conceptual_attributes(objects)
        
        # Define available conceptual relation types
        relation_types = []
        
        # Size-based relations - focus on "the largest" and "the smallest" (singular)
        # This ensures we remove THE ONE largest/smallest object, not all large/small objects
        # CRITICAL: Ensure ABSOLUTE visual distinction - the selected object must stand out
        sizes = [obj["size"] for obj in objects]
        max_size = max(sizes)
        min_size = min(sizes)
        
        # Only add if there's an ABSOLUTE size difference (at least 30 pixels between max and min)
        # AND the selected object is clearly different from the next closest object
        max_size_count = sum(1 for s in sizes if s == max_size)
        min_size_count = sum(1 for s in sizes if s == min_size)
        size_gap = max_size - min_size
        
        # Require MINIMUM 32 pixel gap between max and min for overall distinction
        # (50-18=32 is the maximum possible gap, so we use 32 as threshold)
        MIN_SIZE_GAP = 32
        # Require MINIMUM 20 pixel gap between selected object and next closest (increased for more obviousness)
        # This ensures the selected object is visually OBVIOUS and stands out clearly
        MIN_ISOLATION_GAP = 20
        
        if size_gap >= MIN_SIZE_GAP:  # Overall gap is sufficient (32px is maximum with 18-50 range)
            # Check if largest is isolated (clearly larger than second largest)
            if max_size_count == 1:  # Exactly one largest
                sorted_sizes = sorted(set(sizes), reverse=True)
                if len(sorted_sizes) >= 2:
                    second_largest = sorted_sizes[1]
                    largest_isolation = max_size - second_largest
                    if largest_isolation >= MIN_ISOLATION_GAP:
                        relation_types.append("remove_largest")  # Remove THE largest (singular)
            
            # Check if smallest is isolated (clearly smaller than second smallest)
            if min_size_count == 1:  # Exactly one smallest
                sorted_sizes = sorted(set(sizes))
                if len(sorted_sizes) >= 2:
                    second_smallest = sorted_sizes[1]
                    smallest_isolation = second_smallest - min_size
                    if smallest_isolation >= MIN_ISOLATION_GAP:
                        relation_types.append("remove_smallest")  # Remove THE smallest (singular)
        
        # Similarity-based relations (outlier detection)
        # Focus on "the one that looks different" - ensure only ONE outlier
        # Majority should be VERY clear (at least 70% of objects) for maximum obviousness
        # Check for clear majority group (color+shape combination)
        combo_counts = {}
        for obj in objects:
            combo = (obj["color"], obj["shape"])
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        
        majority_combo = max(combo_counts.items(), key=lambda x: x[1], default=None)
        
        # Only add outlier if there's a VERY clear majority (>=70%) and exactly ONE outlier
        # This ensures the outlier is OBVIOUS and stands out clearly
        if majority_combo and majority_combo[1] >= len(objects) * 0.7:
            # Count how many objects are NOT in the majority
            outlier_count = len(objects) - majority_combo[1]
            if outlier_count == 1:  # Exactly ONE outlier
                relation_types.append("remove_outlier")  # Remove THE one that looks different
        
        # REMOVED: keep_same_shape and keep_same_color
        # These are NOT true relative concepts - they're explicit attribute matching (L1 level)
        # Level 4 should ONLY focus on relative concepts: largest, smallest, outlier
        
        if not relation_types:
            # Fallback to L1 if no suitable L4 rule can be generated
            return self.generate_l1_rule(objects, prompt_index)
        
        # Prefer size-based tasks (remove_largest/smallest) over outlier if both are available
        # This ensures we get a good mix of relative concept types
        size_based_types = [rt for rt in relation_types if rt in ["remove_largest", "remove_smallest"]]
        if size_based_types:
            # If size-based tasks are available, prefer them (70% chance)
            if self.rng.random() < 0.7:
                relation_type = self.rng.choice(size_based_types)
            else:
                # Otherwise choose from all available types
                relation_type = self.rng.choice(relation_types)
        else:
            # No size-based tasks available, choose from what's available
            relation_type = self.rng.choice(relation_types)
        
        # Select target objects based on relation type
        target_object_ids = []
        num_to_remove = 0
        
        if relation_type == "remove_largest":
            # Remove THE largest object (singular, exactly one)
            sizes = [obj["size"] for obj in objects]
            max_size = max(sizes)
            target_object_ids = [obj["id"] for obj in objects if obj["size"] == max_size]
            # Should be exactly one, but ensure it is
            if len(target_object_ids) > 1:
                # If tie, pick the first one (shouldn't happen due to our check)
                target_object_ids = [target_object_ids[0]]
            num_to_remove = len(target_object_ids)
        
        elif relation_type == "remove_smallest":
            # Remove THE smallest object (singular, exactly one)
            sizes = [obj["size"] for obj in objects]
            min_size = min(sizes)
            target_object_ids = [obj["id"] for obj in objects if obj["size"] == min_size]
            # Should be exactly one, but ensure it is
            if len(target_object_ids) > 1:
                # If tie, pick the first one (shouldn't happen due to our check)
                target_object_ids = [target_object_ids[0]]
            num_to_remove = len(target_object_ids)
        
        elif relation_type == "remove_outlier":
            # Remove THE one object that looks different (singular, exactly one)
            # This should be exactly one due to our check above
            target_object_ids = attributes["outlier_objects"]
            # Ensure it's exactly one (should be guaranteed by our check)
            if len(target_object_ids) != 1:
                # Fallback: if somehow multiple, pick the first one
                target_object_ids = [target_object_ids[0]] if target_object_ids else []
            num_to_remove = len(target_object_ids)
        
        
        # Generate prompt based on relation type
        prompt = self._generate_l4_prompt(relation_type, num_to_remove, attributes, objects)
        
        # Create rule
        rule = {
            "level": "L4",
            "relation_type": relation_type,
            "target_object_ids": target_object_ids,
            "num_objects": num_to_remove,
            "attributes": {
                "size_categories": attributes["size_categories"],
                "outlier_objects": attributes["outlier_objects"]
            }
        }
        
        return rule, prompt
    
    def _generate_l4_prompt(self, relation_type: str, num_objects: int,
                           attributes: Dict[str, Any], objects: List[Dict[str, Any]]) -> str:
        """
        Generate prompt text for Level 4 based on conceptual relation type.
        
        Args:
            relation_type: Type of conceptual relation
            num_objects: Number of objects to remove
            attributes: Conceptual attributes dictionary
            objects: List of all objects
            
        Returns:
            Prompt text
        """
        if relation_type == "remove_largest":
            # Always singular: "the largest"
            prompt = "Remove the largest object. Keep all other objects in their exact positions."
        
        elif relation_type == "remove_smallest":
            # Always singular: "the smallest"
            prompt = "Remove the smallest object. Keep all other objects in their exact positions."
        
        elif relation_type == "remove_outlier":
            # Always singular: "the one that looks different"
            prompt = "Remove the object that looks different from the others. Keep all similar objects fixed in their positions."
        
        else:
            # Fallback prompt
            prompt = f"Remove {num_objects} objects based on their conceptual properties. Keep all other objects in their exact positions."
        
        return prompt
    
    def _generate_l3_prompt(self, relation_type: str, num_objects: int) -> str:
        """
        Generate prompt text for Level 3 based on relation type.
        
        Args:
            relation_type: Type of spatial relation
            num_objects: Number of objects to remove
            
        Returns:
            Prompt text
        """
        if relation_type == "leftmost":
            if num_objects == 1:
                prompt = "Remove the leftmost object. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} leftmost objects. Keep all other objects in their exact positions."
        
        elif relation_type == "rightmost":
            if num_objects == 1:
                prompt = "Remove the rightmost object. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} rightmost objects. Keep all other objects in their exact positions."
        
        elif relation_type == "topmost":
            if num_objects == 1:
                prompt = "Remove the topmost object. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} topmost objects. Keep all other objects in their exact positions."
        
        elif relation_type == "bottommost":
            if num_objects == 1:
                prompt = "Remove the bottommost object. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} bottommost objects. Keep all other objects in their exact positions."
        
        elif relation_type == "left_side":
            if num_objects == 1:
                prompt = "Remove the object on the far left side of the screen. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} objects on the far left side of the screen. Keep all other objects in their exact positions."
        
        elif relation_type == "right_side":
            if num_objects == 1:
                prompt = "Remove the object on the far right side of the screen. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} objects on the far right side of the screen. Keep all other objects in their exact positions."
        
        elif relation_type == "top_half":
            prompt = "Remove all objects in the upper half of the image. Keep objects in the lower half unchanged."
        
        elif relation_type == "bottom_half":
            prompt = "Remove all objects in the lower half of the image. Keep objects in the upper half unchanged."
        
        elif relation_type == "farthest_from_center":
            if num_objects == 1:
                prompt = "Move the object farthest from the center out of view. Keep all remaining objects stationary."
            else:
                prompt = f"Move the {num_objects} objects farthest from the center out of view. Keep all remaining objects stationary."
        
        elif relation_type == "nearest_to_center":
            if num_objects == 1:
                prompt = "Remove the object nearest to the center. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} objects nearest to the center. Keep all other objects in their exact positions."
        
        elif relation_type == "top_left_quadrant":
            prompt = "Remove all objects in the top-left quadrant. Keep all other objects in their exact positions."
        
        elif relation_type == "top_right_quadrant":
            prompt = "Remove all objects in the top-right quadrant. Keep all other objects in their exact positions."
        
        elif relation_type == "bottom_left_quadrant":
            prompt = "Remove all objects in the bottom-left quadrant. Keep all other objects in their exact positions."
        
        elif relation_type == "bottom_right_quadrant":
            prompt = "Remove all objects in the bottom-right quadrant. Keep all other objects in their exact positions."
        
        elif relation_type == "corner_closest":
            if num_objects == 1:
                prompt = "Remove the object closest to a corner. Keep all other objects in their exact positions."
            else:
                prompt = f"Remove the {num_objects} objects closest to the corners. Keep all other objects in their exact positions."
        
        else:
            # Fallback prompt
            prompt = f"Remove {num_objects} objects based on spatial relations. Keep all other objects in their exact positions."
        
        return prompt
    
    def generate_l2_rule(self, objects: List[Dict[str, Any]], 
                        prompt_index: int = 0) -> Tuple[Dict[str, Any], str]:
        """
        Generate Level 2 rule: Remove multiple explicitly listed objects by color and shape.
        
        Args:
            objects: List of objects
            prompt_index: Index into PROMPTS_L2 list
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Ensure we have enough objects (at least 3-4 for Level 2)
        if len(objects) < 3:
            # Fallback to L1 if not enough objects
            return self.generate_l1_rule(objects, prompt_index)
        
        # Randomly select 2-3 objects to remove (ensure at least 1 object remains)
        num_to_remove = self.rng.randint(2, min(3, len(objects) - 1))
        selected_objects = self.rng.sample(objects, num_to_remove)
        
        # Build targets list with color and shape combinations
        targets = [
            {"color": obj["color"], "shape": obj["shape"]}
            for obj in selected_objects
        ]
        
        # Get target object IDs - include ALL objects matching each target
        # This ensures "Remove the yellow sphere" removes ALL yellow spheres, not just one
        target_object_ids = []
        for target in targets:
            matching_objs = [obj for obj in objects 
                           if obj["color"] == target["color"] and obj["shape"] == target["shape"]]
            target_object_ids.extend([obj["id"] for obj in matching_objs])
        
        # Remove duplicates while preserving order
        target_object_ids = list(dict.fromkeys(target_object_ids))
        
        # Generate prompt dynamically
        # Format: "Remove the red cube, all green spheres, and the blue pyramid..."
        # Use "all" + plural if multiple objects match the target
        target_descriptions = []
        for target in targets:
            matching_count = len([obj for obj in objects 
                                if obj["color"] == target["color"] and obj["shape"] == target["shape"]])
            
            if matching_count > 1:
                # Multiple objects: use "all" + plural form
                shape_plural = target['shape'] + 's' if not target['shape'].endswith('s') else target['shape']
                target_descriptions.append(f"all {target['color']} {shape_plural}")
            else:
                # Single object: use "the" + singular form
                target_descriptions.append(f"the {target['color']} {target['shape']}")
        
        if len(target_descriptions) == 2:
            prompt = f"Remove {target_descriptions[0]} and {target_descriptions[1]} from the scene. Keep all other objects fixed in their positions."
        else:  # 3 objects
            prompt = f"Remove {target_descriptions[0]}, {target_descriptions[1]}, and {target_descriptions[2]} from the scene. Keep all other objects fixed in their positions."
        
        rule = {
            "level": "L2",
            "targets": targets,
            "target_object_ids": target_object_ids
        }
        
        return rule, prompt


class ObjectSubtractionGenerator:
    """Main generator for object subtraction tasks."""
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256),
                 num_objects_range: Tuple[int, int] = (5, 8)):
        """
        Initialize generator.
        
        Args:
            canvas_size: (width, height) of canvas
            num_objects_range: (min, max) number of objects per scene
        """
        self.canvas_size = canvas_size
        self.num_objects_range = num_objects_range
        self.object_gen = ObjectGenerator(canvas_size=canvas_size)
        self.renderer = SceneRenderer(canvas_size=canvas_size)
        self.rule_gen = RuleGenerator()
    
    def _get_difficulty(self, level: str) -> str:
        """Get difficulty level based on cognitive level."""
        difficulty_map = {
            "L1": "easy",
            "L2": "medium",
            "L3": "hard",
            "L4": "hard"
        }
        return difficulty_map.get(level, "medium")
    
    def generate_single_task(self, task_id: str, level: str = "L1", 
                            seed: Optional[int] = None) -> ObjectSubtractionTaskPair:
        """
        Generate a single object subtraction task.
        
        Args:
            task_id: Unique task identifier
            level: Cognitive level ("L1", "L2", "L3", "L4")
            seed: Random seed for reproducibility
            
        Returns:
            ObjectSubtractionTaskPair
        """
        # Use temporary directory like other tasks
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.rule_gen.rng.seed(seed)
            self.object_gen.rng.seed(seed)
        
        # Generate objects
        # For L4, use wider size range to ensure absolute visual distinction
        # This makes it easier to have large size gaps (>=35 pixels)
        if level == "L4":
            # Use wider size range for L4: 18-50 pixels (instead of default 20-40)
            # This ensures we can easily get 35+ pixel gaps between largest and smallest
            # Max gap possible: 50 - 18 = 32 pixels, but we'll force it to be at least 35
            original_min = self.object_gen.min_size
            original_max = self.object_gen.max_size
            self.object_gen.min_size = 18
            self.object_gen.max_size = 50
        
        # For L4, generate more objects to increase chance of having clear majorities
        # This makes it easier to satisfy L4 conditions (>=60% majority, etc.)
        if level == "L4":
            # Generate 6-8 objects (instead of 5-8) to increase chance of clear majorities
            num_objects = random.randint(6, 8)
        else:
            num_objects = random.randint(self.num_objects_range[0], self.num_objects_range[1])
        
        objects = self.object_gen.generate_objects(num_objects, seed=seed)
        
        # For L4, ensure we have objects with extreme sizes to guarantee >=30 pixel gap
        # AND ensure the extreme objects are isolated (at least 15px from next closest)
        # This ensures "largest" and "smallest" are absolutely obvious and stand out
        if level == "L4":
            sizes = [obj["size"] for obj in objects]
            max_size = max(sizes)
            min_size = min(sizes)
            gap = max_size - min_size
            
            # Ensure overall gap is at least 32 pixels (50-18=32 is maximum possible)
            # Force at least one object to be at max (50) and one at min (18) for maximum gap
            if gap < 32:
                # Find object with max size and force it to maximum (50)
                for obj in objects:
                    if obj["size"] == max_size:
                        obj["size"] = 50
                        # Update area accordingly
                        if obj["shape"] == "cube":
                            obj["area"] = obj["size"] * obj["size"]
                        elif obj["shape"] == "sphere":
                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                        elif obj["shape"] == "pyramid":
                            obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                        else:  # cone
                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                        break
                
                # Find object with min size and force it to minimum (18)
                for obj in objects:
                    if obj["size"] == min_size:
                        obj["size"] = 18
                        # Update area accordingly
                        if obj["shape"] == "cube":
                            obj["area"] = obj["size"] * obj["size"]
                        elif obj["shape"] == "sphere":
                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                        elif obj["shape"] == "pyramid":
                            obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                        else:  # cone
                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                        break
                
                # After forcing max to 50 and min to 18, gap should be at least 32
                # Recalculate to verify
                sizes = [obj["size"] for obj in objects]
                new_gap = max(sizes) - min(sizes)
                # Gap should now be at least 32 (50-18=32)
            
            # After ensuring overall gap, check isolation gaps
            # Recalculate sizes after potential adjustments
            sizes = [obj["size"] for obj in objects]
            unique_sizes = sorted(set(sizes))
            
            # CRITICAL: Ensure largest is UNIQUE and isolated
            # If multiple objects have max_size, make one larger and others smaller
            max_size = max(sizes)
            max_size_count = sum(1 for s in sizes if s == max_size)
            if max_size_count > 1:
                # Multiple objects with max size - make one unique largest
                max_objects = [obj for obj in objects if obj["size"] == max_size]
                # Keep first one as largest, make others smaller
                for i, obj in enumerate(max_objects):
                    if i == 0:
                        # Make this one even larger to ensure uniqueness
                        obj["size"] = min(50, max_size + 15)
                    else:
                        # Make others smaller (at least 15px smaller)
                        obj["size"] = max(18, max_size - 15)
                    # Update area
                    if obj["shape"] == "cube":
                        obj["area"] = obj["size"] * obj["size"]
                    elif obj["shape"] == "sphere":
                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                    elif obj["shape"] == "pyramid":
                        obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                    else:  # cone
                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
            
            # CRITICAL: Ensure smallest is UNIQUE and isolated
            sizes = [obj["size"] for obj in objects]  # Recalculate after adjustments
            min_size = min(sizes)
            min_size_count = sum(1 for s in sizes if s == min_size)
            if min_size_count > 1:
                # Multiple objects with min size - make one unique smallest
                min_objects = [obj for obj in objects if obj["size"] == min_size]
                # Keep first one as smallest, make others larger
                for i, obj in enumerate(min_objects):
                    if i == 0:
                        # Make this one even smaller to ensure uniqueness
                        obj["size"] = max(18, min_size - 15)
                    else:
                        # Make others larger (at least 15px larger)
                        obj["size"] = min(50, min_size + 15)
                    # Update area
                    if obj["shape"] == "cube":
                        obj["area"] = obj["size"] * obj["size"]
                    elif obj["shape"] == "sphere":
                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                    elif obj["shape"] == "pyramid":
                        obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                    else:  # cone
                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
            
            # Now ensure isolation gaps - CRITICAL for L4 to work
            # Strategy: Ensure both largest and smallest are isolated simultaneously
            # If largest is 50 and smallest is 18, we need:
            #   - second_largest <= 30 (for 20px gap from 50)
            #   - second_smallest >= 38 (for 20px gap from 18)
            # This creates a clear separation: [18] ... [38-50] with gap in between
            sizes = [obj["size"] for obj in objects]  # Recalculate
            unique_sizes = sorted(set(sizes))
            
            if len(unique_sizes) >= 2:
                max_size = unique_sizes[-1]
                min_size = unique_sizes[0]
                second_largest = unique_sizes[-2] if len(unique_sizes) >= 2 else max_size
                second_smallest = unique_sizes[1] if len(unique_sizes) >= 2 else min_size
                
                # Adjust to ensure both isolations are satisfied
                # Strategy: Create a clear separation: [18] ... [29] ... [38-50]
                # This ensures: largest (50) is isolated from second (<=29), smallest (18) is isolated from second (>=38)
                if max_size >= 50 and min_size <= 18:
                    # Largest is 50, smallest is 18
                    # Need: second_largest <= 29 (for 21px gap from 50), second_smallest >= 38 (for 20px gap from 18)
                    target_second_largest = 50 - 21  # 29 (ensures 21px gap)
                    target_second_smallest = 18 + 20  # 38 (ensures 20px gap)
                    
                    # Step 1: Adjust all objects >= 30 but < 50 to 29 (for largest isolation)
                    # This creates a clear gap: largest (50) vs others (<=29)
                    for obj in objects:
                        if obj["size"] < 50 and obj["size"] >= 30:
                            obj["size"] = 29
                            # Update area
                            if obj["shape"] == "cube":
                                obj["area"] = obj["size"] * obj["size"]
                            elif obj["shape"] == "sphere":
                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                            elif obj["shape"] == "pyramid":
                                obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                            else:  # cone
                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                    
                    # Step 2: After step 1, ensure smallest isolation
                    # After making all >=30 objects to 29, we need to ensure second_smallest >= 38
                    # Strategy: Make sure at least one object (besides the smallest) is >= 38
                    # If second_smallest is 29 (from step 1), we need to make some objects 38
                    sizes_after_step1 = [obj["size"] for obj in objects]
                    unique_sizes_after = sorted(set(sizes_after_step1))
                    if len(unique_sizes_after) >= 2:
                        min_size_after = unique_sizes_after[0]
                        second_smallest_after = unique_sizes_after[1]
                        smallest_isolation_after = second_smallest_after - min_size_after
                        
                        # If smallest isolation is not satisfied (second_smallest is 29 or less)
                        if smallest_isolation_after < 20:
                            # Need to make second_smallest >= 38
                            # Find objects that are 29 (from step 1) and make at least one of them 38
                            # This ensures smallest isolation while preserving largest isolation
                            objects_at_29 = [obj for obj in objects if obj["size"] == 29]
                            if objects_at_29:
                                # Make the first object at 29 become 38 (ensures smallest isolation)
                                # Keep others at 29 (preserves largest isolation)
                                objects_at_29[0]["size"] = 38
                                # Update area
                                if objects_at_29[0]["shape"] == "cube":
                                    objects_at_29[0]["area"] = objects_at_29[0]["size"] * objects_at_29[0]["size"]
                                elif objects_at_29[0]["shape"] == "sphere":
                                    objects_at_29[0]["area"] = int(np.pi * (objects_at_29[0]["size"] // 2) ** 2)
                                elif objects_at_29[0]["shape"] == "pyramid":
                                    objects_at_29[0]["area"] = int(objects_at_29[0]["size"] * objects_at_29[0]["size"] * 0.433)
                                else:  # cone
                                    objects_at_29[0]["area"] = int(np.pi * (objects_at_29[0]["size"] // 2) ** 2)
                            
                            # Also make any other objects > 18 and < 38 (but not 29) to 38
                            for obj in objects:
                                if obj["size"] > 18 and obj["size"] < 38 and obj["size"] != 29:
                                    obj["size"] = 38
                                    # Update area
                                    if obj["shape"] == "cube":
                                        obj["area"] = obj["size"] * obj["size"]
                                    elif obj["shape"] == "sphere":
                                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                    elif obj["shape"] == "pyramid":
                                        obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                    else:  # cone
                                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                elif max_size >= 50:
                    # Only largest is at extreme, adjust for largest isolation
                    target_second = 50 - 20  # 30
                    for obj in objects:
                        if obj["size"] < 50 and obj["size"] >= target_second:
                            obj["size"] = max(18, target_second - 1)  # 29
                            # Update area
                            if obj["shape"] == "cube":
                                obj["area"] = obj["size"] * obj["size"]
                            elif obj["shape"] == "sphere":
                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                            elif obj["shape"] == "pyramid":
                                obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                            else:  # cone
                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                elif min_size <= 18:
                    # Only smallest is at extreme, adjust for smallest isolation
                    target_second = 18 + 20  # 38
                    for obj in objects:
                        if obj["size"] > min_size and obj["size"] < target_second:
                            obj["size"] = min(50, target_second)  # 38
                            # Update area
                            if obj["shape"] == "cube":
                                obj["area"] = obj["size"] * obj["size"]
                            elif obj["shape"] == "sphere":
                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                            elif obj["shape"] == "pyramid":
                                obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                            else:  # cone
                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
            
            # For L4, we DON'T actively create majority groups here
            # Let the rule generation decide which type of task to create
            # If size-based tasks are available (remove_largest/smallest), they should be preferred
            # Only create majority for outlier if size-based tasks are not available
            # This ensures we get a good mix of task types
        
        # Restore original size range after generation
        if level == "L4":
            self.object_gen.min_size = original_min
            self.object_gen.max_size = original_max
        
        # Generate rule and prompt based on level
        # For L4, retry if fallback to L1 to ensure we get proper L4 tasks
        # Use more retries to ensure we get proper L4 tasks
        max_retries = 10 if level == "L4" else 1
        
        for attempt in range(max_retries):
            if level == "L1":
                rule, prompt = self.rule_gen.generate_l1_rule(objects, 
                                                              prompt_index=DEFAULT_PROMPT_INDEX_L1)
            elif level == "L2":
                rule, prompt = self.rule_gen.generate_l2_rule(objects, 
                                                              prompt_index=DEFAULT_PROMPT_INDEX_L2)
            elif level == "L3":
                rule, prompt = self.rule_gen.generate_l3_rule(objects, 
                                                              canvas_size=self.canvas_size,
                                                              prompt_index=DEFAULT_PROMPT_INDEX_L3)
            elif level == "L4":
                rule, prompt = self.rule_gen.generate_l4_rule(objects, 
                                                              prompt_index=DEFAULT_PROMPT_INDEX_L4)
                # Check if we got a proper L4 rule (not fallback to L1)
                if rule.get("level") == "L4":
                    break  # Success! Got a proper L4 rule
                else:
                    # Fallback to L1 - retry with new objects
                    if attempt < max_retries - 1:
                        # Regenerate objects with different seed
                        retry_seed = seed + 1000 + attempt if seed is not None else None
                        if retry_seed is not None:
                            random.seed(retry_seed)
                            np.random.seed(retry_seed)
                            self.rule_gen.rng.seed(retry_seed)
                            self.object_gen.rng.seed(retry_seed)
                        
                        # Regenerate objects
                        if level == "L4":
                            original_min = self.object_gen.min_size
                            original_max = self.object_gen.max_size
                            self.object_gen.min_size = 18
                            self.object_gen.max_size = 50
                        
                        # For L4 retry, use 6-8 objects to increase chance of clear majorities
                        num_objects = random.randint(6, 8) if level == "L4" else random.randint(self.num_objects_range[0], self.num_objects_range[1])
                        objects = self.object_gen.generate_objects(num_objects, seed=retry_seed)
                        
                        # Apply L4 size adjustments again
                        if level == "L4":
                            sizes = [obj["size"] for obj in objects]
                            max_size = max(sizes)
                            min_size = min(sizes)
                            gap = max_size - min_size
                            
                            if gap < 35:
                                for obj in objects:
                                    if obj["size"] == max_size:
                                        obj["size"] = min(50, obj["size"] + (35 - gap))
                                        if obj["shape"] == "cube":
                                            obj["area"] = obj["size"] * obj["size"]
                                        elif obj["shape"] == "sphere":
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        elif obj["shape"] == "pyramid":
                                            obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                        else:
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        break
                                
                                for obj in objects:
                                    if obj["size"] == min_size:
                                        obj["size"] = max(18, obj["size"] - (35 - gap))
                                        if obj["shape"] == "cube":
                                            obj["area"] = obj["size"] * obj["size"]
                                        elif obj["shape"] == "sphere":
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        elif obj["shape"] == "pyramid":
                                            obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                        else:
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        break
                            
                            # Apply same size adjustments as initial generation
                            sizes = [obj["size"] for obj in objects]
                            max_size = max(sizes)
                            min_size = min(sizes)
                            gap = max_size - min_size
                            
                            if gap < 35:
                                for obj in objects:
                                    if obj["size"] == max_size:
                                        obj["size"] = min(50, obj["size"] + (35 - gap))
                                        if obj["shape"] == "cube":
                                            obj["area"] = obj["size"] * obj["size"]
                                        elif obj["shape"] == "sphere":
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        elif obj["shape"] == "pyramid":
                                            obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                        else:
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        break
                                
                                for obj in objects:
                                    if obj["size"] == min_size:
                                        obj["size"] = max(18, obj["size"] - (35 - gap))
                                        if obj["shape"] == "cube":
                                            obj["area"] = obj["size"] * obj["size"]
                                        elif obj["shape"] == "sphere":
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        elif obj["shape"] == "pyramid":
                                            obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                        else:
                                            obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                        break
                            
                            # Ensure uniqueness and isolation (same logic as initial generation)
                            sizes = [obj["size"] for obj in objects]
                            max_size = max(sizes)
                            min_size = min(sizes)
                            max_size_count = sum(1 for s in sizes if s == max_size)
                            min_size_count = sum(1 for s in sizes if s == min_size)
                            
                            if max_size_count > 1:
                                max_objects = [obj for obj in objects if obj["size"] == max_size]
                                for i, obj in enumerate(max_objects):
                                    if i == 0:
                                        obj["size"] = min(50, max_size + 20)
                                    else:
                                        obj["size"] = max(18, max_size - 15)
                                    if obj["shape"] == "cube":
                                        obj["area"] = obj["size"] * obj["size"]
                                    elif obj["shape"] == "sphere":
                                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                    elif obj["shape"] == "pyramid":
                                        obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                    else:
                                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                            
                            sizes = [obj["size"] for obj in objects]
                            min_size = min(sizes)
                            min_size_count = sum(1 for s in sizes if s == min_size)
                            if min_size_count > 1:
                                min_objects = [obj for obj in objects if obj["size"] == min_size]
                                for i, obj in enumerate(min_objects):
                                    if i == 0:
                                        obj["size"] = max(18, min_size - 20)
                                    else:
                                        obj["size"] = min(50, min_size + 15)
                                    if obj["shape"] == "cube":
                                        obj["area"] = obj["size"] * obj["size"]
                                    elif obj["shape"] == "sphere":
                                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                    elif obj["shape"] == "pyramid":
                                        obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                    else:
                                        obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                            
                            sizes = [obj["size"] for obj in objects]
                            unique_sizes = sorted(set(sizes))
                            
                            if len(unique_sizes) >= 2:
                                max_size = unique_sizes[-1]
                                second_largest = unique_sizes[-2]
                                if max_size - second_largest < 20:
                                    for obj in objects:
                                        if obj["size"] == max_size:
                                            obj["size"] = min(50, second_largest + 20)
                                            if obj["shape"] == "cube":
                                                obj["area"] = obj["size"] * obj["size"]
                                            elif obj["shape"] == "sphere":
                                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                            elif obj["shape"] == "pyramid":
                                                obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                            else:
                                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                            break
                                
                                min_size = unique_sizes[0]
                                second_smallest = unique_sizes[1]
                                if second_smallest - min_size < 20:
                                    for obj in objects:
                                        if obj["size"] == min_size:
                                            obj["size"] = max(18, second_smallest - 20)
                                            if obj["shape"] == "cube":
                                                obj["area"] = obj["size"] * obj["size"]
                                            elif obj["shape"] == "sphere":
                                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                            elif obj["shape"] == "pyramid":
                                                obj["area"] = int(obj["size"] * obj["size"] * 0.433)
                                            else:
                                                obj["area"] = int(np.pi * (obj["size"] // 2) ** 2)
                                            break
                            
                            # Don't actively create majority here - let rule generation decide
                            # This allows size-based tasks to be generated when possible
                            
                            self.object_gen.min_size = original_min
                            self.object_gen.max_size = original_max
                        continue  # Retry with new objects
            else:
                raise ValueError(f"Level {level} not yet implemented")
        
        # For L4, if we still got L1 rule after retries, that's a problem
        # But we'll accept it as a last resort (should be rare with retries)
        if level == "L4" and rule.get("level") != "L4":
            # This should be very rare after retries, but log it
            pass
        
        # Get target object IDs to remove
        remove_ids = rule["target_object_ids"]
        keep_ids = [obj["id"] for obj in objects if obj["id"] not in remove_ids]
        
        # Render first frame (all objects)
        first_path = Path(temp_dir) / f"{task_id}_first.png"
        self.renderer.render_scene(objects, output_path=first_path, show_object_ids=None)
        
        # Render final frame (only kept objects)
        final_path = Path(temp_dir) / f"{task_id}_final.png"
        self.renderer.render_scene(objects, output_path=final_path, show_object_ids=keep_ids)
        
        # Create task pair
        task_pair = ObjectSubtractionTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="ObjectSubtraction",
            level=level,
            object_subtraction_data={
                "objects": objects,
                "rule": rule,
                "remove_object_ids": remove_ids,
                "keep_object_ids": keep_ids,
                "num_objects": num_objects,
                "num_removed": len(remove_ids),
                "num_kept": len(keep_ids)
            },
            difficulty=self._get_difficulty(level),
            created_at=datetime.now().isoformat()
        )
        
        return task_pair


def create_dataset(num_samples: int = 50, levels: List[str] = ["L1", "L2", "L3", "L4"]) -> Dict[str, Any]:
    """
    Create object subtraction dataset - main entry point matching other tasks.
    
    Args:
        num_samples: Number of tasks to generate
        levels: List of cognitive levels to generate ("L1", "L2", "L3", "L4")
        
    Returns:
        Dataset dictionary in standard format
    """
    print(f"🎯 Creating Object Subtraction Dataset")
    print(f"   Total samples: {num_samples}")
    print(f"   Levels: {', '.join(levels)}")
    
    start_time = datetime.now()
    
    generator = ObjectSubtractionGenerator()
    pairs = []
    
    # Distribute samples across levels
    samples_per_level = num_samples // len(levels)
    remainder = num_samples % len(levels)
    
    for level_idx, level in enumerate(levels):
        level_samples = samples_per_level + (1 if level_idx < remainder else 0)
        
        print(f"\n📊 Generating {level_samples} tasks for Level {level}...")
        
        for i in range(level_samples):
            task_id = f"object_subtraction_{level.lower()}_{i:04d}"
            seed = 2025 + level_idx * 10000 + i  # Deterministic seed
            
            try:
                task_pair = generator.generate_single_task(task_id, level=level, seed=seed)
                pairs.append(task_pair)
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{level_samples} tasks...")
            except Exception as e:
                print(f"❌ Error generating task {task_id}: {e}")
                continue
    
    # Convert to dictionary format for consistency with other tasks
    pairs_dict = []
    for pair in pairs:
        pair_dict = {
            "id": pair.id,
            "prompt": pair.prompt,
            "first_image_path": pair.first_image_path,
            "final_image_path": pair.final_image_path,
            "task_category": pair.task_category,
            "level": pair.level,
            "object_subtraction_data": pair.object_subtraction_data,
            "difficulty": pair.difficulty,
            "created_at": pair.created_at
        }
        pairs_dict.append(pair_dict)
    
    # Create dataset dictionary
    dataset = {
        "name": "object_subtraction_tasks",
        "description": f"Object subtraction reasoning tasks for video model evaluation ({len(pairs)} pairs, levels: {', '.join(levels)})",
        "pairs": pairs_dict,
        "metadata": {
            "total_tasks": len(pairs),
            "levels": levels,
            "canvas_size": generator.canvas_size,
            "num_objects_range": generator.num_objects_range,
            "generation_date": datetime.now().isoformat()
        },
        "created_at": datetime.now().isoformat()
    }
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(pairs)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    
    return dataset

