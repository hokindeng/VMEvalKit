"""
RPM (Raven's Progressive Matrices) puzzle generator.

This module generates 3x3 matrix puzzles with various rules and patterns.
"""

import os
import json
import random
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional
import numpy as np


def create_shape(draw: ImageDraw.Draw, shape_type: str, bbox: Tuple[int, int, int, int], 
                 color: str = "black", fill: Optional[str] = None):
    """Draw a shape within the given bounding box."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radius = min(x2 - x1, y2 - y1) // 3
    
    if shape_type == "circle":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], 
                     outline=color, fill=fill, width=2)
    elif shape_type == "square":
        draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], 
                       outline=color, fill=fill, width=2)
    elif shape_type == "triangle":
        points = [
            (cx, cy - radius),
            (cx - radius, cy + radius),
            (cx + radius, cy + radius)
        ]
        draw.polygon(points, outline=color, fill=fill, width=2)
    elif shape_type == "diamond":
        points = [
            (cx, cy - radius),
            (cx - radius, cy),
            (cx, cy + radius),
            (cx + radius, cy)
        ]
        draw.polygon(points, outline=color, fill=fill, width=2)
    elif shape_type == "cross":
        # Vertical line
        draw.line([cx, cy - radius, cx, cy + radius], fill=color, width=3)
        # Horizontal line
        draw.line([cx - radius, cy, cx + radius, cy], fill=color, width=3)
    elif shape_type == "star":
        # Simple 4-pointed star
        points = []
        for i in range(8):
            angle = i * np.pi / 4
            if i % 2 == 0:
                r = radius
            else:
                r = radius // 2
            x = cx + r * np.cos(angle - np.pi / 2)
            y = cy + r * np.sin(angle - np.pi / 2)
            points.append((x, y))
        draw.polygon(points, outline=color, fill=fill, width=2)


class RPMPuzzleGenerator:
    """Generator for RPM-style matrix puzzles."""
    
    def __init__(self, tile_size: int = 192, seed: Optional[int] = None):
        self.tile_size = tile_size
        self.rng = random.Random(seed)
        self.shapes = ["circle", "square", "triangle", "diamond", "cross", "star"]
        self.colors = ["black", "gray", "darkgray"]
        self.fills = [None, "lightgray", "white"]
    
    def generate_pattern_matrix(self) -> Tuple[List[List[Dict]], str]:
        """
        Generate a 3x3 matrix with a specific pattern rule.
        Returns the matrix data and the rule description.
        """
        rule_type = self.rng.choice([
            "shape_progression",
            "number_progression", 
            "rotation",
            "color_pattern",
            "combination"
        ])
        
        if rule_type == "shape_progression":
            return self._generate_shape_progression()
        elif rule_type == "number_progression":
            return self._generate_number_progression()
        elif rule_type == "rotation":
            return self._generate_rotation_pattern()
        elif rule_type == "color_pattern":
            return self._generate_color_pattern()
        else:  # combination
            return self._generate_combination_pattern()
    
    def _generate_shape_progression(self) -> Tuple[List[List[Dict]], str]:
        """Generate a matrix where shapes progress by row or column."""
        shapes = self.rng.sample(self.shapes, 3)
        direction = self.rng.choice(["row", "column"])
        
        matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                if direction == "row":
                    shape = shapes[j]
                else:  # column
                    shape = shapes[i]
                
                row.append({
                    "shapes": [shape],
                    "positions": ["center"],
                    "colors": ["black"],
                    "fills": [None]
                })
            matrix.append(row)
        
        return matrix, f"Shape progression by {direction}"
    
    def _generate_number_progression(self) -> Tuple[List[List[Dict]], str]:
        """Generate a matrix where the number of shapes changes."""
        base_shape = self.rng.choice(self.shapes)
        
        matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                # Number increases across row and down column
                num_shapes = ((i + j) % 3) + 1
                
                positions = []
                if num_shapes == 1:
                    positions = ["center"]
                elif num_shapes == 2:
                    positions = ["left", "right"]
                else:  # 3
                    positions = ["left", "center", "right"]
                
                row.append({
                    "shapes": [base_shape] * num_shapes,
                    "positions": positions,
                    "colors": ["black"] * num_shapes,
                    "fills": [None] * num_shapes
                })
            matrix.append(row)
        
        return matrix, "Number progression pattern"
    
    def _generate_rotation_pattern(self) -> Tuple[List[List[Dict]], str]:
        """Generate a matrix with rotating elements."""
        shapes_set = self.rng.sample(self.shapes, 3)
        
        matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                # Rotate through shapes
                idx = (i + j) % 3
                shape = shapes_set[idx]
                
                row.append({
                    "shapes": [shape],
                    "positions": ["center"],
                    "colors": ["black"],
                    "fills": [None]
                })
            matrix.append(row)
        
        return matrix, "Rotation pattern"
    
    def _generate_color_pattern(self) -> Tuple[List[List[Dict]], str]:
        """Generate a matrix with color/fill patterns."""
        shape = self.rng.choice(self.shapes)
        fills = self.rng.sample(self.fills, 3)
        
        matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                fill_idx = (i + j) % 3
                
                row.append({
                    "shapes": [shape],
                    "positions": ["center"],
                    "colors": ["black"],
                    "fills": [fills[fill_idx]]
                })
            matrix.append(row)
        
        return matrix, "Fill/color pattern"
    
    def _generate_combination_pattern(self) -> Tuple[List[List[Dict]], str]:
        """Generate a matrix with combined rules."""
        shapes = self.rng.sample(self.shapes, 2)
        
        matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                # Combine different shapes based on position
                if (i + j) % 2 == 0:
                    cell_shapes = [shapes[0]]
                else:
                    cell_shapes = [shapes[1]]
                
                # Add more shapes in corners
                if i != 1 and j != 1:
                    cell_shapes.append(shapes[(i + j) % 2])
                
                positions = ["center"] if len(cell_shapes) == 1 else ["left", "right"]
                
                row.append({
                    "shapes": cell_shapes,
                    "positions": positions,
                    "colors": ["black"] * len(cell_shapes),
                    "fills": [None] * len(cell_shapes)
                })
            matrix.append(row)
        
        return matrix, "Combination pattern"
    
    def render_cell(self, cell_data: Dict) -> Image.Image:
        """Render a single cell of the matrix."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), "white")
        draw = ImageDraw.Draw(img)
        
        shapes = cell_data["shapes"]
        positions = cell_data["positions"]
        colors = cell_data["colors"]
        fills = cell_data["fills"]
        
        # Calculate positions for shapes
        for i, (shape, pos, color, fill) in enumerate(zip(shapes, positions, colors, fills)):
            if pos == "center":
                bbox = (self.tile_size // 4, self.tile_size // 4,
                       3 * self.tile_size // 4, 3 * self.tile_size // 4)
            elif pos == "left":
                bbox = (self.tile_size // 8, self.tile_size // 4,
                       3 * self.tile_size // 8, 3 * self.tile_size // 4)
            elif pos == "right":
                bbox = (5 * self.tile_size // 8, self.tile_size // 4,
                       7 * self.tile_size // 8, 3 * self.tile_size // 4)
            else:  # top, bottom, etc.
                bbox = (self.tile_size // 4, self.tile_size // 4,
                       3 * self.tile_size // 4, 3 * self.tile_size // 4)
            
            create_shape(draw, shape, bbox, color, fill)
        
        # Add border
        draw.rectangle([0, 0, self.tile_size - 1, self.tile_size - 1], 
                      outline="black", width=1)
        
        return img
    
    def render_matrix(self, matrix: List[List[Dict]], hide_last: bool = True) -> Image.Image:
        """Render the complete 3x3 matrix."""
        matrix_size = self.tile_size * 3
        img = Image.new("RGB", (matrix_size, matrix_size), "white")
        
        for i in range(3):
            for j in range(3):
                if hide_last and i == 2 and j == 2:
                    # Leave the last cell empty (with question mark)
                    cell_img = Image.new("RGB", (self.tile_size, self.tile_size), "white")
                    draw = ImageDraw.Draw(cell_img)
                    draw.rectangle([0, 0, self.tile_size - 1, self.tile_size - 1], 
                                 outline="black", width=1)
                    draw.text((self.tile_size // 2 - 10, self.tile_size // 2 - 10), 
                            "?", fill="gray")
                else:
                    cell_img = self.render_cell(matrix[i][j])
                
                img.paste(cell_img, (j * self.tile_size, i * self.tile_size))
        
        return img

