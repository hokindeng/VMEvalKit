"""
Prompts for Object Permanence Tasks

Design: Tell model what objects are in the scene, then ask to move occluder 
from current position to the right until it completely exits the frame.
Final frame should have no occluder, objects remain unchanged.
"""

# Easy: Single object
PROMPTS_EASY = [
    "A {color} {shape} is in the scene. An opaque gray panel is on the left side. "
    "Move the panel horizontally from left to right. As the panel moves, it will occlude the {shape} when it passes over it, until the panel completely exits the frame. "
    "The panel maintains its size and shape as it moves, only blocks the view, and does not physically interact with the {shape}. Keep the camera view fixed.",
    
    "There is a {color} {shape} in the scene. An opaque gray panel is positioned on the left. "
    "Move the panel horizontally from left to right. When the panel passes over the {shape}, it will temporarily hide it from view, until the panel is completely out of the frame. "
    "The panel is a visual occluder that maintains its size and shape, and does not touch or move the {shape}. Keep the camera view fixed.",
]

# Medium: Two objects
PROMPTS_MEDIUM = [
    "{object1_description} and {object2_description} are in the scene. An opaque gray panel is on the left side. "
    "Move the panel horizontally from left to right. As the panel moves, it will occlude both objects when it passes over them, until the panel completely exits the frame. "
    "The panel maintains its size and shape as it moves, only blocks the view, and does not physically interact with the objects. Keep the camera view fixed.",
    
    "There are two objects in the scene: {object1_description} and {object2_description}. "
    "An opaque gray panel is positioned on the left. Move the panel horizontally from left to right. "
    "When the panel passes over the objects, it will temporarily hide them from view, until the panel is completely out of the frame. "
    "The panel is a visual occluder that maintains its size and shape, and does not touch or move the objects. Keep the camera view fixed.",
]

# Hard: Multiple objects (3 or more)
PROMPTS_HARD = [
    "{object_list} are in the scene. An opaque gray panel is on the left side. "
    "Move the panel horizontally from left to right. As the panel moves, it will occlude all objects when it passes over them, until the panel completely exits the frame. "
    "The panel maintains its size and shape as it moves, only blocks the view, and does not physically interact with any objects. Keep the camera view fixed.",
    
    "There are multiple objects in the scene: {object_list}. "
    "An opaque gray panel is positioned on the left. Move the panel horizontally from left to right. "
    "When the panel passes over the objects, it will temporarily hide them from view, until the panel is completely out of the frame. "
    "The panel is a visual occluder that maintains its size and shape, and does not touch or move any objects. Keep the camera view fixed.",
]

# Default prompt indices
DEFAULT_PROMPT_INDEX_EASY = 0
DEFAULT_PROMPT_INDEX_MEDIUM = 0
DEFAULT_PROMPT_INDEX_HARD = 0

