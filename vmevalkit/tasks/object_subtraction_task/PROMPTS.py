"""
Prompts for Object Subtraction Tasks

This file centralizes all prompts used for object subtraction tasks.
Organized by cognitive level (L1-L4).
Modify prompts here to experiment with different instruction styles.
"""

# Level 1: Explicit Specificity - Remove objects by explicit visual attributes
PROMPTS_L1 = [
    "Remove all red objects from the scene. Do not do anything to other objects.",
]

# Level 2: Enumerated Selection - Remove multiple explicitly listed objects
PROMPTS_L2 = [
    "Remove the red cube, the green sphere, and the blue pyramid from the scene. Do not do anything to other objects.",
    "Move the yellow cube, the blue sphere, and the red pyramid out of view. Do not do anything to other objects.",
]

# Level 3: Relational Reference - Remove objects using spatial relations
PROMPTS_L3 = [
    "Remove the three objects on the left side of the screen. Do not do anything to other objects.",
    "Move the two objects farthest from the center out of view. Do not do anything to other objects.",
    "Remove all objects in the upper half of the image. Do not do anything to other objects.",
]

# Level 4: Conceptual Abstraction - Remove objects based on semantic properties
PROMPTS_L4 = [
    "Remove all large objects from the scene. Do not do anything to other objects.",
    "Remove the object that looks different from the others. Do not do anything to other objects.",
    "Remove all objects that do not share the same shape. Do not do anything to other objects.",
]

# Default prompt index to use for each level
DEFAULT_PROMPT_INDEX_L1 = 0
DEFAULT_PROMPT_INDEX_L2 = 0
DEFAULT_PROMPT_INDEX_L3 = 0
DEFAULT_PROMPT_INDEX_L4 = 0



