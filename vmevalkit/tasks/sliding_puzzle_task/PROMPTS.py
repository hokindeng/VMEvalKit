"""Centralized prompts for Sliding Puzzle Task."""

# Single prompt with dynamic move count and constraints
PROMPTS = [
    "Complete this sliding puzzle by moving the numbered tiles to their correct positions. "
    "Only {num_moves} move{plural} {is_are} needed. "
    "Slide the tile(s) horizontally or vertically into the empty space. "
    "Keep the camera view fixed and maintain the grid structure unchanged.",
]

DEFAULT_PROMPT_INDEX = 0

