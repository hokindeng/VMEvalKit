"""
Prompts for Mirror Clock Reasoning Tasks

This file centralizes all prompts used for mirror clock tasks.
These prompts are for the future time prediction variant.
The actual prompts are generated dynamically with time delta in mirror_clock_reasoning.py
"""

# Note: Prompts are generated dynamically in the task generator
# to include the specific time delta (e.g., "2 hours", "1 hour and 30 minutes")
# This file is kept for consistency with the framework structure

PROMPTS = [
    # These are template examples - actual prompts include time delta
    "This is a mirrored clock. If the original clock moves forward, what time will it show?",
    "The image shows a horizontally flipped clock. After time passes on the original clock, what will be the new time?",
    "This mirror-reflected clock needs to advance. Show what the original clock will display after this time passes.",
    "From this mirrored clock, determine the original time, then add time. What is the result?",
]


# Difficulty-specific prompts (optional)
# Note: Difficulty is based on visual precision needed, not calculation complexity
DIFFICULTY_SPECIFIC_PROMPTS = {
    'easy': [
        "This horizontally flipped clock shows an hour position. Reverse the flip to show the original.",
        "The clock is mirrored at an exact hour. Show what it looks like unflipped.",
    ],
    'medium': [
        "This mirrored clock has hands at common positions. Reverse the horizontal flip to reveal the original.",
        "Flip this mirror-reflected clock back to show its original appearance.",
    ],
    'hard': [
        "This horizontally flipped clock shows precise hand positions. Reverse the flip to reveal the original clock.",
        "This mirror-reflected clock requires careful attention to hand angles. Show the original by reversing the flip.",
    ]
}


# Default prompt index to use
DEFAULT_PROMPT_INDEX = 0
