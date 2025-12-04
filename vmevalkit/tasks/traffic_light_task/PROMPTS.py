"""
Prompts for Traffic Light Reasoning Tasks

This file centralizes all prompts used for traffic light reasoning tasks.
Each Type has different question templates that test various aspects of temporal reasoning
and coordination understanding.
"""

# Scene description prefix (common to all prompts)
SCENE_PREFIX = "This scene shows a crossroad with two traffic lights. "

# Rule explanation (common to all prompts)
RULE_EXPLANATION = "The two traffic lights are opposite to each other: when one is red, the other is green, and vice versa. "

# Type 1: Basic Countdown Decrement (Simple)
# Tests: Number change understanding, time concept, rule application
TYPE_1_PROMPTS = [
    SCENE_PREFIX + RULE_EXPLANATION + 
    "Currently, Traffic Light A shows red with countdown {countdown_a}. Traffic Light B shows green. " +
    "Generate a video showing the countdown number decrementing from {countdown_a} to 0, then show the final state of both traffic lights."
]

# Type 2: Number Change + Time Understanding (Medium)
# Tests: Larger countdown numbers, complete decrement process, time concept
TYPE_2_PROMPTS = [
    SCENE_PREFIX + RULE_EXPLANATION + 
    "Currently, Traffic Light A shows red with countdown {countdown_a}. Traffic Light B shows green. " +
    "Generate a video showing the countdown number decrementing from {countdown_a} to 0, then show the final state of both traffic lights."
]

# Type 3: Dual Countdown Coordination (Hard)
# Tests: Two countdowns simultaneously, which reaches zero first, coordination understanding
TYPE_3_PROMPTS = [
    SCENE_PREFIX + RULE_EXPLANATION + 
    "Currently, Traffic Light A shows red with countdown {countdown_a}. Traffic Light B shows green with countdown {countdown_b}. " +
    "Generate a video showing both countdown numbers decrementing simultaneously. When any countdown reaches 0, apply the relative rule to switch states. Then show the final state of both traffic lights."
]

# Type 4: Complex Time Calculation (Hard)
# Tests: Multiple state switches, complex time sequence calculation
TYPE_4_PROMPTS = [
    SCENE_PREFIX + RULE_EXPLANATION + 
    "Currently, Traffic Light A shows red with countdown {countdown_a}. Traffic Light B shows green with countdown {countdown_b}. " +
    "Generate a video showing countdown numbers decrementing. When countdown reaches 0, apply the relative rule to switch states. Then show the final state of both traffic lights after {time_elapsed} seconds."
]

# All Type prompts organized by type index (1-4)
TYPE_PROMPTS = {
    1: TYPE_1_PROMPTS,
    2: TYPE_2_PROMPTS,
    3: TYPE_3_PROMPTS,
    4: TYPE_4_PROMPTS
}

# Default prompt index to use (0 for each type)
DEFAULT_PROMPT_INDEX = {
    1: 0,
    2: 0,
    3: 0,
    4: 0
}

