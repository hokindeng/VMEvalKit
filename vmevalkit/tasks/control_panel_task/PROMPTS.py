"""
Prompts for Control Panel Animation Tasks

Design: Test VideoModel's ability to observe and reason about control panel systems.
The model must infer the mapping between lever positions and light colors from the initial image,
then generate animations showing lever movements to achieve target colors.

Key design principles:
- Do not tell the model the specific mapping (left=red, middle=green, right=blue)
- Only provide general rules (each position corresponds to a fixed color)
- Model must observe the initial image to infer the mapping
- Each scene generates 3 independent tasks (one for each target color: red, green, blue)
"""

# Prompt template for control panel tasks
# Placeholders:
# - {num_lights}: Number of lights description (e.g., "two", "three", "four")
# - {target_color}: Target color (e.g., "red", "green", "blue")
# - {num_lights_note}: Optional note for 2-light scenarios
# - {duplicate_color_note}: Optional note for scenarios with duplicate colors (currently unused)
# - {inference_note}: Optional note in task objective
PROMPTS = [
    """Scene structure:

The scene contains {num_lights} indicator lights.

Each indicator light has a unique corresponding control panel directly below it,
and each control panel has a horizontal slider.

The color of each light is controlled solely by the slider position in its corresponding control panel below it.

System rules:

Each indicator light displays only one of three discrete colors at any time:
red, green, or blue. Lights do not display mixed colors, gradients, or other colors.

The slider on each control panel can only stop at three clear discrete positions:
left, middle, and right. The slider cannot stop between two positions.

For any light:
The light's color is completely determined by the position of the slider below it.
No other mechanism or factor can change the light's color.

Each light is independently controlled:
Moving the slider for one light will only change that light's color,
and will not affect other lights.

Mapping rules:

All control panels share a fixed, unique mapping that determines light colors:
'slider position → light color' mapping:
The left, middle, and right positions each correspond to one of three colors,
each position corresponds to one color, and each color corresponds to only one position (one-to-one mapping).

That is:
The left, middle, and right positions of the slider in the control panel determine
which of the three colors (red, green, blue) the light displays,
and which position corresponds to which color must be inferred from the initial image.
{num_lights_note}{duplicate_color_note}
The mapping rules remain consistent throughout the video and do not change over time.

Task objective

Make all indicator lights display {target_color} in the final state.

Observe the initial image to infer the mapping between slider positions and light colors,
then move the sliders to make all lights display {target_color}.{inference_note}

Video generation requirements

Use the image I provide as the initial frame of the video.

The slider movement must be horizontal and can only stop at discrete positions.
During movement, the slider must not tilt, float, jump, or pause in between.
When the slider position changes to the corresponding position, the light color must
immediately and deterministically switch to the color corresponding to that position.

The video must not contain: text, arrows, lines, highlights, selections, UI elements,
or new objects.
Only slider movements and light color changes are allowed.

The camera view must remain completely fixed throughout the entire video:
no rotation, no zooming, no translation."""
]

# Default prompt index
DEFAULT_PROMPT_INDEX = 0
