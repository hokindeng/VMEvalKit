#!/usr/bin/env python3
"""
Direct test of Luma API with our maze images.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from vmevalkit.api_clients.luma_client import LumaDreamMachine

# Test with different settings
client = LumaDreamMachine(
    enhance_prompt=False,  # Disable prompt enhancement
    model="ray-2"
)

# Try with a more photorealistic image
result = client.generate(
    image="test_photorealistic.png",
    text_prompt="Move the red ball along the brown path to reach the red flag at the end.",
    duration=5.0,
    resolution=(1280, 720)
)

print(f"\nSuccess! Video saved to: {result}")
