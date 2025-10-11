#!/usr/bin/env python3
"""
Test script for Google Veo 3.1 integration via WaveSpeed API.

This script demonstrates how to use the Veo 3.1 model for image-to-video generation
with text prompts for reasoning tasks in VMEvalKit.
"""

import asyncio
import os
from pathlib import Path
from vmevalkit.models import Veo31Service, WaveSpeedService, WaveSpeedModel

async def test_veo31_basic():
    """Test basic Veo 3.1 functionality using the convenience class."""
    print("Testing Veo 3.1 via convenience class...")
    
    # Initialize service
    service = Veo31Service()
    
    # Test parameters
    image_path = Path("data/questions/maze_tasks/maze_level_1_image_0.png")
    prompt = "Solve the maze by drawing a red line from the start (green circle) to the end (red circle)"
    output_path = Path("output/veo31_maze_test.mp4")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not image_path.exists():
        print(f"Warning: Test image not found at {image_path}")
        print("Please ensure you have maze images in the data directory")
        return
    
    try:
        # Generate video with Veo 3.1 specific parameters
        result = await service.generate_video(
            prompt=prompt,
            image_path=image_path,
            output_path=output_path,
            resolution="1080p",  # Veo 3.1 supports up to 1080p
            duration=5.0,  # 5 second video
            aspect_ratio="16:9",
            generate_audio=True,  # Generate synchronized audio
            seed=42  # For reproducibility
        )
        
        print(f"✓ Video generated successfully!")
        print(f"  - Output: {result.get('video_path', result.get('video_url'))}")
        print(f"  - Model: {result['model']}")
        print(f"  - Resolution: {result.get('resolution', 'N/A')}")
        print(f"  - Duration: {result.get('duration', 'N/A')}s")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def test_veo31_with_wavespeed_service():
    """Test Veo 3.1 using the general WaveSpeedService class."""
    print("\nTesting Veo 3.1 via WaveSpeedService...")
    
    # Initialize service with Veo 3.1 model
    service = WaveSpeedService(model=WaveSpeedModel.VEO_3_1_I2V)
    
    # Test parameters
    image_path = Path("data/questions/chess_tasks/chess_puzzle_1.png")
    prompt = "Show the best next move for white by animating the piece movement on the chess board"
    output_path = Path("output/veo31_chess_test.mp4")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not image_path.exists():
        print(f"Warning: Test image not found at {image_path}")
        print("Using a simple test case instead")
        # Create a simple test image if chess image doesn't exist
        image_path = Path("data/questions/rotation_tasks/rotation_test.png")
        prompt = "Rotate the object 90 degrees clockwise"
    
    if not image_path.exists():
        print("No test images available. Please add test images to data directory.")
        return
    
    try:
        # Generate video
        result = await service.generate_video(
            prompt=prompt,
            image_path=image_path,
            output_path=output_path,
            resolution="720p",  # Test with 720p
            duration=3.0,  # 3 second video
            generate_audio=False,  # No audio for this test
            negative_prompt="blurry, low quality",  # Avoid low quality results
            seed=-1  # Random seed
        )
        
        print(f"✓ Video generated successfully!")
        print(f"  - Output: {result.get('video_path', result.get('video_url'))}")
        print(f"  - Model: {result['model']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def test_all_wavespeed_models():
    """List all available WaveSpeed models including Veo 3.1."""
    print("\nAvailable WaveSpeed models:")
    print("-" * 40)
    
    for model in WaveSpeedModel:
        model_type = "Google Veo" if "veo" in model.value.lower() else "WaveSpeed WAN"
        print(f"  • {model.value:40} [{model_type}]")
    
    print("-" * 40)
    print(f"Total models: {len(WaveSpeedModel)}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Google Veo 3.1 Integration Test for VMEvalKit")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("WAVESPEED_API_KEY"):
        print("\n⚠️  Warning: WAVESPEED_API_KEY not set in environment")
        print("   Please set your WaveSpeed API key to run generation tests")
        print("   export WAVESPEED_API_KEY='your-api-key-here'\n")
    
    # List available models
    await test_all_wavespeed_models()
    
    # Run tests if API key is available
    if os.getenv("WAVESPEED_API_KEY"):
        await test_veo31_basic()
        await test_veo31_with_wavespeed_service()
    else:
        print("\nSkipping generation tests (no API key)")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
