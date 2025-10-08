#!/usr/bin/env python3
"""
Simple example showing pure inference (no evaluation).

This demonstrates the clean separation:
- Inference: text + image → video
- No scoring, no evaluation, just generation
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.inference import InferenceRunner


def main():
    # Initialize inference runner
    runner = InferenceRunner(output_dir="./outputs")
    
    # Example 1: Run on a task from our dataset
    print("Example 1: Running inference on a maze task")
    result = runner.run_from_task(
        model_name="luma-dream-machine",
        task_data={
            "id": "example_001",
            "prompt": "Move the green dot from start to the red flag.",
            "first_image_path": "data/generated_mazes/irregular_0000_first.png"
        }
    )
    
    if result["status"] == "success":
        print(f"✅ Video generated: {result['video_path']}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # Example 2: Direct inference with custom inputs
    print("\nExample 2: Direct inference")
    result = runner.run(
        model_name="luma-dream-machine",
        image_path="data/generated_mazes/knowwhat_0000_first.png",
        text_prompt="Navigate through the maze to reach the target circle.",
        duration=5.0,
        resolution=(1280, 720)
    )
    
    if result["status"] == "success":
        print(f"✅ Video generated: {result['video_path']}")
    else:
        print(f"❌ Failed: {result['error']}")


if __name__ == "__main__":
    main()
