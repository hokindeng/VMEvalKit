#!/usr/bin/env python3
"""
Simple inference script - NO EVALUATION.

Just runs: text + image → video model → output video
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from vmevalkit.inference import InferenceRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run video generation inference (text + image → video)"
    )
    
    # Model selection
    parser.add_argument(
        "model",
        type=str,
        help="Model name (e.g., luma-dream-machine, google-veo-001)"
    )
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--task-file",
        type=Path,
        help="Path to task JSON file with prompt and first_image_path"
    )
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to input image (use with --prompt)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt (required when using --image)"
    )
    
    # Task selection (when using --task-file)
    parser.add_argument(
        "--task-id",
        type=str,
        help="Specific task ID to run from the dataset"
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Task index to run (0-based, default: 0)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Video duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Resolution as WIDTHxHEIGHT (default: 1280x720)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory (default: ./outputs)"
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        parser.error(f"Invalid resolution format: {args.resolution}")
    
    # Initialize runner
    runner = InferenceRunner(output_dir=str(args.output_dir))
    
    # Run inference
    if args.task_file:
        # Load task from file
        with open(args.task_file, 'r') as f:
            data = json.load(f)
        
        # Handle single task or dataset format
        if "pairs" in data:
            # Dataset format
            pairs = data["pairs"]
            if args.task_id:
                task = next((p for p in pairs if p.get("id") == args.task_id), None)
                if not task:
                    parser.error(f"Task ID not found: {args.task_id}")
            else:
                if args.task_index >= len(pairs):
                    parser.error(f"Task index {args.task_index} out of range")
                task = pairs[args.task_index]
        else:
            # Single task format
            task = data
        
        result = runner.run_from_task(
            model_name=args.model,
            task_data=task,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    else:
        # Direct image + prompt
        if not args.prompt:
            parser.error("--prompt is required when using --image")
        
        result = runner.run(
            model_name=args.model,
            image_path=args.image,
            text_prompt=args.prompt,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    
    # Print result
    if result.get("status") == "success":
        print(f"\n🎬 Video generated: {result['video_path']}")
    else:
        print(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
