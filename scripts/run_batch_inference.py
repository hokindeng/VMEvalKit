#!/usr/bin/env python3
"""
Batch inference script - NO EVALUATION.

Runs inference on multiple tasks from a dataset.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vmevalkit.inference import BatchInferenceRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run batch video generation inference"
    )
    
    # Model(s) selection
    parser.add_argument(
        "models",
        type=str,
        nargs='+',
        help="Model name(s) to run (e.g., luma-dream-machine google-veo-001)"
    )
    
    # Dataset
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset JSON file"
    )
    
    # Task selection
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs='*',
        help="Specific task IDs to run (default: all)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to process"
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
    
    # Parallelization
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
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
    
    # Initialize batch runner
    runner = BatchInferenceRunner(
        output_dir=str(args.output_dir),
        max_workers=args.workers
    )
    
    # Run inference
    if len(args.models) == 1:
        # Single model batch run
        result = runner.run_dataset(
            model_name=args.models[0],
            dataset_path=args.dataset,
            task_ids=args.task_ids,
            max_tasks=args.max_tasks,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    else:
        # Multiple models comparison
        result = runner.run_models_comparison(
            model_names=args.models,
            dataset_path=args.dataset,
            task_ids=args.task_ids,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    
    print("\n✨ Batch inference complete!")


if __name__ == "__main__":
    main()
