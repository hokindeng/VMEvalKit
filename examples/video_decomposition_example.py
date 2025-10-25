#!/usr/bin/env python3
"""
Example script demonstrating video decomposition utility for paper figures.

This script shows how to use the video decomposition utility to create
temporal frame sequences for research papers and presentations.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vmevalkit.utils import decompose_video, create_video_comparison_figure

def example_single_video_decomposition():
    """Example: Decompose a single video into temporal frames."""
    
    print("🎬 Example: Single Video Decomposition")
    print("=" * 50)
    
    # Example video path (you'll need to replace with actual video)
    video_path = "data/outputs/pilot_experiment/openai-sora-2/chess_task/chess_0001/generated_video.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        print("   Please update the video_path variable with a valid video file.")
        return
    
    try:
        # Extract 4 frames horizontally with timestamps
        frames, figure_path = decompose_video(
            video_path=video_path,
            n_frames=4,
            layout="horizontal",
            output_dir="examples/output/",
            create_figure=True,
            save_individual_frames=True,
            add_timestamps=True,
            add_frame_numbers=True,
            title="Chess Task - Temporal Progression",
            dpi=300
        )
        
        print(f"✅ Successfully extracted {len(frames)} frames")
        if figure_path:
            print(f"📊 Created figure: {figure_path}")
        
        # Example: Extract frames in vertical layout
        frames_vertical, figure_vertical = decompose_video(
            video_path=video_path,
            n_frames=6,
            layout="vertical",
            output_dir="examples/output/",
            title="Chess Task - Vertical Timeline"
        )
        
        print(f"📊 Created vertical layout: {figure_vertical}")
        
        # Example: Grid layout for more frames
        frames_grid, figure_grid = decompose_video(
            video_path=video_path,
            n_frames=9,
            layout="grid",
            output_dir="examples/output/",
            title="Chess Task - Grid Layout"
        )
        
        print(f"📊 Created grid layout: {figure_grid}")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")


def example_multi_video_comparison():
    """Example: Compare multiple videos from different models."""
    
    print("\n🎬 Example: Multi-Video Comparison")
    print("=" * 50)
    
    # Example video paths for different models
    video_paths = [
        "data/outputs/pilot_experiment/openai-sora-2/chess_task/chess_0001/generated_video.mp4",
        "data/outputs/pilot_experiment/veo-3.0-generate/chess_task/chess_0001/generated_video.mp4",
        "data/outputs/pilot_experiment/luma-ray-2/chess_task/chess_0001/generated_video.mp4",
    ]
    
    model_names = ["OpenAI Sora", "Google Veo 3.0", "Luma Ray 2"]
    
    # Check if videos exist
    existing_videos = []
    existing_names = []
    for video_path, model_name in zip(video_paths, model_names):
        if Path(video_path).exists():
            existing_videos.append(video_path)
            existing_names.append(model_name)
    
    if not existing_videos:
        print("❌ No video files found for comparison.")
        print("   Please update the video_paths with valid video files.")
        return
    
    try:
        figure_path = create_video_comparison_figure(
            video_paths=existing_videos,
            model_names=existing_names,
            n_frames=4,
            output_dir="examples/output/",
            title="Model Comparison - Chess Task Generation",
            dpi=300
        )
        
        print(f"✅ Created comparison figure: {figure_path}")
        print(f"   Compared {len(existing_videos)} models")
        
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")


def example_custom_paper_figure():
    """Example: Create a custom figure for a research paper."""
    
    print("\n🎬 Example: Custom Paper Figure")
    print("=" * 50)
    
    # Example: Create figures for different tasks
    tasks = ["chess", "maze", "raven", "rotation", "sudoku"]
    
    for task in tasks:
        video_path = f"data/outputs/pilot_experiment/openai-sora-2/{task}_task/{task}_0001/generated_video.mp4"
        
        if Path(video_path).exists():
            try:
                frames, figure_path = decompose_video(
                    video_path=video_path,
                    n_frames=4,
                    layout="horizontal",
                    output_dir=f"examples/output/{task}/",
                    title=f"{task.capitalize()} Task - Temporal Analysis",
                    figure_size=(20, 5),  # Wide format for papers
                    dpi=300,
                    add_timestamps=True,
                    add_frame_numbers=False  # Clean look for papers
                )
                
                print(f"✅ {task.capitalize()}: {figure_path}")
                
            except Exception as e:
                print(f"❌ Error processing {task}: {e}")
        else:
            print(f"⚠️  Video not found for {task} task")


def main():
    """Run all examples."""
    
    print("🎬 VMEvalKit Video Decomposition Examples")
    print("=" * 60)
    print("Creating temporal frame sequences for paper figures...")
    
    # Create output directory
    Path("examples/output").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    example_single_video_decomposition()
    example_multi_video_comparison()
    example_custom_paper_figure()
    
    print("\n✅ All examples completed!")
    print("📁 Check the 'examples/output/' directory for generated figures.")
    print("\n💡 Tips for paper figures:")
    print("   • Use layout='horizontal' for temporal progression")
    print("   • Set dpi=300 or higher for publication quality")
    print("   • Use EPS format for vector graphics in LaTeX")
    print("   • Add meaningful titles and clean timestamps")
    print("   • Consider figure_size for journal column widths")


if __name__ == "__main__":
    main()
