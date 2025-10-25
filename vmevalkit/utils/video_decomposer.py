#!/usr/bin/env python3
"""
Video Decomposition Utility

A utility function to decompose videos into a series of temporal frames
for creating paper figures. Extracts key frames at regular intervals
and optionally arranges them in various layouts.

Usage:
    from vmevalkit.utils.video_decomposer import decompose_video
    
    # Extract 4 frames and create a horizontal timeline figure
    frames, figure_path = decompose_video(
        video_path="path/to/video.mp4",
        n_frames=4,
        output_dir="figures/",
        create_figure=True,
        layout="horizontal"
    )
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal
import logging

logger = logging.getLogger(__name__)

def decompose_video(
    video_path: Union[str, Path],
    n_frames: int = 4,
    output_dir: Optional[Union[str, Path]] = None,
    create_figure: bool = True,
    layout: Literal["horizontal", "vertical", "grid"] = "horizontal",
    figure_size: Tuple[int, int] = (16, 4),
    save_individual_frames: bool = False,
    frame_format: str = "png",
    dpi: int = 300,
    add_timestamps: bool = True,
    add_frame_numbers: bool = True,
    title: Optional[str] = None,
) -> Tuple[List[np.ndarray], Optional[str]]:
    """
    Decompose a video into a series of temporal frames for paper figures.
    
    Args:
        video_path: Path to the input video file
        n_frames: Number of frames to extract (default: 4)
        output_dir: Directory to save outputs (default: same as video)
        create_figure: Whether to create a combined figure (default: True)
        layout: How to arrange frames - "horizontal", "vertical", or "grid"
        figure_size: Size of the combined figure in inches (width, height)
        save_individual_frames: Whether to save individual frame images
        frame_format: Format for saved frames ("png", "jpg", "eps")
        dpi: DPI for saved figures (default: 300 for publication quality)
        add_timestamps: Whether to add timestamp labels to frames
        add_frame_numbers: Whether to add frame number labels
        title: Optional title for the combined figure
    
    Returns:
        Tuple of (list of frame arrays, path to combined figure if created)
    """
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Calculate frame indices to extract (evenly spaced)
        if n_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        # Extract frames
        frames = []
        timestamps = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame at index {frame_idx}")
                continue
            
            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            timestamps.append(timestamp)
            
            # Save individual frame if requested
            if save_individual_frames:
                frame_filename = f"{video_path.stem}_frame_{i:03d}.{frame_format}"
                frame_path = output_dir / frame_filename
                
                if frame_format.lower() == 'eps':
                    # For EPS, we need to use matplotlib
                    plt.figure(figsize=(8, 6))
                    plt.imshow(frame_rgb)
                    plt.axis('off')
                    plt.savefig(frame_path, format='eps', dpi=dpi, bbox_inches='tight',
                               pad_inches=0, facecolor='white')
                    plt.close()
                else:
                    # For PNG/JPG, use OpenCV (faster)
                    cv2.imwrite(str(frame_path), frame)
                
                logger.info(f"Saved frame {i+1}/{len(frame_indices)}: {frame_path}")
        
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        logger.info(f"Successfully extracted {len(frames)} frames")
        
        # Create combined figure if requested
        figure_path = None
        if create_figure:
            figure_path = _create_combined_figure(
                frames=frames,
                timestamps=timestamps,
                video_name=video_path.stem,
                output_dir=output_dir,
                layout=layout,
                figure_size=figure_size,
                dpi=dpi,
                add_timestamps=add_timestamps,
                add_frame_numbers=add_frame_numbers,
                title=title,
                duration=duration
            )
        
        return frames, figure_path
        
    finally:
        cap.release()


def _create_combined_figure(
    frames: List[np.ndarray],
    timestamps: List[float],
    video_name: str,
    output_dir: Path,
    layout: str,
    figure_size: Tuple[int, int],
    dpi: int,
    add_timestamps: bool,
    add_frame_numbers: bool,
    title: Optional[str],
    duration: float
) -> str:
    """Create a combined figure showing all frames in the specified layout."""
    
    n_frames = len(frames)
    
    # Set up matplotlib with professional styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # Determine subplot arrangement
    if layout == "horizontal":
        nrows, ncols = 1, n_frames
        figsize = figure_size
    elif layout == "vertical":
        nrows, ncols = n_frames, 1
        figsize = (figure_size[1], figure_size[0])  # Swap width/height
    elif layout == "grid":
        # Arrange in a roughly square grid
        ncols = int(np.ceil(np.sqrt(n_frames)))
        nrows = int(np.ceil(n_frames / ncols))
        figsize = (figure_size[0], figure_size[0] * nrows / ncols)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    
    # Handle single subplot case
    if n_frames == 1:
        axes = [axes]
    elif layout in ["horizontal", "vertical"]:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:  # grid layout
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    elif layout == "horizontal":
        fig.suptitle(f'Temporal Decomposition: {video_name}', 
                    fontsize=14, fontweight='bold', y=0.95)
    
    # Plot each frame
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.imshow(frame)
        ax.axis('off')
        
        # Add frame number and/or timestamp
        label_text = []
        if add_frame_numbers:
            label_text.append(f"Frame {i+1}")
        if add_timestamps:
            label_text.append(f"t={timestamp:.2f}s")
        
        if label_text:
            ax.text(0.02, 0.98, " | ".join(label_text), 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add subtle border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('gray')
    
    # Hide unused subplots in grid layout
    if layout == "grid":
        for i in range(n_frames, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
    
    # Add temporal arrow for horizontal layout
    if layout == "horizontal" and n_frames > 1:
        # Add arrow below the frames indicating time progression
        fig.text(0.1, 0.02, 'Time Progression', fontsize=11, fontweight='bold')
        fig.text(0.25, 0.02, '→', fontsize=16, fontweight='bold')
        fig.text(0.85, 0.02, f'Duration: {duration:.2f}s', fontsize=10, 
                horizontalalignment='right')
    
    plt.tight_layout()
    if title or (layout == "horizontal"):
        plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save figure in multiple formats for publication
    base_filename = f"{video_name}_decomposed_{n_frames}frames_{layout}"
    
    # Save PNG (high quality)
    png_path = output_dir / f"{base_filename}.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    # Save EPS (vector format for publications)
    eps_path = output_dir / f"{base_filename}.eps"
    plt.savefig(eps_path, format='eps', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    plt.close()
    
    logger.info(f"Created combined figure: {png_path}")
    logger.info(f"Created vector figure: {eps_path}")
    
    return str(png_path)


def create_video_comparison_figure(
    video_paths: List[Union[str, Path]],
    n_frames: int = 4,
    output_dir: Optional[Union[str, Path]] = None,
    model_names: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (16, 10),
    dpi: int = 300,
    title: Optional[str] = None
) -> str:
    """
    Create a comparison figure showing temporal decomposition of multiple videos.
    
    Useful for comparing different AI models' video generation results.
    
    Args:
        video_paths: List of paths to video files
        n_frames: Number of frames to extract from each video
        output_dir: Directory to save the output figure
        model_names: Optional list of model names for labeling (default: use filenames)
        figure_size: Size of the figure in inches
        dpi: DPI for the saved figure
        title: Optional title for the figure
        
    Returns:
        Path to the created comparison figure
    """
    
    if not video_paths:
        raise ValueError("At least one video path must be provided")
    
    video_paths = [Path(p) for p in video_paths]
    n_videos = len(video_paths)
    
    # Set output directory
    if output_dir is None:
        output_dir = video_paths[0].parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use filenames as model names if not provided
    if model_names is None:
        model_names = [p.stem for p in video_paths]
    elif len(model_names) != n_videos:
        raise ValueError("Number of model names must match number of videos")
    
    # Extract frames from all videos
    all_frames = []
    all_timestamps = []
    
    for video_path in video_paths:
        frames, _ = decompose_video(
            video_path=video_path,
            n_frames=n_frames,
            create_figure=False,
            save_individual_frames=False
        )
        
        # Calculate timestamps for this video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        timestamps = [idx / fps if fps > 0 else 0 for idx in frame_indices]
        
        all_frames.append(frames)
        all_timestamps.append(timestamps)
    
    # Set up matplotlib
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 9
    
    # Create subplot grid: rows = videos, cols = frames
    fig, axes = plt.subplots(n_videos, n_frames, figsize=figure_size, facecolor='white')
    
    # Handle single video or single frame cases
    if n_videos == 1:
        axes = [axes] if n_frames > 1 else [[axes]]
    elif n_frames == 1:
        axes = [[ax] for ax in axes]
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    else:
        fig.suptitle('Video Generation Model Comparison - Temporal Decomposition',
                    fontsize=14, fontweight='bold', y=0.95)
    
    # Plot frames for each video
    for video_idx, (frames, timestamps, model_name) in enumerate(zip(all_frames, all_timestamps, model_names)):
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            ax = axes[video_idx][frame_idx]
            ax.imshow(frame)
            ax.axis('off')
            
            # Add model name to first frame of each row
            if frame_idx == 0:
                ax.text(-0.02, 0.5, model_name, transform=ax.transAxes,
                       fontsize=11, fontweight='bold', rotation=90,
                       verticalalignment='center', horizontalalignment='right')
            
            # Add timestamp to frames in first row
            if video_idx == 0:
                ax.text(0.5, -0.02, f't={timestamp:.2f}s', transform=ax.transAxes,
                       fontsize=9, horizontalalignment='center', verticalalignment='top')
            
            # Add subtle border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('gray')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1, bottom=0.1)
    
    # Save figure
    timestamp_str = "_".join([f"{ts:.1f}s" for ts in all_timestamps[0]])
    filename = f"video_comparison_{n_videos}models_{n_frames}frames.png"
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    # Also save EPS version
    eps_path = output_dir / filename.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    plt.close()
    
    logger.info(f"Created video comparison figure: {output_path}")
    
    return str(output_path)


# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Decompose video into temporal frames for paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 4 frames horizontally
  python -m vmevalkit.utils.video_decomposer video.mp4 --frames 4 --layout horizontal
  
  # Create comparison figure for multiple models
  python -m vmevalkit.utils.video_decomposer model1.mp4 model2.mp4 model3.mp4 --comparison
  
  # Extract frames vertically with timestamps
  python -m vmevalkit.utils.video_decomposer video.mp4 --frames 6 --layout vertical --timestamps
        """
    )
    
    parser.add_argument("video_paths", nargs="+", help="Path(s) to video file(s)")
    parser.add_argument("--frames", "-f", type=int, default=4,
                       help="Number of frames to extract (default: 4)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output directory (default: same as video)")
    parser.add_argument("--layout", "-l", choices=["horizontal", "vertical", "grid"],
                       default="horizontal", help="Frame layout (default: horizontal)")
    parser.add_argument("--comparison", "-c", action="store_true",
                       help="Create comparison figure for multiple videos")
    parser.add_argument("--individual", "-i", action="store_true",
                       help="Also save individual frame images")
    parser.add_argument("--no-timestamps", action="store_true",
                       help="Don't add timestamp labels")
    parser.add_argument("--no-frame-numbers", action="store_true",
                       help="Don't add frame number labels")
    parser.add_argument("--title", "-t", type=str,
                       help="Title for the figure")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures (default: 300)")
    
    args = parser.parse_args()
    
    if args.comparison and len(args.video_paths) > 1:
        # Create comparison figure
        create_video_comparison_figure(
            video_paths=args.video_paths,
            n_frames=args.frames,
            output_dir=args.output,
            title=args.title,
            dpi=args.dpi
        )
    else:
        # Process individual videos
        for video_path in args.video_paths:
            frames, figure_path = decompose_video(
                video_path=video_path,
                n_frames=args.frames,
                output_dir=args.output,
                layout=args.layout,
                save_individual_frames=args.individual,
                add_timestamps=not args.no_timestamps,
                add_frame_numbers=not args.no_frame_numbers,
                title=args.title,
                dpi=args.dpi
            )
            
            print(f"✅ Processed {video_path}")
            if figure_path:
                print(f"   📊 Created figure: {figure_path}")
