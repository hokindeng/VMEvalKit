#!/usr/bin/env python3
"""
Video Decomposition Utility

Simple utility to decompose videos into temporal frame sequences for paper figures.
Extracts frames at regular intervals and optionally creates a combined figure.

Usage:
    from vmevalkit.utils import decompose_video
    
    frames, figure_path = decompose_video(
        video_path="path/to/video.mp4",
        n_frames=4,
        layout="horizontal"
    )
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    dpi: int = 300,
    add_timestamps: bool = True,
    add_frame_numbers: bool = False,
    title: Optional[str] = None,
) -> Tuple[List[np.ndarray], Optional[str]]:
    """
    Decompose a video into temporal frames for paper figures.
    
    Args:
        video_path: Path to the input video file
        n_frames: Number of frames to extract (default: 4)
        output_dir: Directory to save outputs (default: same as video)
        create_figure: Whether to create a combined figure (default: True)
        layout: Frame arrangement - "horizontal", "vertical", or "grid"
        figure_size: Figure size in inches (width, height)
        dpi: DPI for saved figures (default: 300 for publication quality)
        add_timestamps: Whether to add timestamp labels
        add_frame_numbers: Whether to add frame number labels (e.g., "Frame 7")
        title: Optional title for the figure
    
    Returns:
        Tuple of (list of frame arrays, path to combined figure if created)
    
    Example:
        >>> frames, fig_path = decompose_video("video.mp4", n_frames=4)
    """
    
    # Validate input and setup paths
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
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
        
        logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        # Calculate frame indices (evenly spaced)
        if n_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        # Extract frames
        frames = []
        timestamps = []
        actual_frame_numbers = []  # Store actual frame indices
        
        for frame_idx in frame_indices:
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
            actual_frame_numbers.append(int(frame_idx))  # Store actual frame number
        
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        logger.info(f"Extracted {len(frames)} frames")
        
        # Create figure if requested
        figure_path = None
        if create_figure:
            figure_path = _create_figure(
                frames, timestamps, actual_frame_numbers, video_path.stem, output_dir, 
                layout, figure_size, dpi, add_timestamps, add_frame_numbers, title, duration
            )
        
        return frames, figure_path
        
    finally:
        cap.release()


def _create_figure(
    frames: List[np.ndarray],
    timestamps: List[float],
    actual_frame_numbers: List[int],
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
    """Create a combined figure showing all frames."""
    
    n_frames = len(frames)
    
    # Setup matplotlib with clean styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    
    # Determine subplot arrangement
    if layout == "horizontal":
        nrows, ncols = 1, n_frames
        figsize = figure_size
    elif layout == "vertical":
        nrows, ncols = n_frames, 1
        figsize = (figure_size[1], figure_size[0])
    elif layout == "grid":
        ncols = int(np.ceil(np.sqrt(n_frames)))
        nrows = int(np.ceil(n_frames / ncols))
        figsize = (figure_size[0], figure_size[0] * nrows / ncols)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    
    # Handle different subplot return types
    if n_frames == 1:
        axes = [axes]
    elif layout in ["horizontal", "vertical"]:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:  # grid layout
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    elif layout == "horizontal":
        fig.suptitle(f'Temporal Decomposition: {video_name}', 
                    fontsize=14, fontweight='bold', y=0.95)
    
    # Plot frames
    for i, (frame, timestamp, frame_num) in enumerate(zip(frames, timestamps, actual_frame_numbers)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.imshow(frame)
        ax.axis('off')
        
        # Add labels: frame number and/or timestamp
        label_parts = []
        if add_frame_numbers:
            label_parts.append(f"Frame {frame_num}")  # Use actual frame number
        if add_timestamps:
            label_parts.append(f"t={timestamp:.2f}s")
        if label_parts:
            # Position label at top of frame with small margin
            ax.text(0.02, 0.98, " | ".join(label_parts),
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
    
    # Hide unused subplots in grid layout
    if layout == "grid":
        for i in range(n_frames, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
    
    # Add time progression indicator for horizontal layout
    if layout == "horizontal" and n_frames > 1:
        fig.text(0.1, 0.02, 'Time →', fontsize=11, fontweight='bold')
        fig.text(0.85, 0.02, f'Duration: {duration:.2f}s', fontsize=10, 
                horizontalalignment='right')
    
    plt.tight_layout()
    if title or layout == "horizontal":
        plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save figure
    base_filename = f"{video_name}_decomposed_{n_frames}frames_{layout}"
    
    # PNG version
    png_path = output_dir / f"{base_filename}.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    # EPS version for papers
    eps_path = output_dir / f"{base_filename}.eps"
    plt.savefig(eps_path, format='eps', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    plt.close()
    
    logger.info(f"Created figure: {png_path}")
    logger.info(f"Created vector figure: {eps_path}")
    
    return str(png_path)