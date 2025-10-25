# Video Decomposer Utility

A comprehensive utility for decomposing videos into temporal frame sequences, specifically designed for creating publication-quality figures in research papers.

## Features

- **Temporal Frame Extraction**: Extract N frames at regular intervals from videos
- **Multiple Layouts**: Arrange frames horizontally, vertically, or in grid format
- **Publication Quality**: High-DPI output in PNG and EPS formats
- **Model Comparison**: Create side-by-side comparisons of multiple videos
- **Customizable Styling**: Add timestamps, frame numbers, titles, and custom layouts
- **Format Support**: Works with MP4, WebM, AVI, MOV video formats

## Quick Start

```python
from vmevalkit.utils import decompose_video

# Extract 4 frames in horizontal layout
frames, figure_path = decompose_video(
    video_path="path/to/your/video.mp4",
    n_frames=4,
    layout="horizontal",
    create_figure=True,
    add_timestamps=True
)
```

## Basic Usage

### Single Video Decomposition

```python
from vmevalkit.utils import decompose_video

# Basic horizontal timeline
frames, figure = decompose_video(
    video_path="video.mp4",
    n_frames=4,
    layout="horizontal",
    output_dir="figures/",
    title="Temporal Progression"
)

# Vertical arrangement with more frames
frames, figure = decompose_video(
    video_path="video.mp4",
    n_frames=6,
    layout="vertical",
    figure_size=(8, 12)
)

# Grid layout for many frames
frames, figure = decompose_video(
    video_path="video.mp4",
    n_frames=9,
    layout="grid",
    add_timestamps=True,
    add_frame_numbers=True
)
```

### Multi-Video Model Comparison

```python
from vmevalkit.utils import create_video_comparison_figure

# Compare different AI models
figure_path = create_video_comparison_figure(
    video_paths=[
        "model1_output.mp4",
        "model2_output.mp4", 
        "model3_output.mp4"
    ],
    model_names=["OpenAI Sora", "Google Veo", "Luma Ray"],
    n_frames=4,
    title="Model Performance Comparison"
)
```

## Command Line Interface

```bash
# Extract 4 frames horizontally
python -m vmevalkit.utils.video_decomposer video.mp4 --frames 4 --layout horizontal

# Create comparison figure
python -m vmevalkit.utils.video_decomposer model1.mp4 model2.mp4 --comparison

# Vertical layout with custom output
python -m vmevalkit.utils.video_decomposer video.mp4 --frames 6 --layout vertical --output ./figures/
```

## Parameters

### `decompose_video()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | str/Path | - | Path to input video file |
| `n_frames` | int | 4 | Number of frames to extract |
| `output_dir` | str/Path | None | Output directory (defaults to video location) |
| `create_figure` | bool | True | Whether to create combined figure |
| `layout` | str | "horizontal" | Layout: "horizontal", "vertical", "grid" |
| `figure_size` | tuple | (16, 4) | Figure size in inches (width, height) |
| `save_individual_frames` | bool | False | Save individual frame images |
| `frame_format` | str | "png" | Format for individual frames |
| `dpi` | int | 300 | DPI for publication quality |
| `add_timestamps` | bool | True | Add timestamp labels |
| `add_frame_numbers` | bool | True | Add frame number labels |
| `title` | str | None | Optional figure title |

### `create_video_comparison_figure()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_paths` | list | - | List of video file paths |
| `n_frames` | int | 4 | Frames per video |
| `output_dir` | str/Path | None | Output directory |
| `model_names` | list | None | Model names for labeling |
| `figure_size` | tuple | (16, 10) | Figure size |
| `dpi` | int | 300 | DPI quality |
| `title` | str | None | Figure title |

## Output Formats

The utility creates figures in multiple formats:

- **PNG**: High-quality raster images for presentations and web
- **EPS**: Vector format for LaTeX and professional publications
- **Individual frames**: Optional separate frame images

## Use Cases for Research Papers

### 1. Temporal Analysis
Show how AI-generated videos evolve over time:
```python
decompose_video("ai_video.mp4", n_frames=5, layout="horizontal", 
               title="Temporal Evolution Analysis")
```

### 2. Model Comparison
Compare different AI models side-by-side:
```python
create_video_comparison_figure(
    ["sora.mp4", "veo.mp4", "luma.mp4"],
    model_names=["Sora", "Veo", "Luma"],
    title="Video Generation Model Comparison"
)
```

### 3. Task Progression
Show progression through complex tasks:
```python
decompose_video("chess_game.mp4", n_frames=8, layout="grid",
               title="Chess Game Progression")
```

### 4. Quality Analysis
Analyze video quality at different timestamps:
```python
decompose_video("output.mp4", n_frames=6, layout="vertical",
               add_timestamps=True, figure_size=(8, 16))
```

## Best Practices for Publications

1. **Resolution**: Use `dpi=300` or higher for print publications
2. **Layout**: Choose `horizontal` for temporal sequences, `grid` for overview
3. **Formats**: Save both PNG (for reviewing) and EPS (for final publication)
4. **Sizing**: Adjust `figure_size` to match journal column widths
5. **Labeling**: Use clear titles and consider timestamp placement
6. **Consistency**: Use same `n_frames` across comparisons

## Example Output

The utility generates professional figures suitable for:
- Research papers and publications
- Conference presentations  
- Technical reports
- Documentation and tutorials

## Requirements

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Python 3.7+

## Installation

The utility is included with VMEvalKit. Install dependencies:

```bash
pip install opencv-python matplotlib numpy
```

## Error Handling

The utility includes comprehensive error handling for:
- Invalid video files
- Unsupported formats
- Missing directories
- Corrupted video data

## Integration with VMEvalKit

This utility is designed to work seamlessly with VMEvalKit's evaluation pipeline:

```python
# Process evaluation outputs
from vmevalkit.utils import decompose_video

# Decompose generated videos for analysis
for model_output in evaluation_results:
    decompose_video(
        model_output.video_path,
        output_dir=f"analysis/{model_output.model_name}/",
        title=f"{model_output.model_name} - {model_output.task}"
    )
```
