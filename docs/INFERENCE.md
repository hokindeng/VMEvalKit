# VMEvalKit Inference Module

This module provides a unified interface for running inference on video generation models at scale, with automatic error handling, resume capability, and structured output management.

## Quick Start

```python
from vmevalkit.runner.inference import InferenceRunner

# Initialize runner with structured output
runner = InferenceRunner(output_dir="output")

# Generate video solution
result = runner.run(
    model_name="luma-ray-2",
    image_path="data/questions/maze_task/maze_0000/first_frame.png",
    text_prompt="Navigate the green dot through the maze corridors to reach the red flag"
)

print(f"Video saved to: {result['inference_dir']}")
# Each inference creates a self-contained folder with:
# - video/: Generated video file
# - question/: Input images and prompt  
# - metadata.json: Complete inference metadata
```

## Core Concepts

### Task Pair: The Fundamental Unit
Every VMEvalKit dataset consists of **Task Pairs** - the basic unit for video reasoning evaluation:

- 📸 **Initial state image** (`first_frame.png` - the reasoning problem)
- 🎯 **Final state image** (`final_frame.png` - the solution/goal state)  
- 📝 **Text prompt** (`prompt.txt` - instructions for video model)
- 📊 **Rich metadata** (`question_metadata.json` - difficulty, task-specific parameters, etc.)

Each task pair is organized in its own folder (`data/questions/{domain}_task/{question_id}/`) containing all four files. Models must generate videos showing the reasoning process from initial → final state.

## Supported Models

VMEvalKit supports **40 models** across **11 families** using a clean modular architecture:

### Commercial APIs (29 models)
- **Luma Dream Machine**: 2 models (`luma-ray-2`, `luma-ray-flash-2`)
- **Google Veo**: 3 models (`veo-2.0-generate`, `veo-3.0-generate`, etc.)
- **Google Veo 3.1**: 4 models (via WaveSpeed, with 720p/1080p variants)
- **WaveSpeed WAN**: 18 models (2.1 & 2.2 variants with LoRA/ultra-fast options)
- **Runway ML**: 3 models (Gen-3A Turbo, Gen-4 Turbo/Aleph)
- **OpenAI Sora**: 2 models (Sora-2, Sora-2-Pro)

### Open-Source Models (11 models)
- **LTX-Video**: 3 models (13B distilled, 13B dev, 2B distilled)
- **HunyuanVideo**: 1 model (high-quality 720p)
- **VideoCrafter**: 1 model (text-guided generation)
- **DynamiCrafter**: 3 models (256p, 512p, 1024p)

All models support **image + text → video** for reasoning evaluation.

## Architecture

VMEvalKit uses a **clean modular architecture** with dynamic loading:

```
vmevalkit/
├── runner/
│   ├── MODEL_CATALOG.py    # 📋 Pure model registry (40 models, 11 families)
│   └── inference.py        # 🎭 Orchestration with dynamic loading
├── models/
│   ├── base.py            # 🔧 Abstract ModelWrapper interface
│   ├── luma_inference.py  # LumaInference + LumaWrapper
│   ├── veo_inference.py   # VeoService + VeoWrapper 
│   └── ...                # Each provider: Service + Wrapper
```

**Key Benefits:**
- **Dynamic Loading**: Models loaded on-demand from catalog
- **Family Organization**: Models grouped by provider families
- **Consistent Interface**: All wrappers inherit from `ModelWrapper`
- **Easy Extension**: Add models without touching core files

## Structured Output System

Each inference creates a **self-contained folder** with all relevant data:

```
output/<model>_<question_id>_<timestamp>/
├── video/
│   └── generated_video.mp4    # Output video
├── question/
│   ├── first_frame.png        # Input image (sent to model)
│   ├── final_frame.png        # Reference image (not sent)
│   ├── prompt.txt             # Text prompt used
│   └── question_metadata.json # Full question data from dataset
└── metadata.json              # Complete inference metadata
```

This structure ensures reproducibility and makes batch analysis easy.

## Running Experiments

### Basic Usage

Generate dataset and run experiments:

```bash
cd /Users/access/VMEvalKit
source venv/bin/activate

# Generate dataset (if needed)
python -m vmevalkit.runner.create_dataset --pairs-per-domain 15

# Run experiment (1 task per domain for testing)
python examples/experiment_2025-10-14.py

# Run all tasks
python examples/experiment_2025-10-14.py --all-tasks
```

### Automatic Resume

The experiment script includes automatic resume capability:

**Features:**
- 🔄 Sequential execution: one model at a time, one task at a time
- ✅ Automatic skip of completed tasks
- 🎯 Selective model execution
- 📁 Directory-based completion tracking

**Usage:**

```bash
# Run all tasks (automatically skips completed ones)
python examples/experiment_2025-10-14.py --all-tasks

# Run specific models only
python examples/experiment_2025-10-14.py --all-tasks --only-model veo-3.0-generate

# Run multiple specific models
python examples/experiment_2025-10-14.py --all-tasks --only-model veo-3.0-generate luma-ray-2
```

**Command Options:**

| Option | Description |
|--------|-------------|
| `--all-tasks` | Run all tasks instead of 1 per domain |
| `--only-model [MODEL ...]` | Run only specified models (others skipped) |

**How It Works:**
- Automatically detects existing output directories
- Skips tasks that already have successful inference results
- To retry failed tasks: manually delete their output directories
- No separate checkpoint files - uses directory presence for tracking

## Python API

### InferenceRunner

The main class for running inference:

```python
from vmevalkit.runner.inference import InferenceRunner

runner = InferenceRunner(
    output_dir="output",  # Where to save results
    timeout=300,          # Timeout in seconds
    max_retries=3         # Number of retries on failure
)
```

### Running Single Inference

```python
result = runner.run(
    model_name="luma-ray-2",
    image_path="path/to/image.png",
    text_prompt="Your prompt here",
    metadata={             # Optional metadata
        "task_type": "maze",
        "difficulty": "hard"
    }
)

# Result contains:
# - inference_dir: Path to output folder
# - video_path: Path to generated video
# - metadata: Complete inference metadata
```

### Batch Processing

```python
# Process multiple tasks
for task in tasks:
    result = runner.run(
        model_name="veo-3.0-generate",
        image_path=task["image"],
        text_prompt=task["prompt"],
        metadata=task.get("metadata", {})
    )
    print(f"Processed {task['id']}: {result['video_path']}")
```

## Model-Specific Configuration

### API Keys

Set up API keys in `.env` file:

```bash
# Commercial APIs
LUMA_API_KEY=your_key_here
WAVESPEED_API_KEY=your_wavespeed_api_key
RUNWAY_API_SECRET=your_runway_secret
OPENAI_API_KEY=your_openai_key

# AWS for S3 storage (optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2
```

### Model Parameters

Different models support different parameters:

```python
# Luma models
result = runner.run(
    model_name="luma-ray-2",
    image_path="image.png",
    text_prompt="prompt",
    model_params={
        "duration": 5,  # Video duration in seconds
        "quality": "high"
    }
)

# Veo models
result = runner.run(
    model_name="veo-3.0-generate",
    image_path="image.png", 
    text_prompt="prompt",
    model_params={
        "resolution": "1080p",
        "fps": 30
    }
)
```

## Error Handling and Recovery

The inference system includes robust error handling:

### Automatic Retries
- Failed inferences are automatically retried up to `max_retries` times
- Exponential backoff between retries
- Different error types handled appropriately

### Resume from Failures
```python
# The system automatically tracks completed tasks
# Re-running the same experiment will skip completed inferences

# To force re-run, delete the output directory:
import shutil
shutil.rmtree("output/luma-ray-2_maze_0000_20250101_120000")
```

### Error Logging
All errors are logged with detailed information:
- Model response
- API errors
- Timeout issues
- File I/O errors

## Tips and Best Practices

1. **Start Small**: Test with 1 task per domain before running full experiments
2. **Monitor API Usage**: Track API costs for commercial models
3. **Check Outputs**: Verify video generation quality before large-scale runs
4. **Use Resume**: Take advantage of automatic resume for long experiments
5. **Structure Metadata**: Include rich metadata for better analysis later

## Extending the System

### Adding Custom Models

See [ADDING_MODELS.md](ADDING_MODELS.md) for detailed instructions on adding new models.

### Custom Output Processing

```python
from vmevalkit.runner.inference import InferenceRunner

class CustomRunner(InferenceRunner):
    def post_process(self, result):
        # Custom processing of inference results
        video_path = result["video_path"]
        # Add your processing logic here
        return result
```

## Related Documentation

- [ADDING_MODELS.md](ADDING_MODELS.md) - How to add new video models
- [EVALUATION.md](EVALUATION.md) - How to evaluate inference results
- [WEB_DASHBOARD.md](WEB_DASHBOARD.md) - Visualizing results in the web interface
