## Installation

Install the required dependencies for open-source models:

```bash
uv pip install transformers diffusers
```

## WAN (Wan-AI)

WAN 2.1 FLF2V (First-Last Frame to Video) is an open-source image-to-video generation model using diffusers.

The model will be automatically downloaded from HuggingFace when first used. No additional installation steps are required.

### Usage

```bash
# Generate videos using WAN model
uv run examples/generate_videos.py --model wan --task chess maze

# Or use the full model name
uv run examples/generate_videos.py --model wan-2.1-flf2v-720p --task chess maze
```

### Model Details

- **Model ID**: `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`
- **Resolution**: Up to 720p (automatically resized based on input image aspect ratio)
- **FPS**: 16 fps
- **Type**: Image-to-video generation

## LTX-Video

### Usage

```bash
uv pip install sentencepiece #  otherwise has error. refer https://huggingface.co/Lightricks/LTX-Video/discussions/96
# Generate videos using LTX-Video model
uv run examples/generate_videos.py --model ltx-video --task chess maze
```