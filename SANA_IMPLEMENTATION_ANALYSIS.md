# SANA Implementation Analysis Report

## Executive Summary

Two different implementations of SANA-Video exist in the VMEvalKit codebase:
1. **Dev Branch Implementation** (by luke, commit f42f74d) - 184 lines, simpler approach
2. **Current Branch Implementation** (feature/implementcogvideo) - 270 lines, more comprehensive

Both implementations use the same underlying `SanaImageToVideoPipeline` from diffusers but differ significantly in their architecture, features, and design philosophy.

---

## Git History Context

```
*   7a9932d (origin/dev) Merge pull request #170 from Video-Reason/diffuers
|\  
| * 48794d2 merge wan diffusers env into one
| * f42f74d add sana model  <-- Dev branch SANA implementation
|/  
* 61a9ee4 refactor eval_prompt.py
| * 9ce0bf1 (HEAD -> feature/implementcogvideo) setup
| * beda2fd implementing cogvideo  <-- Current branch has different SANA implementation
```

**Author of dev implementation:** luke <1263810658@qq.com>  
**Date:** Tue Dec 9 07:22:44 2025 -0800  
**Commit message:** "add sana model"

---

## Detailed Comparison

### 1. Class Naming

| Aspect | Dev Branch | Current Branch |
|--------|-----------|----------------|
| Service Class | `SanaService` | `SanaVideoService` |
| Wrapper Class | `SanaWrapper` | `SanaVideoWrapper` |

**Analysis:** Current branch uses more descriptive names that explicitly indicate "Video" functionality.

---

### 2. Documentation

**Dev Branch:**
- Minimal documentation
- Inline comments: "requires diffuers>=0.36.0" and "takes 22GB vram, 4 mins with single RTX A6000"
- No module-level docstring

**Current Branch:**
- Comprehensive module-level docstring (17 lines)
- Detailed class and method docstrings
- Explains all conditioning modes (Text-to-Video, Image-to-Video, Text+Image-to-Video)
- Lists model variants and references (HuggingFace, GitHub, Diffusers docs)

**Winner:** Current branch - significantly better documentation

---

### 3. Model Loading Strategy

#### Dev Branch (Manual dtype management):
```python
if torch.cuda.is_available():
    self.device = "cuda"
    transformer_dtype = torch.bfloat16
    encoder_dtype = torch.bfloat16
    vae_dtype = torch.float32  # VAE kept at float32
else:
    self.device = "cpu"
    transformer_dtype = torch.float32
    encoder_dtype = torch.float32
    vae_dtype = torch.float32

self.pipe = SanaImageToVideoPipeline.from_pretrained(self.model_id)
self.pipe.transformer.to(transformer_dtype)
self.pipe.text_encoder.to(encoder_dtype)
self.pipe.vae.to(vae_dtype)
self.pipe.to(self.device)
```

**Characteristics:**
- Fine-grained control over component dtypes
- VAE kept at float32 for numerical stability
- Manually moves each component
- More explicit but verbose

#### Current Branch (Unified dtype):
```python
if torch.cuda.is_available():
    self.device = "cuda"
    torch_dtype = torch.bfloat16
else:
    self.device = "cpu"
    torch_dtype = torch.float32

self.pipe = SanaImageToVideoPipeline.from_pretrained(
    self.model_id,
    torch_dtype=torch_dtype
)
self.pipe.to(self.device)
```

**Characteristics:**
- Simpler, unified dtype approach
- Lets diffusers handle component-level dtype management
- More concise
- Relies on diffusers' internal logic

**Analysis:** Dev branch approach may be more stable (VAE at float32), but current branch is cleaner. The dev branch's comment about "22GB vram" suggests they encountered memory issues that led to this optimization.

---

### 4. Image Preprocessing

#### Dev Branch:
```python
def _prepare_image(self, image_path: Union[str, Path]) -> Image.Image:
    image = load_image(str(image_path))
    if image.mode != "RGB":
        image = image.convert("RGB")
    logger.info(f"Prepared image for SANA: {image.size}")
    return image
```

**Features:**
- Loads image
- Converts to RGB
- No resizing
- Returns original size

#### Current Branch:
```python
def _prepare_image(self, image_path: Union[str, Path]):
    from diffusers.utils import load_image
    from PIL import Image
    
    image = load_image(str(image_path))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to model's expected resolution
    target_size = (self.model_constraints["width"], self.model_constraints["height"])
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    logger.info(f"Prepared image: {image.size}")
    return image
```

**Features:**
- Loads image
- Converts to RGB
- **Automatically resizes to model constraints (832x480)**
- Uses LANCZOS resampling for quality
- Lazy imports

**Winner:** Current branch - ensures consistent input dimensions, preventing potential errors

---

### 5. Model Constraints / Defaults

#### Dev Branch:
- No centralized constraints dictionary
- Defaults scattered in method signatures
- Default guidance_scale: **6.0**
- Default num_inference_steps: **50**

#### Current Branch:
```python
self.model_constraints = {
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "fps": 16,
    "guidance_scale": 4.5,
    "num_inference_steps": 20
}
```

**Features:**
- Centralized configuration
- Default guidance_scale: **4.5**
- Default num_inference_steps: **20**
- Easier to maintain and modify

**Winner:** Current branch - better architecture, but note the different hyperparameters

---

### 6. Generation Parameters

#### Dev Branch - Additional Parameters:
- `negative_prompt`: String for negative prompting
- `motion_score`: Integer (default 30) - appended to prompt as " motion score: {value}."
- `seed`: Integer (default 42) - for reproducibility
- `frames`: Parameter name (instead of `num_frames`)

**Motion Score Feature:**
```python
motion_prompt = f" motion score: {motion_score}."
composed_prompt = text_prompt + motion_prompt
```

This is a unique feature that allows users to control motion intensity.

#### Current Branch:
- Uses `num_frames` (more standard naming)
- No negative_prompt support
- No motion_score feature
- No seed control (non-deterministic)

**Analysis:** Dev branch has more advanced features for fine-tuning generation, especially the motion_score which could be valuable for controlling video dynamics.

---

### 7. Wrapper Class Architecture

#### Dev Branch:
```python
class SanaWrapper(ModelWrapper):
    def __init__(self, model: str = "...", output_dir: str = "./data/outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.sana_service = SanaService(model=model)
```

- Calls parent `__init__`
- Relies on parent class for common functionality

#### Current Branch:
```python
class SanaVideoWrapper(ModelWrapper):
    def __init__(self, model: str = "...", output_dir: str = "./data/outputs", **kwargs):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        self.sana_service = SanaVideoService(model=model)
```

- Does NOT call parent `__init__`
- Manually manages attributes
- Creates output directory immediately

**Analysis:** Dev branch follows proper OOP inheritance. Current branch may have issues if parent class has important initialization logic.

---

### 8. Parameter Handling in Wrapper

#### Dev Branch:
```python
fps = kwargs.get("fps", 16)
if "frames" not in kwargs:
    kwargs["frames"] = max(1, int(duration * fps))

kwargs.setdefault("height", 480)
kwargs.setdefault("width", 832)
kwargs.setdefault("guidance_scale", 6.0)
kwargs.setdefault("num_inference_steps", 50)
kwargs.setdefault("motion_score", 30)
kwargs.setdefault("seed", 42)
kwargs.setdefault("fps", fps)
```

- Uses `setdefault()` - keeps user values if provided
- All parameters passed through kwargs
- Simple and straightforward

#### Current Branch:
```python
num_frames = kwargs.pop("num_frames", None)
fps = kwargs.pop("fps", None)
height = kwargs.pop("height", None)
width = kwargs.pop("width", None)
num_inference_steps = kwargs.pop("num_inference_steps", None)
guidance_scale = kwargs.pop("guidance_scale", None)

# Calculate frames from duration if not specified
if num_frames is None:
    fps = fps or self.sana_service.model_constraints["fps"]
    num_frames = int(duration * fps)
```

- Uses `pop()` to extract and remove from kwargs
- Explicit parameter passing to service
- More control but more verbose

**Analysis:** Dev branch approach is cleaner. Current branch's pop() approach prevents parameter conflicts but is more complex.

---

### 9. Import Strategy

#### Dev Branch:
```python
import torch
from PIL import Image
from diffusers import SanaImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
```

- All imports at module level
- Immediate loading

#### Current Branch:
```python
# In _load_model():
import torch
from diffusers import SanaImageToVideoPipeline

# In _prepare_image():
from diffusers.utils import load_image
from PIL import Image
```

- Lazy imports inside methods
- Only loads when needed
- Better for module initialization time

**Winner:** Current branch - lazy imports are generally better practice for optional dependencies

---

### 10. Code Quality & Style

#### Dev Branch:
- 184 lines total
- Simpler, more straightforward
- Less documentation
- Follows parent class patterns

#### Current Branch:
- 270 lines total (46% longer)
- More comprehensive documentation
- Better structured with docstrings
- More defensive programming (image resizing, etc.)

---

## Key Differences Summary

| Feature | Dev Branch | Current Branch |
|---------|-----------|----------------|
| **Lines of Code** | 184 | 270 |
| **Documentation** | Minimal | Comprehensive |
| **Negative Prompts** | ✅ Yes | ❌ No |
| **Motion Score** | ✅ Yes (unique feature) | ❌ No |
| **Seed Control** | ✅ Yes | ❌ No |
| **Image Auto-resize** | ❌ No | ✅ Yes |
| **Model Constraints** | Scattered | ✅ Centralized |
| **Guidance Scale Default** | 6.0 | 4.5 |
| **Inference Steps Default** | 50 | 20 |
| **VAE dtype** | float32 (optimized) | bfloat16 (unified) |
| **Parent Init** | ✅ Called | ❌ Not called |
| **Import Strategy** | Module-level | Lazy imports |
| **Parameter Naming** | `frames` | `num_frames` |

---

## Performance Implications

### Dev Branch:
- **Memory:** 22GB VRAM (per author's comment)
- **Speed:** 4 minutes on RTX A6000
- **Optimization:** VAE kept at float32 for stability
- **Steps:** 50 (more quality, slower)

### Current Branch:
- **Memory:** Unknown (likely similar or slightly higher due to unified bfloat16)
- **Speed:** Likely faster (only 20 steps vs 50)
- **Optimization:** Unified dtype (simpler but potentially less stable)
- **Steps:** 20 (faster, potentially lower quality)

---

## Recommendations

### If merging implementations, consider:

1. **Keep from Dev Branch:**
   - Motion score feature (unique and valuable)
   - Negative prompt support
   - Seed control for reproducibility
   - VAE float32 optimization (if memory is a concern)
   - Proper parent class initialization

2. **Keep from Current Branch:**
   - Comprehensive documentation
   - Centralized model_constraints
   - Automatic image resizing
   - Lazy imports
   - More descriptive class names

3. **Hyperparameter Decision:**
   - Dev: guidance_scale=6.0, steps=50 (higher quality, slower)
   - Current: guidance_scale=4.5, steps=20 (faster, standard)
   - **Recommendation:** Make configurable, default to current branch values for speed

4. **Critical Issue to Address:**
   - Current branch doesn't call `super().__init__()` - this should be fixed

---

## Conclusion

Both implementations are functional but serve slightly different purposes:

- **Dev Branch:** More feature-rich (motion score, negative prompts, seed), optimized for quality and stability
- **Current Branch:** Better documented, cleaner architecture, optimized for speed

The ideal solution would be a **hybrid approach** that combines:
- Current branch's documentation and architecture
- Dev branch's advanced features (motion_score, negative_prompt, seed)
- Dev branch's memory optimization (VAE float32)
- Proper parent class initialization

**Recommendation:** Merge the best of both, with the current branch as the base and dev branch features added as optional parameters.

