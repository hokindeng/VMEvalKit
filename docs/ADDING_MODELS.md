# Adding Models to VMEvalKit

Quick guide for integrating video generation models into VMEvalKit's unified inference system.

## ⚡ Requirements

✅ **MUST support: Image + Text → Video** (essential for reasoning evaluation)  
✅ **Unified interface**: `generate(image_path, text_prompt, duration, output_filename, **kwargs)`  
✅ **Parameter separation**: Constructor for config, generate() for runtime inputs

## 🎯 Integration Steps

### API Models (3 steps)
1. Create `vmevalkit/models/{provider}_inference.py` with Service + Wrapper
2. Register in `vmevalkit/runner/inference.py` 
3. Update `vmevalkit/models/__init__.py`

### Open-Source Models (4 steps)  
1. Add submodule: `git submodule add {repo_url} submodules/{ModelName}`
2. Create wrapper with subprocess/direct calls
3. Register in inference.py
4. Update __init__.py

---

## 📝 Implementation Guide

### API Models

**File**: `vmevalkit/models/{provider}_inference.py`

```python
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import time

class {Provider}Service:
    def __init__(self, model_id: str, api_key: Optional[str] = None, **kwargs):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("{PROVIDER}_API_KEY")
        
    async def generate_video(self, prompt: str, image_path: str, **kwargs):
        # Your API implementation:
        # 1. Prepare request with image + text
        # 2. Make API call  
        # 3. Return (video_bytes, metadata)
        video_bytes = b""  # API response
        metadata = {"request_id": "id", "model": self.model_id}
        return video_bytes, metadata

class {Provider}Wrapper:
    """Wrapper implementing unified interface"""
    def __init__(self, model: str, output_dir: str = "./data/outputs", api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.service = {Provider}Service(model_id=model, api_key=api_key, **kwargs)
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, 
                 duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Unified interface - EXACT signature required"""
        start_time = time.time()
        
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}", "video_path": None}
        
        video_bytes, metadata = asyncio.run(
            self.service.generate_video(prompt=text_prompt, image_path=str(image_path), **kwargs)
        )
        
        if video_bytes:
            if not output_filename:
                output_filename = f"{self.model.replace('/', '_')}_{int(time.time())}.mp4"
            video_path = self.output_dir / output_filename
            video_path.write_bytes(video_bytes)
            return {"success": True, "video_path": str(video_path), 
                    "duration": time.time() - start_time, **metadata}
        
        return {"success": False, "error": "No video generated", "video_path": None}
```

### Open-Source Models

**File**: `vmevalkit/models/{model}_inference.py`

```python
import sys, subprocess, time
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add submodule to path
{MODEL}_PATH = Path(__file__).parent.parent.parent / "submodules" / "{ModelName}"
sys.path.insert(0, str({MODEL}_PATH))

class {Model}Service:
    def __init__(self, model_id: str = "default", output_dir: str = "./data/outputs", **kwargs):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if not {MODEL}_PATH.exists():
            raise FileNotFoundError(f"Run: git submodule update --init {ModelName}")

    def _run_subprocess_inference(self, image_path: str, text_prompt: str, 
                                  output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """For models requiring subprocess execution"""
        start_time = time.time()
        output_path = self.output_dir / (output_filename or f"{self.model_id}_{int(time.time())}.mp4")
        
        cmd = [sys.executable, str({MODEL}_PATH / "inference.py"),
               "--prompt", text_prompt, "--image", str(image_path), "--output", str(output_path)]
        
        try:
            result = subprocess.run(cmd, cwd=str({MODEL}_PATH), 
                                  capture_output=True, text=True, timeout=300)
            success = result.returncode == 0 and output_path.exists()
            return {"success": success, 
                   "video_path": str(output_path) if success else None,
                   "error": result.stderr if not success else None,
                   "duration": time.time() - start_time}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout", "video_path": None}
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, 
                 duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}", "video_path": None}
        
        return self._run_subprocess_inference(str(image_path), text_prompt, 
                                             output_filename=output_filename, **kwargs)

class {Model}Wrapper:
    def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
        self.model = model
        self.service = {Model}Service(model_id=model, output_dir=output_dir, **kwargs)
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, 
                 duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        return self.service.generate(image_path, text_prompt, duration, output_filename, **kwargs)
```

## 🔌 Registration

### 1. Register in `vmevalkit/runner/inference.py`

```python
# Import wrapper
from ..models import {Provider}Wrapper

# Define models
{PROVIDER}_MODELS = {
    "{provider}-model-name": {
        "class": {Provider}Wrapper,
        "model": "actual-model-id",
        "description": "Brief description", 
        "family": "{Provider}"
    }
}

# Add to registries
AVAILABLE_MODELS = {**EXISTING_MODELS, **{PROVIDER}_MODELS}
MODEL_FAMILIES = {**EXISTING_FAMILIES, "{Provider}": {PROVIDER}_MODELS}
```

### 2. Update `vmevalkit/models/__init__.py`

```python
from .{provider}_inference import {Provider}Service, {Provider}Wrapper

__all__ = [..., "{Provider}Service", "{Provider}Wrapper"]
```

## ✅ Testing

```python
from vmevalkit.runner.inference import run_inference

result = run_inference(
    model_name="your-model-name",
    image_path="test.png",
    text_prompt="test prompt",
    output_dir="./data/outputs"
)
print(f"Success: {result.get('success')}")
print(f"Video: {result.get('video_path')}")
```

## 💡 Key Rules

**DO:**
- Constructor: `model`, `api_key`, `output_dir` (stable config)
- generate(): `image_path`, `text_prompt`, `duration`, `output_filename`, `seed` (per-run)
- Use `Path` objects, validate inputs, handle errors
- Follow naming: `{Provider}Wrapper`, `{provider}_inference.py`

**DON'T:**
- Pass runtime args to constructor
- Forget error handling
- Skip input validation

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check `AVAILABLE_MODELS` and `MODEL_FAMILIES` |
| Import error | Update `__init__.py` imports |
| Submodule missing | `git submodule update --init --recursive` |
| API auth fails | Set environment variable: `{PROVIDER}_API_KEY` |

---

Ready to add your model? Follow the steps above and you'll be integrated with VMEvalKit! 🎯