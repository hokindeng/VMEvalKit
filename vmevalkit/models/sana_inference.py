"""SANA-Video Integration for VMEvalKit

Uses SanaImageToVideoPipeline from diffusers for text+image → video generation.
Single backbone (SANA-Video 2B) supports all conditioning modes:
- Text-to-Video
- Image-to-Video  
- Text+Image-to-Video (TextImage-to-Video)

Model variants:
- sana-video-2b-480p: Base short-video model (~5 seconds, 81 frames)
- sana-video-2b-longlive: Extended length via block linear KV-cache

References:
- HuggingFace: https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers
- GitHub: https://github.com/NVlabs/Sana
- Diffusers: https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_video
"""

import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from .base import ModelWrapper

logger = logging.getLogger(__name__)


class SanaVideoService:
    """Service for SANA-Video inference using diffusers pipeline.
    
    Uses SanaImageToVideoPipeline for text+image conditioned video generation.
    The same 2B backbone supports text-only, image-only, and text+image modes.
    """
    
    def __init__(self, model: str = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"):
        self.model_id = model
        self.pipe = None
        self.device = None
        
        # SANA-Video 2B 480p default constraints
        self.model_constraints = {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "fps": 16,
            "guidance_scale": 4.5,
            "num_inference_steps": 20
        }
    
    def _load_model(self):
        """Lazy load the SANA-Video pipeline."""
        if self.pipe is not None:
            return
        
        logger.info(f"Loading SANA-Video model: {self.model_id}")
        import torch
        from diffusers import SanaImageToVideoPipeline
        
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
        logger.info(f"SANA-Video model loaded on {self.device}")
    
    def _prepare_image(self, image_path: Union[str, Path]):
        """Load and prepare image for inference."""
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
    
    def generate_video(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        fps: Optional[int] = None,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt.
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for video generation
            height: Output video height (default: 480)
            width: Output video width (default: 832)
            num_frames: Number of frames to generate (default: 81)
            num_inference_steps: Denoising steps (default: 20)
            guidance_scale: Classifier-free guidance scale (default: 4.5)
            fps: Output video FPS (default: 16)
            output_path: Path to save output video
            **kwargs: Additional pipeline arguments
            
        Returns:
            Dictionary with video_path, frames, and metadata
        """
        start_time = time.time()
        
        self._load_model()
        
        image = self._prepare_image(image_path)
        
        # Apply defaults from model constraints
        height = height or self.model_constraints["height"]
        width = width or self.model_constraints["width"]
        num_frames = num_frames or self.model_constraints["num_frames"]
        num_inference_steps = num_inference_steps or self.model_constraints["num_inference_steps"]
        guidance_scale = guidance_scale or self.model_constraints["guidance_scale"]
        fps = fps or self.model_constraints["fps"]
        
        logger.info(f"Generating video with prompt: {text_prompt[:80]}...")
        logger.info(f"Dimensions: {width}x{height}, frames={num_frames}, steps={num_inference_steps}")
        
        # Generate using SanaImageToVideoPipeline (text+image conditioning)
        output = self.pipe(
            image=image,
            prompt=text_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        frames = output.frames[0]
        
        video_path = None
        if output_path:
            from diffusers.utils import export_to_video
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=fps)
            video_path = str(output_path)
            logger.info(f"Video saved to: {video_path}")
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": video_path,
            "frames": frames,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": duration_taken,
            "model": self.model_id,
            "status": "success" if video_path else "completed",
            "metadata": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "image_size": image.size
            }
        }


class SanaVideoWrapper(ModelWrapper):
    """Wrapper for SANA-Video models conforming to VMEvalKit interface.
    
    Supports both base 480p model and LongLive extended variant.
    """
    
    def __init__(
        self,
        model: str = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        self.sana_service = SanaVideoService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt.
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for video generation
            duration: Desired video duration in seconds (used to calculate frames)
            output_filename: Custom output filename (auto-generated if not provided)
            **kwargs: Additional parameters passed to service
            
        Returns:
            Dictionary with success, video_path, error, duration_seconds,
            generation_id, model, status, and metadata fields
        """
        start_time = time.time()
        
        # Extract parameters from kwargs
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
        
        if not output_filename:
            timestamp = int(time.time())
            safe_model = self.model.replace("/", "-").replace("_", "-")
            output_filename = f"sana_{safe_model}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        result = self.sana_service.generate_video(
            image_path=str(image_path),
            text_prompt=text_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            output_path=output_path,
            **kwargs
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": f"sana_{int(time.time())}",
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "num_frames": result.get("num_frames"),
                "fps": result.get("fps"),
                "sana_result": result
            }
        }

