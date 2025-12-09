import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

import torch
from PIL import Image
from diffusers import SanaImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from .base import ModelWrapper

logger = logging.getLogger(__name__)
# requires diffuers>=0.36.0
# takes 22GB vram, 4 mins with single RTX A6000

class SanaService:
    def __init__(self, model: str = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"):
        self.model_id = model
        self.pipe = None
        self.device = None

    def _load_model(self):
        if self.pipe is not None:
            return

        if torch.cuda.is_available():
            self.device = "cuda"
            transformer_dtype = torch.bfloat16
            encoder_dtype = torch.bfloat16
            vae_dtype = torch.float32
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

        logger.info(f"SANA model loaded on {self.device}")

    def _prepare_image(self, image_path: Union[str, Path]) -> Image.Image:
        image = load_image(str(image_path))

        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.info(f"Prepared image for SANA: {image.size}")
        return image

    def generate_video(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        negative_prompt: str = "",
        motion_score: int = 30,
        frames: int = 81,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        height: int = 480,
        width: int = 832,
        seed: int = 42,
        fps: int = 16,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()

        self._load_model()
        image = self._prepare_image(image_path)

        motion_prompt = f" motion score: {motion_score}."
        composed_prompt = text_prompt + motion_prompt

        generator = torch.Generator(device=self.device).manual_seed(seed)

        frames_output = self.pipe(
            image=image,
            prompt=composed_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            frames=frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]

        video_path = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames_output, str(output_path), fps=fps)
            video_path = str(output_path)
            logger.info(f"SANA video saved to: {video_path}")

        duration_taken = time.time() - start_time

        return {
            "video_path": video_path,
            "frames": frames_output,
            "num_frames": frames,
            "fps": fps,
            "duration_seconds": duration_taken,
            "model": self.model_id,
            "status": "success" if video_path else "completed",
            "metadata": {
                "motion_score": motion_score,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "seed": seed,
            },
        }


class SanaWrapper(ModelWrapper):
    def __init__(
        self,
        model: str = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.sana_service = SanaService(model=model)

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()

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

        if not output_filename:
            timestamp = int(time.time())
            safe_model = self.model.replace("/", "-").replace("_", "-")
            output_filename = f"sana_{safe_model}_{timestamp}.mp4"

        output_path = self.output_dir / output_filename

        result = self.sana_service.generate_video(
            image_path=str(image_path),
            text_prompt=text_prompt,
            output_path=output_path,
            **kwargs,
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
                "sana_result": result,
            },
        }
