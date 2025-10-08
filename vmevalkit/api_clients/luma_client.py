"""
Luma Dream Machine API client implementation.

Luma Dream Machine supports text prompts with image references for video generation,
making it suitable for VMEvalKit's reasoning tasks.
"""

import os
import time
import base64
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
from io import BytesIO
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
import uuid
import boto3

from ..models.base import BaseVideoModel


class LumaAPIError(Exception):
    """Custom exception for Luma API errors."""
    pass


class LumaDreamMachine(BaseVideoModel):
    """
    Luma Dream Machine API implementation.
    
    This model supports text+image→video generation required for reasoning tasks.
    """
    
    BASE_URL = "https://api.lumalabs.ai/dream-machine/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enhance_prompt: bool = True,
        loop_video: bool = False,
        aspect_ratio: str = "16:9",
        model: str = "ray-2",
        s3_bucket: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Luma Dream Machine client.
        
        Args:
            api_key: Luma API key (or set LUMA_API_KEY env var)
            enhance_prompt: Whether to use Luma's prompt enhancement
            loop_video: Whether to create looping videos
            aspect_ratio: Output aspect ratio
        """
        self.api_key = api_key or os.getenv("LUMA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Luma API key required. Set LUMA_API_KEY env var or pass api_key parameter."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.enhance_prompt = enhance_prompt
        self.loop_video = loop_video
        self.aspect_ratio = aspect_ratio
        self.model = model
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET", "vmevalkit")
        
        super().__init__(name="luma_dream_machine", **kwargs)
    
    def supports_text_image_input(self) -> bool:
        """
        Luma Dream Machine supports both text and image inputs.
        
        Returns:
            True - Luma accepts text prompts with image references
        """
        return True
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        text_prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        resolution: tuple = (1280, 720),
        **kwargs
    ) -> str:
        """
        Generate video from text prompt and image using Luma Dream Machine.
        
        Args:
            image: Input image for reference
            text_prompt: Text instructions for video generation
            duration: Video duration in seconds
            fps: Frames per second (Luma may override this)
            resolution: Output resolution
            **kwargs: Additional parameters
            
        Returns:
            Path to generated video file
        """
        # Ensure public image URL per Luma docs (keyframes requires URL)
        image_url = self._ensure_image_url(image)

        # Map resolution to Luma's accepted strings
        _, height = resolution
        if height >= 2160:
            resolution_str = "4k"
        elif height >= 1080:
            resolution_str = "1080"
        elif height >= 720:
            resolution_str = "720p"
        else:
            resolution_str = "540p"

        payload = {
            "prompt": text_prompt,
            "model": self.model,
            "keyframes": {
                "frame0": {
                    "type": "image",
                    "url": image_url
                }
            },
            "enhance_prompt": self.enhance_prompt,
            "loop": self.loop_video,
            "aspect_ratio": self.aspect_ratio,
            "duration": f"{int(duration)}s",
            "resolution": resolution_str
        }

        # Create generation request
        generation_id = self._start_generation(payload=payload)
        
        # Poll for completion
        video_url = self._poll_for_completion(generation_id)
        
        # Download video
        video_path = self._download_video(video_url, generation_id)
        
        return video_path
    
    def _encode_image(self, image: Image.Image) -> bytes:
        """Encode PIL Image to raw PNG bytes for upload."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return buffered.getvalue()

    def _ensure_image_url(self, image: Union[str, Path, Image.Image]) -> str:
        """Return a URL for the image. If local/image object, upload to S3 and return a presigned URL."""
        if isinstance(image, (str, Path)):
            s = str(image)
            if s.startswith("http://") or s.startswith("https://"):
                return s
            pil = self.preprocess_image(image)
        else:
            pil = self.preprocess_image(image)

        data = self._encode_image(pil)
        key = f"uploads/images/{uuid.uuid4().hex}.png"
        session = boto3.session.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
        )
        s3 = session.client("s3")
        s3.put_object(Bucket=self.s3_bucket, Key=key, Body=data, ContentType="image/png")
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': self.s3_bucket, 'Key': key},
            ExpiresIn=3600
        )
        return url
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _start_generation(
        self,
        payload: Dict[str, Any],
    ) -> str:
        """
        Start video generation job.
        
        Returns:
            Generation ID for polling
        """
        endpoint = f"{self.BASE_URL}/generations"
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            # On error, surface response content for debugging
            if not response.ok:
                try:
                    err_body = response.text
                except Exception:
                    err_body = "<no body>"
                raise LumaAPIError(
                    f"Failed to start generation: HTTP {response.status_code} - {err_body}"
                )
            data = response.json()
            return data.get("id") or data.get("generation_id")
        except requests.exceptions.RequestException as e:
            raise LumaAPIError(f"Failed to start generation: {e}")
    
    def _poll_for_completion(
        self,
        generation_id: str,
        max_wait: int = 600,
        poll_interval: int = 5
    ) -> str:
        """
        Poll for generation completion.
        
        Returns:
            URL of generated video
        """
        endpoint = f"{self.BASE_URL}/generations/{generation_id}"
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(endpoint, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                status = data.get("state")
                
                if status == "completed":
                    return data["video"]["url"]
                elif status == "failed":
                    error_msg = data.get("failure_reason", "Unknown error")
                    raise LumaAPIError(f"Generation failed: {error_msg}")
                
                # Still processing
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                raise LumaAPIError(f"Failed to check status: {e}")
        
        raise LumaAPIError(f"Generation timed out after {max_wait} seconds")
    
    def _download_video(self, video_url: str, generation_id: str) -> str:
        """
        Download generated video.
        
        Returns:
            Path to downloaded video file
        """
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        
        video_path = output_dir / f"luma_{generation_id}.mp4"
        
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(video_path)
            
        except requests.exceptions.RequestException as e:
            raise LumaAPIError(f"Failed to download video: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "supports_text_image": True,
            "api": "Luma Dream Machine",
            "capabilities": {
                "text_prompt": True,
                "image_reference": True,
                "max_duration": 10,  # seconds
                "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4"],
                "enhance_prompt": self.enhance_prompt,
                "loop": self.loop_video
            }
        }
