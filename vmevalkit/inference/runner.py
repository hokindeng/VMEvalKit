"""
Simple inference runner for video generation models.

NO EVALUATION - just inference: text + image → video
"""

import os
from pathlib import Path
from typing import Union, Dict, Any, Optional
from datetime import datetime
import json

from ..models.base import BaseVideoModel
from ..core.model_registry import ModelRegistry


class InferenceRunner:
    """
    Simple inference runner that takes text + image and generates video.
    
    This class has NO evaluation logic - it purely runs inference.
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        """
        Initialize inference runner.
        
        Args:
            output_dir: Directory to save generated videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Keep track of runs for logging
        self.runs_log = self.output_dir / "inference_runs.json"
        self.runs = self._load_runs_log()
    
    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        api_key: Optional[str] = None,
        duration: float = 5.0,
        fps: int = 24,
        resolution: tuple = (1280, 720),
        run_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference: text + image → video.
        
        Args:
            model_name: Name of the model to use (e.g., "luma-dream-machine")
            image_path: Path to input image
            text_prompt: Text instructions for video generation
            api_key: Optional API key (uses env vars if not provided)
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Output resolution (width, height)
            run_id: Optional run identifier
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with inference results:
            {
                "video_path": path to generated video,
                "model": model name,
                "prompt": text prompt used,
                "image": input image path,
                "duration": inference duration in seconds,
                "timestamp": when inference was run,
                "run_id": unique identifier for this run
            }
        """
        start_time = datetime.now()
        
        # Generate run ID if not provided
        if not run_id:
            run_id = f"{model_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🚀 Running inference with {model_name}")
        print(f"   Image: {image_path}")
        print(f"   Prompt: {text_prompt}")
        print(f"   Run ID: {run_id}")
        
        try:
            # Load model
            model = ModelRegistry.load_model(
                model_name=model_name,
                api_key=api_key,
                **kwargs
            )
            
            # Run inference
            video_path = model.generate(
                image=image_path,
                text_prompt=text_prompt,
                duration=duration,
                fps=fps,
                resolution=resolution,
                **kwargs
            )
            
            # Calculate duration
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            # Create result
            result = {
                "video_path": str(video_path),
                "model": model_name,
                "prompt": text_prompt,
                "image": str(image_path),
                "duration": duration_seconds,
                "timestamp": start_time.isoformat(),
                "run_id": run_id,
                "status": "success",
                "parameters": {
                    "duration": duration,
                    "fps": fps,
                    "resolution": list(resolution)
                }
            }
            
            print(f"\n✅ Inference complete!")
            print(f"   Video saved: {video_path}")
            print(f"   Time taken: {duration_seconds:.2f}s")
            
        except Exception as e:
            # Log failure
            result = {
                "model": model_name,
                "prompt": text_prompt,
                "image": str(image_path),
                "duration": (datetime.now() - start_time).total_seconds(),
                "timestamp": start_time.isoformat(),
                "run_id": run_id,
                "status": "failed",
                "error": str(e)
            }
            
            print(f"\n❌ Inference failed: {e}")
        
        # Log the run
        self._log_run(result)
        
        return result
    
    def run_from_task(
        self,
        model_name: str,
        task_data: Dict[str, Any],
        api_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference using a task from our generated dataset.
        
        Args:
            model_name: Name of the model to use
            task_data: Task dictionary with 'prompt' and 'first_image_path'
            api_key: Optional API key
            **kwargs: Additional parameters
            
        Returns:
            Inference result dictionary
        """
        prompt = task_data.get("prompt")
        image_path = task_data.get("first_image_path")
        task_id = task_data.get("id", "unknown")
        
        if not prompt or not image_path:
            raise ValueError("Task must have 'prompt' and 'first_image_path'")
        
        return self.run(
            model_name=model_name,
            image_path=image_path,
            text_prompt=prompt,
            api_key=api_key,
            run_id=f"{model_name}_{task_id}",
            **kwargs
        )
    
    def _load_runs_log(self) -> list:
        """Load previous runs from log file."""
        if self.runs_log.exists():
            with open(self.runs_log, 'r') as f:
                return json.load(f)
        return []
    
    def _log_run(self, result: Dict[str, Any]):
        """Log a run result."""
        self.runs.append(result)
        with open(self.runs_log, 'w') as f:
            json.dump(self.runs, f, indent=2)
