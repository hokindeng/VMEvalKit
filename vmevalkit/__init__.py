"""
VMEvalKit - A comprehensive evaluation framework for video reasoning models.

This framework evaluates video generation models on reasoning tasks requiring
both text prompts and image inputs to generate solution videos.
"""

__version__ = "0.1.0"

from .core.evaluator import VMEvaluator
from .core.task_loader import TaskLoader
from .core.model_registry import ModelRegistry
from .inference import InferenceRunner, BatchInferenceRunner

__all__ = [
    "VMEvaluator",
    "TaskLoader", 
    "ModelRegistry",
    "InferenceRunner",
    "BatchInferenceRunner",
    "__version__"
]
