"""
Clean inference module for video generation.

This module handles ONLY inference - no evaluation logic.
Simple flow: text + image → video model → output video
"""

from .runner import InferenceRunner
from .batch_runner import BatchInferenceRunner

__all__ = ["InferenceRunner", "BatchInferenceRunner"]
