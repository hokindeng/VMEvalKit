"""
Batch inference runner for processing multiple tasks.

NO EVALUATION - just batch inference.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

from .runner import InferenceRunner


class BatchInferenceRunner:
    """
    Run inference on multiple tasks in batch.
    
    Simple batch processing - no evaluation logic.
    """
    
    def __init__(self, output_dir: str = "./outputs", max_workers: int = 1):
        """
        Initialize batch runner.
        
        Args:
            output_dir: Directory for outputs
            max_workers: Maximum parallel workers (1 = sequential)
        """
        self.runner = InferenceRunner(output_dir)
        self.max_workers = max_workers
        self.batch_results_dir = Path(output_dir) / "batch_results"
        self.batch_results_dir.mkdir(exist_ok=True, parents=True)
    
    def run_dataset(
        self,
        model_name: str,
        dataset_path: Union[str, Path],
        api_key: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        max_tasks: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on all tasks in a dataset.
        
        Args:
            model_name: Model to use
            dataset_path: Path to dataset JSON file
            api_key: Optional API key
            task_ids: Optional list of specific task IDs to run
            max_tasks: Maximum number of tasks to process
            **kwargs: Additional parameters for inference
            
        Returns:
            Batch results summary
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        pairs = dataset.get("pairs", [])
        if not pairs:
            raise ValueError(f"No pairs found in dataset: {dataset_path}")
        
        # Filter tasks if requested
        if task_ids:
            pairs = [p for p in pairs if p.get("id") in task_ids]
        
        # Limit tasks if requested
        if max_tasks:
            pairs = pairs[:max_tasks]
        
        print(f"\n📦 Batch Inference: {model_name}")
        print(f"   Dataset: {dataset_path}")
        print(f"   Tasks to process: {len(pairs)}")
        print(f"   Workers: {self.max_workers}")
        
        batch_id = f"batch_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = []
        
        # Run inference on each task
        if self.max_workers == 1:
            # Sequential processing
            for task in tqdm(pairs, desc="Processing tasks"):
                result = self.runner.run_from_task(
                    model_name=model_name,
                    task_data=task,
                    api_key=api_key,
                    **kwargs
                )
                results.append(result)
        else:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for task in pairs:
                    future = executor.submit(
                        self.runner.run_from_task,
                        model_name=model_name,
                        task_data=task,
                        api_key=api_key,
                        **kwargs
                    )
                    futures.append(future)
                
                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                 total=len(futures), desc="Processing tasks"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Task failed: {e}")
                        results.append({"status": "failed", "error": str(e)})
        
        # Create batch summary
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        
        batch_summary = {
            "batch_id": batch_id,
            "model": model_name,
            "dataset": str(dataset_path),
            "total_tasks": len(pairs),
            "successful": len(successful),
            "failed": len(failed),
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        # Save batch results
        batch_file = self.batch_results_dir / f"{batch_id}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"\n✅ Batch inference complete!")
        print(f"   Successful: {len(successful)}/{len(pairs)}")
        print(f"   Failed: {len(failed)}")
        print(f"   Results saved: {batch_file}")
        
        return batch_summary
    
    def run_models_comparison(
        self,
        model_names: List[str],
        dataset_path: Union[str, Path],
        api_keys: Optional[Dict[str, str]] = None,
        task_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multiple models on the same dataset for comparison.
        
        Args:
            model_names: List of models to run
            dataset_path: Path to dataset
            api_keys: Optional dict of api keys per model
            task_ids: Optional specific tasks to run
            **kwargs: Additional parameters
            
        Returns:
            Comparison results
        """
        api_keys = api_keys or {}
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🔬 Model Comparison")
        print(f"   Models: {model_names}")
        print(f"   Dataset: {dataset_path}")
        
        model_results = {}
        
        for model_name in model_names:
            print(f"\n--- Running {model_name} ---")
            api_key = api_keys.get(model_name)
            
            try:
                batch_result = self.run_dataset(
                    model_name=model_name,
                    dataset_path=dataset_path,
                    api_key=api_key,
                    task_ids=task_ids,
                    **kwargs
                )
                model_results[model_name] = batch_result
            except Exception as e:
                print(f"Failed to run {model_name}: {e}")
                model_results[model_name] = {"status": "failed", "error": str(e)}
        
        # Create comparison summary
        comparison = {
            "comparison_id": comparison_id,
            "models": model_names,
            "dataset": str(dataset_path),
            "timestamp": datetime.now().isoformat(),
            "model_results": model_results
        }
        
        # Save comparison
        comparison_file = self.batch_results_dir / f"{comparison_id}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n📊 Comparison complete!")
        print(f"   Results saved: {comparison_file}")
        
        return comparison
