"""
Data loader utilities for VMEvalKit web dashboard.
Scans and parses the structured output folders.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def scan_all_outputs(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan all output folders and collect inference results.
    
    Args:
        output_dir: Base output directory (data/outputs)
        
    Returns:
        List of inference result dictionaries
    """
    results = []
    
    if not output_dir.exists():
        return results
    
    # Walk through the output directory structure
    # Structure: output_dir/{model}/{domain}_task/{task_id}/{run_id}/
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Skip special directories
        if model_name in ['logs', 'checkpoints', '.git', '__pycache__']:
            continue
        
        # Look for domain task directories
        for domain_dir in model_dir.iterdir():
            if not domain_dir.is_dir() or not domain_dir.name.endswith('_task'):
                continue
            
            domain = domain_dir.name.replace('_task', '')
            
            # Look for task directories
            for task_dir in domain_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                task_id = task_dir.name
                
                # Look for run directories
                for run_dir in task_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    
                    # Load metadata
                    metadata_file = run_dir / 'metadata.json'
                    if metadata_file.exists():
                        result = load_inference_metadata(run_dir, metadata_file)
                        if result:
                            result['model'] = model_name
                            result['domain'] = domain
                            result['task_id'] = task_id
                            result['run_id'] = run_dir.name
                            result['inference_dir'] = str(run_dir)
                            results.append(result)
    
    # Sort by timestamp (most recent first)
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return results


def load_inference_metadata(run_dir: Path, metadata_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a single inference folder.
    
    Args:
        run_dir: Path to the inference run directory
        metadata_file: Path to the metadata.json file
        
    Returns:
        Dictionary with inference metadata
    """
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract key information
        inference_info = metadata.get('inference', {})
        input_info = metadata.get('input', {})
        output_info = metadata.get('output', {})
        question_data = metadata.get('question_data', {})
        
        # Find video file
        video_dir = run_dir / 'video'
        video_path = None
        if video_dir.exists():
            video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.webm'))
            if video_files:
                video_path = str(video_files[0])
        
        # Find question images
        question_dir = run_dir / 'question'
        first_frame = None
        final_frame = None
        prompt = None
        
        if question_dir.exists():
            first_frame_file = question_dir / 'first_frame.png'
            final_frame_file = question_dir / 'final_frame.png'
            prompt_file = question_dir / 'prompt.txt'
            
            if first_frame_file.exists():
                first_frame = str(first_frame_file)
            if final_frame_file.exists():
                final_frame = str(final_frame_file)
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    prompt = f.read().strip()
        
        return {
            'run_id': inference_info.get('run_id'),
            'model': inference_info.get('model'),
            'timestamp': inference_info.get('timestamp'),
            'status': inference_info.get('status'),
            'success': inference_info.get('status') == 'success',
            'duration_seconds': inference_info.get('duration_seconds'),
            'error': inference_info.get('error'),
            'prompt': prompt or input_info.get('prompt'),
            'video_path': video_path or output_info.get('video_path'),
            'first_frame': first_frame,
            'final_frame': final_frame,
            'question_metadata': question_data,
            'full_metadata': metadata
        }
    except Exception as e:
        print(f"Error loading metadata from {metadata_file}: {e}")
        return None


def get_model_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics per model.
    
    Args:
        results: List of inference results
        
    Returns:
        Dictionary with model statistics
    """
    model_stats = {}
    
    for result in results:
        model = result.get('model', 'unknown')
        if model not in model_stats:
            model_stats[model] = {
                'total': 0,
                'success': 0,
                'failed': 0,
                'total_duration': 0,
                'domains': set()
            }
        
        model_stats[model]['total'] += 1
        if result.get('success', False):
            model_stats[model]['success'] += 1
        else:
            model_stats[model]['failed'] += 1
        
        duration = result.get('duration_seconds', 0)
        if duration:
            model_stats[model]['total_duration'] += duration
        
        domain = result.get('domain')
        if domain:
            model_stats[model]['domains'].add(domain)
    
    # Calculate derived statistics
    for model, stats in model_stats.items():
        stats['success_rate'] = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        stats['avg_duration'] = stats['total_duration'] / stats['total'] if stats['total'] > 0 else 0
        stats['domains'] = list(stats['domains'])  # Convert set to list for JSON serialization
    
    return model_stats


def get_domain_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics per domain.
    
    Args:
        results: List of inference results
        
    Returns:
        Dictionary with domain statistics
    """
    domain_stats = {}
    
    for result in results:
        domain = result.get('domain', 'unknown')
        if domain not in domain_stats:
            domain_stats[domain] = {
                'total': 0,
                'success': 0,
                'failed': 0,
                'models': set()
            }
        
        domain_stats[domain]['total'] += 1
        if result.get('success', False):
            domain_stats[domain]['success'] += 1
        else:
            domain_stats[domain]['failed'] += 1
        
        model = result.get('model')
        if model:
            domain_stats[domain]['models'].add(model)
    
    # Calculate derived statistics
    for domain, stats in domain_stats.items():
        stats['success_rate'] = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        stats['models'] = list(stats['models'])  # Convert set to list
    
    return domain_stats


def get_inference_details(output_dir: Path, inference_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information for a specific inference.
    
    Args:
        output_dir: Base output directory
        inference_id: Inference ID (path relative to output_dir)
        
    Returns:
        Dictionary with detailed inference information
    """
    inference_path = output_dir / inference_id
    metadata_file = inference_path / 'metadata.json'
    
    if not metadata_file.exists():
        return None
    
    return load_inference_metadata(inference_path, metadata_file)


def get_comparison_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepare data for model/task comparison views.
    
    Args:
        results: List of inference results
        
    Returns:
        Dictionary with comparison data
    """
    # Group by task_id and model
    task_model_grid = {}
    
    for result in results:
        task_id = result.get('task_id', 'unknown')
        model = result.get('model', 'unknown')
        
        if task_id not in task_model_grid:
            task_model_grid[task_id] = {}
        
        task_model_grid[task_id][model] = {
            'success': result.get('success', False),
            'duration': result.get('duration_seconds', 0),
            'video_path': result.get('video_path'),
            'inference_dir': result.get('inference_dir')
        }
    
    # Get unique models and tasks
    all_models = sorted(set(r.get('model', 'unknown') for r in results))
    all_tasks = sorted(task_model_grid.keys())
    
    return {
        'task_model_grid': task_model_grid,
        'models': all_models,
        'tasks': all_tasks
    }

