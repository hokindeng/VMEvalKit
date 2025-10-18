"""
Data loader utilities for VMEvalKit web dashboard.
Scans folders and reads prompt.txt files ONLY - NO JSON.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def scan_all_outputs(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan output folders - NO JSON reading.
    
    Structure: output_dir/{model}/{domain}_task/{task_id}/{run_id}/
    
    Deduplicates: If multiple runs exist for the same (model, domain, task_id),
    only keeps the most recent one.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        List of inference result dictionaries (deduplicated)
    """
    all_results = []
    
    if not output_dir.exists():
        print(f"Warning: Output directory does not exist: {output_dir}")
        return all_results
    
    # Scan: {model}/{domain}_task/{task_id}/{run_id}/
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
                
                # Look for run directories and find the most recent one
                run_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
                
                if not run_dirs:
                    continue
                
                # Sort by modification time, most recent first
                run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                
                # Only use the most recent run (deduplication)
                most_recent_run = run_dirs[0]
                
                # Load data from filesystem only
                result = load_from_filesystem(most_recent_run, model_name, domain, task_id)
                if result:
                    all_results.append(result)
                    
                    # Log if there were duplicates
                    if len(run_dirs) > 1:
                        print(f"Note: Found {len(run_dirs)} runs for {model_name}/{domain}/{task_id}, using most recent: {most_recent_run.name}")
    
    # Sort by folder modification time (most recent first)
    all_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    print(f"Found {len(all_results)} unique inference results (deduplicated)")
    return all_results


def load_from_filesystem(run_dir: Path, model_name: str, domain: str, task_id: str) -> Optional[Dict[str, Any]]:
    """
    Load data from filesystem ONLY - no JSON reading.
    
    Reads:
    - Folder names for model/domain/task
    - prompt.txt for prompt text
    - *.png files for images
    - *.mp4 files for videos
    
    Args:
        run_dir: Path to the run directory
        model_name: Model name from folder
        domain: Domain from folder
        task_id: Task ID from folder
        
    Returns:
        Dictionary with inference data
    """
    try:
        # Read prompt from prompt.txt
        prompt = ""
        prompt_file = run_dir / 'question' / 'prompt.txt'
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                prompt = f.read().strip()
        
        # Find first frame image
        first_frame = None
        question_dir = run_dir / 'question'
        if question_dir.exists():
            first_frame_file = question_dir / 'first_frame.png'
            if first_frame_file.exists():
                first_frame = str(first_frame_file)
        
        # Find final frame image
        final_frame = None
        if question_dir.exists():
            final_frame_file = question_dir / 'final_frame.png'
            if final_frame_file.exists():
                final_frame = str(final_frame_file)
        
        # Find video file
        video_path = None
        video_dir = run_dir / 'video'
        if video_dir.exists():
            video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.webm'))
            if video_files:
                video_path = str(video_files[0])
        
        # Get timestamp from folder modification time
        timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()
        
        return {
            'run_id': run_dir.name,
            'model': model_name,
            'domain': domain,
            'task_id': task_id,
            'timestamp': timestamp,
            'prompt': prompt,
            'video_path': video_path,
            'first_frame': first_frame,
            'final_frame': final_frame,
            'inference_dir': str(run_dir)
        }
    except Exception as e:
        print(f"Error loading from {run_dir}: {e}")
        import traceback
        traceback.print_exc()
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
                'domains': set()
            }
        
        model_stats[model]['total'] += 1
        
        domain = result.get('domain')
        if domain:
            model_stats[model]['domains'].add(domain)
    
    # Convert sets to lists
    for model, stats in model_stats.items():
        stats['domains'] = list(stats['domains'])
    
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
                'models': set()
            }
        
        domain_stats[domain]['total'] += 1
        
        model = result.get('model')
        if model:
            domain_stats[domain]['models'].add(model)
    
    # Convert sets to lists
    for domain, stats in domain_stats.items():
        stats['models'] = list(stats['models'])
    
    return domain_stats


def get_hierarchical_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Organize results hierarchically: Models → Domains → Tasks
    
    Args:
        results: List of inference results
        
    Returns:
        Nested dictionary: {model: {domain: [tasks]}}
    """
    hierarchy = {}
    
    for result in results:
        model = result.get('model', 'unknown')
        domain = result.get('domain', 'unknown')
        
        if model not in hierarchy:
            hierarchy[model] = {}
        
        if domain not in hierarchy[model]:
            hierarchy[model][domain] = []
        
        hierarchy[model][domain].append(result)
    
    # Sort tasks within each domain by task_id
    for model in hierarchy:
        for domain in hierarchy[model]:
            hierarchy[model][domain].sort(key=lambda x: x.get('task_id', ''))
    
    return hierarchy


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
    
    if not inference_path.exists():
        return None
    
    # Extract model/domain/task from path
    parts = inference_id.split('/')
    if len(parts) >= 3:
        model = parts[0]
        domain = parts[1].replace('_task', '')
        task_id = parts[2]
        return load_from_filesystem(inference_path, model, domain, task_id)
    
    return None


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
