#!/usr/bin/env python3
"""
Pilot Experiment: Test 6 representative models on ALL human-approved VMEvalKit tasks with parallel execution.

This script runs inference on ALL existing task pairs from each domain (chess, maze, raven, rotation)
that have been approved by human curators (i.e., folders still exist).

Models tested:
- Luma Dream Machine: luma-ray-2
- Google Veo 3.0: veo-3.0-generate  
- Google Veo 3.1 (via WaveSpeed): veo-3.1-720p
- Runway ML: runway-gen4-turbo
- OpenAI Sora: openai-sora-2
- WaveSpeed WAN 2.2: wavespeed-wan-2.2-i2v-720p

Total: ALL approved tasks × 6 models = comprehensive evaluation

Human Curation: Only tasks with existing folders are processed (deleted folders = rejected tasks)

Requirements:
- All necessary API keys configured in environment
- venv activated
- Output directory: ./data/outputs/pilot_experiment/
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.runner.inference import (
    run_inference, 
    AVAILABLE_MODELS, 
    MODEL_FAMILIES,
    InferenceRunner
)


# ========================================
# PILOT EXPERIMENT CONFIGURATION
# ========================================

# Test 6 models including the new Veo 3.1
PILOT_MODELS = {
    "luma-ray-2": "Luma Dream Machine",
    "veo-3.0-generate": "Google Veo 3.0",
    "veo-3.1-720p": "Google Veo 3.1 (via WaveSpeed)",
    "runway-gen4-turbo": "Runway ML",
    "openai-sora-2": "OpenAI Sora",
    "wavespeed-wan-2.2-i2v-720p": "WaveSpeed WAN 2.2",
}

# Questions directory path
QUESTIONS_DIR = Path("data/questions")

# Output directory
OUTPUT_DIR = Path("data/outputs/pilot_experiment")

# Expected domains (for validation)
EXPECTED_DOMAINS = ["chess", "maze", "raven", "rotation"]


# ========================================
# FOLDER-BASED TASK DISCOVERY
# ========================================

def discover_all_tasks_from_folders(questions_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover all human-approved tasks by scanning folder structure directly.
    
    Only tasks with existing folders are processed (deleted folders = rejected tasks).
    Loads metadata from individual question_metadata.json files.
    
    Args:
        questions_dir: Path to questions directory
        
    Returns:
        Dictionary mapping domain to list of task dictionaries
    """
    print(f"🔍 Discovering tasks from folder structure: {questions_dir}")
    
    tasks_by_domain = {}
    total_tasks = 0
    
    # Scan each domain folder
    for domain_dir in sorted(questions_dir.glob("*_task")):
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name.replace("_task", "")
        domain_tasks = []
        
        print(f"   📁 Scanning {domain_dir.name}/")
        
        # Scan each question folder in this domain  
        for question_dir in sorted(domain_dir.glob(f"{domain}_*")):
            if not question_dir.is_dir():
                continue
                
            task_id = question_dir.name
            
            # Check for required files
            metadata_file = question_dir / "question_metadata.json"
            prompt_file = question_dir / "prompt.txt"
            first_image = question_dir / "first_frame.png"
            final_image = question_dir / "final_frame.png"
            
            if not all([metadata_file.exists(), prompt_file.exists(), first_image.exists()]):
                print(f"      ⚠️  Skipping {task_id}: Missing required files")
                continue
            
            # Load metadata from question_metadata.json
            with open(metadata_file, 'r') as f:
                task_metadata = json.load(f)
            
            # Load prompt directly from prompt.txt
            prompt_text = prompt_file.read_text().strip()
            
            # Create standardized task dictionary
            task = {
                "id": task_id,
                "domain": domain,
                "prompt": prompt_text,
                "first_image_path": str(first_image),
                "final_image_path": str(final_image) if final_image.exists() else None,
                # Include original metadata
                **task_metadata
            }
            
            domain_tasks.append(task)
            
        print(f"      ✅ Found {len(domain_tasks)} approved tasks in {domain}")
        tasks_by_domain[domain] = domain_tasks
        total_tasks += len(domain_tasks)
    
    print(f"\n📊 Discovery Summary:")
    print(f"   Total approved tasks: {total_tasks}")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} tasks")
    
    return tasks_by_domain


# ========================================
# INFERENCE EXECUTION
# ========================================

def create_output_structure(base_dir: Path) -> None:
    """Create organized output directory structure for the new system."""
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # With the new structured output, each inference creates its own folder
    # We only need the logs directory at the top level
    (base_dir / "logs").mkdir(exist_ok=True, parents=True)
    
    print(f"📁 Output directory structure ready at: {base_dir}")
    print(f"   Each inference will create a self-contained folder with:")
    print(f"   - video/: Generated video file")
    print(f"   - question/: Input images and prompt")
    print(f"   - metadata.json: Complete inference metadata")


def _ensure_real_png(image_path: str) -> bool:
    """If file is SVG mislabeled as .png, convert to real PNG in-place using CairoSVG."""
    try:
        # Quick check by trying to open as PNG
        Image.open(image_path).verify()
        return True
    except Exception:
        # Fallback: detect SVG text and convert
        try:
            with open(image_path, 'rb') as f:
                head = f.read(1024)
            # Heuristic: look for '<svg' in the head bytes
            if b"<svg" in head.lower():
                import cairosvg
                with open(image_path, 'rb') as f:
                    svg_bytes = f.read()
                cairosvg.svg2png(bytestring=svg_bytes, write_to=image_path)
                # Validate conversion
                Image.open(image_path).verify()
                print(f"   🔧 Converted SVG→PNG in-place: {image_path}")
                return True
        except Exception as e:
            print(f"   ⚠️  Image fix failed for {image_path}: {e}")
    return False


def run_single_inference(
    model_name: str,
    task: Dict[str, Any],
    category: str,
    output_dir: Path,
    runner: Optional[InferenceRunner] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference for a single task-model pair using the new structured output system.
    
    Args:
        model_name: Name of the model to use
        task: Task dictionary from dataset
        category: Task category
        output_dir: Base output directory
        runner: Optional InferenceRunner instance (created if not provided)
        **kwargs: Additional model parameters
        
    Returns:
        Result dictionary with metadata
    """
    task_id = task["id"]
    image_path = task["first_image_path"]
    prompt = task["prompt"]
    
    # Create a unique run_id for this inference
    run_id = f"{model_name}_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n  🎬 Generating: {task_id} with {model_name}")
    print(f"     Image: {image_path}")
    print(f"     Prompt: {prompt[:80]}...")
    
    start_time = datetime.now()
    
    try:
        # Check if image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        # Validate image is a real PNG (auto-fix if it's actually SVG)
        if not _ensure_real_png(image_path):
            raise ValueError(f"Input image invalid or corrupt: {image_path}")
        
        # Use InferenceRunner for structured output
        if runner is None:
            runner = InferenceRunner(output_dir=str(output_dir))
        
        # Run inference with complete question data for structured output
        result = runner.run(
            model_name=model_name,
            image_path=image_path,
            text_prompt=prompt,
            run_id=run_id,
            question_data=task,  # Pass full task data for structured output
            **kwargs
        )
        
        # Add metadata
        result.update({
            "task_id": task_id,
            "category": category,
            "model_name": model_name,
            "model_family": PILOT_MODELS[model_name],
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "success": result.get("status") != "failed"
        })
        
        if result.get("status") != "failed":
            print(f"     ✅ Success! Structured output saved to: {result.get('inference_dir', 'N/A')}")
        else:
            print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed: {str(e)}"
        print(f"     ❌ {error_msg}")
        
        return {
            "task_id": task_id,
            "category": category,
            "model_name": model_name,
            "model_family": PILOT_MODELS[model_name],
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def run_pilot_experiment(
    tasks_by_domain: Dict[str, List[Dict[str, Any]]],
    models: Dict[str, str],
    output_dir: Path,
    skip_existing: bool = True,
    max_workers: int = 6  # Parallel workers (one per model)
) -> Dict[str, Any]:
    """
    Run full pilot experiment with PARALLEL execution on ALL human-approved tasks.
    
    Args:
        tasks_by_domain: Dictionary mapping domain to task lists
        models: Dictionary of model names to test
        output_dir: Base output directory
        skip_existing: Skip tasks that already have outputs
        max_workers: Maximum parallel workers for ThreadPoolExecutor
        
    Returns:
        Dictionary with all results and statistics
    """
    print("=" * 80)
    print("🚀 VMEVAL KIT COMPREHENSIVE EVALUATION (ALL APPROVED TASKS)")
    print("=" * 80)
    print(f"\n📊 Experiment Configuration:")
    print(f"   Models: {len(models)}")
    print(f"   Domains: {len(tasks_by_domain)}")
    print(f"   🔄 Parallel Workers: {max_workers}")
    
    # Calculate totals
    total_tasks = sum(len(tasks) for tasks in tasks_by_domain.values())
    total_generations = total_tasks * len(models)
    
    print(f"\n📈 Task Distribution:")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} approved tasks")
    print(f"   Total approved tasks: {total_tasks}")
    print(f"   Total generations: {total_generations}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Skip existing: {skip_existing}\n")
    
    # Create output structure
    create_output_structure(output_dir)
    
    # Thread-safe results storage
    all_results = []
    results_lock = threading.Lock()
    
    statistics = {
        "total_tasks": total_tasks,
        "total_generations": total_generations,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "by_model": {},
        "by_domain": {}
    }
    stats_lock = threading.Lock()
    
    # Initialize statistics
    for model in models.keys():
        statistics["by_model"][model] = {"completed": 0, "failed": 0, "skipped": 0}
    for domain in tasks_by_domain.keys():
        statistics["by_domain"][domain] = {"completed": 0, "failed": 0, "skipped": 0}
    
    experiment_start = datetime.now()
    
    # Create all inference jobs
    inference_jobs = []
    for domain, tasks in tasks_by_domain.items():
        for task in tasks:
            for model_name in models.keys():
                inference_jobs.append({
                    "model_name": model_name,
                    "task": task,
                    "domain": domain
                })
    
    print(f"📋 Created {len(inference_jobs)} inference jobs")
    print("🚀 Starting parallel execution...\n")
    
    # Create a shared InferenceRunner instance
    runner = InferenceRunner(output_dir=str(output_dir))
    
    # Function to process a single job
    def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
        model_name = job["model_name"]
        task = job["task"]
        domain = job["domain"]
        task_id = task["id"]
        
        # With new structure, check if inference folder already exists
        run_id = f"{model_name}_{task_id}_*"
        existing_dirs = list(output_dir.glob(run_id))
        
        if skip_existing and existing_dirs:
            with stats_lock:
                statistics["skipped"] += 1
                statistics["by_model"][model_name]["skipped"] += 1
                statistics["by_domain"][domain]["skipped"] += 1
            return {
                "task_id": task_id,
                "model_name": model_name,
                "status": "skipped",
                "existing_dir": str(existing_dirs[0])
            }
        
        # Run inference with structured output
        result = run_single_inference(
            model_name=model_name,
            task=task,
            category=domain,  # Pass domain as category for backward compatibility
            output_dir=output_dir,
            runner=runner  # Pass the shared runner instance
        )
        
        # Update statistics and results (thread-safe)
        with results_lock:
            all_results.append(result)
        
        with stats_lock:
            if result["success"]:
                statistics["completed"] += 1
                statistics["by_model"][model_name]["completed"] += 1
                statistics["by_domain"][domain]["completed"] += 1
            else:
                statistics["failed"] += 1
                statistics["by_model"][model_name]["failed"] += 1
                statistics["by_domain"][domain]["failed"] += 1
            
            # Save intermediate results periodically
            if (statistics["completed"] + statistics["failed"]) % 5 == 0:
                save_results(all_results.copy(), statistics.copy(), output_dir, intermediate=True)
        
        return result
    
    # Execute jobs in parallel
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(process_job, job): job for job in inference_jobs}
        
        # Process completed jobs
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed_count += 1
            
            try:
                result = future.result()
                status = result.get("status", "completed" if result.get("success") else "failed")
                print(f"[{completed_count}/{len(inference_jobs)}] {job['task']['id']} × {job['model_name']}: {status}")
            except Exception as exc:
                print(f"[{completed_count}/{len(inference_jobs)}] {job['task']['id']} × {job['model_name']}: ERROR - {exc}")
                with stats_lock:
                    statistics["failed"] += 1
                    statistics["by_model"][job["model_name"]]["failed"] += 1
                    statistics["by_domain"][job["domain"]]["failed"] += 1
    
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    # Final statistics
    statistics["experiment_start"] = experiment_start.isoformat()
    statistics["experiment_end"] = experiment_end.isoformat()
    statistics["duration_seconds"] = duration
    statistics["duration_formatted"] = format_duration(duration)
    
    print(f"\n⚡ Parallel execution completed in {format_duration(duration)}")
    print(f"   Sequential estimate would be: ~{format_duration(duration * max_workers)}")
    
    return {
        "results": all_results,
        "statistics": statistics
    }


# ========================================
# RESULTS MANAGEMENT
# ========================================

def save_results(
    results: List[Dict[str, Any]],
    statistics: Dict[str, Any],
    output_dir: Path,
    intermediate: bool = False
) -> None:
    """Save results and statistics to JSON files."""
    results_dir = output_dir / "logs"
    
    # Save detailed results
    if intermediate:
        results_file = results_dir / "logs_intermediate.json"
    else:
        results_file = results_dir / "logs_final.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save statistics
    stats_file = results_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    if not intermediate:
        print(f"\n✅ Logs saved to: {results_file}")
        print(f"✅ Statistics saved to: {stats_file}")


def generate_summary_report(
    statistics: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate human-readable summary report."""
    report_file = output_dir / "logs" / "SUMMARY.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VMEVAL KIT PILOT EXPERIMENT - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment Date: {statistics['experiment_start']}\n")
        f.write(f"Duration: {statistics['duration_formatted']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Generations: {statistics['total_generations']}\n")
        f.write(f"Completed: {statistics['completed']} ({statistics['completed']/statistics['total_generations']*100:.1f}%)\n")
        f.write(f"Failed: {statistics['failed']} ({statistics['failed']/statistics['total_generations']*100:.1f}%)\n")
        f.write(f"Skipped: {statistics['skipped']} ({statistics['skipped']/statistics['total_generations']*100:.1f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS BY MODEL FAMILY\n")
        f.write("=" * 80 + "\n")
        for model, stats in statistics['by_model'].items():
            f.write(f"\n{model}:\n")
            f.write(f"  Completed: {stats['completed']}\n")
            f.write(f"  Failed: {stats['failed']}\n")
            f.write(f"  Skipped: {stats['skipped']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RESULTS BY DOMAIN\n")
        f.write("=" * 80 + "\n")
        for domain, stats in statistics['by_domain'].items():
            f.write(f"\n{domain.upper()}:\n")
            f.write(f"  Completed: {stats['completed']}\n")
            f.write(f"  Failed: {stats['failed']}\n")
            f.write(f"  Skipped: {stats['skipped']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✅ Summary report saved to: {report_file}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution function."""
    print("🔍 Discovering human-approved tasks from folder structure...")
    
    # Check if questions directory exists
    if not QUESTIONS_DIR.exists():
        print(f"❌ Questions directory not found at: {QUESTIONS_DIR}")
        print("   Please ensure the questions directory exists with task folders.")
        sys.exit(1)
    
    # Discover all approved tasks from folders
    tasks_by_domain = discover_all_tasks_from_folders(QUESTIONS_DIR)
    
    # Verify models are available
    print(f"\n🔍 Verifying {len(PILOT_MODELS)} models for parallel testing...")
    for model_name, family in PILOT_MODELS.items():
        if model_name in AVAILABLE_MODELS:
            print(f"   ✅ {model_name}: {family}")
        else:
            print(f"   ❌ {model_name}: NOT FOUND in available models")
            print(f"      Please check model name or add it to AVAILABLE_MODELS")
            # Don't exit, just warn - some models might not be configured yet
    
    # Check for API keys (warn if missing)
    print(f"\n🔑 Checking API keys...")
    api_keys = {
        "LUMA_API_KEY": os.getenv("LUMA_API_KEY"),
        "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID") or os.getenv("PROJECT_ID"),
        "WAVESPEED_API_KEY": os.getenv("WAVESPEED_API_KEY"),
        "RUNWAYML_API_SECRET": os.getenv("RUNWAYML_API_SECRET"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }
    
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"   ✅ {key_name}: configured")
        else:
            print(f"   ⚠️  {key_name}: not configured (related models may fail)")
    
    print(f"\n{'=' * 80}")
    input("Press ENTER to start the pilot experiment (or Ctrl+C to cancel)...")
    
    # Verify we found tasks
    if not tasks_by_domain or sum(len(tasks) for tasks in tasks_by_domain.values()) == 0:
        print("❌ No approved tasks found. Please check the questions directory structure.")
        sys.exit(1)
    
    # Run experiment
    experiment_results = run_pilot_experiment(
        tasks_by_domain=tasks_by_domain,
        models=PILOT_MODELS,
        output_dir=OUTPUT_DIR,
        skip_existing=True
    )
    
    # Save final results
    print(f"\n{'=' * 80}")
    print("💾 Saving final results...")
    save_results(
        results=experiment_results["results"],
        statistics=experiment_results["statistics"],
        output_dir=OUTPUT_DIR,
        intermediate=False
    )
    
    # Generate summary report
    generate_summary_report(
        statistics=experiment_results["statistics"],
        output_dir=OUTPUT_DIR
    )
    
    # Print final summary
    print(f"\n{'=' * 80}")
    print("🎉 COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"{'=' * 80}")
    stats = experiment_results["statistics"]
    print(f"\n📊 Final Statistics:")
    print(f"   Models tested: {len(PILOT_MODELS)}")
    print(f"   Approved tasks processed: {stats['total_tasks']}")
    print(f"   Total generations: {stats['total_generations']}")
    print(f"   Completed: {stats['completed']} ({stats['completed']/max(stats['total_generations'],1)*100:.1f}%)")
    print(f"   Failed: {stats['failed']} ({stats['failed']/max(stats['total_generations'],1)*100:.1f}%)")
    print(f"   Skipped: {stats['skipped']} ({stats['skipped']/max(stats['total_generations'],1)*100:.1f}%)")
    print(f"   ⏱️ Duration: {stats['duration_formatted']}")
    
    print(f"\n🎯 Results by Domain:")
    for domain, domain_stats in stats['by_domain'].items():
        domain_total = domain_stats['completed'] + domain_stats['failed'] + domain_stats['skipped']
        print(f"   {domain.title()}: {domain_stats['completed']}/{domain_total} completed")
    
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error:")
        print(f"   {str(e)}")
        traceback.print_exc()
        sys.exit(1)
