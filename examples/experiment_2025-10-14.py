#!/usr/bin/env python3
"""
Quick Test Experiment: Test 6 representative models on 1 task per domain with sequential execution.

This script runs inference on 1 task from each domain (chess, maze, raven, rotation, sudoku)
for rapid testing and validation of all model integrations. Models are processed sequentially
(one model at a time), and for each model, tasks are processed one by one.

Models tested:
- Luma Dream Machine: luma-ray-2
- Google Veo 3.0: veo-3.0-generate  
- Google Veo 3.1 (via WaveSpeed): veo-3.1-720p
- Runway ML: runway-gen4-turbo
- OpenAI Sora: openai-sora-2
- WaveSpeed WAN 2.2: wavespeed-wan-2.2-i2v-720p

Total: 5 tasks × 6 models = 30 quick test generations (sequential)

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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback
from PIL import Image
import threading
import time
import signal
import atexit

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
EXPECTED_DOMAINS = ["chess", "maze", "raven", "rotation", "sudoku"]


# ========================================
# PROGRESS TRACKING AND RESUME MECHANISM
# ========================================

class ProgressTracker:
    """
    Manages experiment progress for resume capability.
    
    Tracks:
    - Completed jobs
    - Failed jobs
    - In-progress jobs (cleaned on resume)
    - Statistics
    """
    
    def __init__(self, output_dir: Path, experiment_id: str = None):
        """
        Initialize progress tracker.
        
        Args:
            output_dir: Base output directory
            experiment_id: Unique experiment identifier (auto-generated if None)
        """
        self.output_dir = output_dir
        self.logs_dir = output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate or use experiment ID
        if experiment_id is None:
            experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_id = experiment_id
        
        # Progress file paths
        self.checkpoint_file = self.logs_dir / f"checkpoint_{experiment_id}.json"
        self.progress_file = self.logs_dir / f"progress_{experiment_id}.json"
        
        # Initialize or load progress
        self.progress = self._load_progress()
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Register cleanup handlers
        atexit.register(self.save_progress)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        # Track if we're resuming
        self.is_resume = bool(self.progress.get("jobs_completed", []))
        
        if self.is_resume:
            print(f"📊 Resuming experiment: {experiment_id}")
            print(f"   Completed: {len(self.progress.get('jobs_completed', []))}")
            print(f"   Failed: {len(self.progress.get('jobs_failed', []))}")
            print(f"   Clearing {len(self.progress.get('jobs_in_progress', []))} interrupted jobs")
            # Clear in-progress jobs (they need to be retried)
            self.progress["jobs_in_progress"] = []
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress or initialize new."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                print(f"✅ Loaded checkpoint from: {self.checkpoint_file}")
                return data
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                # Backup corrupted checkpoint
                backup_file = self.checkpoint_file.with_suffix('.backup.json')
                if self.checkpoint_file.exists():
                    self.checkpoint_file.rename(backup_file)
                    print(f"   Backed up to: {backup_file}")
        
        # Initialize new progress
        return {
            "experiment_id": self.experiment_id,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "jobs_completed": [],
            "jobs_failed": [],
            "jobs_in_progress": [],
            "statistics": {}
        }
    
    def save_progress(self, force: bool = False):
        """Save current progress to checkpoint file."""
        with self.lock:
            try:
                self.progress["last_update"] = datetime.now().isoformat()
                
                # Atomic write (write to temp, then rename)
                temp_file = self.checkpoint_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(self.progress, f, indent=2)
                temp_file.replace(self.checkpoint_file)
                
                # Also save detailed progress log
                with open(self.progress_file, 'w') as f:
                    json.dump(self.progress, f, indent=2)
                    
                if force:
                    print(f"💾 Progress saved to: {self.checkpoint_file}")
                    
            except Exception as e:
                print(f"⚠️  Failed to save progress: {e}")
    
    def _handle_interrupt(self, signum, frame):
        """Handle interruption signals gracefully."""
        print("\n\n⚠️  Interrupt detected! Saving progress...")
        self.save_progress(force=True)
        print("✅ Progress saved. You can resume this experiment later.")
        sys.exit(0)
    
    def job_started(self, job_id: str) -> bool:
        """
        Mark a job as started.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if job should proceed, False if already completed
        """
        with self.lock:
            # Check if already completed
            if job_id in self.progress["jobs_completed"]:
                return False
            
            # Mark as in progress
            if job_id not in self.progress["jobs_in_progress"]:
                self.progress["jobs_in_progress"].append(job_id)
            
            return True
    
    def job_completed(self, job_id: str, result: Dict[str, Any]):
        """Mark a job as completed."""
        with self.lock:
            # Remove from in-progress
            if job_id in self.progress["jobs_in_progress"]:
                self.progress["jobs_in_progress"].remove(job_id)
            
            # Add to completed (avoid duplicates)
            if job_id not in self.progress["jobs_completed"]:
                self.progress["jobs_completed"].append(job_id)
            
            # Save periodically (every 5 completions)
            if len(self.progress["jobs_completed"]) % 5 == 0:
                self.save_progress()
    
    def job_failed(self, job_id: str, error: str):
        """Mark a job as failed."""
        with self.lock:
            # Remove from in-progress
            if job_id in self.progress["jobs_in_progress"]:
                self.progress["jobs_in_progress"].remove(job_id)
            
            # Add to failed with error info
            failed_info = {
                "job_id": job_id,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update or add to failed list
            existing_failed = [j for j in self.progress["jobs_failed"] 
                              if isinstance(j, dict) and j.get("job_id") == job_id]
            if not existing_failed:
                self.progress["jobs_failed"].append(failed_info)
    
    def should_skip_job(self, job_id: str) -> bool:
        """Check if a job should be skipped (already completed)."""
        with self.lock:
            return job_id in self.progress["jobs_completed"]
    
    def get_resume_stats(self) -> Dict[str, int]:
        """Get statistics for resume."""
        with self.lock:
            return {
                "completed": len(self.progress["jobs_completed"]),
                "failed": len(self.progress["jobs_failed"]),
                "in_progress": len(self.progress["jobs_in_progress"])
            }
    
    def update_statistics(self, stats: Dict[str, Any]):
        """Update experiment statistics."""
        with self.lock:
            self.progress["statistics"] = stats
            self.save_progress()


# ========================================
# FOLDER-BASED TASK DISCOVERY
# ========================================

def discover_all_tasks_from_folders(questions_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover all human-approved tasks by direct file detection.
    
    Scans for actual PNG and TXT files, then loads supplemental metadata from JSON.
    This approach is more robust as it relies on actual files rather than metadata.
    
    Args:
        questions_dir: Path to questions directory
        
    Returns:
        Dictionary mapping domain to list of task dictionaries
    """
    print(f"🔍 Discovering tasks by scanning actual files: {questions_dir}")
    
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
            
            # Look for actual files (PNG and TXT) - this is our primary source
            prompt_file = question_dir / "prompt.txt"
            first_image = question_dir / "first_frame.png"
            final_image = question_dir / "final_frame.png"
            
            # Required files check
            if not prompt_file.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing prompt.txt")
                continue
                
            if not first_image.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing first_frame.png")
                continue
            
            # Load prompt directly from actual file
            try:
                prompt_text = prompt_file.read_text().strip()
            except Exception as e:
                print(f"      ⚠️  Skipping {task_id}: Cannot read prompt.txt - {e}")
                continue
            
            # Create task dictionary from actual detected files
            task = {
                "id": task_id,
                "domain": domain,
                "prompt": prompt_text,
                "first_image_path": str(first_image.absolute()),
                "final_image_path": str(final_image.absolute()) if final_image.exists() else None
            }
            
            # Load supplemental metadata from JSON files if they exist
            metadata_file = question_dir / "question_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        supplemental_metadata = json.load(f)
                    
                    # Add supplemental data but don't override core fields
                    for key, value in supplemental_metadata.items():
                        if key not in task:  # Don't override our detected values
                            task[key] = value
                            
                except Exception as e:
                    print(f"      ⚠️  Warning: Could not load metadata for {task_id} - {e}")
            
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
            **kwargs  # Clean! No API key filtering needed
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
    experiment_id: str = None,  # For resume capability
    enable_resume: bool = True  # Enable resume mechanism
) -> Dict[str, Any]:
    """
    Run full pilot experiment with SEQUENTIAL execution on ALL human-approved tasks.
    
    Processes one model at a time, and for each model, one task at a time.
    Supports resume capability with checkpoint saves.
    
    Args:
        tasks_by_domain: Dictionary mapping domain to task lists
        models: Dictionary of model names to test
        output_dir: Base output directory
        skip_existing: Skip tasks that already have outputs
        experiment_id: Unique experiment identifier for resume (auto-generated if None)
        enable_resume: Enable checkpoint-based resume mechanism
        
    Returns:
        Dictionary with all results and statistics
    """
    print("=" * 80)
    print("🚀 VMEVAL KIT QUICK TEST (1 TASK PER DOMAIN)")
    print("=" * 80)
    print(f"\n📊 Experiment Configuration:")
    print(f"   Models: {len(models)}")
    print(f"   Domains: {len(tasks_by_domain)}")
    print(f"   🔄 Execution Mode: SEQUENTIAL")
    print(f"   📥 Resume Enabled: {enable_resume}")
    
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
    
    # Initialize progress tracker if resume is enabled
    progress_tracker = None
    if enable_resume:
        progress_tracker = ProgressTracker(output_dir, experiment_id)
        if progress_tracker.is_resume:
            resume_stats = progress_tracker.get_resume_stats()
            print(f"\n📥 RESUMING PREVIOUS EXPERIMENT")
            print(f"   Already completed: {resume_stats['completed']}/{total_generations}")
            print(f"   Failed (will retry): {resume_stats['failed']}")
            print(f"   Interrupted (will retry): {resume_stats['in_progress']}\n")
    
    # Results storage (no longer needs thread safety since we're sequential)
    all_results = []
    
    statistics = {
        "total_tasks": total_tasks,
        "total_generations": total_generations,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "resumed": 0,
        "by_model": {},
        "by_domain": {}
    }
    
    # Initialize statistics
    for model in models.keys():
        statistics["by_model"][model] = {"completed": 0, "failed": 0, "skipped": 0}
    for domain in tasks_by_domain.keys():
        statistics["by_domain"][domain] = {"completed": 0, "failed": 0, "skipped": 0}
    
    # Load existing stats if resuming
    if progress_tracker and progress_tracker.is_resume:
        existing_stats = progress_tracker.progress.get("statistics", {})
        if existing_stats:
            statistics.update(existing_stats)
            statistics["resumed"] = len(progress_tracker.progress.get("jobs_completed", []))
    
    experiment_start = datetime.now()
    
    # Count total jobs for progress tracking
    total_jobs = sum(len(tasks) for tasks in tasks_by_domain.values()) * len(models)
    print(f"📋 Total inference jobs to run: {total_jobs}")
    print("🚀 Starting sequential execution...\n")
    print("   Processing order: Model by model, task by task\n")
    
    # Create a shared InferenceRunner instance
    runner = InferenceRunner(output_dir=str(output_dir))
    
    # Track overall progress
    job_counter = 0
    
    # Sequential execution: model by model, task by task
    for model_name, model_display in models.items():
        print(f"\n{'='*60}")
        print(f"🤖 Processing Model: {model_display} ({model_name})")
        print(f"{'='*60}")
        
        model_start_time = datetime.now()
        model_completed = 0
        model_failed = 0
        model_skipped = 0
        
        # Process all tasks for this model
        for domain, tasks in tasks_by_domain.items():
            print(f"\n  📚 Domain: {domain.title()}")
            
            for task in tasks:
                job_counter += 1
                task_id = task["id"]
                job_id = f"{model_name}_{task_id}"
                
                print(f"    [{job_counter}/{total_jobs}] Processing: {task_id}")
                
                # Check progress tracker first if enabled
                if progress_tracker:
                    if progress_tracker.should_skip_job(job_id):
                        statistics["skipped"] += 1
                        statistics["by_model"][model_name]["skipped"] += 1
                        statistics["by_domain"][domain]["skipped"] += 1
                        model_skipped += 1
                        print(f"      ⏭️  Skipped (already completed in previous run)")
                        continue
                    
                    # Mark job as started
                    if not progress_tracker.job_started(job_id):
                        # Already completed (shouldn't happen in sequential)
                        statistics["skipped"] += 1
                        statistics["by_model"][model_name]["skipped"] += 1
                        statistics["by_domain"][domain]["skipped"] += 1
                        model_skipped += 1
                        print(f"      ⏭️  Skipped (already completed)")
                        continue
                
                # Check if inference folder already exists
                run_id_pattern = f"{model_name}_{task_id}_*"
                existing_dirs = list(output_dir.glob(run_id_pattern))
                
                if skip_existing and existing_dirs and not progress_tracker:
                    statistics["skipped"] += 1
                    statistics["by_model"][model_name]["skipped"] += 1
                    statistics["by_domain"][domain]["skipped"] += 1
                    model_skipped += 1
                    print(f"      ⏭️  Skipped (existing output: {existing_dirs[0].name})")
                    continue
                
                # Run inference with structured output
                result = run_single_inference(
                    model_name=model_name,
                    task=task,
                    category=domain,
                    output_dir=output_dir,
                    runner=runner
                )
                
                # Update progress tracker
                if progress_tracker:
                    if result["success"]:
                        progress_tracker.job_completed(job_id, result)
                    else:
                        progress_tracker.job_failed(job_id, result.get("error", "Unknown error"))
                
                # Update statistics and results
                all_results.append(result)
                
                if result["success"]:
                    statistics["completed"] += 1
                    statistics["by_model"][model_name]["completed"] += 1
                    statistics["by_domain"][domain]["completed"] += 1
                    model_completed += 1
                    print(f"      ✅ Completed successfully")
                else:
                    statistics["failed"] += 1
                    statistics["by_model"][model_name]["failed"] += 1
                    statistics["by_domain"][domain]["failed"] += 1
                    model_failed += 1
                    print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
                
                # Update progress tracker statistics
                if progress_tracker:
                    progress_tracker.update_statistics(statistics)
                
                # Save intermediate results periodically (every 5 completions)
                if (statistics["completed"] + statistics["failed"]) % 5 == 0:
                    save_results(all_results.copy(), statistics.copy(), output_dir, intermediate=True)
                    print(f"      💾 Intermediate results saved")
        
        # Model summary
        model_duration = (datetime.now() - model_start_time).total_seconds()
        print(f"\n  📊 Model {model_display} Summary:")
        print(f"     Completed: {model_completed}")
        print(f"     Failed: {model_failed}")
        print(f"     Skipped: {model_skipped}")
        print(f"     Duration: {format_duration(model_duration)}")
    
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    # Final statistics
    statistics["experiment_start"] = experiment_start.isoformat()
    statistics["experiment_end"] = experiment_end.isoformat()
    statistics["duration_seconds"] = duration
    statistics["duration_formatted"] = format_duration(duration)
    
    # Final save to progress tracker
    if progress_tracker:
        progress_tracker.update_statistics(statistics)
        progress_tracker.save_progress(force=True)
        print(f"\n💾 Final checkpoint saved")
    
    print(f"\n⏱️  Sequential execution completed in {format_duration(duration)}")
    
    if statistics.get("resumed", 0) > 0:
        print(f"   📥 Resumed {statistics['resumed']} previously completed jobs")
    
    return {
        "results": all_results,
        "statistics": statistics,
        "experiment_id": experiment_id or (progress_tracker.experiment_id if progress_tracker else None)
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
    """Main execution function with resume support."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VMEvalKit Quick Test Experiment with Resume Support")
    parser.add_argument("--resume", type=str, help="Resume a previous experiment by ID or 'latest'")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume mechanism entirely")
    parser.add_argument("--experiment-id", type=str, help="Custom experiment ID for this run")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks, not just 1 per domain")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints and exit")
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list_checkpoints:
        logs_dir = OUTPUT_DIR / "logs"
        if logs_dir.exists():
            checkpoints = sorted(logs_dir.glob("checkpoint_*.json"))
            if checkpoints:
                print("📋 Available checkpoints:")
                for cp in checkpoints:
                    try:
                        with open(cp, 'r') as f:
                            data = json.load(f)
                        exp_id = data.get("experiment_id", "unknown")
                        completed = len(data.get("jobs_completed", []))
                        failed = len(data.get("jobs_failed", []))
                        last_update = data.get("last_update", "unknown")
                        print(f"   • {exp_id}:")
                        print(f"     Completed: {completed}, Failed: {failed}")
                        print(f"     Last update: {last_update}")
                        print(f"     File: {cp.name}\n")
                    except Exception as e:
                        print(f"   ⚠️  Could not read {cp.name}: {e}")
            else:
                print("No checkpoints found.")
        else:
            print("No logs directory found.")
        sys.exit(0)
    
    # Determine experiment ID
    experiment_id = args.experiment_id
    if args.resume:
        if args.resume == "latest":
            # Find the latest checkpoint
            logs_dir = OUTPUT_DIR / "logs"
            if logs_dir.exists():
                checkpoints = sorted(logs_dir.glob("checkpoint_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                if checkpoints:
                    latest_cp = checkpoints[0]
                    # Extract experiment ID from filename
                    exp_id_from_file = latest_cp.stem.replace("checkpoint_", "")
                    experiment_id = exp_id_from_file
                    print(f"📥 Resuming latest experiment: {experiment_id}")
                else:
                    print("⚠️  No checkpoints found to resume from.")
                    sys.exit(1)
            else:
                print("⚠️  No logs directory found.")
                sys.exit(1)
        else:
            experiment_id = args.resume
            print(f"📥 Resuming experiment: {experiment_id}")
    
    print("🔍 Discovering human-approved tasks from folder structure...")
    
    # Check if questions directory exists
    if not QUESTIONS_DIR.exists():
        print(f"❌ Questions directory not found at: {QUESTIONS_DIR}")
        print("   Please ensure the questions directory exists with task folders.")
        sys.exit(1)
    
    # Discover all approved tasks from folders
    all_tasks_by_domain = discover_all_tasks_from_folders(QUESTIONS_DIR)
    
    # Choose task set based on args
    tasks_by_domain = {}
    if args.all_tasks:
        tasks_by_domain = all_tasks_by_domain
        print(f"   🎯 Running ALL approved tasks")
    else:
        # Limit to 1 task per domain for quick testing
        for domain, tasks in all_tasks_by_domain.items():
            if tasks:
                tasks_by_domain[domain] = [tasks[0]]  # Take only first task
                print(f"   🎯 Testing with 1 task from {domain}: {tasks[0]['id']}")
            else:
                tasks_by_domain[domain] = []
    
    # Verify models are available
    print(f"\n🔍 Verifying {len(PILOT_MODELS)} models for parallel testing...")
    for model_name, family in PILOT_MODELS.items():
        if model_name in AVAILABLE_MODELS:
            print(f"   ✅ {model_name}: {family}")
        else:
            print(f"   ❌ {model_name}: NOT FOUND in available models")
            print(f"      Please check model name or add it to AVAILABLE_MODELS")
            # Don't exit, just warn - some models might not be configured yet
    
    print(f"\n{'=' * 80}")
    if not args.resume:
        input("Press ENTER to start the quick test experiment (or Ctrl+C to cancel)...")
    
    # Verify we found tasks
    if not tasks_by_domain or sum(len(tasks) for tasks in tasks_by_domain.values()) == 0:
        print("❌ No approved tasks found. Please check the questions directory structure.")
        sys.exit(1)
    
    # Run experiment
    experiment_results = run_pilot_experiment(
        tasks_by_domain=tasks_by_domain,
        models=PILOT_MODELS,
        output_dir=OUTPUT_DIR,
        skip_existing=True,
        experiment_id=experiment_id,
        enable_resume=not args.no_resume
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
    print("🎉 QUICK TEST COMPLETE!")
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
    
    # Show experiment ID for resume
    exp_id = experiment_results.get("experiment_id")
    if exp_id:
        print(f"\n📥 To resume this experiment later, run:")
        print(f"   python {sys.argv[0]} --resume {exp_id}")
        print(f"   or:")
        print(f"   python {sys.argv[0]} --resume latest")
    
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
