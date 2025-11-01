#!/usr/bin/env python3
"""
Create Raven failure figure - USE raven_0013 for Sora
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np

# Configuration
EVAL_DIR = Path("/Users/access/VMEvalKit/data/evaluations/gpt4o-eval/pilot_experiment")
OUTPUT_DIR = Path("/Users/access/VMEvalKit/data/outputs/pilot_experiment")
QUESTIONS_DIR = Path("/Users/access/VMEvalKit/data/questions")
FIGURES_DIR = Path("/tmp/failure_figures")

MODELS = ["openai-sora-2", "veo-3.1-720p", "veo-3.0-generate", "luma-ray-2", "runway-gen4-turbo", "wavespeed-wan-2.2-i2v-720p"]
FAILURE_THRESHOLD = 3

# Manually skip these (they're actually correct despite low score)
SKIP_CASES = {
    "openai-sora-2": ["raven_0000", "raven_0003", "raven_0010"]
}

def find_raven_failure_cases():
    """Find one failure case per model - SKIP known correct cases"""
    failure_cases = {}
    
    print("Scanning raven_task...")
    
    for model in MODELS:
        eval_task_dir = EVAL_DIR / model / "raven_task"
        if not eval_task_dir.exists():
            continue
        
        for task_instance_dir in sorted(eval_task_dir.iterdir()):
            if not task_instance_dir.is_dir():
                continue
            
            task_id = task_instance_dir.name
            
            # SKIP known correct cases
            if model in SKIP_CASES and task_id in SKIP_CASES[model]:
                print(f"  ⊘ Skipping {model} {task_id} (actually correct)")
                continue
            
            eval_file = task_instance_dir / "GPT4OEvaluator.json"
            if not eval_file.exists():
                continue
            
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            score = eval_data.get('result', {}).get('solution_correctness_score', 0)
            
            if score <= FAILURE_THRESHOLD:
                failure_cases[model] = {
                    'task_id': task_id,
                    'score': score,
                }
                print(f"  ✗ {model}: {task_id} (score: {score})")
                break
    
    return failure_cases

def load_video_frames(video_path, num_frames=12):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames if len(frames) == num_frames else None

def load_image(image_path):
    """Load and return image"""
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def format_model_name(model_name):
    """Format model name nicely for display"""
    if 'openai' in model_name:
        return 'Sora'
    elif 'veo-3.1' in model_name:
        return 'Veo 3.1'
    elif 'veo-3.0' in model_name:
        return 'Veo 3.0'
    elif 'luma' in model_name:
        return 'Luma'
    elif 'runway' in model_name:
        return 'Runway'
    elif 'wavespeed' in model_name:
        return 'Wavespeed'
    else:
        return model_name.replace('-', ' ').replace('_', ' ').title()

def create_raven_failure_figure(failure_cases, output_path):
    """Create CLEAN Raven failure figure"""
    
    models_with_failures = [(model, data) for model, data in failure_cases.items() if data]
    
    if not models_with_failures:
        print(f"No failures to visualize")
        return
    
    num_models = len(models_with_failures)
    
    # Create figure
    fig = plt.figure(figsize=(30, 3.5 * num_models), facecolor='white')
    
    fig.suptitle('Raven - Failure Cases', 
                 fontsize=28, fontweight='bold', y=0.995)
    
    # Create main grid
    gs = gridspec.GridSpec(num_models, 15,
                          hspace=0.3, wspace=0.2,
                          left=0.12, right=0.98,
                          top=0.94, bottom=0.02,
                          width_ratios=[1, 1, 0.3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    for model_idx, (model_name, failure_data) in enumerate(models_with_failures):
        task_id = failure_data['task_id']
        
        # Load data
        question_dir = QUESTIONS_DIR / "raven_task" / task_id
        output_model_dir = OUTPUT_DIR / model_name / "raven_task" / task_id
        
        first_frame = load_image(question_dir / "first_frame.png")
        expected_final = load_image(question_dir / "final_frame.png")
        
        # Find video
        video_frames = None
        if output_model_dir.exists():
            output_folders = list(output_model_dir.iterdir())
            if output_folders:
                video_dir = output_folders[0] / "video"
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4"))
                    if video_files:
                        video_frames = load_video_frames(video_files[0])
        
        # Model name
        model_display = format_model_name(model_name)
        y_pos = (num_models - model_idx - 0.5) / num_models * 0.92 + 0.04
        
        fig.text(0.05, y_pos, model_display,
                fontsize=16, fontweight='bold', 
                ha='left', va='center',
                transform=fig.transFigure,
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor='lightblue', 
                         alpha=0.7, 
                         edgecolor='darkblue', 
                         linewidth=2))
        
        # Column 0: Initial State
        ax_first = plt.subplot(gs[model_idx, 0])
        if first_frame is not None:
            ax_first.imshow(first_frame)
        ax_first.set_title('Initial State', fontsize=11, fontweight='bold', pad=5)
        ax_first.axis('off')
        
        # Column 1: Expected Final
        ax_expected = plt.subplot(gs[model_idx, 1])
        if expected_final is not None:
            ax_expected.imshow(expected_final)
        ax_expected.set_title('Expected Final', fontsize=11, fontweight='bold', pad=5)
        ax_expected.axis('off')
        
        # Column 2: EMPTY (gap)
        
        # Columns 3-14: 12 video frames
        if video_frames:
            for i, frame in enumerate(video_frames):
                ax_frame = plt.subplot(gs[model_idx, 3 + i])
                ax_frame.imshow(frame)
                ax_frame.axis('off')
    
    plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created: {output_path}")

def main():
    print("=" * 60)
    print("Finding Raven failure cases (skipping known correct)...")
    print("=" * 60)
    
    failure_cases = find_raven_failure_cases()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Creating corrected Raven failure figure...")
    print("=" * 60)
    
    if failure_cases:
        output_file = FIGURES_DIR / "raven_task_failures.png"
        create_raven_failure_figure(failure_cases, output_file)
        
        output_eps = FIGURES_DIR / "raven_task_failures.eps"
        create_raven_failure_figure(failure_cases, output_eps)
    
    print("\n" + "=" * 60)
    print(f"✓ Corrected Raven figure saved to: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

