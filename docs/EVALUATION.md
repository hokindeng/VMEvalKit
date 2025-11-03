# VMEvalKit Evaluation Module

This module provides comprehensive evaluation methods for assessing video generation models' reasoning capabilities, along with powerful analysis tools for processing and visualizing results.

## Table of Contents
- [Available Evaluators](#available-evaluators)
- [Analysis Tools](#analysis-tools)
- [Evaluation Criteria](#evaluation-criteria)
- [Output Structure](#output-structure)
- [Command-Line Interface](#command-line-interface)
- [Custom Evaluators](#custom-evaluators)
- [API Reference](#api-reference)
- [Resume Capability](#resume-capability)
- [Analysis Workflow](#analysis-workflow)
- [Tips and Best Practices](#tips-and-best-practices)

## Available Evaluators

### 1. Human Evaluator
Interactive web interface for human annotation of generated videos.

**Features:**
- Gradio-based web interface
- Side-by-side display of input and generated video
- Structured evaluation criteria
- Real-time progress tracking with model-specific statistics
- Export results as JSON
- **Automatic skip of already evaluated tasks** (checks evaluations folder)
- **In-interface annotator name input** (no command-line setup needed)
- Visual queue status with completion percentages
- Refresh capability to reload evaluation queue

**Usage:**
```bash
# Run human evaluation interface (evaluates entire pilot experiment)
# Automatically skips tasks that have any existing evaluation
python examples/run_evaluation.py human

# Or use the command-line runner
python -m vmevalkit.runner.evaluate human --annotator "John Doe" --port 7860 --share
```

When launched, the interface will:
- Prompt for the annotator name
- Automatically skip tasks with existing evaluations
- Show progress and statistics by model
- Display completion percentages

**Python Usage:**
```python
from vmevalkit.eval import HumanEvaluator

evaluator = HumanEvaluator(
    experiment_name="pilot_experiment"
)
# Automatically skips tasks with existing evaluations in data/evaluations/
# Annotator name is set within the Gradio interface

evaluator.launch_interface(share=True, port=7860)
```

### 2. GPT-4O Evaluator
Automatic evaluation using OpenAI's GPT-4O vision model.

**Features:**
- Compares final frame of generated video with ground truth
- Direct assessment of whether the model answered the question correctly
- **Task-specific evaluation prompts** for each domain
- Batch processing of all models with async execution
- Detailed scoring and explanations
- **Resume capability** - saves after each task evaluation
- Progress logging with skip statistics

**Task-Specific Guidance:**
- **Chess**: Checks if final board position matches expected position after correct move
- **Maze**: Verifies complete path from start to end matches expected solution
- **Rotation**: Checks if final rotation angle and position match expected result
- **Raven**: Verifies pattern completion matches expected pattern
- **Sudoku**: Checks if numbers placed match expected solution

**Usage:**
```bash
# Evaluate entire pilot experiment (all models, all tasks)
python examples/run_evaluation.py gpt4o

# Or use the command-line runner with specific options
python -m vmevalkit.runner.evaluate gpt4o --experiment pilot_experiment --temperature 0.1
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Install dependencies: `opencv-python`, `httpx`

## Analysis Tools

### Performance Visualization (`analysis/plot.py`)
Creates professional visualizations of evaluation results.

**Features:**
- Overall model performance ranking charts
- Domain-specific performance heatmaps
- Score distribution analysis
- Binary success rate calculation (scores 4-5 = success)
- Sophisticated color schemes and typography
- Export to PNG and EPS formats

**Usage:**
```bash
# Analyze human evaluation results
python analysis/plot.py --eval-folder data/evaluations/human-eval/

# Analyze GPT-4O evaluation results
python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/
```

**Outputs:**
1. **Overall Performance Bar Chart**: Shows success rates for all models
2. **Domain Performance Heatmap**: Model × Domain matrix visualization
3. **Score Distribution Plots**: Distribution of scores per model
4. **Detailed Statistics CSV**: Complete performance metrics

### Statistical Comparison Tool (`analysis/stats.py`)
Comprehensive statistical analysis comparing GPT-4O and human evaluations.

**Features:**
- Paired t-test and Wilcoxon signed-rank test
- Multiple correlation analyses (Pearson, Spearman, Kendall)
- Cohen's kappa for inter-rater reliability
- Bootstrap confidence intervals
- Convergence analysis to find equivalence threshold
- Task-type specific analysis
- Publication-ready scatter plots

**Usage:**
```bash
# Run full statistical comparison
python analysis/stats.py
```

**Statistical Tests Performed:**
1. **Basic Statistics**: Mean, std, median for both evaluators
2. **Paired T-Test**: Tests if means are significantly different
3. **Wilcoxon Test**: Non-parametric alternative
4. **Correlation Analysis**: Measures agreement strength
5. **Inter-rater Reliability**: Cohen's kappa coefficient
6. **Bootstrap CI**: 95% confidence intervals for mean difference
7. **Convergence Analysis**: Finds sample size for statistical equivalence

## Evaluation Criteria

### Scoring Scale (1-5)
Both evaluators use a 1-5 scale for solution correctness:

**Solution Correctness Score**: 
- **1**: Completely wrong solution - no understanding of task
- **2**: Mostly incorrect with minor correct elements
- **3**: Partially correct (about half correct)
- **4**: Mostly correct with minor errors ✓
- **5**: Perfect solution ✓

### Binary Grading System
**Important**: For final analysis, scores are converted to binary:
- **Success (Correct)**: Scores 4 and 5
- **Failure (Incorrect)**: Scores 1, 2, and 3

This binary system is used in all performance calculations and visualizations.

## Output Structure

### Directory Structure
Evaluations are saved in `data/evaluations/` following this structure:
```
data/evaluations/
├── human-eval/              # Human evaluations
│   └── pilot_experiment/
│       ├── luma-ray-2/
│       │   ├── chess_task/
│       │   │   ├── chess_0000/
│       │   │   │   └── human-eval.json
│       │   │   └── ...
│       │   └── ...
│       └── ...
├── gpt4o-eval/              # GPT-4O evaluations  
│   └── pilot_experiment/
│       ├── GPT4OEvaluator_all_models.json  # Aggregated results
│       └── [same structure as human-eval]
└── custom-eval/             # Custom evaluations
```

### JSON File Format
Each evaluation JSON file contains:
```json
{
  "metadata": {
    "evaluator": "human-eval",        // or "GPT4OEvaluator"
    "annotator": "John Doe",           // (human only)
    "timestamp": "2024-10-14T12:00:00",
    "model_name": "luma-ray-2",
    "task_type": "chess_task",
    "task_id": "chess_0000"
  },
  "result": {
    "solution_correctness_score": 5,  // 1-5 scale
    "explanation": "Perfect solution", // (GPT-4O only)
    "comments": "User comments",      // (human only)
    "evaluation_type": "final_frame_comparison", // (GPT-4O only)
    "status": "completed"              // or "failed", "skipped"
  }
}
```

## Command-Line Interface

### Using the Runner Module
The `vmevalkit.runner.evaluate` module provides a unified CLI:

```bash
# Human evaluation with options
python -m vmevalkit.runner.evaluate human \
  --experiment pilot_experiment \
  --annotator "John Doe" \
  --port 7860 \
  --share

# GPT-4O evaluation with options
python -m vmevalkit.runner.evaluate gpt4o \
  --experiment pilot_experiment \
  --output-dir data/evaluations \
  --temperature 0.1
```

### Environment Variables
```bash
# Required for GPT-4O evaluation
export OPENAI_API_KEY="your-api-key"

# Optional: Set default paths
export VMEVAL_DATA_DIR="/path/to/data"
export VMEVAL_OUTPUT_DIR="/path/to/evaluations"
```

## Custom Evaluators

To create a custom evaluator, follow the pattern of existing evaluators:

```python
from pathlib import Path
from datetime import datetime
import json

class MyEvaluator:
    def __init__(self, output_dir="data/evaluations/my-eval", 
                 experiment_name="pilot_experiment"):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = Path("data/outputs") / experiment_name
    
    def _has_evaluation(self, model_name, task_type, task_id):
        """Check if task already evaluated (for resume capability)."""
        eval_path = self.output_dir / self.experiment_name / model_name / task_type / task_id
        eval_file = eval_path / "MyEvaluator.json"
        return eval_file.exists()
    
    def evaluate_single(self, model_name, task_type, task_id, video_path):
        # Your evaluation logic here
        return {
            "solution_correctness_score": 5,  # 1-5 scale
            "explanation": "The solution perfectly solves the task",
            "status": "completed"
        }
    
    def _save_result(self, model_name, task_type, task_id, eval_result):
        """Save evaluation result in standard format."""
        output_path = self.output_dir / self.experiment_name / model_name / task_type / task_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "MyEvaluator.json", 'w') as f:
            json.dump({
                "metadata": {
                    "evaluator": "MyEvaluator",
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "task_type": task_type,
                    "task_id": task_id
                },
                "result": eval_result
            }, f, indent=2)
    
    def evaluate_model(self, model_name):
        """Evaluate all tasks for a model with resume capability."""
        # Iterate through tasks, skip if already evaluated
        # Save results using _save_result method
        pass
```

Note: Evaluators are standalone classes - no base class required. Ensure your evaluator:
1. Saves results in the standard JSON format
2. Implements resume capability via `_has_evaluation()` check
3. Uses consistent directory structure
4. Includes proper metadata

## API Reference

### HumanEvaluator

**Parameters:**
- `output_dir`: Directory for saving evaluations (default: "data/evaluations/human-eval")
- `experiment_name`: Name of experiment to evaluate (default: "pilot_experiment")

**Methods:**
- `launch_interface(share, port)`: Start web interface with annotator name input
- `_load_evaluation_queue()`: Load tasks, skipping already evaluated ones
- `_get_queue_status_text()`: Generate detailed queue status with model breakdown
- `_save_evaluation()`: Save evaluation result in standard format

**Automatic Features:**
- Checks for existing `*-eval.json` files
- Automatically skips evaluated tasks
- Real-time queue status updates
- Model-specific progress tracking

### GPT4OEvaluator

**Parameters:**
- `output_dir`: Directory for saving evaluations (default: "data/evaluations/gpt4o-eval")
- `experiment_name`: Name of experiment to evaluate (default: "pilot_experiment")
- `api_key`: OpenAI API key (defaults to OPENAI_API_KEY env var)
- `model`: GPT model to use (default: "gpt-4o")
- `temperature`: Temperature for responses (default: 0.1)

**Methods:**
- `extract_final_frame(video_path)`: Extract the final frame from video
- `create_prompt(task_type)`: Generate task-specific evaluation prompts
- `evaluate_single_async()`: Async evaluation of single video
- `evaluate_model_async()`: Async evaluation of all tasks for a model
- `evaluate_all_models()`: Evaluate all models in experiment
- `_has_evaluation()`: Check if task already evaluated (resume support)
- `_save_single_result()`: Save result immediately for resume capability

**Async Features:**
- Uses httpx for async API calls
- Batch processing with progress logging
- Automatic retry on failures

### EvaluationComparator

**Location:** `analysis/stats.py`

**Methods:**
- `load_evaluations()`: Load both GPT-4O and human evaluations
- `prepare_paired_data()`: Match evaluations for comparison
- `basic_statistics()`: Calculate descriptive statistics
- `paired_t_test()`: Test for significant differences
- `correlation_analysis()`: Calculate correlation coefficients
- `inter_rater_reliability()`: Compute Cohen's kappa
- `bootstrap_confidence_intervals()`: Generate confidence intervals
- `convergence_analysis()`: Find statistical equivalence threshold
- `plot_comparisons()`: Create publication-ready visualizations

## Resume Capability

### Human Evaluation
- **Automatic Resume**: Skips already evaluated tasks
- **Detection Method**: Checks for any `*-eval.json` files
- **Progress Tracking**: Shows tasks skipped and remaining
- **Re-evaluation**: Manually delete evaluation files to re-evaluate

### GPT-4O Evaluation
- **Automatic Resume**: Skips already evaluated tasks (saves after each)
- **Detection Method**: Checks for `GPT4OEvaluator.json` files
- **Progress Logging**: Reports skipped, evaluated, and failed counts
- **Interrupt Safe**: Can interrupt with Ctrl+C and resume later

## Analysis Workflow

### Complete Evaluation Pipeline
```bash
# 1. Run inference to generate videos
python examples/experiment_2025-10-14.py

# 2. Run human evaluation (interactive)
python examples/run_evaluation.py human

# 3. Run GPT-4O evaluation (automatic)
python examples/run_evaluation.py gpt4o

# 4. Visualize results
python analysis/plot.py --eval-folder data/evaluations/human-eval/
python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/

# 5. Compare human vs GPT-4O statistically
python analysis/stats.py

# 6. View generated visualizations
open analysis/statistics/gpt4o_vs_human_comparison.png
open analysis/performance_by_domain.png
```

### Output Files Generated
1. **Evaluation JSONs**: Individual task evaluations
2. **Performance Charts**: PNG and EPS visualizations
3. **Statistical Results**: JSON with comparison metrics
4. **CSV Reports**: Detailed performance tables

## Tips and Best Practices

### For Human Evaluation
- **Session Management**: Complete evaluations in one session when possible
- **Annotator Consistency**: Use same annotator for related tasks
- **Comments**: Add detailed comments for edge cases or ambiguous results
- **Progress Monitoring**: Use refresh button to update queue status
- **Backup**: Evaluations are saved immediately after submission

### For GPT-4O Evaluation
- **Cost Management**: Monitor API usage for large experiments
- **Temperature**: Use low temperature (0.1) for consistency
- **Ground Truth**: Only tasks with final frame ground truth will be evaluated
- **Batch Processing**: Let it run uninterrupted for efficiency
- **Resume on Failure**: Safe to restart after interruptions

### For Analysis
- **Binary Grading**: Remember scores 4-5 are considered success
- **Statistical Significance**: Use stats.py to validate comparisons
- **Visualization**: Use plot.py for publication-ready figures
- **Domain Analysis**: Check performance variations across task types

### General Best Practices
1. **Order of Operations**: Always run inference → evaluation → analysis
2. **Experiment Naming**: Use descriptive experiment names for organization
3. **Version Control**: Track evaluation results in git for reproducibility
4. **Documentation**: Document any custom evaluation criteria
5. **Validation**: Compare human and GPT-4O results for validation
6. **Error Handling**: Check logs for failed evaluations
7. **Data Backup**: Regularly backup evaluation results

## Troubleshooting

### Common Issues and Solutions

**Human Evaluation Interface Not Loading:**
- Check if port 7860 is available
- Try using `--share` flag for public URL
- Verify Gradio is installed: `pip install gradio`

**GPT-4O API Errors:**
- Verify OPENAI_API_KEY is set correctly
- Check API quota and rate limits
- Ensure opencv-python is installed

**Missing Evaluations:**
- Check evaluation directory structure
- Verify experiment name matches
- Look for error messages in logs

**Statistical Analysis Failures:**
- Ensure paired evaluations exist
- Check for sufficient sample size
- Verify data format consistency