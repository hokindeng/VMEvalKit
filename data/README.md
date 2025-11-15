---
license: mit
task_categories:
  - video-classification
  - visual-question-answering
  - image-to-video
tags:
  - video-generation
  - reasoning
  - benchmark
  - evaluation
  - multimodal
language:
  - en
size_categories:
  - n<1K
pretty_name: "Video Models Start to Solve: Benchmark Results"
---

# Video Models Start to Solve 🎥🧠

A comprehensive benchmark evaluating video generation models on complex reasoning tasks across five cognitive domains.

## 📊 Dataset Overview

This dataset contains the complete evaluation suite from the VMEvalKit benchmark, including:
- **75 reasoning task pairs** across 5 cognitive domains
- **Inference results** from 6 state-of-the-art video generation models
- **Dual evaluations**: Human annotations and GPT-4O automated scoring

### Dataset Statistics

| Component | Size | Count | Description |
|-----------|------|-------|-------------|
| Questions | 5.8 MB | 75 task pairs | Benchmark input tasks with images and prompts |
| Outputs | 656 MB | ~4,500 videos | Model-generated videos (MP4 format) |
| Evaluations | 3.5 MB | ~9,000 scores | Human and GPT-4O evaluation results |

**Total Dataset Size**: ~665 MB

## 🎯 Cognitive Domains

The benchmark evaluates five distinct reasoning capabilities:

| Domain | Tasks | Description | Example |
|--------|-------|-------------|---------|
| **Chess** | 15 | Strategic thinking and tactical pattern recognition | Mate-in-1 puzzles |
| **Maze** | 15 | Spatial reasoning and navigation planning | Path-finding through complex mazes |
| **Raven** | 15 | Abstract reasoning and pattern completion | Progressive matrix completion |
| **Rotation** | 15 | 3D mental rotation and spatial visualization | Camera rotation around 3D objects |
| **Sudoku** | 15 | Logical reasoning and constraint satisfaction | Sudoku solving steps |

## 🤖 Evaluated Models

Results from 6 state-of-the-art video generation models (as of Jan 2025):

1. **OpenAI Sora 2** - Turbo variant
2. **Luma Ray 2** - Latest flagship model
3. **Runway Gen-4 Turbo** - High-speed generation
4. **Google Veo 3.0 Generate** - Standard quality
5. **Google Veo 3.1 720p** - Enhanced quality
6. **Wan 2.2 I2V 720p** - Image-to-video specialist

Each model generated 75 videos (one per task), totaling **450 videos**.

## 📁 Dataset Structure

```
Video-Models-Solve/
├── questions/                           # Input benchmark tasks
│   ├── vmeval_dataset.json             # Master dataset manifest
│   ├── chess_task/                     # 15 chess puzzles
│   │   └── chess_XXXX/
│   │       ├── first_frame.png         # Initial state
│   │       ├── final_frame.png         # Target state
│   │       ├── first_frame_padded_1280_720.png  # Padded for video generation
│   │       ├── prompt.txt              # Text instructions
│   │       └── question_metadata.json  # Task metadata
│   ├── maze_task/                      # 15 maze challenges
│   ├── raven_task/                     # 15 Raven matrices
│   ├── rotation_task/                  # 15 rotation tasks
│   └── sudoku_task/                    # 15 sudoku puzzles
│
├── outputs/                             # Model inference results
│   └── pilot_experiment/               # Experiment version
│       ├── inference_log.json          # Global inference metadata
│       └── [model-name]/               # Per-model results
│           └── [domain]_task/
│               └── [task_id]/
│                   └── [run_id]/       # Timestamped run
│                       ├── video/
│                       │   └── model_output.mp4  # Generated video
│                       ├── question/
│                       │   ├── prompt.txt
│                       │   └── first_frame.png
│                       └── metadata.json         # Generation parameters
│
└── evaluations/                         # Evaluation results
    ├── gpt4o-eval/                     # GPT-4O automated scoring
    │   └── pilot_experiment/
    │       └── [model-name]/
    │           └── [domain]_task/
    │               └── [task_id]/
    │                   └── GPT4OEvaluator.json
    └── human-eval/                     # Human annotations
        └── pilot_experiment/
            └── [model-name]/
                └── [domain]_task/
                    └── [task_id]/
                        └── human-eval.json
```

## 🔧 Usage

### Using with VMEvalKit

```bash
# Clone the VMEvalKit repository
git clone https://github.com/Video-Models-Solve/VMEvalKit.git
cd VMEvalKit

# Run analysis
python analysis/stats.py
python analysis/plot.py
```

### Analyzing Results

```python
import json
from pathlib import Path

# Load evaluation results
eval_path = Path("evaluations/gpt4o-eval/pilot_experiment")
for model_dir in eval_path.iterdir():
    model_name = model_dir.name
    for task_dir in model_dir.glob("*/"):
        for eval_file in task_dir.glob("*/GPT4OEvaluator.json"):
            with open(eval_file) as f:
                results = json.load(f)
                print(f"{model_name}: {results['score']}")
```

## 📊 Key Results

### Overall Performance (GPT-4O Evaluation)

| Model | Overall Score | Chess | Maze | Raven | Rotation | Sudoku |
|-------|--------------|-------|------|-------|----------|--------|
| OpenAI Sora 2 | 81.3% | 100% | 80% | 80% | 73.3% | 73.3% |
| Google Veo 3.1 | 77.3% | 100% | 80% | 60% | 80% | 66.7% |
| Luma Ray 2 | 74.7% | 86.7% | 93.3% | 60% | 66.7% | 66.7% |
| Runway Gen-4 | 74.7% | 100% | 80% | 60% | 60% | 73.3% |
| Google Veo 3.0 | 73.3% | 93.3% | 73.3% | 53.3% | 80% | 66.7% |
| Wan 2.2 I2V | 69.3% | 86.7% | 73.3% | 60% | 60% | 66.7% |

### Domain Difficulty Ranking

1. **Chess** (Easy) - 94.4% average success
2. **Maze** (Easy) - 80.0% average success
3. **Rotation** (Medium) - 70.0% average success
4. **Sudoku** (Medium) - 68.9% average success
5. **Raven** (Hard) - 62.2% average success

## 🏆 Citation

If you use this dataset or code, please cite:

```bibtex
@article{video-models-solve-2025,
  title={Video Models Start to Solve: Evaluating Reasoning Capabilities in Video Generation},
  author={[Your Authors]},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

This dataset is released under the MIT License.

## 🔗 Links

- **Paper**: [Coming Soon]
- **Code**: https://github.com/Video-Models-Solve/VMEvalKit
- **Leaderboard**: [Coming Soon]
- **Demo**: [Coming Soon]

## 🙏 Acknowledgments

This benchmark was created to advance the understanding of reasoning capabilities in video generation models. We thank:
- OpenAI, Google, Luma, Runway, and other model providers
- The open-source community for evaluation tools
- Human annotators for providing ground truth evaluations

## 📝 Version History

- **v1.0** (January 2025) - Initial release
  - 75 task pairs across 5 domains
  - 6 model evaluations
  - Human and GPT-4O dual scoring

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/Video-Models-Solve/VMEvalKit/blob/main/CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions or issues, please open an issue on [GitHub](https://github.com/Video-Models-Solve/VMEvalKit/issues).

