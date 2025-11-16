# VideoThinkBench Integration

This document explains how to download and use the [VideoThinkBench dataset](https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench) in VMEvalKit.

## 📋 Overview

**VideoThinkBench** is a comprehensive benchmark for evaluating video generation models' reasoning capabilities. It contains **~4,149 tasks** across 5 subsets covering vision-centric and text-centric reasoning tasks.

- **Source**: [OpenMOSS-Team/VideoThinkBench](https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench)
- **License**: MIT License
- **Citation**: [arXiv:2511.04570](https://arxiv.org/abs/2511.04570)

## 📊 Dataset Subsets

VideoThinkBench consists of 5 subsets:

### Vision-Centric Tasks

1. **ARC_AGI_2** (1,000 tasks)
   - Abstract reasoning tasks requiring few-shot learning
   - Pattern recognition in input-output grids

2. **Eyeballing_Puzzles** (1,050 tasks)
   - Spatial reasoning and visual estimation
   - Drawing tasks requiring geometric reasoning

3. **Visual_Puzzles** (496 tasks)
   - Pattern recognition and visual logic problems
   - Color, shape, and size reasoning

4. **Mazes** (150 tasks)
   - Path-finding and navigation challenges
   - Spatial planning

### Text-Centric Tasks

5. **Text_Centric_Tasks** (1,453 tasks)
   - Mathematical reasoning (MATH, GSM8K, AIME, MathVista, MathVision)
   - Multimodal understanding (MMMU, MMBench)
   - General knowledge (MMLU, MMLU-Pro)
   - Scientific reasoning (GPQA-diamond, SuperGPQA)

## 🚀 Quick Start

Download all VideoThinkBench subsets:

```bash
python examples/create_questions.py --task videothinkbench
```

Download specific subsets:

```bash
# Download just ARC-AGI-2 and Mazes
python examples/create_questions.py --task arc_agi_2 mazes

# Download all 5 subsets individually
python examples/create_questions.py --task arc_agi_2 eyeballing_puzzles visual_puzzles mazes text_centric_tasks
```

Mix with generated tasks:

```bash
# Download VideoThinkBench + generate Chess and Sudoku tasks
python examples/create_questions.py --task videothinkbench chess sudoku --pairs-per-domain 50
```

## 📁 Output Structure

The downloaded data will be organized in the VMEvalKit format:

```
data/questions/
├── arc_agi_2_task/
│   ├── 182e5d0f/
│   │   ├── first_frame.png
│   │   ├── final_frame.png
│   │   ├── prompt.txt
│   │   └── question_metadata.json
│   └── ...
├── eyeballing_puzzles_task/
│   └── ...
├── visual_puzzles_task/
│   └── ...
├── mazes_task/
│   └── ...
├── text_centric_tasks_task/
│   └── ...
└── videothinkbench_dataset.json  # Master metadata file
```

Each task folder contains:
- **first_frame.png**: Input image
- **final_frame.png**: Expected output/solution image (if available)
- **prompt.txt**: Task prompt/instructions
- **question_metadata.json**: Complete metadata including task ID, domain, source info

## 📜 License & Attribution

### VideoThinkBench License

VideoThinkBench is licensed under the **MIT License** by OpenMOSS-Team.

When using VideoThinkBench, please cite:

```bibtex
@article{tong2025thinkingwithvideo,
    title={Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm},
    author={Jingqi Tong and Yurong Mou and Hangcheng Li and Mingzhe Li and Yongzhuo Yang and Ming Zhang and Qiguang Chen and Tianyi Liang and Xiaomeng Hu and Yining Zheng and Xinchi Chen and Jun Zhao and Xuanjing Huang and Xipeng Qiu},
    journal={arXiv preprint arXiv:2511.04570},
    year={2025}
}
```

### VMEvalKit Usage

When using VideoThinkBench data downloaded through VMEvalKit:

1. **Retain attribution**: Keep the source information in `question_metadata.json`
2. **Include license**: Ensure the MIT license is acknowledged
3. **Cite both**: Cite both VideoThinkBench (above) and VMEvalKit papers

### Redistribution

If you redistribute VideoThinkBench data downloaded via VMEvalKit:

- ✅ **Allowed**: Share the data with proper attribution
- ✅ **Allowed**: Modify the data for research purposes
- ✅ **Allowed**: Use commercially (MIT License permits this)
- ⚠️ **Required**: Include MIT License text and attribution
- ⚠️ **Required**: Cite the VideoThinkBench paper

## 🔍 Available Domains

Use `--list-domains` to see all available domains:

```bash
python examples/create_questions.py --list-domains
```

This will show:
- **videothinkbench**: Meta-task that downloads all 5 subsets
- **arc_agi_2**: ARC-AGI-2 subset (1k tasks)
- **eyeballing_puzzles**: Eyeballing Puzzles subset (1.05k tasks)
- **visual_puzzles**: Visual Puzzles subset (496 tasks)
- **mazes**: Mazes subset (150 tasks)
- **text_centric_tasks**: Text-Centric Tasks subset (1.45k tasks)
- Plus 6 original VMEvalKit domains (chess, maze, raven, rotation, sudoku, object_subtraction)

## 🛠 Advanced Usage

### Verify Downloaded Data

Read and analyze downloaded data:

```bash
python examples/create_questions.py --read-only
```

### Custom Output Directory

```bash
python examples/create_questions.py --task videothinkbench --output-dir /custom/path
```

### Integration with Video Generation

After downloading, use the data for video generation:

```bash
# Generate videos for all VideoThinkBench tasks
python examples/generate_videos.py --task videothinkbench

# Generate videos for specific subset
python examples/generate_videos.py --task arc_agi_2
```

## 📚 References

- **VideoThinkBench Dataset**: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench
- **Paper**: https://arxiv.org/abs/2511.04570
- **Project**: https://github.com/OpenMOSS/VideoThinkBench (if available)

## ⚠️ Important Notes

1. **Download Time**: The complete dataset (~4.1k tasks) may take several minutes to download depending on your internet connection.

2. **Storage**: Ensure you have sufficient disk space (~500MB-1GB for images and metadata).

3. **Dependencies**: Requires `datasets` and `PIL` libraries:
   ```bash
   pip install datasets Pillow
   ```

4. **License Compliance**: The MIT License is permissive, but always include proper attribution when using or redistributing the data.

5. **Data Updates**: VideoThinkBench may be updated on HuggingFace. Re-download to get the latest version.

## 🤝 Contributing

If you find issues with the VideoThinkBench integration or have suggestions:

1. Check if the issue is with VMEvalKit integration or the source dataset
2. For VMEvalKit issues: Open an issue in the VMEvalKit repository
3. For source dataset issues: Contact the VideoThinkBench authors

## 📞 Support

For questions about:
- **VideoThinkBench dataset**: Contact OpenMOSS-Team
- **VMEvalKit integration**: Open an issue in VMEvalKit repository
- **License questions**: Review the MIT License text or consult legal counsel

