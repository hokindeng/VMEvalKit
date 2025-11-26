#!/bin/bash

echo "Creating questions..."
python examples/create_questions.py --task chess maze --pairs-per-domain 5

# for svd , need to install diffusers
pip install diffusers
python examples/generate_videos.py --model svd --task chess maze

# for generate all video for all task.
python examples/generate_videos.py --model svd --task chess maze --all-tasks

