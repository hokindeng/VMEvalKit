#!/bin/bash


LOG_FILE="lmdeploy_server.log"
score_log_file="score.log"

pip install lmdeploy timm peft>=0.17.0 openai
CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server OpenGVLab/InternVL3-8B \
--chat-template internvl2_5 \
--server-port 23333 \
--tp 1 >> $LOG_FILE 2>&1 & # takes 30GB vram.
# redirect to log

echo "Waiting for server..."
while ! nc -z localhost 23333; do
  sleep 1
done
echo "Server is ready."

python examples/score_videos.py internvl >> $score_log_file 2>&1