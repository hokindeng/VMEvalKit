#!/bin/bash

echo "Installing VMEvalKit..."

git clone https://github.com/hokindeng/VMEvalKit.git
cd VMEvalKit

git submodule update --init --recursive

cp env.template .env

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .


pip install lmdeploy timm peft openai