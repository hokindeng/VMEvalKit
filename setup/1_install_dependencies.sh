#!/bin/bash
##############################################################################
# Step 1: Install All Dependencies
#
# Creates 4 isolated virtual environments in envs/:
#   venv_main         - torch 2.5.1 (LTX, SVD, WAN, Morphic)
#   venv_hunyuan      - torch 2.0.0 (HunyuanVideo)
#   venv_dynamicrafter - torch 2.0.0 (DynamiCrafter)
#   venv_videocrafter  - torch 2.0.0 (VideoCrafter)
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

print_header "Step 1: Installing Dependencies"

ensure_dir "${ENVS_DIR}"

# ============================================================================
# VENV 1: Main (torch 2.5.1) - LTX, SVD, WAN, Morphic
# ============================================================================
print_step "[1/4] Creating venv_main (torch 2.5.1)..."

python3 -m venv "${ENVS_DIR}/venv_main"
activate_venv "venv_main"

pip install -q --upgrade pip setuptools wheel

# Core PyTorch stack
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Diffusers ecosystem
pip install -q diffusers transformers accelerate sentencepiece
pip install -q xformers==0.0.28.post3 einops decord av omegaconf

# Vision utilities
pip install -q opencv-python opencv-contrib-python

# Core utilities
pip install -q numpy Pillow pandas matplotlib tqdm
pip install -q pydantic pydantic-settings python-dotenv
pip install -q requests httpx aiohttp tenacity boto3

# Media processing
pip install -q imageio imageio-ffmpeg moviepy cairosvg ftfy

deactivate
print_success "venv_main ready (LTX, SVD, WAN, Morphic - 10 models)"

# ============================================================================
# VENV 2: Hunyuan (torch 2.0.0)
# ============================================================================
print_step "[2/4] Creating venv_hunyuan (torch 2.0.0)..."

python3 -m venv "${ENVS_DIR}/venv_hunyuan"
activate_venv "venv_hunyuan"

pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

pip install -q diffusers==0.31.0 transformers==4.39.3 accelerate==1.1.1
pip install -q opencv-python einops imageio imageio-ffmpeg
pip install -q safetensors peft loguru
pip install -q Pillow numpy pandas tqdm pydantic python-dotenv requests

deactivate
print_success "venv_hunyuan ready (HunyuanVideo - 1 model)"

# ============================================================================
# VENV 3: DynamiCrafter (torch 2.0.0)
# ============================================================================
print_step "[3/4] Creating venv_dynamicrafter (torch 2.0.0)..."

python3 -m venv "${ENVS_DIR}/venv_dynamicrafter"
activate_venv "venv_dynamicrafter"

pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

pip install -q decord einops imageio omegaconf opencv-python
pip install -q Pillow pytorch_lightning PyYAML tqdm transformers
pip install -q moviepy av xformers==0.0.18 gradio timm
pip install -q pandas pydantic python-dotenv requests

deactivate
print_success "venv_dynamicrafter ready (DynamiCrafter - 3 models)"

# ============================================================================
# VENV 4: VideoCrafter (torch 2.0.0)
# ============================================================================
print_step "[4/4] Creating venv_videocrafter (torch 2.0.0)..."

python3 -m venv "${ENVS_DIR}/venv_videocrafter"
activate_venv "venv_videocrafter"

pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

pip install -q omegaconf pytorch-lightning einops transformers
pip install -q opencv-python imageio av moviepy
pip install -q Pillow numpy pandas tqdm pydantic python-dotenv requests

deactivate
print_success "venv_videocrafter ready (VideoCrafter - 1 model)"

# ============================================================================
# Summary
# ============================================================================
print_header "✅ All 4 Virtual Environments Created"

echo "   venv_main          → LTX, SVD, WAN, Morphic (10 models)"
echo "   venv_hunyuan       → HunyuanVideo (1 model)"
echo "   venv_dynamicrafter → DynamiCrafter 256/512/1024 (3 models)"
echo "   venv_videocrafter  → VideoCrafter2 (1 model)"
echo ""
echo "Next: ./setup/2_download_checkpoints.sh"
