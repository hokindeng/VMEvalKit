#!/bin/bash
##############################################################################
# Step 2: Download Model Checkpoints
#
# Downloads checkpoints for submodule-based models (~24GB total):
#   DynamiCrafter 256/512/1024, VideoCrafter2
#
# Note: Diffusers models (LTX, SVD, WAN, Hunyuan) auto-download on first use
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

print_header "Step 2: Downloading Model Checkpoints"

echo "Total download size: ~24GB"
echo "Time estimate: 10-30 minutes (network dependent)"
echo ""

download_checkpoint() {
    local rel_path="$1"
    local url="$2"
    local size="$3"
    local idx="$4"
    local total="$5"
    
    local full_path="${SUBMODULES_DIR}/${rel_path}"
    local dir_path="$(dirname "$full_path")"
    local name="$(basename "$(dirname "$rel_path")")"
    
    print_download "[${idx}/${total}] ${name} - ${size}..."
    
    ensure_dir "$dir_path"
    
    if [[ -f "$full_path" ]]; then
        print_skip "Already exists"
    else
        wget -q --show-progress -c "$url" -O "$full_path"
        print_success "Downloaded"
    fi
    echo ""
}

# Download all checkpoints from config
total=${#CHECKPOINTS[@]}
idx=0

for entry in "${CHECKPOINTS[@]}"; do
    idx=$((idx + 1))
    
    IFS='|' read -r path url size <<< "$entry"
    download_checkpoint "$path" "$url" "$size" "$idx" "$total"
done

print_header "✅ All Checkpoints Downloaded"

echo "   DynamiCrafter 256  → $(get_checkpoint_size "DynamiCrafter/checkpoints/dynamicrafter_256_v1/model.ckpt")"
echo "   DynamiCrafter 512  → $(get_checkpoint_size "DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt")"
echo "   DynamiCrafter 1024 → $(get_checkpoint_size "DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt")"
echo "   VideoCrafter2      → $(get_checkpoint_size "VideoCrafter/checkpoints/base_512_v2/model.ckpt")"
echo ""
echo "Next: ./setup/3a_validate_opensource.sh"
