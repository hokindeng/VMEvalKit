#!/bin/bash
##############################################################################
# Quick Status Check - Shows setup status for open-source and commercial
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Load .env for commercial API checks
load_env_file

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    VMEvalKit Status                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Open-Source Status
# ============================================================================

print_section "OPEN-SOURCE MODELS"

echo ""
echo "📦 Virtual Environments:"
VENV_OK=0
VENV_TOTAL=4
for venv in venv_main venv_hunyuan venv_dynamicrafter venv_videocrafter; do
    if venv_exists "$venv"; then
        print_success "${venv}"
        VENV_OK=$((VENV_OK + 1))
    else
        print_error "${venv}"
    fi
done
echo "   Status: ${VENV_OK}/${VENV_TOTAL} ready"

echo ""
echo "💾 Model Checkpoints:"
CKPT_OK=0
CKPT_TOTAL=${#CHECKPOINTS[@]}
for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r path url size_desc <<< "$entry"
    name="$(basename "$(dirname "$path")")"
    
    if checkpoint_exists "$path"; then
        print_success "${name}"
        CKPT_OK=$((CKPT_OK + 1))
    else
        print_error "${name}"
    fi
done
echo "   Status: ${CKPT_OK}/${CKPT_TOTAL} ready"

# ============================================================================
# Commercial Status
# ============================================================================

print_section "COMMERCIAL MODELS"

echo ""
echo "🔑 API Keys:"

declare -A API_KEYS=(
    ["LUMA_API_KEY"]="Luma"
    ["GOOGLE_APPLICATION_CREDENTIALS"]="Google Veo"
    ["WAVESPEED_API_KEY"]="WaveSpeed"
    ["RUNWAY_API_SECRET"]="Runway"
    ["OPENAI_API_KEY"]="OpenAI"
)

API_OK=0
API_TOTAL=${#API_KEYS[@]}
for key in "${!API_KEYS[@]}"; do
    provider="${API_KEYS[$key]}"
    if check_api_key "$key"; then
        print_success "${provider} (${key})"
        API_OK=$((API_OK + 1))
    else
        print_skip "${provider} (${key} not set)"
    fi
done
echo "   Status: ${API_OK}/${API_TOTAL} configured"

# ============================================================================
# Test Results
# ============================================================================

print_section "TEST RESULTS"

echo ""
if [[ -d "${TESTS_DIR}/outputs/model_validation" ]]; then
    VIDEO_COUNT=$(find "${TESTS_DIR}/outputs/model_validation" \( -name "*.mp4" -o -name "*.webm" \) 2>/dev/null | wc -l)
    MODEL_COUNT=$(find "${TESTS_DIR}/outputs/model_validation" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    print_success "Tests run: ${MODEL_COUNT} models, ${VIDEO_COUNT} videos"
else
    print_warning "Tests not run yet"
fi

# ============================================================================
# Inference Progress
# ============================================================================

print_section "INFERENCE PROGRESS"

echo ""
OUTPUT_DIR="${VMEVAL_ROOT}/data/outputs/pilot_experiment"
if [[ -d "$OUTPUT_DIR" ]]; then
    TOTAL_VIDEOS=$(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    TARGET=26288  # 16 models × 1,643 tasks
    PROGRESS=$(python3 -c "print(f'{100*${TOTAL_VIDEOS}/${TARGET}:.1f}%')" 2>/dev/null || echo "N/A")
    print_info "Videos generated: ${TOTAL_VIDEOS}"
    print_info "Target: ${TARGET}"
    print_info "Progress: ${PROGRESS}"
else
    print_warning "No inference run yet"
fi

# ============================================================================
# Next Actions
# ============================================================================

print_section "NEXT ACTIONS"

echo ""
if [[ $VENV_OK -lt $VENV_TOTAL ]]; then
    echo "  1. Complete setup:"
    echo "     ./setup/RUN_SETUP.sh"
elif [[ ! -d "${TESTS_DIR}/outputs/model_validation" ]]; then
    echo "  1. Test models (optional):"
    echo "     ./setup/4_test_models.sh --opensource"
    echo ""
    echo "  2. Or start inference:"
    echo "     ./run_all_models.sh --parallel"
else
    echo "  ✅ Ready! Run inference:"
    echo "     ./run_all_models.sh --parallel"
fi
echo ""
