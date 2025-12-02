#!/bin/bash
##############################################################################
# Step 3b: Validate Commercial API Setup
#
# Validates commercial model API keys in .env file:
#   - LUMA_API_KEY          → Luma Ray 2, Ray 2 Flash
#   - GOOGLE_APPLICATION_CREDENTIALS → Veo 2, Veo 3.0
#   - WAVESPEED_API_KEY     → Veo 3.1 Flash, WaveSpeed WAN
#   - RUNWAY_API_SECRET     → Runway Gen4 Turbo
#   - OPENAI_API_KEY        → OpenAI Sora
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

print_header "Step 3b: Validating Commercial API Setup"

# Load .env file
load_env_file

CONFIGURED=0
MISSING=0

# ============================================================================
# API Key Definitions
# ============================================================================
declare -A API_KEYS=(
    ["LUMA_API_KEY"]="Luma Ray 2, Ray 2 Flash"
    ["GOOGLE_APPLICATION_CREDENTIALS"]="Veo 2, Veo 3.0 Generate"
    ["WAVESPEED_API_KEY"]="Veo 3.1 Flash, WaveSpeed WAN"
    ["RUNWAY_API_SECRET"]="Runway Gen4 Turbo"
    ["OPENAI_API_KEY"]="OpenAI Sora"
)

# ============================================================================
# Check Each API Key
# ============================================================================
print_section "API Key Status"

for key in "${!API_KEYS[@]}"; do
    models="${API_KEYS[$key]}"
    
    if check_api_key "$key"; then
        # Mask the key value for display
        value="${!key}"
        masked="${value:0:8}...${value: -4}"
        print_success "${key} → ${models}"
        print_info "   Value: ${masked}"
        CONFIGURED=$((CONFIGURED + 1))
    else
        print_warning "${key} - NOT SET"
        print_info "   Missing: ${models}"
        MISSING=$((MISSING + 1))
    fi
    echo ""
done

# ============================================================================
# Model Availability Summary
# ============================================================================
print_section "Commercial Model Availability"

for entry in "${COMMERCIAL_MODELS[@]}"; do
    IFS='|' read -r model env_var <<< "$entry"
    
    if check_api_key "$env_var"; then
        print_success "${model}"
    else
        print_skip "${model} (requires ${env_var})"
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo ""
TOTAL_KEYS=${#API_KEYS[@]}

if [[ $CONFIGURED -eq $TOTAL_KEYS ]]; then
    print_header "✅ ALL ${TOTAL_KEYS} API KEYS CONFIGURED"
    echo "All ${#COMMERCIAL_MODELS[@]} commercial models ready!"
else
    print_header "⚠️  ${CONFIGURED}/${TOTAL_KEYS} API Keys Configured"
    echo ""
    echo "To configure missing keys, add them to ${VMEVAL_ROOT}/.env:"
    echo ""
    for key in "${!API_KEYS[@]}"; do
        if ! check_api_key "$key"; then
            echo "  ${key}=your_key_here"
        fi
    done
    echo ""
    echo "Note: Commercial models are optional. Open-source models work without API keys."
fi

echo ""
echo "Next: ./setup/4_test_models.sh"
exit 0

