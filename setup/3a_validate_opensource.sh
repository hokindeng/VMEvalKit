#!/bin/bash
##############################################################################
# Step 3a: Validate Open-Source Setup
#
# Validates all open-source components:
#   - Virtual environments exist and work
#   - Model checkpoints downloaded
#   - Python imports succeed
#   - Test assets available
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

print_header "Step 3a: Validating Open-Source Setup"

ERRORS=0

# ============================================================================
# Check 1: Virtual Environments
# ============================================================================
print_section "[1/4] Virtual Environments"

for venv in venv_main venv_hunyuan venv_dynamicrafter venv_videocrafter; do
    if venv_exists "$venv"; then
        version=$("${ENVS_DIR}/${venv}/bin/python" --version 2>&1)
        print_success "${venv} (${version})"
    else
        print_error "${venv} - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# Check 2: Model Checkpoints
# ============================================================================
print_section "[2/4] Model Checkpoints"

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r path url size_desc <<< "$entry"
    name="$(basename "$(dirname "$path")")"
    
    if checkpoint_exists "$path"; then
        actual_size=$(get_checkpoint_size "$path")
        print_success "${name} (${actual_size})"
    else
        print_error "${name} - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# Check 3: Python Imports
# ============================================================================
print_section "[3/4] Python Imports"

# venv_main imports
activate_venv "venv_main"
if python3 -c "import torch, diffusers, transformers" 2>/dev/null; then
    torch_ver=$(python3 -c "import torch; print(torch.__version__)")
    diffusers_ver=$(python3 -c "import diffusers; print(diffusers.__version__)")
    print_success "venv_main: torch ${torch_ver}, diffusers ${diffusers_ver}"
else
    print_error "venv_main imports failed"
    ERRORS=$((ERRORS + 1))
fi
deactivate

# venv_hunyuan imports
activate_venv "venv_hunyuan"
if python3 -c "import torch, loguru" 2>/dev/null; then
    torch_ver=$(python3 -c "import torch; print(torch.__version__)")
    print_success "venv_hunyuan: torch ${torch_ver}, loguru OK"
else
    print_error "venv_hunyuan imports failed"
    ERRORS=$((ERRORS + 1))
fi
deactivate

# venv_dynamicrafter imports
activate_venv "venv_dynamicrafter"
if python3 -c "import torch, pytorch_lightning" 2>/dev/null; then
    torch_ver=$(python3 -c "import torch; print(torch.__version__)")
    print_success "venv_dynamicrafter: torch ${torch_ver}, pytorch_lightning OK"
else
    print_error "venv_dynamicrafter imports failed"
    ERRORS=$((ERRORS + 1))
fi
deactivate

# venv_videocrafter imports
activate_venv "venv_videocrafter"
if python3 -c "import torch, omegaconf" 2>/dev/null; then
    torch_ver=$(python3 -c "import torch; print(torch.__version__)")
    print_success "venv_videocrafter: torch ${torch_ver}, omegaconf OK"
else
    print_error "venv_videocrafter imports failed"
    ERRORS=$((ERRORS + 1))
fi
deactivate

# ============================================================================
# Check 4: Test Assets
# ============================================================================
print_section "[4/4] Test Assets"

for i in 1 2; do
    test_dir="${TESTS_DIR}/assets/example_question_${i}"
    if [[ -f "${test_dir}/first_frame.png" ]] && [[ -f "${test_dir}/prompt.txt" ]]; then
        print_success "example_question_${i}"
    else
        print_error "example_question_${i} - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo ""
if [[ $ERRORS -eq 0 ]]; then
    print_header "✅ OPEN-SOURCE VALIDATION PASSED"
    echo "All ${#OPENSOURCE_MODELS[@]} open-source models ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Validate commercial APIs: ./setup/3b_validate_commercial.sh"
    echo "  2. Test models: ./setup/4_test_models.sh"
    exit 0
else
    print_header "❌ VALIDATION FAILED - ${ERRORS} error(s)"
    echo "Fix the issues above, then re-run this script."
    exit 1
fi

