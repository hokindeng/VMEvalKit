#!/bin/bash
##############################################################################
# Step 4: Test Models - Generate Sample Videos
#
# Tests models by generating 2 videos each from example questions.
# Supports testing open-source, commercial, or all models.
#
# Usage:
#   ./setup/4_test_models.sh                  # Test all available
#   ./setup/4_test_models.sh --opensource     # Test only open-source
#   ./setup/4_test_models.sh --commercial     # Test only commercial
#   ./setup/4_test_models.sh --model ltx-video # Test specific model
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Parse arguments
TEST_MODE="all"
SPECIFIC_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --opensource)
            TEST_MODE="opensource"
            shift
            ;;
        --commercial)
            TEST_MODE="commercial"
            shift
            ;;
        --model)
            TEST_MODE="specific"
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--opensource|--commercial|--model MODEL_NAME]"
            exit 1
            ;;
    esac
done

# Load environment for commercial API keys
load_env_file

print_header "Step 4: Testing Models"

# Verify test questions exist
for i in 1 2; do
    test_dir="${TESTS_DIR}/assets/example_question_${i}"
    if [[ ! -d "$test_dir" ]]; then
        print_error "Test questions not found in ${test_dir}"
        exit 1
    fi
done
print_success "Test questions ready"
echo ""

# Setup output directories
TEST_OUTPUT="${TESTS_DIR}/outputs/model_validation"
LOGS_OUTPUT="${LOGS_DIR}/model_tests"
ensure_dir "$TEST_OUTPUT"
ensure_dir "$LOGS_OUTPUT"

# Prepare test data structure
DATA_TEST_DIR="${VMEVAL_ROOT}/data/questions/test_task"
ensure_dir "${DATA_TEST_DIR}/test_example_1"
ensure_dir "${DATA_TEST_DIR}/test_example_2"
cp "${TESTS_DIR}/assets/example_question_1/"* "${DATA_TEST_DIR}/test_example_1/"
cp "${TESTS_DIR}/assets/example_question_2/"* "${DATA_TEST_DIR}/test_example_2/"

# ============================================================================
# Build Model Test List
# ============================================================================

declare -a MODELS_TO_TEST=()

build_opensource_list() {
    for entry in "${OPENSOURCE_MODELS[@]}"; do
        IFS='|' read -r model venv <<< "$entry"
        MODELS_TO_TEST+=("${model}|${venv}|opensource")
    done
}

build_commercial_list() {
    for entry in "${COMMERCIAL_MODELS[@]}"; do
        IFS='|' read -r model env_var <<< "$entry"
        if check_api_key "$env_var"; then
            MODELS_TO_TEST+=("${model}|venv_main|commercial")
        fi
    done
}

case $TEST_MODE in
    opensource)
        print_info "Testing OPEN-SOURCE models only"
        build_opensource_list
        ;;
    commercial)
        print_info "Testing COMMERCIAL models only (with configured API keys)"
        build_commercial_list
        ;;
    specific)
        print_info "Testing specific model: ${SPECIFIC_MODEL}"
        venv=$(get_venv_for_model "$SPECIFIC_MODEL")
        MODELS_TO_TEST+=("${SPECIFIC_MODEL}|${venv}|specific")
        ;;
    all)
        print_info "Testing ALL available models"
        build_opensource_list
        build_commercial_list
        ;;
esac

TOTAL=${#MODELS_TO_TEST[@]}

if [[ $TOTAL -eq 0 ]]; then
    print_error "No models to test. Check API keys for commercial models."
    exit 1
fi

echo "   Models to test: ${TOTAL}"
echo "   Videos per model: 2"
echo "   Expected output: $((TOTAL * 2)) videos"
echo ""

# ============================================================================
# Test Each Model
# ============================================================================

PASSED=0
FAILED=0
SKIPPED=0
IDX=0

for entry in "${MODELS_TO_TEST[@]}"; do
    IDX=$((IDX + 1))
    IFS='|' read -r model venv model_type <<< "$entry"
    
    print_section "[${IDX}/${TOTAL}] ${model} (${venv})"
    
    # Check venv exists
    if ! venv_exists "$venv"; then
        print_error "Virtual environment ${venv} not found"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Run test with timeout
    activate_venv "$venv"
    
    set +e
    timeout 600 python "${VMEVAL_ROOT}/examples/generate_videos.py" \
        --model "$model" \
        --task-id test_example_1 test_example_2 \
        --output-dir "$TEST_OUTPUT" \
        > "${LOGS_OUTPUT}/${model}.log" 2>&1
    
    EXIT_CODE=$?
    set -e
    
    deactivate
    
    # Count generated videos
    VIDEO_COUNT=$(find "$TEST_OUTPUT" -path "*/${model}/*" \( -name "*.mp4" -o -name "*.webm" \) 2>/dev/null | wc -l)
    
    # Evaluate result
    if [[ $EXIT_CODE -eq 0 ]] && [[ $VIDEO_COUNT -ge 2 ]]; then
        print_success "PASSED - ${VIDEO_COUNT} videos generated"
        PASSED=$((PASSED + 1))
    elif [[ $EXIT_CODE -eq 124 ]]; then
        print_warning "TIMEOUT (>10 min)"
        FAILED=$((FAILED + 1))
    elif [[ $VIDEO_COUNT -gt 0 ]]; then
        print_warning "PARTIAL - only ${VIDEO_COUNT} video(s)"
        FAILED=$((FAILED + 1))
    else
        print_error "FAILED - see ${LOGS_OUTPUT}/${model}.log"
        FAILED=$((FAILED + 1))
    fi
done

# Cleanup test data
rm -rf "${DATA_TEST_DIR}"

# ============================================================================
# Summary
# ============================================================================

print_header "TEST RESULTS"

echo "   Total tested: ${TOTAL}"
echo "   ✅ Passed:    ${PASSED}"
echo "   ❌ Failed:    ${FAILED}"
echo "   ⏭️  Skipped:  ${SKIPPED}"
echo ""

TESTED=$((TOTAL - SKIPPED))
if [[ $TESTED -gt 0 ]]; then
    SUCCESS_RATE=$(python3 -c "print(f'{100*${PASSED}/${TESTED}:.1f}%')" 2>/dev/null || echo "N/A")
    echo "   Success rate: ${SUCCESS_RATE}"
fi

echo ""
echo "📁 Test videos: ${TEST_OUTPUT}"
echo "📝 Logs: ${LOGS_OUTPUT}"
echo ""

if [[ $PASSED -eq $TESTED ]]; then
    echo "✅ ALL TESTED MODELS WORKING!"
    exit 0
else
    echo "⚠️  Some models failed. Check logs for details."
    exit 1
fi

