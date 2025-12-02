#!/bin/bash
##############################################################################
# VMEvalKit - Master Setup Script
#
# ONE command to set up everything:
#   ./setup/RUN_SETUP.sh
#
# Options:
#   --skip-download    Skip checkpoint downloads
#   --skip-validate    Skip validation
#   --yes              Skip confirmation prompt
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Parse arguments
SKIP_DOWNLOAD=false
SKIP_VALIDATE=false
AUTO_YES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)  SKIP_DOWNLOAD=true; shift ;;
        --skip-validate)  SKIP_VALIDATE=true; shift ;;
        --yes|-y)         AUTO_YES=true; shift ;;
        *)                shift ;;
    esac
done

# ============================================================================
# Welcome
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              VMEvalKit - Complete Setup                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will set up:"
echo "  • 4 virtual environments (torch 2.5.1 + 2.0.0)"
echo "  • 15 open-source model dependencies"
echo "  • ~24GB model checkpoints"
echo ""
echo "Estimated time: 30-60 minutes"
echo "Disk space needed: ~50GB"
echo ""

if [[ "$AUTO_YES" != "true" ]]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 1
    fi
fi

START_TIME=$(date +%s)

# ============================================================================
# Execute Setup Steps
# ============================================================================

# Step 1: Install dependencies
"${SCRIPT_DIR}/1_install_dependencies.sh"

# Step 2: Download checkpoints (optional)
if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
    "${SCRIPT_DIR}/2_download_checkpoints.sh"
else
    print_info "Skipping checkpoint download (--skip-download)"
fi

# Step 3: Validate (optional)
if [[ "$SKIP_VALIDATE" != "true" ]]; then
    "${SCRIPT_DIR}/3a_validate_opensource.sh"
    "${SCRIPT_DIR}/3b_validate_commercial.sh" || true  # Don't fail on missing API keys
else
    print_info "Skipping validation (--skip-validate)"
fi

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ SETUP COMPLETE!                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Test open-source models:"
echo "   ./setup/4_test_models.sh --opensource"
echo ""
echo "2. Test commercial models (if API keys configured):"
echo "   ./setup/4_test_models.sh --commercial"
echo ""
echo "3. Run full inference:"
echo "   ./run_all_models.sh --parallel"
echo ""
echo "4. Run single model:"
echo "   source envs/venv_main/bin/activate"
echo "   python examples/generate_videos.py --model ltx-video --all-tasks"
echo ""
