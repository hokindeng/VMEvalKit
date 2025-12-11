#!/usr/bin/env python3
"""
Test script for CogVideoX integration in VMEvalKit.

Usage:
    python3 test_cogvideox.py --model cogvideox-5b-i2v
    python3 test_cogvideox.py --model cogvideox1.5-5b-i2v
"""

import sys
from pathlib import Path

# Add VMEvalKit to path
sys.path.insert(0, str(Path(__file__).parent))

from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES


def test_catalog_registration():
    """Test that models are registered in the catalog."""
    print("=" * 70)
    print("TEST 1: Catalog Registration")
    print("=" * 70)
    
    cogvideox_models = ['cogvideox-5b-i2v', 'cogvideox1.5-5b-i2v']
    all_passed = True
    
    for model in cogvideox_models:
        if model in AVAILABLE_MODELS:
            print(f"✅ {model} registered in AVAILABLE_MODELS")
            config = AVAILABLE_MODELS[model]
            print(f"   - Family: {config['family']}")
            print(f"   - Model ID: {config['model']}")
            print(f"   - Wrapper: {config['wrapper_class']}")
            print(f"   - Service: {config['service_class']}")
            print(f"   - Resolution: {config['args']['resolution']}")
            print(f"   - Frames: {config['args']['num_frames']} @ {config['args']['fps']}fps")
            print(f"   - Guidance Scale: {config['args']['guidance_scale']}")
        else:
            print(f"❌ {model} NOT registered")
            all_passed = False
    
    # Check family registration
    if 'CogVideoX' in MODEL_FAMILIES:
        print(f"\n✅ CogVideoX family registered with {len(MODEL_FAMILIES['CogVideoX'])} models")
    else:
        print(f"\n❌ CogVideoX family NOT registered")
        all_passed = False
    
    return all_passed


def test_model_loading():
    """Test that models can be loaded via dynamic import."""
    print("\n" + "=" * 70)
    print("TEST 2: Dynamic Model Loading")
    print("=" * 70)
    
    try:
        from vmevalkit.runner.inference import _load_model_wrapper
        
        all_passed = True
        for model_name in ['cogvideox-5b-i2v', 'cogvideox1.5-5b-i2v']:
            try:
                wrapper_class = _load_model_wrapper(model_name)
                print(f"✅ {model_name}: Successfully loaded {wrapper_class.__name__}")
            except ImportError as e:
                if "torch" in str(e) or "diffusers" in str(e):
                    print(f"⚠️  {model_name}: Import skipped (dependencies not installed yet)")
                    print(f"   Run: bash setup/install_model.sh {model_name}")
                else:
                    print(f"❌ {model_name}: Failed to load - {e}")
                    all_passed = False
            except Exception as e:
                print(f"❌ {model_name}: Failed to load - {e}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ Failed to import inference module: {e}")
        return False


def test_pydantic_models():
    """Test Pydantic model definitions."""
    print("\n" + "=" * 70)
    print("TEST 3: Pydantic Model Validation")
    print("=" * 70)
    
    try:
        from vmevalkit.models.cogvideox_inference import CogVideoXConfig, GenerationResult
        
        # Test config creation
        config = CogVideoXConfig(
            model_id="THUDM/CogVideoX-5b-I2V",
            resolution=(720, 480),
            num_frames=49,
            fps=8,
            guidance_scale=6.0
        )
        print(f"✅ CogVideoXConfig created successfully")
        print(f"   - Resolution: {config.resolution}")
        print(f"   - Frames: {config.num_frames} @ {config.fps}fps")
        print(f"   - Guidance Scale: {config.guidance_scale}")
        
        # Test result model
        result = GenerationResult(
            success=True,
            video_path="/path/to/video.mp4",
            error=None,
            duration_seconds=45.2,
            generation_id="test_123",
            model="test-model",
            status="success",
            metadata={"test": "data"}
        )
        print(f"✅ GenerationResult created successfully")
        print(f"   - Success: {result.success}")
        print(f"   - Status: {result.status}")
        
        # Test validation
        try:
            bad_config = CogVideoXConfig(
                model_id="test",
                resolution=(100, 100),  # Too small
                num_frames=49,
                fps=8
            )
            print(f"❌ Validation failed to catch invalid resolution")
            return False
        except ValueError as e:
            print(f"✅ Resolution validation working: {str(e)[:50]}...")
        
        return True
        
    except ImportError as e:
        if "pydantic" in str(e):
            print(f"⚠️  Pydantic not installed (will be installed during setup)")
            return True
        else:
            print(f"❌ Failed to import CogVideoX models: {e}")
            return False
    except Exception as e:
        print(f"❌ Pydantic test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CogVideoX Integration Tests for VMEvalKit")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Catalog Registration", test_catalog_registration()))
    results.append(("Dynamic Model Loading", test_model_loading()))
    results.append(("Pydantic Models", test_pydantic_models()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Install models:")
        print("   bash setup/install_model.sh cogvideox-5b-i2v")
        print("   bash setup/install_model.sh cogvideox1.5-5b-i2v")
        print("\n2. Test inference:")
        print("   python3 examples/generate_videos.py --model cogvideox-5b-i2v --task-id tests_0001")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

