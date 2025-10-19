#!/usr/bin/env python3
"""
Quick test script to validate the new modular VMEvalKit architecture.
"""

import sys
from pathlib import Path

# Add VMEvalKit to path
sys.path.insert(0, str(Path(__file__).parent))

def test_catalog_import():
    """Test that we can import the MODEL_CATALOG."""
    print("🔍 Testing catalog import...")
    
    try:
        from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES
        print(f"✅ Imported catalog with {len(AVAILABLE_MODELS)} models")
        print(f"✅ Found {len(MODEL_FAMILIES)} families: {list(MODEL_FAMILIES.keys())}")
        return True
    except Exception as e:
        print(f"❌ Catalog import failed: {e}")
        return False

def test_dynamic_loading():
    """Test dynamic loading of model wrappers."""
    print("\n🔍 Testing dynamic loading...")
    
    try:
        from vmevalkit.runner.inference import _load_model_wrapper
        
        # Test loading a few different model types
        test_models = [
            "luma-ray-2",
            "veo-2.0-generate", 
            "wavespeed-wan-2.2-i2v-720p",
            "ltx-video-13b-distilled"
        ]
        
        for model_name in test_models:
            try:
                wrapper_class = _load_model_wrapper(model_name)
                print(f"✅ Successfully loaded {model_name}: {wrapper_class.__name__}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Dynamic loading test failed: {e}")
        return False

def test_runner_creation():
    """Test that InferenceRunner can be created."""
    print("\n🔍 Testing runner creation...")
    
    try:
        from vmevalkit.runner.inference import InferenceRunner
        runner = InferenceRunner(output_dir="./test_output")
        
        print(f"✅ Created InferenceRunner")
        
        # Test model listing
        models = runner.list_models()
        families = runner.list_models_by_family()
        
        print(f"✅ Listed {len(models)} models")
        print(f"✅ Listed {len(families)} families")
        
        return True
    except Exception as e:
        print(f"❌ Runner creation failed: {e}")
        return False

def test_base_class_inheritance():
    """Test that wrapper classes properly inherit from ModelWrapper."""
    print("\n🔍 Testing base class inheritance...")
    
    try:
        from vmevalkit.models.base import ModelWrapper
        from vmevalkit.models.luma_inference import LumaWrapper
        from vmevalkit.models.veo_inference import VeoWrapper
        
        # Check inheritance
        assert issubclass(LumaWrapper, ModelWrapper), "LumaWrapper should inherit from ModelWrapper"
        assert issubclass(VeoWrapper, ModelWrapper), "VeoWrapper should inherit from ModelWrapper"
        
        print(f"✅ Wrapper classes properly inherit from ModelWrapper")
        return True
    except Exception as e:
        print(f"❌ Base class inheritance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing VMEvalKit New Modular Architecture\n")
    
    tests = [
        test_catalog_import,
        test_dynamic_loading, 
        test_runner_creation,
        test_base_class_inheritance
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! New architecture is working! 🎉")
        return 0
    else:
        print("⚠️  Some tests failed. Architecture needs fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
