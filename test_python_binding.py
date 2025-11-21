#!/usr/bin/env python3
"""
Quick test script to verify the Python binding works correctly
"""

import sys
from pathlib import Path

# Add library path
sys.path.insert(0, str(Path(__file__).parent / "bin"))

def test_import():
    """Test that we can import the library"""
    print("🧪 Test 1: Importing library...")
    try:
        import libneurogen
        print("✓ Library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Build the library with: make lib")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🧪 Test 2: Creating model...")
    try:
        import libneurogen
        model = libneurogen.NeuroGenModel(
            vocab_size=1000,
            embedding_dim=128,
            gpu_device=0
        )
        print("✓ Model created successfully")
        return True, model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False, None

def test_forward_pass(model):
    """Test forward pass"""
    print("\n🧪 Test 3: Forward pass...")
    try:
        # Create dummy embedding
        embedding = [0.1] * 128
        output = model.forward(embedding)
        print(f"✓ Forward pass successful (output size: {len(output)})")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_training_step(model):
    """Test training step"""
    print("\n🧪 Test 4: Training step...")
    try:
        # Simple sequence: [1, 2, 3, 4, 5]
        input_ids = [1, 2, 3, 4]
        target_ids = [2, 3, 4, 5]
        
        loss = model.train_step(input_ids, target_ids)
        print(f"✓ Training step successful (loss: {loss:.4f})")
        return True
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

def test_generation(model):
    """Test text generation"""
    print("\n🧪 Test 5: Text generation...")
    try:
        prompt_ids = [1, 2, 3]
        generated = model.generate(
            prompt_token_ids=prompt_ids,
            max_length=10,
            temperature=1.0
        )
        print(f"✓ Generation successful (length: {len(generated)})")
        print(f"   Input:  {prompt_ids}")
        print(f"   Output: {generated}")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def test_statistics(model):
    """Test statistics retrieval"""
    print("\n🧪 Test 6: Statistics...")
    try:
        stats = model.get_statistics()
        print(f"✓ Statistics retrieved ({len(stats)} metrics)")
        print("   Sample metrics:")
        for i, (key, value) in enumerate(list(stats.items())[:5]):
            print(f"     {key}: {value:.6f}")
        return True
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
        return False

def test_checkpoint(model):
    """Test checkpoint save/load"""
    print("\n🧪 Test 7: Checkpoint save/load...")
    try:
        checkpoint_path = "test_checkpoint.bin"
        
        # Save
        model.save_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Load
        model.load_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        
        # Cleanup
        import os
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        return True
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("NeuroGen Python Binding Test Suite")
    print("="*60)
    
    # Test import
    if not test_import():
        print("\n❌ Cannot proceed without successful import")
        print("   1. Make sure you've built the library: make lib")
        print("   2. Check that bin/libneurogen.so exists")
        return
    
    # Test model creation
    success, model = test_model_creation()
    if not success:
        print("\n❌ Cannot proceed without model creation")
        return
    
    # Run remaining tests
    tests = [
        ("Forward Pass", lambda: test_forward_pass(model)),
        ("Training Step", lambda: test_training_step(model)),
        ("Generation", lambda: test_generation(model)),
        ("Statistics", lambda: test_statistics(model)),
        ("Checkpoint", lambda: test_checkpoint(model)),
    ]
    
    passed = 2  # Import and model creation
    total = 2 + len(tests)
    
    for name, test_func in tests:
        if test_func():
            passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Test Summary: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✓ All tests passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Install Python dependencies: pip install -r requirements.txt")
        print("  2. Run training: python train_slimpajama.py")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("   Check the error messages above for details")

if __name__ == "__main__":
    main()

