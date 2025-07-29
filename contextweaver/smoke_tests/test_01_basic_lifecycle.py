"""
Test 01: Basic Engine Lifecycle

Tests basic contextweaver initialization, start, stop operations.

python -m contextweaver.smoke_tests.test_01_basic_lifecycle
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from contextweaver import ContextWeaver
from contextweaver.strategies import ConservativeStrategy, AggressiveStrategy


async def test_basic_initialization():
    """Test basic engine initialization"""
    print("\n=== Test 1: Basic Initialization ===")
    
    # Create engine with defaults
    engine = ContextWeaver()
    
    # Check defaults
    assert engine.is_active == False
    assert engine.strategy is not None
    assert len(engine.detectors) == 3  # Default detectors
    print("✓ Engine created with default configuration")
    
    # Create with custom strategy
    custom_engine = ContextWeaver(strategy=AggressiveStrategy())
    assert custom_engine.strategy.name == "aggressive"
    print("✓ Engine created with custom strategy")
    
    return True


async def test_start_stop_cycle():
    """Test start/stop operations"""
    print("\n=== Test 2: Start/Stop Cycle ===")
    
    engine = ContextWeaver()
    
    # Start engine
    await engine.start()
    assert engine.is_active == True
    print("✓ Engine started successfully")
    
    # Start again (should be idempotent)
    await engine.start()
    assert engine.is_active == True
    print("✓ Double start handled correctly")
    
    # Stop engine
    await engine.stop()
    assert engine.is_active == False
    print("✓ Engine stopped successfully")
    
    # Stop again (should be idempotent)
    await engine.stop()
    assert engine.is_active == False
    print("✓ Double stop handled correctly")
    
    return True


async def test_multiple_engines():
    """Test multiple engine instances"""
    print("\n=== Test 3: Multiple Engine Instances ===")
    
    # Create multiple engines with different strategies
    conservative_engine = ContextWeaver(strategy=ConservativeStrategy())
    aggressive_engine = ContextWeaver(strategy=AggressiveStrategy())
    
    # Start both
    await conservative_engine.start()
    await aggressive_engine.start()
    
    assert conservative_engine.is_active == True
    assert aggressive_engine.is_active == True
    assert conservative_engine.strategy.name != aggressive_engine.strategy.name
    print("✓ Multiple engines running independently")
    
    # Stop them
    await conservative_engine.stop()
    await aggressive_engine.stop()
    print("✓ Multiple engines stopped successfully")
    
    return True


async def test_engine_with_context_manager():
    """Test engine as async context manager pattern"""
    print("\n=== Test 4: Context Manager Pattern ===")
    
    # Note: This test assumes we'll add __aenter__ and __aexit__ to ContextWeaver
    # For now, we'll simulate the pattern
    
    engine = ContextWeaver()
    
    try:
        await engine.start()
        assert engine.is_active == True
        print("✓ Engine started in context")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
    finally:
        await engine.stop()
        assert engine.is_active == False
        print("✓ Engine cleaned up properly")
    
    return True


async def main():
    """Run all basic lifecycle tests"""
    print("ContextWeaver Basic Lifecycle Tests")
    print("=" * 50)
    
    tests = [
        test_basic_initialization,
        test_start_stop_cycle,
        test_multiple_engines,
        test_engine_with_context_manager
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\nSummary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All lifecycle tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)