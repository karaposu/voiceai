"""
Run all ContextWeaver smoke tests

Usage: python -m contextweaver.smoke_tests.run_all_tests
"""

import asyncio
import sys
import os
import importlib
import time
from pathlib import Path


async def run_test_module(module_name: str):
    """Run a single test module"""
    print(f"\n{'='*60}")
    print(f"Running {module_name}")
    print(f"{'='*60}")
    
    try:
        # Import and run the module
        module = importlib.import_module(f".{module_name}", package="contextweaver.smoke_tests")
        
        # Get the main function
        if hasattr(module, 'main'):
            start_time = time.time()
            result = await module.main()
            elapsed = time.time() - start_time
            
            status = "PASSED" if result == 0 else "FAILED"
            print(f"\n{module_name}: {status} (took {elapsed:.2f}s)")
            return result == 0
        else:
            print(f"✗ {module_name} has no main() function")
            return False
            
    except Exception as e:
        print(f"✗ Failed to run {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all smoke tests"""
    print("ContextWeaver Smoke Test Suite")
    print("=" * 60)
    print("Running all tests...\n")
    
    # Find all test files
    test_dir = Path(__file__).parent
    test_files = sorted([
        f.stem for f in test_dir.glob("test_*.py")
        if f.stem != "run_all_tests"
    ])
    
    if not test_files:
        print("✗ No test files found!")
        return 1
    
    print(f"Found {len(test_files)} test modules:")
    for test in test_files:
        print(f"  - {test}")
    
    # Run all tests
    results = []
    start_time = time.time()
    
    for test_file in test_files:
        result = await run_test_module(test_file)
        results.append((test_file, result))
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Total time: {total_time:.2f}s")
    
    print("\nDetailed results:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print("\n✓ All ContextWeaver smoke tests passed!")
        print("\nContextWeaver is ready for use with:")
        print("- Intelligent context detection")
        print("- Flexible injection strategies")
        print("- High performance (< 50ms cycles)")
        print("- Thread-safe operations")
        print("- Automatic cleanup and expiration")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)