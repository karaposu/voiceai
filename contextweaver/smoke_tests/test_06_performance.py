"""
Test 06: Performance and Scale

Tests contextweaver performance under load.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from contextweaver import ContextWeaver
from contextweaver.strategies import AggressiveStrategy
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority
import time
import random
from datetime import datetime
import gc


async def test_many_contexts():
    """Test with many contexts"""
    print("\n=== Test 1: Many Contexts (100+) ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Add 100 contexts
    start_time = time.time()
    
    for i in range(100):
        context = ContextToInject(
            information={"id": i, "data": f"context_{i}" * 10},  # Some data
            strategy={"param": random.random()},
            attention={"focus": f"item_{i}"},
            priority=random.choice(list(ContextPriority)),
            timing=random.choice(list(InjectionTiming)),
            ttl_seconds=random.randint(60, 300) if i % 3 == 0 else None
        )
        engine.add_context(context)
    
    add_time = time.time() - start_time
    print(f"✓ Added 100 contexts in {add_time:.3f}s ({add_time/100*1000:.2f}ms per context)")
    
    # Test retrieval performance
    state = {"message_count": 5}
    
    start_time = time.time()
    relevant = engine.get_relevant_contexts(state, max_items=10)
    retrieve_time = time.time() - start_time
    
    print(f"✓ Retrieved top 10 contexts in {retrieve_time*1000:.2f}ms")
    assert len(relevant) == 10
    
    # Test injection check performance
    start_time = time.time()
    result = await engine.check_injection(state)
    check_time = time.time() - start_time
    
    print(f"✓ Injection check completed in {check_time*1000:.2f}ms")
    
    await engine.stop()
    return True


async def test_rapid_updates():
    """Test rapid context updates"""
    print("\n=== Test 2: Rapid Context Updates ===")
    
    engine = ContextWeaver(strategy=AggressiveStrategy())
    await engine.start()
    
    # Simulate rapid context updates
    update_count = 1000
    start_time = time.time()
    
    for i in range(update_count):
        # Alternate between adding and removing
        if i % 2 == 0:
            context = engine.add_raw_context(
                information={"counter": i},
                timing=InjectionTiming.LAZY,
                priority=ContextPriority.LOW
            )
            if i % 4 == 0 and hasattr(locals(), 'last_context_id'):
                engine.remove_context(last_context_id)
            last_context_id = context.context_id
        
        # Periodic cleanup simulation
        if i % 100 == 0:
            state = {"counter": i}
            await engine.check_injection(state)
    
    elapsed = time.time() - start_time
    ops_per_sec = update_count / elapsed
    
    print(f"✓ Processed {update_count} operations in {elapsed:.2f}s ({ops_per_sec:.0f} ops/sec)")
    assert ops_per_sec > 100  # Should handle at least 100 ops/sec
    
    await engine.stop()
    return True


async def test_concurrent_operations():
    """Test concurrent access"""
    print("\n=== Test 3: Concurrent Operations ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Define concurrent tasks
    async def add_contexts(prefix, count):
        for i in range(count):
            engine.add_raw_context(
                information={f"{prefix}_id": i},
                timing=InjectionTiming.NEXT_PAUSE
            )
            await asyncio.sleep(0.001)  # Small delay
    
    async def check_injections(count):
        results = []
        for i in range(count):
            state = {"iteration": i}
            result = await engine.check_injection(state)
            results.append(result is not None)
            await asyncio.sleep(0.005)
        return results
    
    async def get_stats_periodically(count):
        stats = []
        for i in range(count):
            stats.append(engine.get_stats())
            await asyncio.sleep(0.01)
        return stats
    
    # Run concurrent operations
    start_time = time.time()
    
    results = await asyncio.gather(
        add_contexts("taskA", 50),
        add_contexts("taskB", 50),
        check_injections(20),
        get_stats_periodically(10),
        return_exceptions=True
    )
    
    elapsed = time.time() - start_time
    
    # Check for errors
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 0, f"Concurrent operations failed: {errors}"
    
    print(f"✓ Concurrent operations completed in {elapsed:.2f}s without errors")
    
    await engine.stop()
    return True


async def test_memory_efficiency():
    """Test memory usage"""
    print("\n=== Test 4: Memory Efficiency ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Force garbage collection for baseline
    gc.collect()
    
    # Add many large contexts
    large_data = "x" * 1000  # 1KB string
    
    for i in range(500):
        context = ContextToInject(
            information={"id": i, "large_data": large_data},
            strategy={"large_config": large_data},
            ttl_seconds=5,  # Short TTL for cleanup test
            timing=InjectionTiming.LAZY
        )
        engine.add_context(context)
    
    print(f"✓ Added 500 contexts with ~1KB data each")
    
    # Trigger cleanup of expired contexts
    await asyncio.sleep(0.1)
    
    # Add marker to track cleanup
    initial_count = len(engine.available_context)
    
    # Wait for expiration and trigger cleanup
    await asyncio.sleep(5.1)
    state = {}
    await engine.check_injection(state)  # Triggers cleanup
    
    final_count = len(engine.available_context)
    print(f"✓ Cleanup removed {initial_count - final_count} expired contexts")
    assert final_count < initial_count
    
    await engine.stop()
    return True


async def test_detection_cycle_performance():
    """Test full detection cycle performance"""
    print("\n=== Test 5: Detection Cycle Performance ===")
    
    engine = ContextWeaver(strategy=AggressiveStrategy())
    await engine.start()
    
    # Add realistic contexts
    for i in range(20):
        engine.add_raw_context(
            information={"topic": f"topic_{i%5}", "detail": f"detail_{i}"},
            strategy={"approach": f"strategy_{i%3}"},
            timing=random.choice([
                InjectionTiming.IMMEDIATE,
                InjectionTiming.NEXT_PAUSE,
                InjectionTiming.LAZY
            ]),
            priority=random.choice([
                ContextPriority.HIGH,
                ContextPriority.MEDIUM,
                ContextPriority.LOW
            ])
        )
    
    # Create realistic state
    state = type('State', (), {
        'audio': type('Audio', (), {
            'vad_active': False,
            'silence_duration_ms': 1500
        })(),
        'messages': [
            type('Message', (), {'content': f"Message {i}"})()
            for i in range(5)
        ],
        'current_turn': type('Turn', (), {
            'user_message': type('Message', (), {'content': "Question"})(),
            'assistant_message': None,
            'started_at': datetime.now()
        })(),
        'metrics': type('Metrics', (), {'interruption_count': 1})(),
        'turns': [type('Turn', (), {'is_complete': True})() for _ in range(3)]
    })()
    
    # Measure full cycle performance
    cycle_times = []
    
    for _ in range(100):
        start = time.time()
        result = await engine.check_injection(state)
        cycle_times.append((time.time() - start) * 1000)
    
    avg_cycle_ms = sum(cycle_times) / len(cycle_times)
    max_cycle_ms = max(cycle_times)
    min_cycle_ms = min(cycle_times)
    
    print(f"✓ Detection cycle performance:")
    print(f"  - Average: {avg_cycle_ms:.2f}ms")
    print(f"  - Min: {min_cycle_ms:.2f}ms")
    print(f"  - Max: {max_cycle_ms:.2f}ms")
    
    assert avg_cycle_ms < 50  # Should average under 50ms
    assert max_cycle_ms < 100  # No cycle should exceed 100ms
    
    await engine.stop()
    return True


async def test_strategy_switching():
    """Test performance of strategy switching"""
    print("\n=== Test 6: Strategy Switching Performance ===")
    
    from contextweaver.strategies import ConservativeStrategy, AdaptiveStrategy
    
    strategies = [
        ConservativeStrategy(),
        AggressiveStrategy(),
        AdaptiveStrategy()
    ]
    
    # Add contexts once
    contexts = []
    for i in range(30):
        contexts.append(ContextToInject(
            information={"id": i},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.NEXT_PAUSE
        ))
    
    # Test each strategy
    for strategy in strategies:
        engine = ContextWeaver(strategy=strategy)
        await engine.start()
        
        # Add all contexts
        for ctx in contexts:
            engine.add_context(ctx)
        
        # Time decision making
        state = {"test": True}
        
        start_time = time.time()
        decisions = 0
        
        while time.time() - start_time < 1.0:  # Run for 1 second
            await engine.check_injection(state)
            decisions += 1
        
        print(f"✓ {strategy.name} strategy: {decisions} decisions/second")
        
        await engine.stop()
    
    return True


async def main():
    """Run all performance tests"""
    print("ContextWeaver Performance Tests")
    print("=" * 50)
    
    tests = [
        test_many_contexts,
        test_rapid_updates,
        test_concurrent_operations,
        test_memory_efficiency,
        test_detection_cycle_performance,
        test_strategy_switching
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
        print("✓ All performance tests passed!")
        print("\nPerformance characteristics:")
        print("- Handles 100+ contexts efficiently")
        print("- Processes 100+ operations per second")
        print("- Detection cycles under 50ms average")
        print("- Concurrent operations safe")
        print("- Memory efficient with automatic cleanup")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)