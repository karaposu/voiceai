"""
Test 05: Injection Decision Flow

Tests the complete injection decision flow with real scenarios.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from contextweaver import ContextWeaver
from contextweaver.strategies import ConservativeStrategy, AggressiveStrategy
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority
from datetime import datetime
import time


async def test_basic_injection_flow():
    """Test basic injection decision flow"""
    print("\n=== Test 1: Basic Injection Flow ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Add context
    context = ContextToInject(
        information={"message": "Hello from injection"},
        priority=ContextPriority.HIGH,
        timing=InjectionTiming.NEXT_PAUSE
    )
    engine.add_context(context)
    
    # Create state that should trigger injection
    state = type('State', (), {
        'audio': type('Audio', (), {'vad_active': False})(),  # Silent
        'messages': [
            type('Message', (), {'content': "User said something"})()
        ]
    })()
    
    # Wait for silence detection
    await asyncio.sleep(1.5)  # Let silence detector trigger
    
    # Check injection
    injected_context = await engine.check_injection(state)
    
    assert injected_context is not None
    assert injected_context.context_id == context.context_id
    assert injected_context.injection_count == 1
    print("✓ Basic injection flow works")
    
    await engine.stop()
    return True


async def test_timing_based_injection():
    """Test different timing modes"""
    print("\n=== Test 2: Timing-Based Injection ===")
    
    engine = ContextWeaver(strategy=AggressiveStrategy())
    await engine.start()
    
    # Add contexts with different timings
    immediate = ContextToInject(
        information={"timing": "immediate"},
        priority=ContextPriority.CRITICAL,
        timing=InjectionTiming.IMMEDIATE
    )
    
    lazy = ContextToInject(
        information={"timing": "lazy"},
        priority=ContextPriority.LOW,
        timing=InjectionTiming.LAZY
    )
    
    next_pause = ContextToInject(
        information={"timing": "next_pause"},
        priority=ContextPriority.MEDIUM,
        timing=InjectionTiming.NEXT_PAUSE
    )
    
    engine.add_context(immediate)
    engine.add_context(lazy)
    engine.add_context(next_pause)
    
    # State with good detection
    state = type('State', (), {
        'audio': type('Audio', (), {'vad_active': False})(),
        'messages': []
    })()
    
    # Check what gets injected (should be immediate due to priority + timing)
    await asyncio.sleep(0.6)  # Some detection time
    injected = await engine.check_injection(state)
    
    # Aggressive strategy with critical priority should inject immediate
    assert injected is not None
    print(f"✓ Injected context with timing: {injected.information['timing']}")
    
    await engine.stop()
    return True


async def test_condition_based_injection():
    """Test conditional injection"""
    print("\n=== Test 3: Condition-Based Injection ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Context with conditions
    conditional = ContextToInject(
        information={"help": "context"},
        conditions={
            "topics_include": ["support", "help"],
            "conversation_length_min": 2
        },
        timing=InjectionTiming.ON_TRIGGER,
        priority=ContextPriority.HIGH
    )
    
    engine.add_context(conditional)
    
    # State that doesn't meet conditions
    state1 = {
        'current_topics': ["general"],
        'message_count': 1,
        'audio': type('Audio', (), {'vad_active': False})(),
        'messages': []
    }
    
    result1 = await engine.check_injection(state1)
    assert result1 is None
    print("✓ Correctly skips context with unmet conditions")
    
    # State that meets conditions
    state2 = {
        'current_topics': ["support", "billing"],
        'message_count': 3,
        'audio': type('Audio', (), {'vad_active': False})(),
        'messages': []
    }
    
    # Need to wait for detection
    await asyncio.sleep(1.1)
    result2 = await engine.check_injection(state2)
    
    # Conservative strategy might still not inject, but context should be valid
    relevant = engine.get_relevant_contexts(state2)
    assert len(relevant) > 0
    assert relevant[0].context_id == conditional.context_id
    print("✓ Context becomes available when conditions are met")
    
    await engine.stop()
    return True


async def test_priority_competition():
    """Test multiple contexts competing"""
    print("\n=== Test 4: Priority Competition ===")
    
    engine = ContextWeaver(strategy=AggressiveStrategy())
    await engine.start()
    
    # Add multiple contexts
    contexts = [
        ContextToInject(
            information={"priority": "critical", "id": 1},
            priority=ContextPriority.CRITICAL,
            timing=InjectionTiming.LAZY
        ),
        ContextToInject(
            information={"priority": "high", "id": 2},
            priority=ContextPriority.HIGH,
            timing=InjectionTiming.IMMEDIATE  # Better timing
        ),
        ContextToInject(
            information={"priority": "medium", "id": 3},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.IMMEDIATE
        )
    ]
    
    for ctx in contexts:
        engine.add_context(ctx)
    
    # Let detectors run
    state = type('State', (), {
        'audio': type('Audio', (), {'vad_active': False})(),
        'messages': []
    })()
    
    await asyncio.sleep(0.6)
    injected = await engine.check_injection(state)
    
    # Should inject based on combined priority + timing score
    assert injected is not None
    print(f"✓ Injected context with priority: {injected.information['priority']}")
    
    await engine.stop()
    return True


async def test_injection_with_expiry():
    """Test injection with TTL and expiry"""
    print("\n=== Test 5: Injection with Expiry ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Add contexts with different TTLs
    short_lived = ContextToInject(
        information={"ttl": "short"},
        ttl_seconds=1,
        priority=ContextPriority.HIGH,
        timing=InjectionTiming.IMMEDIATE
    )
    
    long_lived = ContextToInject(
        information={"ttl": "long"},
        ttl_seconds=10,
        priority=ContextPriority.MEDIUM,
        timing=InjectionTiming.NEXT_PAUSE
    )
    
    engine.add_context(short_lived)
    engine.add_context(long_lived)
    
    # Wait for short one to expire
    await asyncio.sleep(1.2)
    
    state = type('State', (), {
        'audio': type('Audio', (), {'vad_active': False})(),
        'messages': []
    })()
    
    # Check what's available
    relevant = engine.get_relevant_contexts(state)
    
    # Only long-lived should remain
    assert len(relevant) == 1
    assert relevant[0].information["ttl"] == "long"
    print("✓ Expired contexts are automatically cleaned")
    
    await engine.stop()
    return True


async def test_real_conversation_simulation():
    """Simulate a real conversation flow"""
    print("\n=== Test 6: Real Conversation Simulation ===")
    
    engine = ContextWeaver(strategy=AggressiveStrategy(min_interval_seconds=1))
    await engine.start()
    
    # Track injections
    injections = []
    
    # Simulate conversation phases
    
    # Phase 1: Greeting
    greeting_context = engine.add_raw_context(
        information={"phase": "greeting", "name": "John"},
        strategy={"tone": "friendly"},
        timing=InjectionTiming.IMMEDIATE,
        priority=ContextPriority.HIGH
    )
    
    state = type('State', (), {
        'audio': type('Audio', (), {'vad_active': False})(),
        'messages': [
            type('Message', (), {'content': "Hello"})()
        ],
        'message_count': 1
    })()
    
    await asyncio.sleep(0.1)
    result = await engine.check_injection(state)
    if result:
        injections.append(result.information.get("phase"))
    
    # Phase 2: Topic introduced
    await asyncio.sleep(1.1)  # Pass cooldown
    
    support_context = engine.add_raw_context(
        information={"phase": "support", "department": "billing"},
        strategy={"patience": "high"},
        attention={"focus": "problem_solving"},
        timing=InjectionTiming.NEXT_PAUSE,
        priority=ContextPriority.HIGH,
        conditions={"topics_include": ["billing", "payment"]}
    )
    
    state.messages.append(
        type('Message', (), {'content': "I have a billing issue"})()
    )
    state.current_topics = ["billing", "support"]
    state.message_count = 2
    
    result = await engine.check_injection(state)
    if result:
        injections.append(result.information.get("phase"))
    
    # Phase 3: Urgency detected
    await asyncio.sleep(1.1)
    
    urgent_context = engine.add_raw_context(
        information={"phase": "urgent", "escalate": True},
        strategy={"urgency": "high"},
        timing=InjectionTiming.IMMEDIATE,
        priority=ContextPriority.CRITICAL
    )
    
    state.messages.append(
        type('Message', (), {'content': "This is urgent! I need help now!"})()
    )
    
    result = await engine.check_injection(state)
    if result:
        injections.append(result.information.get("phase"))
    
    print(f"✓ Conversation phases injected: {injections}")
    assert len(injections) >= 2  # At least some injections happened
    
    # Check stats
    stats = engine.get_stats()
    print(f"✓ Final stats: {stats['context_items']} contexts, "
          f"{stats['injection_rate_per_min']:.2f} injections/min")
    
    await engine.stop()
    return True


async def main():
    """Run all injection flow tests"""
    print("ContextWeaver Injection Flow Tests")
    print("=" * 50)
    
    tests = [
        test_basic_injection_flow,
        test_timing_based_injection,
        test_condition_based_injection,
        test_priority_competition,
        test_injection_with_expiry,
        test_real_conversation_simulation
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
        print("✓ All injection flow tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)