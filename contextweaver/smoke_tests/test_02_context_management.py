"""
Test 02: Context Management

Tests adding, removing, and managing ContextToInject objects.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from contextweaver import ContextWeaver
from contextengine.schema import (
    ContextToInject, InjectionTiming, ContextPriority,
    create_immediate_context, create_conditional_context
)
import time


async def test_add_and_remove_context():
    """Test adding and removing contexts"""
    print("\n=== Test 1: Add and Remove Context ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Create context
    context = ContextToInject(
        information={"user_name": "John", "account_type": "premium"},
        strategy={"tone": "friendly", "formality": "casual"},
        attention={"focus": "problem_solving"},
        timing=InjectionTiming.NEXT_PAUSE,
        priority=ContextPriority.MEDIUM
    )
    
    # Add context
    engine.add_context(context)
    assert len(engine.available_context) == 1
    assert context.context_id in engine.available_context
    print("✓ Context added successfully")
    
    # Remove context
    engine.remove_context(context.context_id)
    assert len(engine.available_context) == 0
    print("✓ Context removed successfully")
    
    await engine.stop()
    return True


async def test_add_raw_context():
    """Test adding context from raw data"""
    print("\n=== Test 2: Add Raw Context ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Add raw context
    created_context = engine.add_raw_context(
        information={"session_type": "support"},
        strategy={"patience": "high"},
        attention={"listen_for": ["complaint", "issue"]},
        timing=InjectionTiming.IMMEDIATE,
        priority=ContextPriority.HIGH,
        source="test_suite"
    )
    
    assert len(engine.available_context) == 1
    assert created_context.context_id in engine.available_context
    assert created_context.source == "test_suite"
    assert created_context.priority == ContextPriority.HIGH
    print("✓ Raw context converted and added successfully")
    
    await engine.stop()
    return True


async def test_context_expiration():
    """Test context TTL and expiration"""
    print("\n=== Test 3: Context Expiration ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Create context with short TTL
    expiring_context = ContextToInject(
        information={"temporary": "data"},
        ttl_seconds=1,  # 1 second TTL
        timing=InjectionTiming.LAZY
    )
    
    # Create permanent context
    permanent_context = ContextToInject(
        information={"permanent": "data"},
        timing=InjectionTiming.LAZY
    )
    
    engine.add_context(expiring_context)
    engine.add_context(permanent_context)
    assert len(engine.available_context) == 2
    print("✓ Added expiring and permanent contexts")
    
    # Wait for expiration
    await asyncio.sleep(1.1)
    
    # Check injection (should trigger cleanup)
    result = await engine.check_injection({})
    
    # Verify expired context was removed
    assert len(engine.available_context) == 1
    assert expiring_context.context_id not in engine.available_context
    assert permanent_context.context_id in engine.available_context
    print("✓ Expired context automatically removed")
    
    await engine.stop()
    return True


async def test_context_priorities():
    """Test context priority ordering"""
    print("\n=== Test 4: Context Priority Ordering ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Add contexts with different priorities
    contexts = [
        ContextToInject(
            information={"priority": "low"},
            priority=ContextPriority.LOW,
            timing=InjectionTiming.NEXT_PAUSE
        ),
        ContextToInject(
            information={"priority": "critical"},
            priority=ContextPriority.CRITICAL,
            timing=InjectionTiming.NEXT_PAUSE
        ),
        ContextToInject(
            information={"priority": "medium"},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.NEXT_PAUSE
        ),
        ContextToInject(
            information={"priority": "high"},
            priority=ContextPriority.HIGH,
            timing=InjectionTiming.NEXT_PAUSE
        )
    ]
    
    for ctx in contexts:
        engine.add_context(ctx)
    
    # Get relevant contexts
    relevant = engine.get_relevant_contexts({}, max_items=10)
    
    # Check ordering (should be by priority)
    assert len(relevant) == 4
    assert relevant[0].information["priority"] == "critical"
    assert relevant[1].information["priority"] == "high"
    assert relevant[2].information["priority"] == "medium"
    assert relevant[3].information["priority"] == "low"
    print("✓ Contexts correctly ordered by priority")
    
    await engine.stop()
    return True


async def test_injection_counting():
    """Test injection count and limits"""
    print("\n=== Test 5: Injection Counting ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Create context with injection limit
    limited_context = ContextToInject(
        information={"message": "limited"},
        max_injections=2,
        timing=InjectionTiming.LAZY,
        priority=ContextPriority.LOW
    )
    
    engine.add_context(limited_context)
    
    # First injection
    limited_context.mark_injected()
    assert limited_context.injection_count == 1
    print("✓ First injection counted")
    
    # Second injection
    limited_context.mark_injected()
    assert limited_context.injection_count == 2
    print("✓ Second injection counted")
    
    # Check if should inject (should be false now)
    should_inject = limited_context.should_inject({})
    assert should_inject == False
    print("✓ Context respects injection limit")
    
    # Cleanup should remove it
    await engine.check_injection({})
    assert limited_context.context_id not in engine.available_context
    print("✓ Over-injected context removed")
    
    await engine.stop()
    return True


async def test_conditional_contexts():
    """Test context conditions"""
    print("\n=== Test 6: Conditional Contexts ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Create conditional context
    topic_context = create_conditional_context(
        conditions={
            "topics_include": ["support", "help"],
            "conversation_length_min": 2
        },
        information={"mode": "support"},
        strategy={"patience": "high"}
    )
    
    engine.add_context(topic_context)
    
    # Test with non-matching state
    state1 = {
        "current_topics": ["general"],
        "message_count": 1
    }
    assert topic_context.should_inject(state1) == False
    print("✓ Context correctly rejects non-matching conditions")
    
    # Test with matching state
    state2 = {
        "current_topics": ["support", "billing"],
        "message_count": 3
    }
    assert topic_context.should_inject(state2) == True
    print("✓ Context correctly accepts matching conditions")
    
    await engine.stop()
    return True


async def test_context_factory_functions():
    """Test context factory functions"""
    print("\n=== Test 7: Context Factory Functions ===")
    
    engine = ContextWeaver()
    await engine.start()
    
    # Test immediate context factory
    immediate = create_immediate_context(
        information={"alert": "urgent"},
        strategy={"urgency": "high"}
    )
    assert immediate.timing == InjectionTiming.IMMEDIATE
    assert immediate.priority == ContextPriority.HIGH
    print("✓ Immediate context factory works")
    
    # Test conditional context factory
    conditional = create_conditional_context(
        conditions={"user_mood": "frustrated"},
        information={"empathy": "high"}
    )
    assert conditional.timing == InjectionTiming.ON_TRIGGER
    assert conditional.priority == ContextPriority.MEDIUM
    assert "user_mood" in conditional.conditions
    print("✓ Conditional context factory works")
    
    await engine.stop()
    return True


async def main():
    """Run all context management tests"""
    print("ContextWeaver Context Management Tests")
    print("=" * 50)
    
    tests = [
        test_add_and_remove_context,
        test_add_raw_context,
        test_context_expiration,
        test_context_priorities,
        test_injection_counting,
        test_conditional_contexts,
        test_context_factory_functions
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
        print("✓ All context management tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)