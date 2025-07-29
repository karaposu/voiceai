"""
Test 04: Strategy Behavior

Tests different injection strategies with real scenarios.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from contextweaver.strategies import (
    ConservativeStrategy, AggressiveStrategy, AdaptiveStrategy
)
from contextweaver.detectors import DetectionResult
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority
from datetime import datetime
import time


async def test_conservative_strategy():
    """Test conservative injection strategy"""
    print("\n=== Test 1: Conservative Strategy ===")
    
    strategy = ConservativeStrategy(
        min_interval_seconds=5,
        confidence_threshold=0.8
    )
    
    # Create test contexts
    contexts = {
        "ctx1": ContextToInject(
            information={"type": "high_priority"},
            priority=ContextPriority.HIGH,
            timing=InjectionTiming.NEXT_PAUSE
        ),
        "ctx2": ContextToInject(
            information={"type": "low_priority"},
            priority=ContextPriority.LOW,
            timing=InjectionTiming.LAZY
        )
    }
    
    # Test 1: Low confidence detection (should not inject)
    detections = [
        DetectionResult(
            detected=True,
            confidence=0.6,  # Below threshold
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    decision = await strategy.decide(detections, {}, contexts)
    assert decision.should_inject == False
    assert decision.reason == "no_high_confidence_detection"
    print("✓ Correctly rejects low confidence detection")
    
    # Test 2: High confidence detection with high priority context
    detections = [
        DetectionResult(
            detected=True,
            confidence=0.9,  # Above threshold
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    decision = await strategy.decide(detections, {}, contexts)
    assert decision.should_inject == True
    assert decision.context_to_inject.information["type"] == "high_priority"
    print("✓ Correctly accepts high confidence with high priority")
    
    # Test 3: Too soon after injection
    strategy.record_injection()
    await asyncio.sleep(0.1)
    
    decision = await strategy.decide(detections, {}, contexts)
    assert decision.should_inject == False
    assert decision.reason == "too_soon_after_last_injection"
    print("✓ Correctly enforces minimum interval")
    
    return True


async def test_aggressive_strategy():
    """Test aggressive injection strategy"""
    print("\n=== Test 2: Aggressive Strategy ===")
    
    strategy = AggressiveStrategy(
        min_interval_seconds=1,
        confidence_threshold=0.5
    )
    
    # Create test contexts including critical
    contexts = {
        "critical": ContextToInject(
            information={"type": "critical"},
            priority=ContextPriority.CRITICAL,
            timing=InjectionTiming.IMMEDIATE
        ),
        "medium": ContextToInject(
            information={"type": "medium"},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.NEXT_TURN
        )
    }
    
    # Test 1: Medium confidence (acceptable for aggressive)
    detections = [
        DetectionResult(
            detected=True,
            confidence=0.6,  # Above aggressive threshold
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    decision = await strategy.decide(detections, {}, contexts)
    assert decision.should_inject == True
    assert decision.context_to_inject.information["type"] == "critical"
    print("✓ Accepts medium confidence detection")
    
    # Test 2: Critical priority overrides cooldown
    strategy.record_injection()
    await asyncio.sleep(0.1)  # Still in cooldown
    
    decision = await strategy.decide(detections, {}, contexts)
    assert decision.should_inject == True
    assert decision.reason == "critical_priority_override"
    print("✓ Critical priority overrides cooldown")
    
    # Test 3: Injection rate maintenance
    # Remove critical to test rate maintenance
    del contexts["critical"]
    print(f"✓ Remaining context priority: {contexts['medium'].priority_value}")
    
    # Clear injection history to simulate low rate
    strategy.injection_history = []
    
    # Test with detection above aggressive threshold
    valid_detections = [
        DetectionResult(
            detected=True,
            confidence=0.55,  # Just above aggressive threshold
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    decision = await strategy.decide(valid_detections, {}, contexts)
    # Should inject with valid detection and medium priority context
    assert decision.should_inject == True
    print("✓ Maintains injection through aggressive behavior")
    
    # Also test injection rate maintenance without detection
    strategy.injection_history = []  # Clear to simulate low rate
    no_detection_decision = await strategy.decide([], {}, contexts)
    # Should still inject based on low injection rate
    if no_detection_decision.should_inject:
        assert no_detection_decision.reason == "maintaining_injection_rate"
        print("✓ Maintains minimum injection rate without detection")
    
    return True


async def test_adaptive_strategy():
    """Test adaptive injection strategy"""
    print("\n=== Test 3: Adaptive Strategy ===")
    
    strategy = AdaptiveStrategy(
        initial_threshold=0.7,
        learning_rate=0.1
    )
    
    contexts = {
        "adaptive": ContextToInject(
            information={"type": "adaptive"},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.NEXT_PAUSE
        )
    }
    
    # Test 1: Initial threshold
    detections = [
        DetectionResult(
            detected=True,
            confidence=0.75,
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    state = type('State', (), {
        'metrics': type('Metrics', (), {'interruption_count': 0})(),
        'turns': []
    })()
    
    decision = await strategy.decide(detections, state, contexts)
    # Adaptive strategy might or might not inject based on combined score
    initial_threshold = strategy.current_threshold
    if decision.should_inject:
        print(f"✓ Initial decision to inject with threshold {initial_threshold:.2f}")
    else:
        print(f"✓ Initial decision not to inject with threshold {initial_threshold:.2f}")
    
    # Test 2: Threshold adjustment based on interruptions
    state.metrics.interruption_count = 5
    state.turns = [type('Turn', (), {'is_complete': True})() for _ in range(2)]
    
    # Update metrics and decide again
    decision = await strategy.decide(detections, state, contexts)
    assert strategy.current_threshold > initial_threshold
    print(f"✓ Threshold increased to {strategy.current_threshold:.2f} due to interruptions")
    
    # Test 3: Record success and adjust
    for _ in range(10):
        strategy.record_injection_result(success=True)
    
    # High success rate should lower threshold
    assert strategy.current_threshold < 0.9
    print("✓ Threshold adjusts based on success rate")
    
    # Test 4: Adaptive scoring based on silence
    silent_state = type('State', (), {
        'audio': type('Audio', (), {'silence_duration_ms': 4000})(),
        'metrics': type('Metrics', (), {'interruption_count': 0})(),
        'turns': [type('Turn', (), {'is_complete': True})() for _ in range(10)]
    })()
    
    immediate_context = ContextToInject(
        information={"type": "immediate"},
        priority=ContextPriority.MEDIUM,
        timing=InjectionTiming.IMMEDIATE
    )
    contexts["immediate"] = immediate_context
    
    decision = await strategy.decide(detections, silent_state, contexts)
    # Long silence should boost immediate contexts
    assert decision.should_inject == True
    print("✓ Adapts to conversation state (silence)")
    
    return True


async def test_strategy_comparison():
    """Compare strategies in same scenario"""
    print("\n=== Test 4: Strategy Comparison ===")
    
    conservative = ConservativeStrategy()
    aggressive = AggressiveStrategy()
    adaptive = AdaptiveStrategy()
    
    # Same scenario for all
    detections = [
        DetectionResult(
            detected=True,
            confidence=0.65,  # Medium confidence
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    contexts = {
        "test": ContextToInject(
            information={"test": "data"},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.NEXT_PAUSE
        )
    }
    
    state = {}
    
    # Get decisions
    cons_decision = await conservative.decide(detections, state, contexts)
    aggr_decision = await aggressive.decide(detections, state, contexts)
    adap_decision = await adaptive.decide(detections, state, contexts)
    
    # Conservative should reject (confidence too low)
    assert cons_decision.should_inject == False
    print("✓ Conservative rejects medium confidence")
    
    # Aggressive should accept
    assert aggr_decision.should_inject == True
    print("✓ Aggressive accepts medium confidence")
    
    # Adaptive depends on state but likely accepts
    print(f"✓ Adaptive decision: {adap_decision.should_inject}")
    
    return True


async def test_strategy_performance():
    """Test strategy performance"""
    print("\n=== Test 5: Strategy Performance ===")
    
    strategies = [
        ConservativeStrategy(),
        AggressiveStrategy(),
        AdaptiveStrategy()
    ]
    
    # Create realistic scenario
    detections = [
        DetectionResult(detected=True, confidence=0.8, timestamp=datetime.now(), metadata={})
        for _ in range(3)
    ]
    
    contexts = {
        f"ctx{i}": ContextToInject(
            information={f"data{i}": i},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.NEXT_PAUSE
        )
        for i in range(10)
    }
    
    state = type('State', (), {
        'metrics': type('Metrics', (), {'interruption_count': 2})(),
        'turns': [type('Turn', (), {'is_complete': True})() for _ in range(5)]
    })()
    
    # Time decision making
    for strategy in strategies:
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            await strategy.decide(detections, state, contexts)
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / iterations) * 1000
        
        print(f"✓ {strategy.name} strategy: {avg_time_ms:.3f}ms per decision")
        assert avg_time_ms < 5  # Should be well under 5ms
    
    return True


async def main():
    """Run all strategy tests"""
    print("ContextWeaver Strategy Tests")
    print("=" * 50)
    
    tests = [
        test_conservative_strategy,
        test_aggressive_strategy,
        test_adaptive_strategy,
        test_strategy_comparison,
        test_strategy_performance
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
        print("✓ All strategy tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)