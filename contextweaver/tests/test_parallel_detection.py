#!/usr/bin/env python3
"""
Test Parallel Detection Performance

Verifies that parallel detection is working and provides performance improvements.
"""

import asyncio
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from contextweaver import ContextWeaver
from contextweaver.strategies import AdaptiveStrategy
from contextweaver.detectors import BaseDetector, DetectionResult
from datetime import datetime


class SlowDetector(BaseDetector):
    """A detector that simulates slow processing"""
    
    def __init__(self, name: str, delay_ms: int = 10):
        super().__init__(threshold=0.5)
        self.name = name
        self.delay_ms = delay_ms
    
    async def detect(self, state) -> DetectionResult:
        """Simulate slow detection"""
        # Simulate processing time
        await asyncio.sleep(self.delay_ms / 1000)
        
        return DetectionResult(
            detected=True,
            confidence=0.8,
            timestamp=datetime.now(),
            metadata={"detector": self.name, "delay_ms": self.delay_ms}
        )


async def test_parallel_vs_sequential():
    """Compare parallel and sequential detection performance"""
    print("Testing Parallel Detection Performance")
    print("=" * 50)
    
    # Create detectors with varying delays
    detectors = [
        SlowDetector("Fast", delay_ms=5),
        SlowDetector("Medium", delay_ms=10),
        SlowDetector("Slow", delay_ms=15),
        SlowDetector("Slower", delay_ms=20),
        SlowDetector("Slowest", delay_ms=25)
    ]
    
    # Calculate expected sequential time
    total_sequential_time = sum(d.delay_ms for d in detectors)
    max_detector_time = max(d.delay_ms for d in detectors)
    
    print(f"\nDetectors: {len(detectors)}")
    print(f"Expected sequential time: ~{total_sequential_time}ms")
    print(f"Expected parallel time: ~{max_detector_time}ms")
    print(f"Expected speedup: {total_sequential_time/max_detector_time:.1f}x")
    
    # Test with parallel detection (current implementation)
    context_weaver = ContextWeaver(
        strategy=AdaptiveStrategy(),
        detectors=detectors
    )
    
    await context_weaver.start()
    
    # Mock state
    mock_state = type('State', (), {
        'messages': [],
        'audio': type('Audio', (), {'vad_active': False})()
    })()
    
    # Measure parallel detection time
    print("\n--- Parallel Detection Test ---")
    parallel_times = []
    
    for i in range(10):
        start = time.perf_counter()
        result = await context_weaver.check_injection(mock_state)
        elapsed = (time.perf_counter() - start) * 1000
        parallel_times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f}ms")
    
    avg_parallel = sum(parallel_times) / len(parallel_times)
    print(f"\nAverage parallel time: {avg_parallel:.2f}ms")
    
    # Test sequential detection for comparison
    print("\n--- Sequential Detection Test (Simulated) ---")
    sequential_times = []
    
    for i in range(10):
        start = time.perf_counter()
        # Simulate sequential execution
        for detector in detectors:
            await detector.detect(mock_state)
        elapsed = (time.perf_counter() - start) * 1000
        sequential_times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f}ms")
    
    avg_sequential = sum(sequential_times) / len(sequential_times)
    print(f"\nAverage sequential time: {avg_sequential:.2f}ms")
    
    # Results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Sequential: {avg_sequential:.2f}ms")
    print(f"Parallel: {avg_parallel:.2f}ms")
    print(f"Speedup: {avg_sequential/avg_parallel:.2f}x")
    print(f"Time saved: {avg_sequential - avg_parallel:.2f}ms ({((avg_sequential - avg_parallel)/avg_sequential)*100:.1f}%)")
    
    # Verify parallel is faster
    assert avg_parallel < avg_sequential, "Parallel should be faster than sequential"
    assert avg_parallel < max_detector_time * 1.5, "Parallel time should be close to slowest detector"
    
    print("\n✅ Parallel detection is working correctly!")
    
    await context_weaver.stop()


async def test_timeout_handling():
    """Test that timeout handling works correctly"""
    print("\n\nTesting Timeout Handling")
    print("=" * 50)
    
    class HangingDetector(BaseDetector):
        """A detector that hangs"""
        async def detect(self, state) -> DetectionResult:
            # Simulate hanging
            await asyncio.sleep(1)  # 1 second delay
            return DetectionResult(
                detected=True,
                confidence=1.0,
                timestamp=datetime.now()
            )
    
    # Mix of normal and hanging detectors
    detectors = [
        SlowDetector("Normal1", delay_ms=5),
        HangingDetector(),  # This will timeout
        SlowDetector("Normal2", delay_ms=10),
    ]
    
    context_weaver = ContextWeaver(
        strategy=AdaptiveStrategy(),
        detectors=detectors
    )
    
    await context_weaver.start()
    
    mock_state = type('State', (), {'messages': []})()
    
    # This should complete quickly despite hanging detector
    start = time.perf_counter()
    result = await context_weaver.check_injection(mock_state)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"Detection completed in {elapsed:.2f}ms")
    print("(Despite having a 1-second hanging detector)")
    
    # Should timeout at 50ms default + some overhead
    assert elapsed < 100, f"Should timeout quickly, but took {elapsed}ms"
    
    print("\n✅ Timeout handling is working correctly!")
    
    await context_weaver.stop()


async def test_error_resilience():
    """Test that errors in one detector don't affect others"""
    print("\n\nTesting Error Resilience")
    print("=" * 50)
    
    class ErrorDetector(BaseDetector):
        """A detector that throws errors"""
        async def detect(self, state) -> DetectionResult:
            raise ValueError("Simulated detector error")
    
    class WorkingDetector(BaseDetector):
        """A detector that works normally"""
        async def detect(self, state) -> DetectionResult:
            return DetectionResult(
                detected=True,
                confidence=0.9,
                timestamp=datetime.now(),
                metadata={"working": True}
            )
    
    detectors = [
        WorkingDetector(),
        ErrorDetector(),  # This will error
        WorkingDetector(),
    ]
    
    context_weaver = ContextWeaver(
        strategy=AdaptiveStrategy(),
        detectors=detectors
    )
    
    await context_weaver.start()
    
    mock_state = type('State', (), {'messages': []})()
    
    # Should complete despite error
    result = await context_weaver.check_injection(mock_state)
    
    print("Detection completed successfully despite detector error")
    print("\n✅ Error resilience is working correctly!")
    
    await context_weaver.stop()


async def test_real_world_performance():
    """Test with real detectors to show actual performance gain"""
    print("\n\nTesting Real-World Performance")
    print("=" * 50)
    
    from contextweaver.detectors import (
        SilenceDetector,
        PauseDetector,
        TopicChangeDetector,
        ResponseTimingDetector,
        ConversationFlowDetector
    )
    
    # Real detectors
    detectors = [
        SilenceDetector(enable_prediction=True),
        PauseDetector(),
        TopicChangeDetector(),
        ResponseTimingDetector(),
        ConversationFlowDetector()
    ]
    
    context_weaver = ContextWeaver(
        strategy=AdaptiveStrategy(),
        detectors=detectors
    )
    
    await context_weaver.start()
    
    # Create realistic state
    from voxon.state import ConversationState, ConversationStatus, Message, SpeakerRole
    
    state = ConversationState(
        status=ConversationStatus.CONNECTED,
        messages=[
            Message(role=SpeakerRole.USER, content="Hello, how are you?"),
            Message(role=SpeakerRole.ASSISTANT, content="I'm doing well, thank you!"),
            Message(role=SpeakerRole.USER, content="Can you help me with Python?")
        ],
        turns=[]
    )
    
    # Create state with audio
    class StateWithAudio:
        def __init__(self, conversation_state):
            self.status = conversation_state.status
            self.messages = conversation_state.messages
            self.turns = conversation_state.turns
            self.audio = type('Audio', (), {
                'vad_active': False,
                'is_listening': True
            })()
    
    state = StateWithAudio(state)
    
    # Measure performance
    times = []
    for i in range(50):
        start = time.perf_counter()
        result = await context_weaver.check_injection(state)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Real detector performance (50 runs):")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms")
    print(f"  Max: {max_time:.2f}ms")
    
    # Should be very fast with parallel execution
    assert avg_time < 20, f"Average detection time too high: {avg_time}ms"
    
    print("\n✅ Real-world performance is excellent!")
    
    await context_weaver.stop()


async def main():
    """Run all parallel detection tests"""
    await test_parallel_vs_sequential()
    await test_timeout_handling()
    await test_error_resilience()
    await test_real_world_performance()
    
    print("\n" + "=" * 50)
    print("ALL PARALLEL DETECTION TESTS PASSED! 🎉")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())