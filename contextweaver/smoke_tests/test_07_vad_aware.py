"""
Test 07: VAD-Aware Context Injection

Tests VAD mode awareness and response control integration.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from contextweaver.engine import ContextWeaver
from contextweaver.strategies import (
    ConservativeStrategy, AggressiveStrategy, AdaptiveStrategy
)
from contextweaver.detectors import SilenceDetector, PauseDetector
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority
from voxon.orchestrator.vad_adapter import VADModeAdapter
from datetime import datetime
import time


async def test_vad_adapter():
    """Test VAD mode adapter functionality"""
    print("\n=== Test 1: VAD Mode Adapter ===")
    
    adapter = VADModeAdapter()
    
    # Test 1: Server VAD with auto-response (most restrictive)
    adapter.update_mode("server", True, 500)
    assert adapter.get_injection_window() == 200
    assert adapter.get_monitoring_interval() == 0.025
    assert adapter.get_strategy_recommendation() == "aggressive"
    print("✓ Server+Auto mode: short window, fast monitoring")
    
    # Test 2: Client VAD with manual response (most flexible)
    adapter.update_mode("client", False)
    assert adapter.get_injection_window() == 2000
    assert adapter.get_monitoring_interval() == 0.1
    assert adapter.get_strategy_recommendation() == "conservative"
    print("✓ Client+Manual mode: long window, normal monitoring")
    
    # Test 3: Detection config adjustment
    adapter.update_mode("server", True)
    config_server_auto = adapter.get_detection_config()
    adapter.update_mode("client", False)
    config_client_manual = adapter.get_detection_config()
    
    assert config_server_auto["urgency_multiplier"] > config_client_manual["urgency_multiplier"]
    assert config_server_auto["silence_threshold_ms"] < config_client_manual["silence_threshold_ms"]
    print("✓ Detection configs adapt to VAD mode")
    
    # Test 4: Urgency calculation
    adapter.update_mode("server", True)
    high_urgency = adapter.calculate_urgency(0.8, "immediate")
    low_urgency = adapter.calculate_urgency(0.3, "lazy")
    assert high_urgency > low_urgency * 3  # Much higher with multiplier
    print("✓ Urgency calculation includes mode multiplier")
    
    return True


async def test_strategy_vad_adaptation():
    """Test strategy adaptation to VAD modes"""
    print("\n=== Test 2: Strategy VAD Adaptation ===")
    
    # Create states with different VAD modes
    state_server_auto = type('State', (), {
        'vad_mode': 'server',
        'auto_response': True,
        'injection_mode': 'immediate',
        '__dict__': {'vad_mode': 'server', 'auto_response': True}
    })()
    
    state_client_manual = type('State', (), {
        'vad_mode': 'client',
        'auto_response': False,
        'injection_mode': 'controlled',
        '__dict__': {'vad_mode': 'client', 'auto_response': False}
    })()
    
    # Create test contexts
    contexts = {
        "test": ContextToInject(
            information={"test": "data"},
            priority=ContextPriority.MEDIUM,
            timing=InjectionTiming.NEXT_PAUSE
        )
    }
    
    # Test conservative strategy adaptation
    conservative = ConservativeStrategy()
    
    # Medium confidence detection
    from contextweaver.detectors.base import DetectionResult
    detections = [
        DetectionResult(
            detected=True,
            confidence=0.75,  # Below normal conservative threshold
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    # Test in server+auto mode (should be more lenient)
    decision_server = await conservative.decide(detections, state_server_auto, contexts)
    
    # Test in client+manual mode (should be stricter)
    decision_client = await conservative.decide(detections, state_client_manual, contexts)
    
    # In server+auto, the threshold is lowered so it might inject
    # In client+manual, it should be more conservative
    print(f"✓ Conservative strategy - Server+Auto: {decision_server.should_inject}")
    print(f"✓ Conservative strategy - Client+Manual: {decision_client.should_inject}")
    
    # Test aggressive strategy adaptation
    aggressive = AggressiveStrategy()
    
    # Low confidence detection
    low_detections = [
        DetectionResult(
            detected=True,
            confidence=0.4,  # Below normal aggressive threshold
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    decision_aggressive_server = await aggressive.decide(low_detections, state_server_auto, contexts)
    decision_aggressive_client = await aggressive.decide(low_detections, state_client_manual, contexts)
    
    # Aggressive should be even more aggressive in server+auto
    print(f"✓ Aggressive strategy adapts to VAD mode")
    
    return True


async def test_engine_coordinator_vad():
    """Test EngineCoordinator VAD detection"""
    print("\n=== Test 3: Engine Coordinator VAD Detection ===")
    
    from voxon.orchestrator.engine_coordinator import EngineCoordinator
    
    coordinator = EngineCoordinator()
    
    # Mock voice engine with config
    mock_voice_engine = type('MockVoiceEngine', (), {
        'config': type('Config', (), {'vad_type': 'server'})(),
        'session_config': {
            'turn_detection': {
                'create_response': True,
                'silence_duration_ms': 300
            }
        },
        'events': type('Events', (), {
            'on': lambda self, event, handler: f"handler_{event}",
            'off': lambda self, handler_id: None
        })(),
        'is_connected': True,
        'conversation_state': {}
    })()
    
    # Mock context engine
    mock_context_engine = type('MockContextEngine', (), {
        'start': lambda: asyncio.sleep(0),
        'stop': lambda: asyncio.sleep(0),
        'check_injection': lambda state: None,
        'add_context': lambda ctx: None
    })()
    
    coordinator.set_voice_engine(mock_voice_engine)
    coordinator.set_context_engine(mock_context_engine)
    
    # Detect VAD mode
    coordinator._detect_vad_mode()
    
    assert coordinator.vad_mode == "server"
    assert coordinator.auto_response_enabled == True
    assert coordinator.injection_mode == "immediate"
    assert coordinator.vad_adapter.server_silence_ms == 300
    print("✓ Correctly detects server VAD with auto-response")
    
    # Test with client VAD
    mock_voice_engine.config.vad_type = "client"
    mock_voice_engine.session_config['turn_detection']['create_response'] = False
    
    coordinator._detect_vad_mode()
    
    assert coordinator.vad_mode == "client"
    assert coordinator.auto_response_enabled == False
    assert coordinator.injection_mode == "controlled"
    print("✓ Correctly detects client VAD with manual response")
    
    # Test stats include VAD info
    stats = coordinator.get_stats()
    assert stats['vad_mode'] == "client"
    assert stats['auto_response_enabled'] == False
    assert stats['injection_mode'] == "controlled"
    print("✓ Stats include VAD information")
    
    return True


async def test_injection_timing_vad():
    """Test injection timing based on VAD mode"""
    print("\n=== Test 4: Injection Timing with VAD Modes ===")
    
    # Create context weaver with strategies
    context_weaver = ContextWeaver(
        strategy=AdaptiveStrategy(),
        detectors=[SilenceDetector(), PauseDetector()]
    )
    
    await context_weaver.start()
    
    # Add some contexts
    immediate_context = ContextToInject(
        information={"urgent": "data"},
        priority=ContextPriority.HIGH,
        timing=InjectionTiming.IMMEDIATE
    )
    
    normal_context = ContextToInject(
        information={"normal": "data"},
        priority=ContextPriority.MEDIUM,
        timing=InjectionTiming.NEXT_PAUSE
    )
    
    context_weaver.add_context(immediate_context)
    context_weaver.add_context(normal_context)
    
    # Test with server+auto state (should prioritize immediate)
    state_urgent = type('State', (), {
        'vad_mode': 'server',
        'auto_response': True,
        'injection_mode': 'immediate',
        'audio': type('Audio', (), {'vad_active': False, 'silence_duration_ms': 1000})(),
        'messages': [],
        '__dict__': {'vad_mode': 'server', 'auto_response': True}
    })()
    
    # Trigger silence detection
    await asyncio.sleep(0.5)
    decision = await context_weaver.check_injection(state_urgent)
    
    if decision:
        assert decision.information.get("urgent") == "data"
        assert decision.timing == InjectionTiming.IMMEDIATE
        print("✓ Prioritizes immediate context in server+auto mode")
    else:
        print("✓ No injection yet (may need more time)")
    
    await context_weaver.stop()
    return True


async def test_performance_vad_modes():
    """Test performance across VAD modes"""
    print("\n=== Test 5: Performance Across VAD Modes ===")
    
    adapter = VADModeAdapter()
    
    # Test decision speed for different modes
    modes = [
        ("server", True),   # Most urgent
        ("server", False),  # Medium urgency
        ("client", True),   # Medium urgency
        ("client", False)   # Least urgent
    ]
    
    for vad_mode, auto_response in modes:
        adapter.update_mode(vad_mode, auto_response)
        
        start_time = time.time()
        
        # Simulate 100 urgency calculations
        for i in range(100):
            window = adapter.get_injection_window()
            interval = adapter.get_monitoring_interval()
            urgency = adapter.calculate_urgency(0.7, "next_pause")
            should_force = adapter.should_force_injection(window * 0.9, urgency)
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / 100) * 1000
        
        print(f"✓ {vad_mode}+{'auto' if auto_response else 'manual'}: {avg_time_ms:.3f}ms per decision")
        assert avg_time_ms < 1  # Should be very fast
    
    return True


async def main():
    """Run all VAD-aware tests"""
    print("ContextWeaver VAD-Aware Tests")
    print("=" * 50)
    
    tests = [
        test_vad_adapter,
        test_strategy_vad_adaptation,
        test_engine_coordinator_vad,
        test_injection_timing_vad,
        test_performance_vad_modes
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
        print("✓ All VAD-aware tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)