#!/usr/bin/env python3
"""
Phase 2 Integration Test: Response Control

Tests the ResponseController, InjectionWindowManager, and enhanced
EngineCoordinator working together for intelligent context injection.
"""

import asyncio
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from voxon.orchestrator import (
    ResponseController, ResponseMode,
    InjectionWindowManager, WindowType
)
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority


class TestPhase2ResponseControl:
    """Test response control and injection window management"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.response_controller = ResponseController()
        self.window_manager = InjectionWindowManager()
    
    async def run_all_tests(self):
        """Run all Phase 2 tests"""
        print("\n=== Phase 2: Response Control Integration Tests ===\n")
        
        tests = [
            self.test_response_controller_basic,
            self.test_injection_window_manager,
            self.test_vad_mode_coordination,
            self.test_injection_timing,
            self.test_window_lifecycle,
            self.test_response_triggering,
            self.test_queue_management,
            self.test_metrics_tracking
        ]
        
        for test in tests:
            try:
                await test()
                self.passed += 1
                print(f"✓ {test.__name__}")
            except Exception as e:
                self.failed += 1
                print(f"✗ {test.__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n=== Results: {self.passed} passed, {self.failed} failed ===")
        return self.failed == 0
    
    async def test_response_controller_basic(self):
        """Test basic ResponseController functionality"""
        controller = ResponseController()
        
        # Test mode setting
        await controller.set_response_mode(ResponseMode.MANUAL, auto_response=False)
        assert controller.response_mode == ResponseMode.MANUAL
        assert controller.max_injection_delay_ms == 2000
        
        await controller.set_response_mode(ResponseMode.AUTOMATIC, auto_response=True)
        assert controller.response_mode == ResponseMode.AUTOMATIC
        assert controller.max_injection_delay_ms == 200
        
        # Test injection request
        requested = await controller.request_injection(
            context="Test context",
            priority=8,
            deadline_ms=1000
        )
        assert requested is True
        assert len(controller.injection_queue) == 1
        
        # Test window calculation
        window = await controller.get_injection_window()
        assert window['has_pending'] is True
        assert window['pending_priority'] == 8
        assert window['response_mode'] == 'automatic'
    
    async def test_injection_window_manager(self):
        """Test InjectionWindowManager functionality"""
        manager = InjectionWindowManager()
        
        # Test VAD mode update
        manager.update_vad_mode("server", auto_response=True)
        assert manager.vad_mode == "server"
        assert manager.auto_response is True
        assert manager.pre_response_buffer == 50
        
        # Test window creation on user input end
        manager.on_user_input_end()
        # Wait for window to become active (post-input delay)
        await asyncio.sleep(0.3)
        assert len(manager.active_windows) >= 1
        post_input_windows = [w for w in manager.active_windows if w.window_type == WindowType.POST_INPUT]
        assert len(post_input_windows) == 1
        
        # Test window creation on AI response end
        manager.on_ai_response_end()
        # Wait for window to become active
        await asyncio.sleep(0.2)
        windows = [w for w in manager.active_windows if w.window_type == WindowType.TURN_TRANSITION]
        assert len(windows) == 1
        
        # Test best window selection
        # Clean up any expired windows first
        manager._cleanup_windows()
        
        # Create a fresh window for testing
        manager.create_immediate_window(duration_ms=1000)
        best = manager.get_best_window(required_time_ms=100)
        assert best is not None
        assert best.can_inject(100)
    
    async def test_vad_mode_coordination(self):
        """Test coordination between different VAD modes"""
        controller = ResponseController()
        manager = InjectionWindowManager()
        
        # Test server VAD + auto response (fast injection)
        await controller.set_response_mode(ResponseMode.AUTOMATIC, auto_response=True)
        manager.update_vad_mode("server", auto_response=True)
        
        window = await controller.get_injection_window()
        assert window['window_ms'] <= 100
        assert window['urgency'] == 'immediate'
        
        # Test client VAD + manual response (relaxed injection)
        await controller.set_response_mode(ResponseMode.MANUAL, auto_response=False)
        manager.update_vad_mode("client", auto_response=False)
        
        window = await controller.get_injection_window()
        assert window['window_ms'] >= 1000
        assert window['urgency'] == 'relaxed'
    
    async def test_injection_timing(self):
        """Test injection timing and execution"""
        controller = ResponseController()
        injection_executed = False
        
        async def mock_inject(context: str):
            nonlocal injection_executed
            injection_executed = True
            await asyncio.sleep(0.05)  # Simulate injection time
        
        # Request injection
        await controller.request_injection("Test context", priority=9)
        
        # Execute injection
        success = await controller.execute_injection(
            inject_callback=mock_inject
        )
        
        assert success is True
        assert injection_executed is True
        assert controller.metrics['successful_injections'] == 1
    
    async def test_window_lifecycle(self):
        """Test injection window lifecycle"""
        manager = InjectionWindowManager()
        
        # Create immediate window
        window = manager.create_immediate_window(duration_ms=500)
        assert window.window_type == WindowType.IMMEDIATE
        assert window.is_active is True
        assert window.priority == 10
        
        # Test window expiration
        await asyncio.sleep(0.6)  # Wait for window to expire
        assert window.is_active is False
        
        # Test window cleanup
        manager._cleanup_windows()
        assert len(manager.active_windows) == 0
    
    async def test_response_triggering(self):
        """Test response trigger coordination"""
        controller = ResponseController()
        response_triggered = False
        
        async def mock_trigger_response():
            nonlocal response_triggered
            response_triggered = True
        
        # Test manual mode (should trigger)
        await controller.set_response_mode(ResponseMode.MANUAL)
        assert controller.should_trigger_response() is True
        
        # Test automatic mode (should not trigger)
        await controller.set_response_mode(ResponseMode.AUTOMATIC)
        assert controller.should_trigger_response() is False
        
        # Test execution with trigger
        await controller.request_injection("Context")
        await controller.execute_injection(
            inject_callback=lambda ctx: asyncio.sleep(0),
            trigger_response_callback=mock_trigger_response
        )
        
        # In manual mode, response should be triggered
        await controller.set_response_mode(ResponseMode.MANUAL)
        response_triggered = False
        await controller.request_injection("Context")
        await controller.execute_injection(
            inject_callback=lambda ctx: asyncio.sleep(0),
            trigger_response_callback=mock_trigger_response
        )
        assert response_triggered is True
    
    async def test_queue_management(self):
        """Test injection queue management"""
        controller = ResponseController()
        
        # Test priority ordering
        await controller.request_injection("Low priority", priority=3)
        await controller.request_injection("High priority", priority=9)
        await controller.request_injection("Medium priority", priority=5)
        
        assert len(controller.injection_queue) == 3
        assert controller.injection_queue[0].priority == 9
        assert controller.injection_queue[1].priority == 5
        assert controller.injection_queue[2].priority == 3
        
        # Test queue limit
        for i in range(10):
            await controller.request_injection(f"Context {i}", priority=1)
        
        # Should reject when queue is full
        rejected = await controller.request_injection("Overflow", priority=1)
        assert rejected is False
    
    async def test_metrics_tracking(self):
        """Test metrics collection"""
        controller = ResponseController()
        manager = InjectionWindowManager()
        
        # Test response controller metrics
        await controller.request_injection("Test 1")
        await controller.request_injection("Test 2", deadline_ms=10)  # Very short deadline
        await asyncio.sleep(0.02)  # Let second request expire
        
        # Execute first injection
        await controller.execute_injection(
            inject_callback=lambda ctx: asyncio.sleep(0)
        )
        
        # Try to execute second (should be expired)
        await controller.execute_injection(
            inject_callback=lambda ctx: asyncio.sleep(0)
        )
        
        metrics = controller.get_metrics()
        assert metrics['total_injections'] >= 1
        assert metrics['successful_injections'] >= 1
        assert metrics['expired_injections'] >= 1
        
        # Test window manager metrics
        manager.on_user_input_end()
        manager.on_ai_response_end()
        
        window_metrics = manager.get_metrics()
        assert window_metrics['windows_created'] == 2
        assert window_metrics['active_windows'] >= 1


async def main():
    """Run Phase 2 tests"""
    tester = TestPhase2ResponseControl()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)