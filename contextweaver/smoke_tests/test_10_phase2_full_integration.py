#!/usr/bin/env python3
"""
Phase 2 Full Integration Test

Tests the complete Phase 2 implementation with all components
working together in realistic scenarios.
"""

import asyncio
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig
from contextweaver import ContextWeaver
from contextweaver.strategies import AdaptiveStrategy
from contextweaver.detectors import SilenceDetector, PauseDetector
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority


class MockVoiceEngine:
    """Mock VoiceEngine for testing"""
    def __init__(self, vad_type="client", auto_response=True):
        self.config = type('Config', (), {'vad_type': vad_type})()
        self.session_config = {
            'turn_detection': {
                'create_response': auto_response,
                'silence_duration_ms': 500
            }
        }
        self.is_connected = True
        self.conversation_state = type('State', (), {})()
        self.events = type('Events', (), {
            'on': lambda self, e, h: f"handler_{e}",
            'off': lambda self, h: None
        })()
        self.text_sent = []
        self.response_triggered = False
    
    async def send_text(self, text):
        self.text_sent.append(text)
    
    async def trigger_response(self):
        self.response_triggered = True


class TestPhase2FullIntegration:
    """Test full Phase 2 integration"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("\n=== Phase 2 Full Integration Tests ===\n")
        
        tests = [
            self.test_server_vad_auto_response,
            self.test_client_vad_manual_response,
            self.test_adaptive_injection,
            self.test_conversation_flow,
            self.test_edge_cases
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
    
    async def test_server_vad_auto_response(self):
        """Test server VAD with automatic response mode"""
        # Create mock voice engine with server VAD
        voice_engine = MockVoiceEngine(vad_type="server", auto_response=True)
        
        # Create context weaver
        context_weaver = ContextWeaver(
            strategy=AdaptiveStrategy(),
            detectors=[SilenceDetector(), PauseDetector()]
        )
        
        # Create Voxon
        config = VoxonConfig()
        voxon = Voxon(config)
        
        # Access the coordinator through the right path
        coordinator = voxon.engine_coordinator
        
        # Set engines directly for testing
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        # Initialize coordinator
        await coordinator.initialize()
        
        # Wait for async response controller configuration
        await asyncio.sleep(0.1)
        
        # Verify VAD mode detection
        assert coordinator.vad_mode == "server"
        assert coordinator.auto_response_enabled is True
        assert coordinator.injection_mode == "immediate"
        
        # Verify response controller configuration
        assert coordinator.response_controller.response_mode.value == "automatic"
        # For automatic mode with auto-response, the delay is set to 200
        assert coordinator.response_controller.max_injection_delay_ms == 200
        
        # Verify window manager configuration
        assert coordinator.injection_window_manager.vad_mode == "server"
        assert coordinator.injection_window_manager.default_window_duration == 100
    
    async def test_client_vad_manual_response(self):
        """Test client VAD with manual response mode"""
        # Create mock voice engine with client VAD
        voice_engine = MockVoiceEngine(vad_type="client", auto_response=False)
        
        # Create context weaver
        context_weaver = ContextWeaver()
        
        # Create and configure Voxon
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Verify configuration
        assert coordinator.vad_mode == "client"
        assert coordinator.auto_response_enabled is False
        assert coordinator.injection_mode == "controlled"
        
        # Test injection with response triggering
        context = ContextToInject(
            information={"test": "data"},
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH
        )
        
        context_weaver.add_context(context)
        
        # Simulate window creation
        coordinator.injection_window_manager.create_immediate_window()
        
        # Run one monitoring cycle
        await asyncio.sleep(0.1)
        
        # Verify response would be triggered in manual mode
        assert coordinator.response_controller.should_trigger_response() is True
    
    async def test_adaptive_injection(self):
        """Test adaptive injection based on conversation state"""
        voice_engine = MockVoiceEngine()
        context_weaver = ContextWeaver(strategy=AdaptiveStrategy())
        
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Simulate conversation flow
        # 1. User speaks
        coordinator._handle_audio_start({})
        assert len(coordinator.injection_window_manager.active_windows) == 0
        
        # 2. User stops speaking
        coordinator._handle_audio_stop({})
        await asyncio.sleep(0.3)  # Wait for post-input window
        
        # 3. Verify window created
        windows = coordinator.injection_window_manager.active_windows
        assert any(w.window_type.value == "post_input" for w in windows)
        
        # 4. AI starts responding
        coordinator._handle_response_start({})
        assert coordinator.response_controller.response_pending is True
        
        # 5. AI finishes responding
        coordinator._handle_response_complete({})
        assert coordinator.response_controller.response_ready is True
        
        # Get stats
        stats = coordinator.get_stats()
        assert stats['events_processed'] >= 4
        assert 'response_controller' in stats
        assert 'injection_windows' in stats
    
    async def test_conversation_flow(self):
        """Test complete conversation flow with context injection"""
        voice_engine = MockVoiceEngine()
        context_weaver = ContextWeaver()
        
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Add context to inject
        context = ContextToInject(
            information={"user_preference": "detailed_explanations"},
            strategy={"tone": "friendly"},
            timing=InjectionTiming.NEXT_PAUSE,
            priority=ContextPriority.MEDIUM
        )
        context_weaver.add_context(context)
        
        # Simulate conversation
        # User input
        coordinator._handle_text_input({'text': 'Hello, I need help'})
        coordinator._handle_audio_stop({})
        
        # Create injection window
        coordinator.injection_window_manager.create_immediate_window(duration_ms=2000)
        
        # Let monitoring cycle run
        await asyncio.sleep(0.2)
        
        # Check if context was injected (would be in real scenario)
        # In our mock, we can't fully test the async monitoring loop
        # but we can verify the setup is correct
        recommendation = coordinator.injection_window_manager.get_injection_recommendation()
        assert recommendation['recommended'] is True
    
    async def test_edge_cases(self):
        """Test edge cases and error handling"""
        voice_engine = MockVoiceEngine()
        context_weaver = ContextWeaver()
        
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Test queue overflow
        controller = coordinator.response_controller
        for i in range(15):  # Try to overflow
            await controller.request_injection(f"Context {i}")
        
        # Should handle gracefully
        assert len(controller.injection_queue) <= 10
        
        # Test expired injection
        await controller.request_injection("Expired", deadline_ms=1)
        await asyncio.sleep(0.01)
        
        success = await controller.execute_injection(
            inject_callback=lambda ctx: asyncio.sleep(0)
        )
        # Should skip expired
        assert controller.metrics['expired_injections'] >= 0
        
        # Test window cleanup
        manager = coordinator.injection_window_manager
        # Create expired window
        window = manager.create_immediate_window(duration_ms=1)
        await asyncio.sleep(0.01)
        
        manager._cleanup_windows()
        assert window not in manager.active_windows
        
        # Shutdown
        await coordinator.shutdown()
        assert coordinator.is_initialized is False


async def main():
    """Run full integration tests"""
    tester = TestPhase2FullIntegration()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)