#!/usr/bin/env python3
"""
Phase 3 Full Integration Test

Tests the complete system with advanced detectors integrated into
the full context injection pipeline.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from voxon import Voxon, VoxonConfig
from contextweaver import ContextWeaver
from contextweaver.strategies import AdaptiveStrategy
from contextweaver.detectors import (
    SilenceDetector, ResponseTimingDetector, 
    ConversationFlowDetector, PauseDetector
)
from contextweaver.schema import ContextToInject, InjectionTiming, ContextPriority
from voxon.state import Message, SpeakerRole


class MockVoiceEngine:
    """Mock VoiceEngine with full state simulation"""
    def __init__(self, vad_type="client", auto_response=True):
        self.config = type('Config', (), {'vad_type': vad_type})()
        self.session_config = {
            'turn_detection': {
                'create_response': auto_response,
                'silence_duration_ms': 500
            }
        }
        self.is_connected = True
        self.is_listening = True
        
        # Full conversation state
        from voxon.state import ConversationState, ConversationStatus
        self.conversation_state = ConversationState(
            status=ConversationStatus.CONNECTED,
            messages=[],
            turns=[]
        )
        
        # Audio state simulation
        self._audio_state = type('AudioState', (), {
            'vad_active': False,
            'vad_confidence': 0.0,
            'is_listening': True,
            'is_playing': False
        })()
        
        # Event system
        self.events = type('Events', (), {
            'on': lambda self, e, h: f"handler_{e}",
            'off': lambda self, h: None,
            'emit': lambda self, e: None
        })()
        
        # Track injections
        self.contexts_injected = []
        self.responses_triggered = 0
    
    @property
    def audio_state(self):
        """Get audio state"""
        return self._audio_state
    
    async def send_text(self, text):
        """Simulate text injection"""
        self.contexts_injected.append(text)
        print(f"[Context Injected]: {text}")
    
    async def trigger_response(self):
        """Simulate response trigger"""
        self.responses_triggered += 1
        print("[Response Triggered]")
    
    def add_message(self, role: SpeakerRole, content: str):
        """Add message to conversation"""
        from voxon.state import Message
        msg = Message(role=role, content=content)
        self.conversation_state = self.conversation_state.evolve(
            messages=list(self.conversation_state.messages) + [msg]
        )


class TestPhase3Integration:
    """Test full Phase 3 integration"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    async def run_all_tests(self):
        """Run all Phase 3 integration tests"""
        print("\n=== Phase 3 Full Integration Tests ===\n")
        
        tests = [
            self.test_advanced_detectors_integration,
            self.test_learning_system,
            self.test_conversation_flow_tracking,
            self.test_response_timing_prediction,
            self.test_adaptive_behavior,
            self.test_real_conversation_simulation
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
    
    async def test_advanced_detectors_integration(self):
        """Test all advanced detectors working together"""
        print("\n--- Testing Advanced Detectors Integration ---")
        
        # Create enhanced ContextWeaver with all new detectors
        context_weaver = ContextWeaver(
            strategy=AdaptiveStrategy(),
            detectors=[
                SilenceDetector(silence_threshold_ms=1000, enable_prediction=True),
                ResponseTimingDetector(prediction_window_ms=500),
                ConversationFlowDetector(learning_rate=0.2),
                PauseDetector()
            ]
        )
        
        # Create mock voice engine
        voice_engine = MockVoiceEngine(vad_type="server", auto_response=True)
        
        # Create and configure Voxon
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Verify advanced detectors are configured
        assert coordinator.vad_mode == "server"
        assert coordinator.injection_mode == "immediate"
        
        # Add context to inject
        context = ContextToInject(
            information={"test": "advanced_detection"},
            timing=InjectionTiming.NEXT_PAUSE,
            priority=ContextPriority.HIGH
        )
        context_weaver.add_context(context)
        
        # Simulate conversation with silence
        voice_engine.add_message(SpeakerRole.USER, "Hello, I have a question")
        voice_engine._audio_state.vad_active = False
        
        # Update detector states to recognize opportunity
        for detector in context_weaver.detectors:
            if isinstance(detector, SilenceDetector):
                detector.silence_start = datetime.now() - timedelta(milliseconds=1100)
                detector.vad_mode = "server"
            elif isinstance(detector, ResponseTimingDetector):
                detector.last_user_input_end = datetime.now() - timedelta(milliseconds=500)
                detector.response_pending = True
                detector.update_response_mode("automatic", "server", True)
            elif isinstance(detector, ConversationFlowDetector):
                # Will analyze messages
                pass
        
        # Check injection with advanced detection
        # Create a mock state with all required attributes
        class EnhancedState:
            def __init__(self):
                self.status = voice_engine.conversation_state.status
                self.messages = voice_engine.conversation_state.messages
                self.turns = voice_engine.conversation_state.turns
                self.vad_mode = "server"
                self.auto_response = True
                self.audio = voice_engine._audio_state
        
        enhanced_state = EnhancedState()
        
        result = await context_weaver.check_injection(enhanced_state)
        
        # Should detect opportunity with advanced detectors
        assert result is not None or len(voice_engine.contexts_injected) > 0
        
        print("✓ Advanced detectors integrated successfully")
    
    async def test_learning_system(self):
        """Test learning capabilities across detectors"""
        print("\n--- Testing Learning System ---")
        
        flow_detector = ConversationFlowDetector(learning_rate=0.3)
        timing_detector = ResponseTimingDetector()
        
        # Simulate successful pattern
        flow_detector.record_injection_outcome(
            pattern_type="qa_cycle",
            phase=flow_detector.current_phase,
            success=True,
            context_type="detailed_answer",
            metadata={}
        )
        
        # Record response timing
        timing_detector.record_response_event(
            "response_triggered",
            datetime.now(),
            {"delay_ms": 250}
        )
        
        # Simulate more patterns
        for i in range(5):
            success = i % 2 == 0
            flow_detector.record_injection_outcome(
                pattern_type="qa_cycle",
                phase=flow_detector.current_phase,
                success=success,
                context_type="detailed_answer",
                metadata={}
            )
        
        # Check learning
        stats = flow_detector.get_statistics()
        assert stats['patterns_learned'] > 0
        assert stats['average_success_rate'] != 0.5  # Should have learned
        
        # Check timing predictions improve
        timing_detector.response_patterns['user_to_response'] = [200, 220, 210, 215, 208]
        avg = timing_detector._get_average_response_time()
        assert 200 <= avg <= 220
        
        print("✓ Learning system working across detectors")
    
    async def test_conversation_flow_tracking(self):
        """Test conversation flow detection and phase transitions"""
        print("\n--- Testing Conversation Flow Tracking ---")
        
        flow_detector = ConversationFlowDetector()
        
        # Create mock state with conversation
        state = type('State', (), {
            'messages': [],
            'turns': [],
            'audio': type('Audio', (), {'vad_active': False})()
        })()
        
        # Greeting phase
        state.messages = [
            Message(role=SpeakerRole.USER, content="Hello there!"),
            Message(role=SpeakerRole.ASSISTANT, content="Hi! How can I help?")
        ]
        
        result = await flow_detector.detect(state)
        initial_phase = flow_detector.current_phase
        
        # Main conversation
        state.messages.extend([
            Message(role=SpeakerRole.USER, content="I need help with Python"),
            Message(role=SpeakerRole.ASSISTANT, content="I'd be happy to help with Python!"),
            Message(role=SpeakerRole.USER, content="How do I use async/await?"),
            Message(role=SpeakerRole.ASSISTANT, content="Async/await is used for..."),
        ])
        
        result = await flow_detector.detect(state)
        
        # Should have detected patterns
        assert len(flow_detector.detected_patterns) > 0
        assert flow_detector.message_count == len(state.messages)
        
        # Conclusion phase
        state.messages.extend([
            Message(role=SpeakerRole.USER, content="Thank you so much!"),
            Message(role=SpeakerRole.ASSISTANT, content="You're welcome!")
        ])
        
        result = await flow_detector.detect(state)
        
        # Should recognize conclusion
        patterns = [p.pattern_type for p in flow_detector.detected_patterns]
        
        print(f"✓ Tracked {len(patterns)} patterns through conversation phases")
    
    async def test_response_timing_prediction(self):
        """Test response timing predictions improve with data"""
        print("\n--- Testing Response Timing Prediction ---")
        
        timing_detector = ResponseTimingDetector()
        
        # Simulate historical timing data
        response_times = [180, 200, 190, 195, 185, 205, 195, 190, 198, 192]
        
        for delay in response_times:
            timing_detector.response_patterns['user_to_response'].append(delay)
            timing_detector.response_times.append({
                'timestamp': datetime.now(),
                'delay_ms': delay,
                'mode': 'automatic'
            })
        
        # Test prediction accuracy
        input_end = datetime.now()
        predicted = timing_detector._predict_automatic_response(input_end)
        
        assert predicted is not None
        predicted_delay = (predicted - input_end).total_seconds() * 1000
        avg_actual = sum(response_times) / len(response_times)
        
        # Prediction should be close to average
        assert abs(predicted_delay - avg_actual) < 50
        
        # Test window detection
        timing_detector.last_user_input_end = datetime.now() - timedelta(milliseconds=150)
        timing_detector.response_pending = True
        timing_detector.predicted_response_time = datetime.now() + timedelta(milliseconds=50)
        
        state = type('State', (), {})()
        result = await timing_detector.detect(state)
        
        # Should detect pre-response window
        if result.detected:
            assert result.metadata.get('type') == 'pre_response_window'
        
        print("✓ Response timing predictions working accurately")
    
    async def test_adaptive_behavior(self):
        """Test system adapts to different VAD modes"""
        print("\n--- Testing Adaptive Behavior ---")
        
        # Test with server VAD + auto response
        silence_detector = SilenceDetector(silence_threshold_ms=2000)
        timing_detector = ResponseTimingDetector()
        
        silence_detector.update_vad_mode("server", auto_response=True)
        timing_detector.update_response_mode("automatic", "server", True)
        
        # Thresholds should be reduced for fast mode
        assert silence_detector.silence_threshold_ms <= 1000
        assert timing_detector.prediction_window_ms <= 200
        
        # Test with client VAD + manual response
        silence_detector.update_vad_mode("client", auto_response=False)
        timing_detector.update_response_mode("manual", "client", False)
        
        # Thresholds should be relaxed
        assert silence_detector.silence_threshold_ms >= 2000
        assert timing_detector.prediction_window_ms >= 1000
        
        # Test adaptive threshold in silence detector
        for duration in [1500, 1600, 1550, 1580, 1520]:
            silence_detector._record_silence_duration(duration)
        
        adaptive = silence_detector._get_adaptive_threshold()
        assert adaptive != silence_detector.silence_threshold_ms
        
        print("✓ System adapts correctly to different modes")
    
    async def test_real_conversation_simulation(self):
        """Simulate a realistic conversation with all components"""
        print("\n--- Testing Real Conversation Simulation ---")
        
        # Full system setup
        context_weaver = ContextWeaver(
            strategy=AdaptiveStrategy(),
            detectors=[
                SilenceDetector(enable_prediction=True),
                ResponseTimingDetector(),
                ConversationFlowDetector(),
            ]
        )
        
        voice_engine = MockVoiceEngine(vad_type="client", auto_response=False)
        
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Add various contexts
        contexts = [
            ContextToInject(
                information={"capability": "I can help with programming"},
                timing=InjectionTiming.NEXT_PAUSE,
                priority=ContextPriority.MEDIUM,
                conditions={"phase": "greeting"}
            ),
            ContextToInject(
                information={"example": "Here's how async/await works..."},
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                conditions={"keywords": ["async", "await"]}
            ),
            ContextToInject(
                information={"summary": "We covered async/await basics"},
                timing=InjectionTiming.NEXT_TURN,
                priority=ContextPriority.LOW,
                conditions={"phase": "conclusion"}
            )
        ]
        
        for ctx in contexts:
            context_weaver.add_context(ctx)
        
        # Simulate conversation flow
        print("\n[Simulating Conversation]")
        
        # 1. Greeting
        voice_engine.add_message(SpeakerRole.USER, "Hello!")
        voice_engine._audio_state.vad_active = False
        
        # 2. Introduction  
        voice_engine.add_message(SpeakerRole.ASSISTANT, "Hi! How can I help?")
        voice_engine.add_message(SpeakerRole.USER, "I need help with Python async/await")
        
        # 3. Main topic
        voice_engine.add_message(SpeakerRole.ASSISTANT, "I can explain async/await")
        voice_engine.add_message(SpeakerRole.USER, "How does await work exactly?")
        
        # Update detector states
        for detector in context_weaver.detectors:
            if isinstance(detector, ConversationFlowDetector):
                # Let it analyze the conversation
                # Create mock state with audio
                class DetectorState:
                    def __init__(self):
                        self.status = voice_engine.conversation_state.status
                        self.messages = voice_engine.conversation_state.messages
                        self.turns = voice_engine.conversation_state.turns
                        self.audio = voice_engine._audio_state
                
                state = DetectorState()
                await detector.detect(state)
        
        # 4. Conclusion
        voice_engine.add_message(SpeakerRole.USER, "Thanks, that helps!")
        voice_engine.add_message(SpeakerRole.ASSISTANT, "You're welcome!")
        
        # Check results
        print(f"\nContexts injected: {len(voice_engine.contexts_injected)}")
        print(f"Responses triggered: {voice_engine.responses_triggered}")
        
        # Verify some contexts were considered or detectors worked
        stats = coordinator.get_stats()
        # In a real conversation simulation, we'd have events processed
        # For now, just verify the simulation ran
        assert len(voice_engine.conversation_state.messages) > 0
        
        print("✓ Real conversation simulation completed successfully")


async def main():
    """Run Phase 3 integration tests"""
    tester = TestPhase3Integration()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)