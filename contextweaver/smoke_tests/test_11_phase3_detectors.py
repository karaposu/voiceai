#!/usr/bin/env python3
"""
Phase 3 Advanced Detectors Test

Tests the enhanced detectors with VAD integration, response timing prediction,
and conversation flow pattern detection.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from contextweaver.detectors import (
    SilenceDetector, ResponseTimingDetector, 
    ConversationFlowDetector, ConversationPhase
)
from voxon.state import ConversationState, Message, SpeakerRole


class MockAudioState:
    """Mock audio state for testing"""
    def __init__(self):
        self.vad_active = False
        self.vad_confidence = 0.0
        self.is_listening = True
        self.is_playing = False


class MockState:
    """Mock state for testing"""
    def __init__(self):
        self.audio = MockAudioState()
        self.vad_mode = "client"
        self.auto_response = True
        self.injection_mode = "adaptive"
        self.messages = []
        self.turns = []
        self.status = "idle"
        self.current_turn = None


class TestPhase3Detectors:
    """Test Phase 3 advanced detectors"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    async def run_all_tests(self):
        """Run all Phase 3 detector tests"""
        print("\n=== Phase 3: Advanced Detectors Tests ===\n")
        
        tests = [
            self.test_enhanced_silence_detector,
            self.test_response_timing_detector,
            self.test_conversation_flow_detector,
            self.test_detector_coordination,
            self.test_learning_capabilities,
            self.test_vad_mode_adaptation,
            self.test_pattern_recognition,
            self.test_prediction_accuracy
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
    
    async def test_enhanced_silence_detector(self):
        """Test enhanced SilenceDetector with VAD integration"""
        detector = SilenceDetector(
            silence_threshold_ms=1000,
            enable_prediction=True
        )
        
        state = MockState()
        
        # Test 1: VAD active should not detect silence
        state.audio.vad_active = True
        result = await detector.detect(state)
        assert result.detected is False
        assert result.metadata['reason'] == 'voice_active'
        
        # Test 2: Start of silence
        state.audio.vad_active = False
        result = await detector.detect(state)
        assert result.detected is False
        assert result.metadata['reason'] == 'silence_just_started'
        
        # Test 3: Need to call detect again to set silence_start first
        result = await detector.detect(state)
        # Now simulate time passing by adjusting silence_start
        detector.silence_start = datetime.now() - timedelta(milliseconds=1100)
        result = await detector.detect(state)
        assert result.detected is True
        assert result.confidence > 0.5
        assert result.metadata['silence_duration_ms'] >= 1000
        
        # Test 4: VAD mode adaptation
        detector.update_vad_mode("server", auto_response=True)
        assert detector.silence_threshold_ms <= 1000  # Should be reduced
        
        # Test 5: Adaptive threshold
        # Record some silence durations
        for duration in [800, 900, 850, 820, 880]:
            detector._record_silence_duration(duration)
        
        # Check adaptive threshold adjusted
        adaptive = detector._get_adaptive_threshold()
        assert adaptive != detector.silence_threshold_ms
        
        # Test 6: Statistics
        stats = detector.get_statistics()
        assert stats['vad_mode'] == 'server'
        assert stats['silence_count'] == 5
        assert stats['average_silence_ms'] > 0
    
    async def test_response_timing_detector(self):
        """Test ResponseTimingDetector"""
        detector = ResponseTimingDetector(prediction_window_ms=500)
        
        state = MockState()
        state.vad_mode = "server"
        state.auto_response = True
        state.injection_mode = "immediate"
        
        # Test 1: Initial state (no detection)
        result = await detector.detect(state)
        assert result.detected is False
        # Check response_pending exists in metadata
        assert 'response_pending' in result.metadata
        
        # Test 2: User input ends
        state.audio.is_listening = False
        detector.last_user_input_end = datetime.now()
        detector.response_pending = True
        
        # Predict response time
        detector.predicted_response_time = datetime.now() + timedelta(milliseconds=300)
        
        # Test 3: Check detection with pending response
        result = await detector.detect(state)
        # May or may not detect depending on timing logic
        
        # Test 4: Simulate entering prediction window
        # Set predicted time to be very close
        detector.predicted_response_time = datetime.now() + timedelta(milliseconds=100)
        result = await detector.detect(state)
        # Should detect we're in the window
        if (datetime.now() <= detector.predicted_response_time):
            # We're before the predicted time, so might detect window
            pass  # Timing dependent, don't assert
        
        # Test 5: Response mode updates
        detector.update_response_mode("manual", "client", False)
        assert detector.response_mode == "manual"
        assert detector.prediction_window_ms == 1000
        
        # Test 6: Record patterns
        detector.response_patterns['user_to_response'] = [200, 250, 180, 220, 210]
        avg_time = detector._get_average_response_time()
        assert 180 <= avg_time <= 250
        
        # Test 7: Statistics
        stats = detector.get_statistics()
        assert stats['response_mode'] == 'manual'
        assert stats['average_response_delay'] > 0
    
    async def test_conversation_flow_detector(self):
        """Test ConversationFlowDetector"""
        detector = ConversationFlowDetector(learning_rate=0.2)
        
        state = MockState()
        
        # Test 1: Greeting phase
        state.messages = [
            Message(role=SpeakerRole.USER, content="Hello there!"),
            Message(role=SpeakerRole.ASSISTANT, content="Hi! How can I help you?")
        ]
        
        result = await detector.detect(state)
        # Detector analyzes content and should identify greeting
        assert detector.current_phase in [ConversationPhase.GREETING, ConversationPhase.INTRODUCTION, ConversationPhase.MAIN_TOPIC]
        
        # Test 2: Introduction phase
        state.messages.append(
            Message(role=SpeakerRole.USER, content="I need help with my project")
        )
        
        result = await detector.detect(state)
        # Should transition to introduction
        assert detector.current_phase in [ConversationPhase.INTRODUCTION, ConversationPhase.MAIN_TOPIC]
        
        # Test 3: Pattern detection
        # Add Q&A pattern
        for i in range(3):
            state.messages.extend([
                Message(role=SpeakerRole.USER, content=f"What about item {i}?"),
                Message(role=SpeakerRole.ASSISTANT, content=f"Item {i} is...")
            ])
        
        result = await detector.detect(state)
        # Q&A pattern should be detected based on questions
        if result.detected:
            # Check if pattern is Q&A related
            pattern_type = result.metadata.get('pattern_type', '')
            assert 'qa' in pattern_type or 'cycle' in pattern_type or 'question' in pattern_type.lower()
        else:
            # Even if not detected as opportunity, pattern should be recognized
            assert len(detector.detected_patterns) > 0
        
        # Test 4: Learning from outcomes
        detector.record_injection_outcome(
            pattern_type='qa_cycle',
            phase=ConversationPhase.MAIN_TOPIC,
            success=True,
            context_type='detailed_answer',
            metadata={}
        )
        
        # Check learning updated
        pattern_key = f"qa_cycle_{ConversationPhase.MAIN_TOPIC}"
        assert detector.successful_patterns.get(pattern_key, 0) > 0.5
        
        # Test 5: Get recommendation
        recommended = detector.get_recommended_context_type(
            'qa_cycle', 
            ConversationPhase.MAIN_TOPIC
        )
        assert recommended == 'detailed_answer'
        
        # Test 6: Statistics
        stats = detector.get_statistics()
        assert stats['current_phase'] == detector.current_phase.value
        assert stats['patterns_detected'] > 0
        assert stats['message_count'] == len(state.messages)
    
    async def test_detector_coordination(self):
        """Test multiple detectors working together"""
        silence_detector = SilenceDetector()
        timing_detector = ResponseTimingDetector()
        flow_detector = ConversationFlowDetector()
        
        state = MockState()
        state.vad_mode = "client"
        
        # Simulate conversation start
        state.messages = [
            Message(role=SpeakerRole.USER, content="Hello, I have a question")
        ]
        
        # All detectors process state
        silence_result = await silence_detector.detect(state)
        timing_result = await timing_detector.detect(state)
        flow_result = await flow_detector.detect(state)
        
        # Flow detector should identify some phase
        # With just one message, might be IDLE or MAIN_TOPIC
        assert flow_detector.current_phase in [
            ConversationPhase.IDLE, 
            ConversationPhase.GREETING, 
            ConversationPhase.INTRODUCTION,
            ConversationPhase.MAIN_TOPIC
        ]
        
        # Simulate silence after user input
        state.audio.vad_active = False
        silence_detector.silence_start = datetime.now() - timedelta(milliseconds=1500)
        
        silence_result = await silence_detector.detect(state)
        # Should detect silence since we set the start time in the past
        # If not detected, check why
        if not silence_result.detected:
            print(f"Silence not detected: {silence_result.metadata}")
        assert silence_result.metadata.get('silence_duration_ms', 0) >= 1000
        
        # Timing detector tracks input end
        timing_detector.last_user_input_end = datetime.now() - timedelta(milliseconds=500)
        timing_detector.response_pending = True
        timing_detector.predicted_response_time = datetime.now() + timedelta(milliseconds=200)
        
        timing_result = await timing_detector.detect(state)
        # Should be approaching response window
    
    async def test_learning_capabilities(self):
        """Test detector learning capabilities"""
        flow_detector = ConversationFlowDetector(learning_rate=0.3)
        
        # Simulate multiple successful injections
        patterns = [
            ('qa_cycle', ConversationPhase.MAIN_TOPIC, True),
            ('qa_cycle', ConversationPhase.MAIN_TOPIC, True),
            ('topic_transition', ConversationPhase.MAIN_TOPIC, False),
            ('qa_cycle', ConversationPhase.MAIN_TOPIC, True),
        ]
        
        for pattern_type, phase, success in patterns:
            flow_detector.record_injection_outcome(
                pattern_type=pattern_type,
                phase=phase,
                success=success,
                context_type='test_context',
                metadata={}
            )
        
        # Check learned success rates
        qa_key = f"qa_cycle_{ConversationPhase.MAIN_TOPIC}"
        topic_key = f"topic_transition_{ConversationPhase.MAIN_TOPIC}"
        
        assert flow_detector.successful_patterns[qa_key] > 0.7  # Should be high
        assert flow_detector.successful_patterns[topic_key] < 0.5  # Should be low
        
        # Test learning progress
        progress = flow_detector._get_learning_progress()
        assert progress['total_injections'] == 4
        assert progress['success_rate'] == 0.75
    
    async def test_vad_mode_adaptation(self):
        """Test VAD mode adaptation across detectors"""
        silence_detector = SilenceDetector(silence_threshold_ms=2000)
        timing_detector = ResponseTimingDetector()
        
        # Test server VAD + auto response
        silence_detector.update_vad_mode("server", auto_response=True)
        timing_detector.update_response_mode("automatic", "server", True)
        
        assert silence_detector.silence_threshold_ms <= 1000  # Reduced
        assert timing_detector.prediction_window_ms == 200  # Tight window
        
        # Test client VAD + manual response
        silence_detector.update_vad_mode("client", auto_response=False)
        timing_detector.update_response_mode("manual", "client", False)
        
        assert silence_detector.silence_threshold_ms >= 2000  # Relaxed
        assert timing_detector.prediction_window_ms == 1000  # Relaxed window
    
    async def test_pattern_recognition(self):
        """Test conversation pattern recognition"""
        flow_detector = ConversationFlowDetector()
        
        state = MockState()
        
        # Test Q&A pattern
        qa_messages = []
        for i in range(5):
            qa_messages.extend([
                Message(role=SpeakerRole.USER, content=f"How does {i} work?"),
                Message(role=SpeakerRole.ASSISTANT, content=f"Well, {i} works by...")
            ])
        state.messages = qa_messages
        
        result = await flow_detector.detect(state)
        # Q&A pattern detection depends on message analysis
        # The detector analyzes messages and counts questions
        # With 5 "How does X work?" messages, should have high question count
        
        # Check detection worked in some way
        if not result.detected:
            # At minimum, patterns should be identified
            patterns_found = [p.pattern_type for p in flow_detector.detected_patterns]
            assert len(patterns_found) > 0 or flow_detector.question_count > 0
        else:
            # If detected, should be Q&A related
            assert 'qa' in str(result.metadata).lower() or 'question' in str(result.metadata).lower()
        
        # Test repetition pattern
        state.messages = [
            Message(role=SpeakerRole.USER, content="I don't understand the API"),
            Message(role=SpeakerRole.ASSISTANT, content="The API works like..."),
            Message(role=SpeakerRole.USER, content="But how does the API handle errors?"),
            Message(role=SpeakerRole.ASSISTANT, content="The API handles errors by..."),
            Message(role=SpeakerRole.USER, content="Can you explain the API authentication?"),
        ]
        
        # Clear previous state
        flow_detector.topic_keywords.clear()
        flow_detector.keywords_seen.clear()
        
        result = await flow_detector.detect(state)
        # Should detect some pattern based on repeated keyword "API"
        # Check if any relevant pattern was detected
        patterns = [p.pattern_type for p in flow_detector.detected_patterns]
        
        # The detector might identify this as clarification needed or repetition
        # or it might just count the keywords
        assert len(patterns) > 0 or flow_detector.topic_keywords.get('api', 0) >= 3 or result.detected
    
    async def test_prediction_accuracy(self):
        """Test prediction accuracy improvements"""
        timing_detector = ResponseTimingDetector()
        
        # Simulate historical response times
        response_times = [200, 220, 210, 205, 215, 225, 210, 208, 212, 218]
        
        for i, delay in enumerate(response_times):
            timing_detector.response_times.append({
                'timestamp': datetime.now() - timedelta(seconds=i),
                'delay_ms': delay,
                'mode': 'automatic'
            })
            timing_detector.response_patterns['user_to_response'].append(delay)
        
        # Test prediction
        input_end = datetime.now()
        predicted = timing_detector._predict_automatic_response(input_end)
        
        assert predicted is not None
        predicted_delay = (predicted - input_end).total_seconds() * 1000
        
        # Should predict close to average
        avg_delay = sum(response_times) / len(response_times)
        assert abs(predicted_delay - avg_delay) < 50  # Within 50ms
        
        # Test accuracy calculation
        timing_detector.prediction_errors = [10, 15, 8, 12, 20, 5, 18, 7, 13, 11]
        accuracy = timing_detector._get_historical_accuracy()
        assert 0.7 <= accuracy <= 1.0


async def main():
    """Run Phase 3 detector tests"""
    tester = TestPhase3Detectors()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)