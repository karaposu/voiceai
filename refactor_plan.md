# VoiceEngine Refactor Plan

## 🎯 Objective

Complete architectural refactor of VoiceEngine to prepare for context injection capabilities. This refactor prioritizes clean architecture over legacy compatibility during development, with a final validation that all features still work.

## 🚀 Refactor Strategy

**Approach**: "Rip and Replace" with Safety Net
- Create new architecture alongside old code
- Migrate features systematically  
- Validate with comprehensive tests
- Switch over when complete
- Delete old code

## 📅 Timeline: 5 Weeks

### Week 1: Foundation & Provider Abstraction

#### Day 1-2: Project Setup
- [ ] Create new branch: `feature/voice-engine-refactor`
- [ ] Set up new folder structure:
  ```
  voicechatengine/
  ├── v2/                    # New architecture
  │   ├── core/
  │   ├── providers/
  │   ├── events/
  │   ├── state/
  │   └── tests/
  ├── voice_engine.py        # Current (will be replaced)
  └── base_engine.py         # Current (will be replaced)
  ```
- [ ] Set up test infrastructure for v2

#### Day 3-4: Provider Abstraction
- [ ] Create `BaseProvider` abstract class
- [ ] Implement `OpenAIProvider` with all current functionality
- [ ] Create `MockProvider` for testing
- [ ] Write smoke tests for providers

#### Day 5: Provider Integration Tests
- [ ] Create integration tests for OpenAI provider
- [ ] Validate all current OpenAI features work
- [ ] Document provider interface

### Week 2: Event System & State Management

#### Day 1-2: Event Architecture
- [ ] Implement `EventEmitter` base class
- [ ] Define comprehensive event types:
  ```python
  class EventType(Enum):
      # Connection events
      CONNECTION_ESTABLISHED = "connection.established"
      CONNECTION_LOST = "connection.lost"
      
      # Audio events
      AUDIO_CHUNK_RECEIVED = "audio.chunk.received"
      AUDIO_CHUNK_SENT = "audio.chunk.sent"
      
      # Conversation events
      TURN_STARTED = "conversation.turn.started"
      TURN_COMPLETED = "conversation.turn.completed"
      SILENCE_DETECTED = "conversation.silence.detected"
      TOPIC_CHANGED = "conversation.topic.changed"
      
      # Timing events (for context injection)
      INJECTION_WINDOW_OPENED = "timing.window.opened"
      INJECTION_WINDOW_CLOSED = "timing.window.closed"
  ```
- [ ] Create event handlers and subscription system
- [ ] Write smoke tests for event system

#### Day 3-4: Unified State Management
- [ ] Design `ConversationState` dataclass with all needed fields
- [ ] Implement `StateManager` with atomic updates
- [ ] Add state persistence capabilities
- [ ] Create state transition smoke tests

#### Day 5: State-Event Integration
- [ ] Connect state changes to event emission
- [ ] Implement state snapshots for debugging
- [ ] Performance test state updates (target: <1ms per update)

### Week 3: Conversation Tracking & Timing

#### Day 1-2: Conversation Tracker
- [ ] Implement `MessageHistory` with ring buffer for memory efficiency
- [ ] Create `TurnDetector` for speaker changes
- [ ] Add `TopicTracker` with basic keyword extraction
- [ ] Build conversation metrics calculator

#### Day 3-4: Timing Infrastructure
- [ ] Implement `SilenceDetector` with configurable thresholds
- [ ] Create `InjectionWindowDetector` for natural pause detection
- [ ] Add `ConversationFlowAnalyzer` for pattern detection
- [ ] Write timing accuracy smoke tests

#### Day 5: Integration Testing
- [ ] Create smoke test: `test_04_conversation_tracking.py`
- [ ] Test with real audio samples from test_voice.wav
- [ ] Validate timing accuracy > 95%
- [ ] Measure tracking overhead < 1ms per audio chunk

### Week 4: Architecture Assembly & Migration

#### Day 1-2: Core Engine Rewrite
- [ ] Create new `VoiceEngineV2` class with clean architecture
- [ ] Implement all current VoiceEngine public methods
- [ ] Add backward compatibility layer
- [ ] Write migration guide

#### Day 3-4: Feature Parity Testing
- [ ] Migrate all smoke tests to work with V2
- [ ] Create feature parity smoke test suite:
  ```python
  def test_audio_streaming():
      print("\n=== Test: Audio Streaming ===")
      # Test implementation
      
  def test_text_messaging():
      print("\n=== Test: Text Messaging ===")
      # Test implementation
      
  def test_callbacks():
      print("\n=== Test: Event Callbacks ===")
      # Test implementation
      
  def test_metrics():
      print("\n=== Test: Metrics Collection ===")
      # Test implementation
  ```
- [ ] Fix any feature gaps

#### Day 5: Performance Validation
- [ ] Benchmark v1 vs v2 performance
- [ ] Memory usage comparison
- [ ] Latency impact analysis
- [ ] Optimize hot paths

### Week 5: Testing, Documentation & Cutover

#### Day 1-2: Comprehensive Smoke Testing
- [ ] Update all existing smoke tests to work with V2:
  - `test_01_audio_engine_basics.py` → `test_01_voxstream_basics.py`
  - `test_05_base_engine_audio.py` → Update for V2 architecture
  - `test_08_voice_engine_audio.py` → Full V2 validation
- [ ] Create new V2 smoke tests:
  ```
  smoke_tests/v2/
  ├── test_01_provider_abstraction.py
  ├── test_02_event_system.py
  ├── test_03_state_management.py
  ├── test_04_conversation_tracking.py
  ├── test_05_timing_detection.py
  ├── test_06_feature_parity.py
  ├── test_07_performance_validation.py
  └── test_08_full_integration.py
  ```
- [ ] Add stress test scenarios to smoke tests
- [ ] Create test runner script for V2 tests

#### Day 3: Documentation
- [ ] Architecture documentation with diagrams
- [ ] Migration guide for existing users
- [ ] API reference for new components
- [ ] Performance tuning guide

#### Day 4: Cutover Preparation
- [ ] Create cutover checklist
- [ ] Prepare rollback plan
- [ ] Update all imports to use V2
- [ ] Final integration test run

#### Day 5: Cutover & Cleanup
- [ ] Replace old VoiceEngine with V2
- [ ] Run full test suite
- [ ] Monitor for issues
- [ ] Delete old code after validation period

## 🧪 Test Strategy

### Smoke Test Strategy

#### Core Principle
All tests follow the smoke test pattern used in the project - standalone Python scripts that validate functionality through actual usage rather than unit test frameworks.

#### Test Organization
```
voicechatengine/smoke_tests/v2/
├── test_01_provider_abstraction.py
├── test_02_event_system.py
├── test_03_state_management.py
├── test_04_conversation_tracking.py
├── test_05_timing_detection.py
├── test_06_feature_parity.py
├── test_07_performance_validation.py
└── test_08_full_integration.py
```

### New Smoke Tests to Create

1. **Provider Abstraction Tests** (`test_01_provider_abstraction.py`)
   ```python
   # Test provider switching
   async def test_provider_switching():
       print("\n=== Test: Provider Switching ===")
       engine = VoiceEngineV2()
       
       # Start with OpenAI
       await engine.set_provider("openai")
       await engine.connect()
       print("✓ Connected to OpenAI")
       
       # Switch to Mock
       await engine.set_provider("mock")
       await engine.connect()
       print("✓ Switched to Mock provider")
       
       return True
   ```

2. **Event System Tests** (`test_02_event_system.py`)
   ```python
   # Test event emission and handling
   async def test_event_flow():
       print("\n=== Test: Event Flow ===")
       events_received = []
       
       engine = VoiceEngineV2()
       engine.on_event = lambda e: events_received.append(e)
       
       # Trigger various events
       await engine.connect()
       await engine.send_text("Hello")
       await engine.disconnect()
       
       print(f"✓ Received {len(events_received)} events")
       return len(events_received) >= 3
   ```

3. **State Management Tests** (`test_03_state_management.py`)
   ```python
   # Test state consistency
   def test_state_consistency():
       print("\n=== Test: State Consistency ===")
       
       state = ConversationState()
       
       # Simulate conversation
       state.add_message(Message(role="user", content="Hello"))
       state.add_message(Message(role="assistant", content="Hi there"))
       
       assert state.message_count == 2
       assert state.current_turn == 2
       print("✓ State tracking accurate")
       
       return True
   ```

4. **Timing Detection Tests** (`test_05_timing_detection.py`)
   ```python
   # Test silence detection
   def test_silence_detection():
       print("\n=== Test: Silence Detection ===")
       
       detector = SilenceDetector()
       
       # Test with silent audio
       silent_audio = bytes(4800)  # 100ms silence
       result = detector.process(silent_audio)
       
       assert result.is_silence == True
       print(f"✓ Detected silence: {result.duration_ms}ms")
       
       return True
   ```

## 📊 Success Criteria

### Functional Requirements
- [ ] All current VoiceEngine features working
- [ ] Provider switching capability demonstrated
- [ ] Event system handling 1000+ events/second
- [ ] State updates < 1ms latency
- [ ] Conversation tracking with full history

### Performance Requirements
- [ ] No increase in audio latency (target: < 50ms)
- [ ] Memory usage < 100MB for 1-hour conversation
- [ ] CPU usage < 5% during normal operation
- [ ] Event processing < 1ms per event

### Quality Requirements
- [ ] All smoke tests passing (both existing and new V2 tests)
- [ ] No memory leaks in 24-hour stress test
- [ ] Clean architecture validation (no circular dependencies)
- [ ] Performance benchmarks meet targets

## 📈 Validation Approach

### Feature Parity Validation
Create `test_06_feature_parity.py` that validates all current features:
```python
def run_all_tests():
    print("=" * 50)
    print("Feature Parity Validation")
    print("=" * 50)
    
    tests = [
        test_audio_streaming,
        test_text_messaging,
        test_event_callbacks,
        test_metrics_collection,
        test_error_handling,
        test_reconnection,
        test_vad_integration,
        test_session_management
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nFeature Parity: {passed}/{len(tests)} tests passed")
    return passed == len(tests)
```

### Performance Validation
Create `test_07_performance_validation.py` that ensures no regression:
```python
def test_latency_comparison():
    print("\n=== Test: Latency Comparison ===")
    
    # Test V1
    v1_latencies = measure_v1_latencies()
    
    # Test V2
    v2_latencies = measure_v2_latencies()
    
    # Compare
    latency_increase = (v2_latencies.avg - v1_latencies.avg) / v1_latencies.avg
    
    print(f"V1 avg latency: {v1_latencies.avg:.2f}ms")
    print(f"V2 avg latency: {v2_latencies.avg:.2f}ms")
    print(f"Difference: {latency_increase*100:.1f}%")
    
    return latency_increase < 0.05  # Allow max 5% increase
```

## 🔄 Rollback Plan

If critical issues discovered post-cutover:
1. Revert imports to old VoiceEngine
2. Tag V2 for fixing
3. Deploy hotfix using old code
4. Fix V2 issues offline
5. Re-attempt cutover

## 📈 Monitoring Plan

Post-refactor monitoring:
- Error rates comparison (v1 vs v2)
- Performance metrics dashboard
- Memory usage tracking
- User-reported issues
- Event system health

## 🎯 Definition of Done

The refactor is complete when:
1. All current features work identically
2. All smoke tests pass (new and existing)
3. Performance meets or exceeds v1
4. Documentation is complete
5. Team sign-off on architecture
6. 1 week of stable production usage

## 💡 Future Ready

This refactor prepares VoiceEngine for:
- Context injection (immediate next step)
- Multi-provider support (Anthropic, Google)
- Advanced timing algorithms
- Plugin architecture
- Real-time analytics
- Conversation recording/replay

## 🏁 Final Validation

Run the complete smoke test suite:
```bash
# Run all V2 smoke tests
python -m voicechatengine.smoke_tests.v2.run_all_tests

# Expected output:
# ✓ Provider Abstraction: PASS
# ✓ Event System: PASS
# ✓ State Management: PASS
# ✓ Conversation Tracking: PASS
# ✓ Timing Detection: PASS
# ✓ Feature Parity: PASS
# ✓ Performance: PASS
# ✓ Full Integration: PASS
# 
# All systems ready for context injection!
```

This refactor transforms VoiceEngine into a clean, modular architecture with comprehensive smoke test coverage, ready for context injection while ensuring all current functionality continues to work.