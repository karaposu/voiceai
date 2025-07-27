# Context Injection Expansion for VoiceEngine

## 🎯 Executive Summary

VoiceEngine currently handles voice conversations but lacks context injection capabilities. This document outlines how to extend VoiceEngine to support the sophisticated ContextEngine architecture while maintaining clean separation of concerns and backward compatibility.

## 🔍 Current State Analysis

### What VoiceEngine Does Well
- Manages WebSocket connections to AI providers
- Handles audio streaming (via VoxStream)
- Provides event-driven callbacks
- Tracks metrics and usage

### What's Missing
- No mechanism to inject context mid-conversation
- No awareness of conversation state beyond basic metrics
- No integration points for external context sources
- No timing-aware injection opportunities

## 🏗️ Proposed Architecture

### Core Design Principles

1. **Non-Invasive Integration**: Context injection should be optional - VoiceEngine works without it
2. **Event-Driven Communication**: VoiceEngine emits events, ContextEngine listens and responds
3. **Provider Abstraction**: Context injection works regardless of AI provider
4. **Zero Latency Impact**: Context injection happens asynchronously without blocking audio

### Integration Layers

```
ContextEngine (creates contexts)
      ↓ [ContextToInject objects]
ContextInjectionEngine (manages delivery)
      ↓ [Injection Events]
VoiceEngine (injects into conversation)
      ↓ [Provider-specific format]
AI Provider (OpenAI, Anthropic, etc.)
```

## 📊 Required Additions to VoiceEngine

### 1. Context Injection Interface

VoiceEngine needs a clean interface for receiving and injecting contexts:

```python
class ContextInjectionInterface:
    async def inject_context(self, context: ContextToInject) -> InjectionResult
    def can_inject_now(self) -> bool
    def get_injection_windows(self) -> List[InjectionWindow]
```

**Why**: Provides a standard way for ContextInjectionEngine to deliver contexts without knowing VoiceEngine internals.

### 2. Conversation State Exposure

VoiceEngine must expose rich conversation state for context conditions:

```python
@dataclass
class ConversationState:
    # Timing information
    current_turn: int
    last_speech_end: float
    silence_duration: float
    conversation_duration: float
    
    # Content information
    current_topics: List[str]
    recent_keywords: List[str]
    detected_emotions: Dict[str, float]
    
    # Interaction patterns
    turn_taking_pattern: str  # "balanced", "user_dominant", etc.
    response_latencies: List[float]
    interruption_count: int
```

**Why**: ContextToInject objects need this state to evaluate their conditions and determine if they should be injected.

### 3. Injection Timing Events

VoiceEngine needs to emit events at natural injection points:

```python
class InjectionOpportunityEvent:
    type: InjectionTiming  # NEXT_TURN, NEXT_PAUSE, etc.
    window_duration_ms: int
    confidence: float
```

Events to emit:
- `silence_detected` - Natural pause in conversation
- `turn_completed` - Speaker change
- `topic_shift` - Conversation topic changed
- `emotion_change` - Emotional tone shifted
- `buffer_low` - Good time for injection without latency

**Why**: ContextInjectionEngine needs to know when injection windows occur to deliver contexts with appropriate timing.

### 4. Provider-Agnostic Injection Layer

Abstract the actual injection mechanism:

```python
class ProviderInjector(ABC):
    @abstractmethod
    async def inject(self, formatted_context: str) -> bool
    
    @abstractmethod
    def format_context(self, context: ContextToInject) -> str
```

**Why**: Different providers have different context injection methods:
- OpenAI: `conversation.item.create` with system message
- Anthropic: XML-formatted context blocks
- Google: Metadata in request headers

### 5. Injection Queue Management

VoiceEngine needs an internal queue for contexts awaiting injection:

```python
class InjectionQueue:
    def add(self, context: ContextToInject)
    def get_next_for_timing(self, timing: InjectionTiming) -> Optional[ContextToInject]
    def clear_expired()
```

**Why**: Contexts arrive asynchronously but must be injected at specific moments. The queue holds them until the right opportunity.

### 6. Streaming Integration Points

For real-time streaming, VoiceEngine needs hook points:

- **Pre-audio hooks**: Before sending audio to provider
- **Mid-stream hooks**: During silence or pauses
- **Post-response hooks**: After receiving complete response

**Why**: Enables non-blocking context injection that doesn't interrupt the audio stream.

## 🔄 Integration Flow

### Initialization Phase
1. VoiceEngine starts with optional ContextInjectionEngine reference
2. Registers injection interface if provided
3. Begins emitting conversation state events

### Runtime Flow
1. **VoiceEngine** processes audio and emits state events
2. **ContextEngine** creates ContextToInject based on events
3. **ContextInjectionEngine** queues contexts by timing
4. **VoiceEngine** signals injection opportunities
5. **ContextInjectionEngine** delivers appropriate contexts
6. **VoiceEngine** formats and injects into provider stream

### Example Interaction
```
User speaks → 
  VoiceEngine emits 'speech_started' →
    ContextEngine begins analysis →
  VoiceEngine emits 'speech_ended' →
    Fast context ready (50ms) →
  VoiceEngine emits 'turn_completed' →
    ContextInjectionEngine delivers context →
  VoiceEngine injects before AI response
```

## 🎯 Implementation Strategy

### Phase 1: Foundation (Week 1-2)
1. Add ConversationState tracking
2. Implement basic event emission
3. Create ContextInjectionInterface

### Phase 2: Integration (Week 2-3)
1. Add injection queue
2. Implement provider formatters
3. Create injection timing logic

### Phase 3: Advanced Features (Week 3-4)
1. Add streaming hook points
2. Implement retry logic
3. Add performance monitoring

### Phase 4: Testing & Optimization (Week 4-5)
1. Integration tests with ContextEngine
2. Latency impact analysis
3. Provider-specific optimizations

## 🚨 Critical Considerations

### Latency Management
- Context injection must NEVER block audio processing
- Use async operations with timeouts
- Skip injection rather than delay response

### Provider Differences
- OpenAI: Supports mid-conversation system messages
- Anthropic: Prefers context in initial prompt
- Google: Uses conversation metadata

### Failure Handling
- Graceful degradation if injection fails
- Continue conversation without context
- Log failures for analysis

### Token Management
- Track token usage for context
- Compress contexts if needed
- Priority-based context selection

## 📈 Success Metrics

1. **Zero Latency Impact**: P99 latency unchanged
2. **Injection Success Rate**: >95% for critical contexts  
3. **Natural Flow**: No noticeable conversation disruption
4. **Provider Coverage**: Works with all major providers

## 🔮 Future Enhancements

### Multi-Modal Context
- Image context for vision-enabled models
- Audio context for sound analysis
- Document context for reference

### Predictive Injection
- Anticipate context needs
- Pre-load contexts before needed
- Speculative context preparation

### Context Feedback Loop
- Track which contexts improved conversation
- Learn optimal injection timing
- Personalize context selection

## 🎬 Conclusion

Adding context injection to VoiceEngine requires careful design to maintain real-time performance while enabling sophisticated context delivery. The proposed architecture achieves this through:

1. **Clean Interfaces**: Well-defined boundaries between systems
2. **Event-Driven Design**: Loosely coupled communication
3. **Timing Awareness**: Natural injection points
4. **Provider Abstraction**: Works with any AI provider
5. **Performance First**: Never compromise latency

This expansion transforms VoiceEngine from a simple voice handler into an intelligent conversation orchestrator capable of dynamic context injection while maintaining its core strength in real-time audio processing.