# VoiceAI Documentation

This directory contains comprehensive documentation for all major modules in the VoiceAI system.

## Architecture Overview

The VoiceAI system consists of four main modules working together:

```
┌─────────────────────────────────────────────────────┐
│                    Applications                      │
│         (Customer Support, Tutors, Assistants)       │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                      Voxon                           │
│         (High-level Orchestration Layer)             │
│   • Coordinates engines                              │
│   • Manages conversation state                       │
│   • Handles timing and synchronization               │
└────────┬─────────────────────────┬──────────────────┘
         │                         │
┌────────▼──────────┐    ┌────────▼──────────────────┐
│   VoxEngine       │    │    ContextWeaver          │
│ (Voice I/O Layer) │    │ (Context Injection Layer) │
│                   │    │                           │
│ • Audio streaming │    │ • Detects opportunities  │
│ • Speech ↔ Text   │    │ • Manages contexts       │
│ • VAD handling    │    │ • Learning system        │
└───────────────────┘    └────────┬──────────────────┘
                                  │
                         ┌────────▼──────────────────┐
                         │    ContextEngine         │
                         │  (Schema & Data Layer)   │
                         │                          │
                         │ • Context data structure │
                         │ • Priority system        │
                         │ • Timing control         │
                         └────────────────────────────┘
```

## Module Documentation

### Core Modules

1. **[VoxEngine](voxengine.md)** - Low-level voice I/O engine
   - Real-time audio streaming
   - Voice Activity Detection (VAD)
   - OpenAI Realtime API integration
   - Modern event system ([details](events.md))
   - Provider abstraction ([details](providers.md))

2. **[ContextWeaver](contextweaver.md)** - Intelligent context injection system
   - Multiple detection strategies
   - Parallel detection execution ([details](parallel_detection.md))
   - Learning from conversation patterns
   - VAD-aware timing optimization
   - Built-in detectors:
     - SilenceDetector - VAD-aware silence detection
     - PauseDetector - Natural conversation pauses
     - TopicChangeDetector - Semantic topic transitions
     - ResponseTimingDetector - Response prediction
     - ConversationFlowDetector - Pattern learning

3. **[Voxon](voxon.md)** - High-level orchestration layer
   - Engine coordination
   - State management
   - Response control
   - Timing synchronization
   - Session management ([details](sessions.md))
   - Conversation templates

4. **[ContextEngine](contextengine.md)** - Context data structures
   - Schema definitions
   - Priority system
   - Timing enumerations
   - Condition framework

### System Documentation

5. **[Event System](events.md)** - Complete event reference
   - Modern EventType system
   - Event flow patterns
   - Advanced event handling
   - Migration from callbacks

6. **[Provider System](providers.md)** - Voice AI backend abstraction
   - Available providers (OpenAI, Mock, Custom)
   - Provider interface
   - Implementation guide
   - Provider comparison

7. **[Session Management](sessions.md)** - Conversation continuity
   - Session lifecycle
   - Memory management
   - User adaptation
   - Analytics and monitoring

8. **[Parallel Detection](parallel_detection.md)** - Performance optimization
   - Why parallel detection matters
   - Implementation details
   - Performance benchmarks
   - Real-world impact

## Key Concepts

### Voice Activity Detection (VAD)
- **Client VAD**: Detection happens on client side, more control
- **Server VAD**: Detection by AI provider, lower latency
- Affects injection timing and strategies

### Context Injection
- **Timing**: When to inject (immediate, next pause, on topic, etc.)
- **Priority**: Importance levels (critical, high, medium, low, background)
- **Conditions**: Requirements for injection (keywords, sentiment, etc.)
- **Learning**: System improves timing based on outcomes

### Response Control
- **Automatic**: AI responds automatically after silence
- **Manual**: Explicit trigger required (`response.create`)
- **Hybrid**: Adaptive based on conversation flow

## Integration Patterns

### Basic Setup
```python
# 1. Create engines
voice_engine = VoiceEngine(config)
context_weaver = ContextWeaver(strategy, detectors)

# 2. Create orchestrator
voxon = Voxon(voxon_config)
coordinator = voxon.engine_coordinator

# 3. Connect engines
coordinator.voice_engine = voice_engine
coordinator.context_engine = context_weaver

# 4. Initialize
await coordinator.initialize()
```

### Context Flow
```
User Speech → VoxEngine → State Update → Voxon → Enhanced State
                                                        ↓
                                                  ContextWeaver
                                                        ↓
                                                 Injection Decision
                                                        ↓
Context Injection ← VoxEngine ← Voxon ← Context To Inject
```

### Enhanced State

The "Enhanced State" is created by Voxon by enriching the basic state from VoxEngine with additional context:

**Basic State (from VoxEngine):**
- Audio information (VAD active, silence duration)
- Connection status
- Raw conversation data

**Enhanced State (after Voxon processing):**
- **Original data** - All VoxEngine state data
- **Conversation context** - Full history, turns, metrics
- **VAD configuration** - Detected mode (client/server) and injection strategy
- **Timing analysis** - Calculated injection windows based on VAD mode
- **Response control** - Current response mode and permissions
- **Metadata** - Additional context for intelligent decisions

This enrichment enables ContextWeaver to make more informed injection decisions based on the complete conversation context rather than just audio state.

## Performance Characteristics

### Latency Metrics
- **Detection Latency**: 
  - Sequential: ~45ms (5 detectors)
  - Parallel: <10ms (5 detectors)
  - Per detector: 5-15ms typical
- **Injection Window**: 
  - Server VAD + Auto: 50-100ms (tight)
  - Server VAD + Manual: 200-500ms (controlled)
  - Client VAD + Auto: 200-500ms (moderate)
  - Client VAD + Manual: 500-2000ms (relaxed)
- **Context Processing**: 2-5ms per context
- **State Enhancement**: <1ms

### Resource Usage
- **Memory Usage**: 
  - Base: ~50MB
  - Per conversation: +2-5MB
  - Context queue: ~1KB per context
  - Detector memory: 10-20MB total
- **CPU Usage**: 
  - Idle: <1%
  - Active conversation: 5-10%
  - Parallel detection spike: 15-20%
  - Learning operations: +5%

### Scalability
- **Concurrent Conversations**: Limited by VoxEngine instances
- **Context Queue Size**: Tested up to 1000 contexts
- **Message History**: Efficient up to 10,000 messages
- **Detector Count**: Optimal 3-7, tested up to 20

## Common Use Cases

1. **Customer Support** - FAQ injection, escalation, empathy
2. **Education** - Hints, examples, encouragement  
3. **Technical Help** - Code snippets, documentation, debugging
4. **Healthcare** - Disclaimers, reminders, emergency info
5. **Entertainment** - Story elements, game hints, trivia

## Best Practices

1. Start with conservative injection strategies
2. Use parallel detection for performance
3. Set appropriate TTLs on time-sensitive contexts
4. Monitor learning system outcomes
5. Test with different VAD configurations
6. Handle edge cases gracefully

## Debugging Tips

1. Enable debug logging for timing analysis
2. Monitor event flow between modules
3. Check detector statistics
4. Verify state consistency
5. Track injection success rates

## Quick Start Guide

```python
from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig
from contextweaver import ContextWeaver, AdaptiveStrategy
from contextweaver.detectors import SilenceDetector, ConversationFlowDetector
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority

# 1. Initialize engines
voice_engine = VoiceEngine(VoiceEngineConfig(
    api_key="your-openai-key",
    vad_type="server"  # or "client"
))

context_weaver = ContextWeaver(
    strategy=AdaptiveStrategy(),
    detectors=[SilenceDetector(), ConversationFlowDetector()],
    parallel_detection=True
)

# 2. Create orchestrator
voxon = Voxon(VoxonConfig(enable_context_injection=True))
coordinator = voxon.engine_coordinator
coordinator.voice_engine = voice_engine
coordinator.context_engine = context_weaver

# 3. Add some context
context = ContextToInject(
    information={"greeting": "Hello! How can I help you today?"},
    timing=InjectionTiming.IMMEDIATE,
    priority=ContextPriority.HIGH
)
context_weaver.add_context(context)

# 4. Start conversation
await coordinator.initialize()
await voice_engine.connect()

# System handles everything automatically!
```

## Glossary

- **VAD (Voice Activity Detection)** - Technology that detects when someone is speaking
  - **Client VAD** - Detection happens on user's device, more control
  - **Server VAD** - Detection by AI provider, lower latency
- **Enhanced State** - VoxEngine state enriched with conversation context by Voxon
- **Injection Window** - Time period when context can be injected without interrupting
- **Detection Cycle** - One complete run of all detectors checking for opportunities
- **Response Control** - Managing when and how the AI responds (automatic vs manual)

## Troubleshooting Guide

### Common Issues and Solutions

**1. Context not being injected**
- Check VAD mode and injection windows
- Verify context conditions match current state
- Enable debug logging: `logging.getLogger('contextweaver').setLevel(logging.DEBUG)`
- Ensure detectors are running: check `context_weaver.is_active()`

**2. Injection timing conflicts**
- Review VAD mode compatibility (server VAD = tight windows)
- Consider switching to manual response mode for complex injections
- Increase detection cycle frequency in Voxon config
- Use `InjectionTiming.IMMEDIATE` for critical contexts

**3. Poor detection performance**
- Enable parallel detection: `parallel_detection=True`
- Reduce detector timeout: `detection_timeout_ms=30`
- Profile slow detectors: `context_weaver.get_detection_stats()`
- Remove unnecessary detectors from the pipeline

**4. Learning system not improving**
- Ensure outcome recording: `detector.record_injection_outcome(...)`
- Check learning rate: `AdaptiveStrategy(learning_rate=0.3)`
- Verify sufficient conversation history
- Reset learning data if patterns changed significantly

**5. State synchronization issues**
- Ensure Voxon is properly initialized
- Check event listener connections
- Verify both engines are connected to coordinator
- Monitor state update frequency

### Debug Logging Configuration

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Module-specific debugging
logging.getLogger('voxengine').setLevel(logging.DEBUG)
logging.getLogger('contextweaver').setLevel(logging.DEBUG)
logging.getLogger('voxon').setLevel(logging.DEBUG)

# Performance profiling
logging.getLogger('contextweaver.performance').setLevel(logging.DEBUG)
```

### Performance Tuning Tips

1. **For Low Latency**
   - Use server VAD with automatic response
   - Enable parallel detection
   - Reduce detector count
   - Lower detection timeout

2. **For High Accuracy**
   - Use client VAD with manual response
   - Enable all detectors
   - Increase detection timeout
   - Enable learning systems

3. **For Complex Applications**
   - Use hybrid approach with mode switching
   - Implement custom detectors for domain logic
   - Fine-tune injection strategies
   - Monitor and adapt based on metrics

## Additional Resources

- [Event System Documentation](events.md) - Complete event reference
- [Parallel Detection Deep Dive](parallel_detection.md) - Performance optimization
- [API Reference](contextweaver.md#available-endpoints) - ContextWeaver APIs
- [Architecture Decisions](../what_voxon_is.md) - Design philosophy
- [Implementation Examples](contextweaver.md#example-usage) - Code samples

For detailed information about each module, see the individual documentation files.