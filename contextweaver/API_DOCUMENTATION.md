# ContextWeaver API Documentation

## Overview

ContextWeaver is a sophisticated context injection system for AI conversations that intelligently adds relevant information at optimal moments without interrupting the natural flow.

## Quick Start

```python
from contextweaver import ContextWeaver
from contextweaver.strategies import AdaptiveStrategy
from contextweaver.detectors import SilenceDetector, ConversationFlowDetector
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority

# Initialize ContextWeaver
context_weaver = ContextWeaver(
    strategy=AdaptiveStrategy(),
    detectors=[
        SilenceDetector(enable_prediction=True),
        ConversationFlowDetector(learning_rate=0.2)
    ]
)

# Start the engine
await context_weaver.start()

# Add context
context = ContextToInject(
    information={"fact": "The sky is blue"},
    timing=InjectionTiming.NEXT_PAUSE,
    priority=ContextPriority.MEDIUM
)
context_weaver.add_context(context)

# Check for injection opportunity
result = await context_weaver.check_injection(conversation_state)
if result:
    # Inject the context
    await voice_engine.send_text(result.information)
```

## Core Components

### 1. ContextWeaver Engine

The main orchestrator that coordinates detectors and strategies.

```python
class ContextWeaver:
    def __init__(
        self,
        strategy: Optional[InjectionStrategy] = None,
        detectors: Optional[List[BaseDetector]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ContextWeaver.
        
        Args:
            strategy: Injection strategy (default: ConservativeStrategy)
            detectors: List of detectors (default: standard set)
            logger: Custom logger instance
        """
```

#### Methods

- `async start()`: Start the context injection engine
- `async stop()`: Stop the engine
- `add_context(context: ContextToInject)`: Add context for injection
- `remove_context(context_id: str)`: Remove context by ID
- `async check_injection(state) -> Optional[ContextToInject]`: Check if injection should occur
- `get_relevant_contexts(state, max_items=5) -> List[ContextToInject]`: Get relevant contexts

### 2. Injection Strategies

#### ConservativeStrategy
Cautious approach, only injects during clear pauses.
```python
strategy = ConservativeStrategy(threshold=0.8)
```

#### AggressiveStrategy
Proactive approach, injects more frequently.
```python
strategy = AggressiveStrategy(threshold=0.5)
```

#### AdaptiveStrategy
Learns from conversation patterns and adapts.
```python
strategy = AdaptiveStrategy(
    initial_threshold=0.6,
    learning_rate=0.1
)
```

### 3. Detectors

#### SilenceDetector
Detects silence periods and predicts optimal injection points.
```python
detector = SilenceDetector(
    silence_threshold_ms=2000,
    enable_prediction=True,
    history_size=10
)
```

#### PauseDetector
Identifies natural conversation pauses.
```python
detector = PauseDetector(
    pause_threshold_ms=500,
    min_confidence=0.7
)
```

#### TopicChangeDetector
Detects topic transitions.
```python
detector = TopicChangeDetector(
    similarity_threshold=0.3,
    keyword_weight=0.7
)
```

#### ResponseTimingDetector
Predicts when AI responses will occur.
```python
detector = ResponseTimingDetector(
    prediction_window_ms=500,
    history_size=20
)
```

#### ConversationFlowDetector
Analyzes conversation patterns and learns.
```python
detector = ConversationFlowDetector(
    learning_rate=0.2,
    pattern_memory_size=50
)
```

### 4. Context Schema

#### ContextToInject
```python
@dataclass
class ContextToInject:
    # Core dimensions
    information: Dict[str, Any]  # What to inject
    strategy: Dict[str, Any]     # How to behave
    attention: Dict[str, Any]    # What to focus on
    
    # Control
    timing: InjectionTiming      # When to inject
    conditions: Dict[str, Any]   # Requirements
    priority: ContextPriority    # Importance
    
    # Metadata
    context_id: str             # Unique ID
    ttl_seconds: Optional[int]  # Time to live
    max_injections: Optional[int]  # Injection limit
```

#### InjectionTiming
```python
class InjectionTiming(Enum):
    IMMEDIATE = "immediate"      # ASAP
    NEXT_TURN = "next_turn"     # Speaker change
    NEXT_PAUSE = "next_pause"   # Natural pause
    ON_TOPIC = "on_topic"       # Topic match
    ON_TRIGGER = "on_trigger"   # Specific trigger
    SCHEDULED = "scheduled"     # Time-based
    LAZY = "lazy"              # Whenever convenient
    MANUAL = "manual"          # Explicit only
```

#### ContextPriority
```python
class ContextPriority(Enum):
    CRITICAL = 10    # Must be delivered
    HIGH = 8         # Important
    MEDIUM = 5       # Standard
    LOW = 3          # Nice to have
    BACKGROUND = 1   # Only if idle
```

## Integration with Voxon

```python
from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig

# Initialize components
voice_engine = VoiceEngine(VoiceEngineConfig(
    api_key="your-key",
    vad_type="server"
))

context_weaver = ContextWeaver(
    strategy=AdaptiveStrategy(),
    detectors=[...]
)

# Create Voxon orchestrator
voxon = Voxon(VoxonConfig())
coordinator = voxon.engine_coordinator
coordinator.voice_engine = voice_engine
coordinator.context_engine = context_weaver

# Initialize with VAD awareness
await coordinator.initialize()

# The system now automatically:
# - Detects VAD mode and adapts strategies
# - Manages injection windows
# - Controls response timing
# - Learns from patterns
```

## Advanced Usage

### 1. Custom Detectors

```python
from contextweaver.detectors import BaseDetector, DetectionResult

class CustomDetector(BaseDetector):
    async def detect(self, state) -> DetectionResult:
        # Your detection logic
        if self._should_detect(state):
            return DetectionResult(
                detected=True,
                confidence=0.9,
                timestamp=datetime.now(),
                metadata={"reason": "custom"}
            )
        return DetectionResult(detected=False, ...)
```

### 2. Learning System

```python
# Record injection outcomes
flow_detector.record_injection_outcome(
    pattern_type="qa_cycle",
    phase=ConversationPhase.MAIN_TOPIC,
    success=True,
    context_type="detailed_answer",
    metadata={}
)

# Get recommendations
context_type = flow_detector.get_recommended_context_type(
    pattern_type="qa_cycle",
    phase=ConversationPhase.MAIN_TOPIC
)
```

### 3. Conditional Context

```python
context = ContextToInject(
    information={"help": "Available commands..."},
    timing=InjectionTiming.NEXT_PAUSE,
    priority=ContextPriority.HIGH,
    conditions={
        "keywords": ["help", "commands"],
        "confidence": 0.8,
        "phase": "main_topic"
    }
)
```

## Performance Considerations

1. **Detector Count**: More detectors increase accuracy but add latency
2. **History Size**: Larger history improves predictions but uses more memory
3. **Learning Rate**: Higher rates adapt faster but may be unstable
4. **Context Queue**: Monitor queue size to prevent memory issues

## Best Practices

1. **Start Conservative**: Begin with ConservativeStrategy and adjust
2. **Monitor Learning**: Track pattern detection accuracy
3. **Set TTLs**: Always set TTL for time-sensitive contexts
4. **Priority Balance**: Use CRITICAL sparingly
5. **Test VAD Modes**: Verify behavior in different VAD configurations

## Error Handling

```python
try:
    result = await context_weaver.check_injection(state)
except Exception as e:
    logger.error(f"Injection check failed: {e}")
    # Fallback behavior
```

## Metrics and Monitoring

```python
# Get detector statistics
stats = detector.get_statistics()

# Get coordinator stats
coord_stats = coordinator.get_stats()

# Monitor injection success
if result:
    metrics.increment("injection.success")
else:
    metrics.increment("injection.skipped")
```