# ContextWeaver Documentation

## What is this for?

ContextWeaver is an intelligent context injection system that enhances AI conversations by adding relevant information at optimal moments without interrupting the natural flow. It uses multiple detectors to identify injection opportunities and strategies to decide when and what to inject. The system learns from conversation patterns to improve its timing and relevance over time.

## What it requires

### Dependencies
- **Python 3.8+**
- **Core Libraries:**
  - `asyncio` - Asynchronous detector execution
  - `datetime` - Timing and timestamps
  - `statistics` - Learning and analysis
  - `logging` - Debug and monitoring

### Integration Requirements
- **Conversation State** - Access to current conversation state
- **Event System** - For real-time conversation monitoring

### Configuration
```python
ContextWeaver(
    strategy=AdaptiveStrategy(),        # Injection strategy
    detectors=[...],                    # List of detectors
    parallel_detection=True,            # Enable parallel execution
    detection_timeout_ms=50             # Timeout per detector
)
```

## Limitations

1. **Timing Constraints**
   - Detection must complete within timeout window (default 50ms)
   - Cannot inject during active speech (VAD active)
   - Limited by voice engine response time

2. **Context Management**
   - Contexts expire based on TTL
   - Memory usage scales with context queue size
   - Maximum injection frequency limits

3. **Learning System**
   - Requires conversation history for accuracy
   - Learning rate affects adaptation speed
   - Pattern recognition limited to implemented detectors

4. **Detection Accuracy**
   - Dependent on quality of conversation state
   - May miss opportunities in rapid conversations
   - False positives possible in edge cases

## Possible Use Cases

1. **Customer Support**
   - Inject FAQ answers during pauses
   - Add policy information when relevant
   - Provide escalation options on frustration

2. **Educational Assistants**
   - Inject hints after thinking pauses
   - Add examples when concepts mentioned
   - Provide encouragement during struggles

3. **Technical Support**
   - Inject code examples for programming help
   - Add documentation links contextually
   - Provide debugging steps on errors

4. **Healthcare Assistants**
   - Inject medication reminders
   - Add medical disclaimers when needed
   - Provide emergency contacts contextually

5. **Sales & Marketing**
   - Inject product information
   - Add promotional offers at right moments
   - Provide comparison data when shopping

6. **Gaming & Entertainment**
   - Inject story elements
   - Add hints for puzzles
   - Provide lore information

## Available Endpoints

### Main Class: `ContextWeaver`

```python
class ContextWeaver:
    # Lifecycle
    async def start() -> None
    async def stop() -> None
    
    # Context Management
    def add_context(context: ContextToInject) -> None
    def remove_context(context_id: str) -> None
    def add_raw_context(**kwargs) -> ContextToInject
    
    # Detection
    async def check_injection(state: Any) -> Optional[ContextToInject]
    def get_relevant_contexts(state: Any, max_items: int = 5) -> List[ContextToInject]
    
    # Configuration
    @property
    def is_active() -> bool
    @property
    def available_context() -> Dict[str, ContextToInject]
```

### Detector Interface

```python
class BaseDetector:
    async def detect(state: Any) -> DetectionResult
    def get_statistics() -> Dict[str, Any]
    
class DetectionResult:
    detected: bool
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
```

### Strategy Interface

```python
class InjectionStrategy:
    async def decide(
        detections: List[DetectionResult],
        state: Any,
        available_context: Dict[str, ContextToInject]
    ) -> InjectionDecision
    
    def record_injection() -> None
    def get_injection_rate(window_seconds: int) -> float
```

## Interfaces

### 1. **Detectors**

**Built-in Detectors:**

1. **`SilenceDetector`** - VAD-aware silence detection
   - Monitors audio state for silence periods
   - Adjusts thresholds based on VAD mode
   - Predicts upcoming silence windows
   - Config: `silence_threshold_ms`, `enable_prediction`

2. **`PauseDetector`** - Natural conversation pause identification
   - Detects meaningful pauses vs brief hesitations
   - Analyzes pause patterns in conversation
   - Differentiates thinking pauses from turn-taking
   - Config: `min_pause_ms`, `max_pause_ms`

3. **`TopicChangeDetector`** - Semantic topic transition detection
   - Compares message similarity using embeddings
   - Identifies topic shifts and new subjects
   - Tracks topic history throughout conversation
   - Config: `similarity_threshold`, `lookback_messages`

4. **`ResponseTimingDetector`** - AI response timing prediction
   - Predicts when AI will start responding
   - Adapts to different response modes
   - Calculates safe injection windows
   - Config: `response_mode`, `prediction_confidence`

5. **`ConversationFlowDetector`** - Pattern learning and analysis
   - Learns from conversation patterns
   - Identifies recurring interaction cycles
   - Adapts injection strategy based on success
   - Config: `learning_rate`, `pattern_memory_size`

**Custom Detector Interface:**
```python
class CustomDetector(BaseDetector):
    async def detect(self, state) -> DetectionResult:
        # Your detection logic
        return DetectionResult(...)
```

### 2. **Strategies**

**Built-in Strategies:**
- `ConservativeStrategy` - Cautious, only clear opportunities
- `AggressiveStrategy` - Proactive, more frequent injection
- `AdaptiveStrategy` - Learns and adapts to conversation

**Custom Strategy Interface:**
```python
class CustomStrategy(InjectionStrategy):
    async def decide(self, detections, state, contexts) -> InjectionDecision:
        # Your decision logic
        return InjectionDecision(...)
```

### 3. **Context Schema**

ContextWeaver includes built-in schema definitions for context data:

```python
from contextweaver import ContextToInject, InjectionTiming, ContextPriority

@dataclass
class ContextToInject:
    # Content
    information: Dict[str, Any]      # What to inject
    strategy: Dict[str, Any]         # How to behave
    attention: Dict[str, Any]        # What to focus on
    
    # Control
    timing: InjectionTiming          # When to inject
    conditions: Dict[str, Any]       # Requirements
    priority: ContextPriority        # Importance
    
    # Metadata
    context_id: str                  # Unique ID
    ttl_seconds: Optional[int]       # Expiration
    max_injections: Optional[int]    # Limit
```

**InjectionTiming Options:**
- `IMMEDIATE` - Inject as soon as possible
- `NEXT_TURN` - At speaker change
- `NEXT_PAUSE` - During next silence
- `ON_TOPIC` - When topic matches
- `ON_TRIGGER` - On specific event
- `SCHEDULED` - At specific time
- `LAZY` - Whenever convenient
- `MANUAL` - Only when requested

**ContextPriority Levels:**
- `CRITICAL` (10) - Must be delivered
- `HIGH` (8) - Important
- `MEDIUM` (5) - Standard
- `LOW` (3) - Nice to have
- `BACKGROUND` (1) - Only if nothing else

## Integration Points

### 1. **Voxon Integration**
```python
# Voxon coordinates ContextWeaver
coordinator = voxon.engine_coordinator
coordinator.context_engine = context_weaver
await coordinator.initialize()

# Automatic VAD mode detection
# Response timing coordination
# Injection window management
```

### 2. **VoxEngine Integration**
```python
# Inject context through voice engine
if context := await context_weaver.check_injection(state):
    await voice_engine.send_text(context.information)
```

### 3. **Event System Integration**
```python
# Monitor conversation events
@voice_engine.events.on('conversation.updated')
async def check_context(event):
    context = await context_weaver.check_injection(event.state)
```

### 4. **Learning System Integration**
```python
# Record outcomes for learning
detector.record_injection_outcome(
    pattern_type="qa_cycle",
    phase=phase,
    success=True,
    context_type="answer",
    metadata={...}
)
```

## Edge Cases Covered

1. **Timing Edge Cases**
   - Rapid conversation switches
   - Overlapping detection windows
   - Simultaneous injection opportunities
   - VAD mode changes mid-conversation

2. **Detection Failures**
   - Detector timeouts (50ms limit)
   - Detector exceptions isolated
   - Partial detection results handled
   - Graceful degradation

3. **Context Conflicts**
   - Priority-based resolution
   - Timing conflict handling
   - Condition evaluation failures
   - Expired context cleanup

4. **Performance Edge Cases**
   - High context volume handling
   - Slow detector mitigation
   - Memory pressure management
   - CPU spike protection

5. **Learning Edge Cases**
   - Cold start (no history)
   - Outlier pattern handling
   - Conversation style changes
   - Reset and retraining

6. **State Inconsistencies**
   - Missing state properties
   - Malformed conversation data
   - Audio state unavailable
   - Partial state updates

7. **Concurrency Issues**
   - Parallel detection safety
   - State mutation protection
   - Event ordering guarantees
   - Resource contention

## Example Usage

### Basic Setup
```python
from contextweaver import ContextWeaver
from contextweaver.strategies import AdaptiveStrategy
from contextweaver.detectors import SilenceDetector, ConversationFlowDetector

# Initialize with parallel detection
context_weaver = ContextWeaver(
    strategy=AdaptiveStrategy(learning_rate=0.2),
    detectors=[
        SilenceDetector(enable_prediction=True),
        ConversationFlowDetector()
    ],
    parallel_detection=True
)

await context_weaver.start()
```

### Adding Context
```python
from contextweaver import ContextToInject, InjectionTiming, ContextPriority

# Add knowledge base context
context = ContextToInject(
    information={
        "policy": "30-day return policy",
        "details": "Items must be unused"
    },
    timing=InjectionTiming.ON_TOPIC,
    priority=ContextPriority.HIGH,
    conditions={"keywords": ["return", "refund"]},
    ttl_seconds=3600
)
context_weaver.add_context(context)
```

### Checking for Injection
```python
# In conversation loop
@voice_engine.events.on('user.silence')
async def on_silence(event):
    context = await context_weaver.check_injection(conversation_state)
    if context:
        await voice_engine.send_text(context.information['policy'])
```

### Learning Integration
```python
# Basic outcome recording
flow_detector = context_weaver.detectors[1]
flow_detector.record_injection_outcome(
    pattern_type="policy_question",
    phase=ConversationPhase.MAIN_TOPIC,
    success=True,
    context_type="policy_info"
)

# Advanced learning with metadata
flow_detector.record_injection_outcome(
    pattern_type="technical_support",
    phase=ConversationPhase.PROBLEM_SOLVING,
    success=True,
    context_type="code_example",
    metadata={
        "user_satisfaction": 0.9,
        "resolution_time": 45,
        "injection_delay_ms": 150,
        "follow_up_needed": False
    }
)

# Adaptive strategy learning
strategy = AdaptiveStrategy(learning_rate=0.2)
strategy.learn_from_conversation({
    "total_injections": 5,
    "successful_injections": 4,
    "user_interruptions": 1,
    "avg_response_time": 230
})

# Pattern analysis
patterns = flow_detector.get_learned_patterns()
print(f"Most successful pattern: {patterns['most_successful']}")
print(f"Optimal timing: {patterns['avg_successful_timing_ms']}ms")
```

### Performance Optimization with Parallel Detection

```python
# Sequential detection (slow)
context_weaver = ContextWeaver(
    strategy=AdaptiveStrategy(),
    detectors=[...],
    parallel_detection=False  # ~45ms per cycle
)

# Parallel detection (fast) 
context_weaver = ContextWeaver(
    strategy=AdaptiveStrategy(),
    detectors=[...],
    parallel_detection=True,  # ~10ms per cycle
    detection_timeout_ms=50   # Prevent slow detectors
)

# Monitor performance
stats = context_weaver.get_detection_stats()
print(f"Avg detection time: {stats['avg_detection_ms']}ms")
print(f"Slowest detector: {stats['slowest_detector']}")
```