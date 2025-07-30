# ContextEngine Documentation

## What is this for?

ContextEngine provides the schema and data structures for representing contextual information that can be injected into AI conversations. It defines the `ContextToInject` class and related enums that standardize how context is structured, prioritized, and controlled. This module serves as the contract between context providers and the injection system.

## What it requires

### Dependencies
- **Python 3.8+**
- **Core Libraries:**
  - `dataclasses` - For data structure definitions
  - `enum` - For enumeration types
  - `uuid` - For unique context IDs
  - `datetime` - For timing and expiration

### No External Dependencies
- Pure Python implementation
- No network requirements
- No API keys needed
- Minimal memory footprint

## Limitations

1. **Data Structure Constraints**
   - Fixed schema for context representation
   - Dictionary-based information storage
   - No built-in validation beyond types

2. **Timing Limitations**
   - Predefined timing options (enum)
   - No sub-second precision for TTL
   - Limited conditional logic

3. **Priority System**
   - Fixed priority levels (5 levels)
   - No dynamic priority adjustment
   - Simple numeric comparison

4. **Condition Evaluation**
   - Basic dictionary-based conditions
   - No complex boolean logic
   - Limited to key-value matching

## Possible Use Cases

1. **Information Injection**
   - Facts and data ("The capital of France is Paris")
   - API responses (weather, stock prices)
   - Database query results
   - Knowledge base entries

2. **Behavioral Modification**
   - Tone adjustment ("Be more empathetic")
   - Response style ("Use bullet points")
   - Language preferences
   - Personality traits

3. **Attention Direction**
   - Focus areas ("Prioritize security concerns")
   - Important topics ("User mentioned pricing")
   - Goal tracking ("Help user complete purchase")
   - Context awareness

4. **System Messages**
   - Error notifications
   - Status updates
   - Warning messages
   - Debug information

5. **Dynamic Content**
   - Personalized responses
   - User preferences
   - Session context
   - Historical data

## Available Endpoints

### Main Class: `ContextToInject`

```python
@dataclass
class ContextToInject:
    # Core content dimensions
    information: Dict[str, Any] = field(default_factory=dict)
    strategy: Dict[str, Any] = field(default_factory=dict)
    attention: Dict[str, Any] = field(default_factory=dict)
    
    # Control fields
    timing: InjectionTiming = InjectionTiming.NEXT_PAUSE
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: Union[ContextPriority, int] = ContextPriority.MEDIUM
    
    # Metadata
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[int] = None
    source: Optional[str] = None
    
    # Injection tracking
    injection_count: int = 0
    last_injected_at: Optional[float] = None
    max_injections: Optional[int] = None
    
    # Methods
    def is_expired() -> bool
    def is_empty() -> bool
    def should_inject(state: Dict[str, Any]) -> bool
    def mark_injected() -> None
    def to_dict() -> Dict[str, Any]
    @property
    def priority_value() -> int
```

### Enumerations

```python
class InjectionTiming(Enum):
    IMMEDIATE = "immediate"          # Inject ASAP
    NEXT_TURN = "next_turn"         # At speaker change  
    NEXT_PAUSE = "next_pause"       # During silence
    ON_TOPIC = "on_topic"           # When topic matches
    ON_TRIGGER = "on_trigger"       # On specific event
    SCHEDULED = "scheduled"         # At specific time
    LAZY = "lazy"                   # Whenever convenient
    MANUAL = "manual"               # Only when requested

class ContextPriority(Enum):
    CRITICAL = 10     # Must be delivered
    HIGH = 8          # Important
    MEDIUM = 5        # Standard
    LOW = 3           # Nice to have
    BACKGROUND = 1    # Only if nothing else
```

## Interfaces

### 1. **Context Creation**

```python
# Full creation
context = ContextToInject(
    information={"fact": "Paris is the capital of France"},
    strategy={"tone": "educational"},
    attention={"focus": "geography"},
    timing=InjectionTiming.ON_TOPIC,
    conditions={"keywords": ["France", "capital"]},
    priority=ContextPriority.HIGH,
    ttl_seconds=3600,
    source="knowledge_base"
)

# Minimal creation
context = ContextToInject(
    information={"message": "Hello!"}
)
```

### 2. **Condition Interface**

```python
# Simple conditions
conditions = {
    "keywords": ["help", "support"],
    "min_confidence": 0.8,
    "user_type": "premium"
}

# Complex conditions (evaluated by injection system)
conditions = {
    "any_keyword": ["error", "problem", "issue"],
    "all_keywords": ["payment", "failed"],
    "sentiment": "negative",
    "silence_duration_ms": 3000
}

# Advanced conditions with operators
conditions = {
    "or": [
        {"keywords": ["help", "support"]},
        {"sentiment": "confused"},
        {"silence_duration_ms": {"gt": 5000}}
    ],
    "and": [
        {"user_type": "premium"},
        {"conversation_length": {"lt": 10}}
    ]
}

# State-based conditions
conditions = {
    "vad_mode": "server",
    "auto_response": True,
    "message_count": {"gte": 5},
    "last_message_type": "question",
    "time_since_last_injection": {"gt": 30}
}
```

### 3. **Information Structure**

```python
# Flat information
information = {
    "message": "Here's the information you requested",
    "data": "specific_value"
}

# Nested information
information = {
    "product": {
        "name": "Widget",
        "price": 29.99,
        "features": ["A", "B", "C"]
    },
    "availability": True
}

# Action-based information
information = {
    "action": "escalate",
    "target": "human_agent",
    "reason": "Complex issue"
}
```

## Integration Points

### 1. **ContextWeaver Integration**
```python
# ContextWeaver stores and manages ContextToInject objects
context_weaver.add_context(context)

# Checks conditions and timing
if context.should_inject(state):
    return context
```

### 2. **VoxEngine Integration**
```python
# Information is sent as text
await voice_engine.send_text(context.information['message'])

# Or formatted from complex data
text = format_context(context.information)
await voice_engine.send_text(text)
```

### 3. **Application Integration**
```python
# Applications create contexts
def create_help_context(topic):
    return ContextToInject(
        information=load_help_text(topic),
        timing=InjectionTiming.IMMEDIATE,
        priority=ContextPriority.HIGH,
        source="help_system"
    )

# Dynamic context generation
def generate_context(user_query):
    data = database.query(user_query)
    return ContextToInject(
        information={"results": data},
        conditions={"query_complete": True}
    )
```

### 4. **Learning System Integration**
```python
# Track injection success
context.mark_injected()
if user_satisfied:
    record_success(context.context_id)
```

## Edge Cases Covered

1. **Priority Normalization**
   - Integer priorities converted to enums
   - Out-of-range values handled
   - Consistent comparison behavior

2. **Expiration Handling**
   - TTL-based expiration
   - Injection count limits
   - Automatic cleanup triggers

3. **Empty Context**
   - Detection of empty information
   - Validation helpers
   - Injection prevention

4. **ID Uniqueness**
   - UUID generation for each context
   - No collision possibility
   - Traceable contexts

5. **Time Tracking**
   - Creation timestamp
   - Last injection tracking
   - Age calculation

6. **Condition Failures**
   - Graceful handling of missing keys
   - Type mismatches
   - Evaluation errors

7. **Serialization**
   - Clean dictionary conversion
   - JSON compatibility
   - Metadata preservation

## Example Usage

### Basic Context Creation
```python
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority

# Simple informational context
info_context = ContextToInject(
    information={"tip": "Say 'help' for assistance"},
    timing=InjectionTiming.LAZY,
    priority=ContextPriority.LOW
)

# Urgent system message
alert_context = ContextToInject(
    information={"alert": "System maintenance in 5 minutes"},
    timing=InjectionTiming.IMMEDIATE,
    priority=ContextPriority.CRITICAL,
    ttl_seconds=300  # Expires in 5 minutes
)
```

### Conditional Context
```python
# Context that only injects on specific conditions
support_context = ContextToInject(
    information={
        "message": "I can help with that error",
        "solution": "Try restarting the application"
    },
    timing=InjectionTiming.NEXT_PAUSE,
    conditions={
        "keywords": ["error", "not working", "broken"],
        "sentiment": "frustrated"
    },
    priority=ContextPriority.HIGH,
    source="support_system"
)
```

### Complex Behavioral Context
```python
# Context that modifies AI behavior
behavior_context = ContextToInject(
    information={
        "user_name": "John",
        "preference": "formal communication"
    },
    strategy={
        "tone": "professional",
        "verbosity": "concise",
        "use_technical_terms": True
    },
    attention={
        "priority_topics": ["pricing", "features"],
        "avoid_topics": ["competitors"],
        "time_constraint": "5 minutes"
    },
    timing=InjectionTiming.IMMEDIATE,
    priority=ContextPriority.HIGH,
    max_injections=1  # Only inject once
)
```

### Tracking and Management
```python
# Check if context should be injected
state = {"keywords": ["error"], "sentiment": "frustrated"}
if support_context.should_inject(state):
    # Inject the context
    inject(support_context)
    support_context.mark_injected()

# Check expiration
if info_context.is_expired():
    remove_context(info_context.context_id)

# Get context details
context_dict = alert_context.to_dict()
print(f"Context {context_dict['context_id']} created at {context_dict['created_at']}")
```

## How Context Injection Actually Works

The `behavior_context` example above defines WHAT to inject, but the actual injection mechanism is handled by ContextWeaver. Here's the complete flow:

### 1. Context Registration
```python
# Application adds context to ContextWeaver
context_weaver.add_context(behavior_context)
# Context is now stored in context_weaver.available_context dict
```

### 2. Timing Evaluation

The `timing=InjectionTiming.IMMEDIATE` parameter is evaluated by ContextWeaver during its detection cycle:

```python
# Inside ContextWeaver.check_injection() - happens continuously
async def check_injection(self, state):
    # Run detectors (silence, pause, topic change, etc.)
    detections = await self._run_parallel_detection(state)
    
    # Strategy evaluates timing for each available context
    for context in self.available_context.values():
        if context.timing == InjectionTiming.IMMEDIATE:
            # Inject as soon as possible (next check cycle)
            # Usually within 50-100ms
        elif context.timing == InjectionTiming.NEXT_PAUSE:
            # Wait for SilenceDetector to trigger
        elif context.timing == InjectionTiming.ON_TOPIC:
            # Wait for TopicChangeDetector + keyword match
        # ... other timing modes
```

### 3. Injection Execution

When timing conditions are met:

```python
# ContextWeaver returns the context to inject
context = await context_weaver.check_injection(state)
if context:  # This is our behavior_context
    # Voxon/Application extracts and formats the information
    injection_text = format_for_injection(context)
    
    # Send through VoxEngine
    await voice_engine.send_text(injection_text)
    
    # Mark as injected (respects max_injections=1)
    context.mark_injected()
```

### 4. How Behavior Context Affects the AI

The `behavior_context` has three dimensions that affect the AI differently:

```python
# 1. Information - Direct facts to inject
information = {
    "user_name": "John",
    "preference": "formal communication"
}
# → Becomes: "The user's name is John and prefers formal communication"

# 2. Strategy - HOW the AI should behave
strategy = {
    "tone": "professional",
    "verbosity": "concise",
    "use_technical_terms": True
}
# → Becomes: "Please adopt a professional tone, be concise, and use technical terminology"

# 3. Attention - WHAT to focus on
attention = {
    "priority_topics": ["pricing", "features"],
    "avoid_topics": ["competitors"],
    "time_constraint": "5 minutes"
}
# → Becomes: "Focus on pricing and features. Avoid mentioning competitors. Keep this under 5 minutes."
```

### 5. Complete Example with Actual Injection

```python
# Step 1: Create behavioral context
behavior_context = ContextToInject(
    information={"user_name": "John", "is_premium": True},
    strategy={"tone": "professional", "response_style": "bullet points"},
    attention={"focus": "technical details", "urgency": "high"},
    timing=InjectionTiming.IMMEDIATE,
    priority=ContextPriority.CRITICAL
)

# Step 2: Add to ContextWeaver
context_weaver.add_context(behavior_context)

# Step 3: In the conversation loop (handled by Voxon)
@voice_engine.events.on('conversation.started')
async def on_conversation_start(event):
    # ContextWeaver checks for immediate contexts
    context = await context_weaver.check_injection(state)
    
    if context and context.timing == InjectionTiming.IMMEDIATE:
        # Format the complete behavioral instruction
        prompt = f"""
        User Information: {json.dumps(context.information)}
        Communication Style: {json.dumps(context.strategy)}
        Focus Areas: {json.dumps(context.attention)}
        
        Please acknowledge and apply these parameters.
        """
        
        # Inject into conversation
        await voice_engine.send_text(prompt)
        
        # AI receives this and adjusts behavior accordingly

# Step 4: Subsequent AI responses will reflect the behavior
# User: "Tell me about your product"
# AI: "Good afternoon, John. Here are our key features:
#      • Feature A: Technical specification...
#      • Feature B: Implementation details...
#      [Professional tone, bullet points, technical focus]"
```

For more details on the detection and timing mechanisms, see:
- [ContextWeaver Detection Process](contextweaver.md#detection) - Lines 126-174
- [ContextWeaver Strategy Decision](contextweaver.md#interfaces) - Lines 156-166
- [Voxon Orchestration](voxon.md#integration-points) - Lines 185-220