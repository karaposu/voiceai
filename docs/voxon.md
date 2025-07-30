# Voxon Documentation

## What is this for?

Voxon is the **Conversation Intelligence Layer** that orchestrates - but does not implement - the core functionality of voice conversations. It coordinates between VoxEngine (voice I/O) and ContextWeaver (context injection) to create intelligent, context-aware conversations while maintaining a clear separation of concerns.

### What Voxon IS:

**1. Conversation Orchestrator**
- Manages the flow of multi-turn conversations
- Coordinates between VoxEngine and ContextWeaver
- Handles conversation state transitions and lifecycle
- Provides high-level conversation abstractions

**2. Intelligence Layer**
- Conversation memory and history management
- Turn-taking logic and interruption handling
- Context awareness and topic tracking
- Conversation analytics and insights

**3. High-Level Abstractions**
- Simplified API for building conversational applications
- Pre-built conversation patterns (Q&A, multi-turn dialogues)
- Session management across multiple conversations
- User preference and adaptation layer

### What Voxon IS NOT:

**1. NOT a Voice Engine**
- Doesn't handle audio I/O directly
- Doesn't manage WebSocket connections
- Doesn't deal with audio codecs or streaming

**2. NOT a Context Engine**
- Doesn't implement RAG or knowledge retrieval
- Doesn't manage vector databases
- Doesn't handle prompt engineering directly

**3. NOT Low-Level Infrastructure**
- Doesn't implement transport protocols
- Doesn't handle device management
- Doesn't deal with real-time audio processing

This separation allows each component to excel at its core responsibility while Voxon provides the intelligence to coordinate them effectively.

## Direct Responsibilities vs Delegation

### What Voxon Handles Directly:
```python
# Direct responsibilities
- Conversation flow control
- Turn management and timing
- Memory persistence
- Conversation templates
- Multi-engine coordination
- High-level event aggregation
- Conversation metrics
- Session continuity
- State enhancement for context decisions
```

### What Voxon Delegates:
```python
# Delegates to VoxEngine
- Audio input/output
- Real-time streaming
- Voice activity detection
- Audio device management
- Low-level connection handling
- Speech-to-text/text-to-speech

# Delegates to ContextWeaver
- Context retrieval timing
- Knowledge base queries
- Dynamic prompt construction
- RAG operations
- Embedding management
- Injection opportunity detection
```

## What it requires

### Dependencies
- **Python 3.8+**
- **Core Components:**
  - `VoxEngine` - For voice I/O capabilities
  - `ContextWeaver` - For intelligent context injection
  - `EventEmitter` - For event-driven architecture

### Configuration
```python
VoxonConfig(
    # Conversation settings
    enable_context_injection=True,
    enable_learning=True,
    
    # Timing settings
    monitor_interval=0.1,           # How often to check for injection
    
    # Identity/personality (optional)
    identity=Identity(...)
)
```

### Required Engines
```python
# Voxon requires both engines to be configured
voxon = Voxon(config)
coordinator = voxon.engine_coordinator
coordinator.voice_engine = voice_engine      # Required
coordinator.context_engine = context_weaver   # Required
await coordinator.initialize()
```

## Limitations

1. **Engine Dependencies**
   - Requires both VoxEngine and ContextWeaver
   - Cannot function with partial engine setup
   - Engine compatibility requirements

2. **State Management**
   - Immutable state requires careful update patterns
   - State synchronization overhead
   - Memory usage scales with conversation length

3. **Coordination Overhead**
   - Additional latency from orchestration layer
   - Event propagation delays
   - Monitoring interval limits reaction time

4. **Complexity**
   - More complex than direct engine usage
   - Requires understanding of all three layers
   - Debugging across layers can be challenging

## Possible Use Cases

1. **Enterprise Voice Assistants**
   - Multi-department support systems
   - Intelligent call routing
   - Context-aware help desks

2. **Educational Platforms**
   - Adaptive tutoring systems
   - Language learning applications
   - Interactive course assistants

3. **Healthcare Applications**
   - Patient intake systems
   - Medication adherence assistants
   - Mental health support bots

4. **Customer Service**
   - Intelligent IVR systems
   - Support ticket creation
   - FAQ automation with context

5. **Developer Tools**
   - Code review assistants
   - Documentation helpers
   - Debugging companions

6. **Entertainment**
   - Interactive storytelling
   - Game NPCs with memory
   - Virtual companions

## Available Endpoints

### Main Class: `Voxon`

```python
class Voxon:
    # Properties
    engine_coordinator: EngineCoordinator
    config: VoxonConfig
    
    # Methods (typically accessed via engine_coordinator)
```

### EngineCoordinator Class

```python
class EngineCoordinator:
    # Lifecycle
    async def initialize() -> None
    async def shutdown() -> None
    
    # Engine Management
    voice_engine: VoiceEngine
    context_engine: ContextWeaver
    
    # VAD Configuration
    def update_vad_configuration(vad_type: str, auto_response: bool) -> None
    
    # Response Control
    def set_response_mode(mode: str, auto_response: bool) -> None
    async def trigger_manual_response() -> bool
    
    # State Access
    def get_conversation_state() -> ConversationState
    def get_stats() -> Dict[str, Any]
    
    # Properties
    @property
    def vad_mode() -> str
    @property
    def auto_response() -> bool
    @property
    def injection_mode() -> str
```

### State Management

```python
# ConversationState (immutable)
@dataclass(frozen=True)
class ConversationState:
    conversation_id: str
    status: ConversationStatus
    messages: List[Message]
    turns: List[Turn]
    metrics: ConversationMetrics
    
    # Methods
    def evolve(**changes) -> ConversationState
    def to_dict() -> Dict[str, Any]
    
    # Properties
    @property
    def is_active() -> bool
    @property
    def message_count() -> int
    @property
    def last_message() -> Optional[Message]
```

## Interfaces

### 1. **Engine Interfaces**

```python
# Voice Engine Interface
voice_engine: VoiceEngine
- send_text(text: str)
- send_audio(audio: bytes)
- events: EventEmitter

# Context Engine Interface  
context_engine: ContextWeaver
- add_context(context: ContextToInject)
- check_injection(state) -> Optional[ContextToInject]
```

### 2. **Component Interfaces**

**VADModeAdapter**
```python
class VADModeAdapter:
    def analyze_vad_mode(config) -> Tuple[str, bool]
    def get_injection_mode(vad_mode, auto_response) -> str
    def get_timing_params(vad_mode, auto_response) -> Dict
```

**ResponseController**
```python
class ResponseController:
    async def should_trigger_response(state) -> bool
    async def execute_injection(inject_cb, trigger_cb) -> bool
    def update_mode(response_mode, vad_mode, auto_response)
```

**InjectionWindowManager**
```python
class InjectionWindowManager:
    def calculate_window(silence_duration, vad_mode, auto_response) -> Dict
    def should_force_inject(timing, vad_mode) -> Tuple[bool, bool]
```

### 3. **Event System**

```python
# Voxon subscribes to VoxEngine events
voice_engine.events.on('conversation.updated', handle_update)
voice_engine.events.on('audio.vad', handle_vad)
voice_engine.events.on('error', handle_error)

# Voxon emits orchestration events
coordinator.events.emit('injection.completed', context)
coordinator.events.emit('mode.changed', mode_info)
```

## Integration Points

### 1. **VoxEngine Integration**
```python
# Voxon manages VoxEngine lifecycle
coordinator.voice_engine = voice_engine
await coordinator.initialize()  # Initializes voice engine

# Monitors voice events
# Injects context via voice engine
# Manages response timing
```

### 2. **ContextWeaver Integration**
```python
# Voxon coordinates context injection
coordinator.context_engine = context_weaver

# Provides enhanced state to ContextWeaver
# Manages injection timing windows
# Handles learning feedback
```

### 3. **State Flow**
```python
VoxEngine → ConversationState → Voxon → Enhanced State → ContextWeaver
                                  ↓
                            Decision Making
                                  ↓
                            Context Injection → VoxEngine
```

**Enhanced State Creation Details:**

```python
# Basic state from VoxEngine
basic_state = {
    "audio": {
        "vad_active": False,
        "silence_duration_ms": 1500,
        "last_audio_timestamp": 1234567890
    },
    "connection": {
        "status": "connected",
        "latency_ms": 45
    }
}

# Voxon enhances with:
enhanced_state = {
    # All original state data
    **basic_state,
    
    # Conversation context
    "conversation": {
        "conversation_id": "conv_abc123",
        "status": "active",
        "messages": [...],  # Full message history
        "turns": [...],     # Speaker turns
        "metrics": {
            "duration_seconds": 120,
            "message_count": 15,
            "turn_count": 8,
            "avg_message_length": 45
        }
    },
    
    # VAD configuration analysis
    "vad_info": {
        "mode": "server",              # Detected from config
        "auto_response": True,         # Response behavior
        "injection_mode": "tight",     # Strategy recommendation
        "confidence": 0.95             # Detection confidence
    },
    
    # Timing analysis
    "timing": {
        "injection_window_ms": 100,       # Calculated window
        "window_type": "tight",           # Classification
        "window_start": 1234567890,       # Window timing
        "window_end": 1234567990,
        "should_force": False,            # Force injection
        "should_trigger_after": False     # Post-injection response
    },
    
    # Response control state
    "response_control": {
        "mode": "automatic",
        "can_interrupt": False,
        "pending_response": False,
        "last_response_at": 1234567000,
        "response_count": 7
    },
    
    # Additional metadata
    "metadata": {
        "enhancement_timestamp": 1234567891,
        "voxon_version": "1.0.0",
        "active_detectors": 5,
        "injection_history": [...]
    }
}
```

### 4. **External Integration**
```python
# Application level
app = MyVoiceApp()
app.voxon = Voxon(config)

# Configure engines
app.voxon.engine_coordinator.voice_engine = create_voice_engine()
app.voxon.engine_coordinator.context_engine = create_context_weaver()

# Start conversation
await app.voxon.engine_coordinator.initialize()
```

## Edge Cases Covered

1. **Engine Coordination**
   - Handles engine initialization order
   - Manages engine failure recovery
   - Prevents race conditions between engines
   - Synchronizes state updates

2. **VAD Mode Transitions**
   - Smooth transitions between VAD modes
   - Adapts injection strategy dynamically
   - Maintains conversation continuity
   - Handles mid-conversation changes

3. **State Consistency**
   - Immutable state prevents corruption
   - Atomic state updates
   - State rollback on errors
   - Version tracking for debugging

4. **Timing Conflicts**
   - Prevents injection during speech
   - Manages overlapping windows
   - Prioritizes user experience
   - Handles rapid mode switches

5. **Resource Management**
   - Proper engine cleanup
   - Memory-efficient state storage
   - Event listener cleanup
   - Circular reference prevention

6. **Error Propagation**
   - Catches engine errors
   - Provides meaningful error context
   - Maintains operation despite errors
   - Logs for debugging

7. **Performance**
   - Async coordination without blocking
   - Efficient state updates
   - Minimal overhead monitoring
   - Optimized event handling

## Conversation Templates

Voxon provides pre-built conversation patterns:

```python
# Customer Support Template
conversation = await voxon.start_conversation(
    template="customer_support",
    context={
        "user_id": "123",
        "product": "VoiceAI",
        "support_tier": "premium"
    }
)

# Educational Tutor Template
conversation = await voxon.start_conversation(
    template="educational_tutor",
    context={
        "subject": "mathematics",
        "level": "intermediate",
        "learning_style": "visual"
    }
)

# Medical Assistant Template
conversation = await voxon.start_conversation(
    template="medical_assistant",
    context={
        "patient_id": "456",
        "appointment_type": "follow_up",
        "privacy_mode": True
    }
)
```

### Session Management

```python
# Session continuity across conversations
session = voxon.create_session(user_id="user_123")

# First conversation
conv1 = await session.start_conversation()
await conv1.send_message("Remember my name is John")
await session.save()

# Later conversation (remembers context)
conv2 = await session.resume_conversation()
response = await conv2.send_message("What's my name?")
# AI responds: "Your name is John"

# Session analytics
analytics = session.get_analytics()
print(f"Total conversations: {analytics['conversation_count']}")
print(f"Average duration: {analytics['avg_duration_seconds']}s")
print(f"Topics discussed: {analytics['topics']}")
```

## Example Usage

### Basic Setup
```python
from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig
from contextweaver import ContextWeaver

# Create Voxon orchestrator
voxon = Voxon(VoxonConfig(
    enable_context_injection=True,
    enable_learning=True
))

# Configure engines
voice_engine = VoiceEngine(VoiceEngineConfig(
    api_key="sk-...",
    vad_type="server"
))

context_weaver = ContextWeaver(
    strategy=AdaptiveStrategy(),
    detectors=[...]
)

# Connect engines
coordinator = voxon.engine_coordinator
coordinator.voice_engine = voice_engine
coordinator.context_engine = context_weaver

# Initialize (detects VAD mode, configures strategies)
await coordinator.initialize()
```

### Managing Conversation
```python
# Start conversation
await voice_engine.connect()

# Voxon automatically:
# - Monitors conversation state
# - Checks for injection opportunities  
# - Manages response timing
# - Coordinates engines

# Change modes dynamically
coordinator.set_response_mode("manual", auto_response=False)

# Get conversation state
state = coordinator.get_conversation_state()
print(f"Messages: {state.message_count}")
```

### Advanced Features
```python
# Manual response control
if coordinator.response_controller:
    should_respond = await coordinator.response_controller.should_trigger_response(state)
    if should_respond:
        await coordinator.trigger_manual_response()

# Get performance stats
stats = coordinator.get_stats()
print(f"Events processed: {stats['events_processed']}")
print(f"Injection mode: {stats['injection_mode']}")

# VAD mode transitions
# Scenario: Switch from natural conversation to precise control
coordinator.set_response_mode("manual", auto_response=False)
coordinator.update_vad_configuration("client", auto_response=False)

# Scenario: High-latency network, switch to client VAD
if network_latency > 200:
    coordinator.update_vad_configuration("client", auto_response=True)
    
# Scenario: Critical context injection needed
if critical_context_pending:
    # Temporarily switch to manual control
    coordinator.set_response_mode("manual", auto_response=False)
    await inject_critical_context()
    # Switch back to automatic
    coordinator.set_response_mode("automatic", auto_response=True)
```

### VAD Mode Transition Examples

```python
# Example 1: Adaptive VAD based on conversation phase
@voice_engine.events.on('conversation.phase_change')
async def adapt_vad_mode(event):
    if event.phase == "greeting":
        # Fast, natural responses for greeting
        coordinator.update_vad_configuration("server", True)
    elif event.phase == "complex_query":
        # More control for complex interactions
        coordinator.update_vad_configuration("client", False)
    elif event.phase == "closing":
        # Back to natural for goodbye
        coordinator.update_vad_configuration("server", True)

# Example 2: Quality-based adaptation
@voice_engine.events.on('audio.quality_check')
async def adapt_to_quality(event):
    if event.packet_loss > 5:
        # Switch to client VAD for reliability
        coordinator.update_vad_configuration("client", True)
    elif event.background_noise > threshold:
        # Client VAD handles noise better
        coordinator.update_vad_configuration("client", False)
```

### Clean Shutdown
```python
# Voxon handles proper cleanup
await coordinator.shutdown()
# - Stops monitoring
# - Cleans up engines
# - Removes event listeners
```

## Key Design Principles

1. **Separation of Concerns**: Voxon orchestrates but doesn't duplicate engine functionality
2. **Engine Agnostic**: Can work with different voice/context engine implementations
3. **High-Level Focus**: Provides abstractions for common conversation patterns
4. **Stateful Management**: Maintains conversation context across turns
5. **Observable**: Rich events for monitoring conversation flow

### Architecture Benefits

This separation allows:
- **VoxEngine** to focus on being the best voice I/O engine
- **ContextWeaver** to focus on intelligent context retrieval
- **Voxon** to focus on making conversations feel natural and intelligent

Each component can be developed, tested, and scaled independently while Voxon ensures they work together seamlessly.