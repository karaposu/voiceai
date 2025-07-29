# Voxon: Conversation Intelligence Layer

## What Voxon IS:

**1. Conversation Orchestrator**
- Manages the flow of multi-turn conversations
- Coordinates between voxengine (voice I/O) and contextinjectionengine (context management)
- Handles conversation state transitions and lifecycle

**2. Intelligence Layer**
- Conversation memory and history management
- Turn-taking logic and interruption handling
- Context awareness and topic tracking
- Conversation analytics and insights

**3. High-Level Abstractions**
- Simplified API for building conversational applications
- Pre-built conversation patterns (Q&A, multi-turn dialogues, etc.)
- Session management across multiple conversations
- User preference and adaptation layer

## What Voxon IS NOT:

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

## What Voxon Handles Directly:

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
```

## What Voxon Abstracts Away:

```python
# Delegates to voxengine
- Audio input/output
- Real-time streaming
- Voice activity detection
- Audio device management
- Low-level connection handling

# Delegates to contextinjectionengine
- Context retrieval
- Knowledge base queries
- Dynamic prompt construction
- RAG operations
- Embedding management
```

## Architecture Example:

```python
# Voxon orchestrates but doesn't implement low-level details
voxon = Voxon()

# Configure the engines it will orchestrate
voxon.set_voice_engine(voxengine)
voxon.set_context_engine(contextinjectionengine)

# High-level conversation API
conversation = await voxon.start_conversation(
    template="customer_support",
    context={"user_id": "123", "product": "VoiceAI"},
    memory_mode="persistent"
)

# Voxon handles the coordination
response = await conversation.send_message("I need help with my order")
# Under the hood:
# 1. Voxon asks contextinjectionengine for relevant context
# 2. Voxon sends enhanced prompt to voxengine
# 3. Voxon manages conversation state and history
# 4. Voxon returns unified response
```

## Key Design Principles:

1. **Separation of Concerns**: Voxon orchestrates but doesn't duplicate functionality
2. **Engine Agnostic**: Can work with different voice/context engine implementations
3. **High-Level Focus**: Provides abstractions for common conversation patterns
4. **Stateful Management**: Maintains conversation context across turns
5. **Observable**: Rich events for monitoring conversation flow

This separation allows:
- voxengine to focus on being the best voice I/O engine
- contextinjectionengine to focus on intelligent context retrieval
- voxon to focus on making conversations feel natural and intelligent