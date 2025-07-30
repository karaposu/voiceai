# Session Management Documentation

## Overview

Session management in VoiceAI provides conversation continuity, user preferences, and analytics across multiple interactions. Sessions are managed at the Voxon level and coordinate state across VoxEngine and ContextWeaver.

## Session Concepts

### Session Lifecycle

```
Session Created
    ↓
Conversation 1 Started → Active → Ended
    ↓
Session Saved (memory persisted)
    ↓
Conversation 2 Started (memory restored) → Active → Ended
    ↓
Session Closed
```

### Session Components

1. **Session State** - User preferences, history, context
2. **Conversation Memory** - What was discussed across conversations  
3. **Learning Data** - Patterns and preferences learned
4. **Analytics** - Usage statistics and insights

## Session Configuration

### SessionConfig

```python
from voxengine import SessionConfig

@dataclass
class SessionConfig:
    # Identity
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    
    # AI Configuration
    instructions: str = "You are a helpful assistant"
    voice: str = "alloy"
    language: Optional[str] = None
    
    # Behavior
    temperature: float = 0.8
    max_output_tokens: int = 4096
    enable_functions: bool = False
    
    # Memory
    memory_mode: Literal["ephemeral", "persistent", "hybrid"] = "hybrid"
    max_memory_size: int = 10000  # tokens
    
    # Features
    enable_learning: bool = True
    enable_analytics: bool = True
    enable_adaptation: bool = True
```

### Session Presets

```python
from voxengine import SessionPresets

# Conversation-focused preset
config = SessionPresets.CONVERSATION
# - Natural temperature (0.8)
# - Persistent memory
# - Learning enabled

# Task-focused preset  
config = SessionPresets.TASK_ORIENTED
# - Lower temperature (0.3)
# - Ephemeral memory
# - Function calling enabled

# Educational preset
config = SessionPresets.EDUCATIONAL
# - Balanced temperature (0.5)
# - Hybrid memory
# - Adaptation enabled

# Customer support preset
config = SessionPresets.CUSTOMER_SUPPORT
# - Professional tone
# - Persistent memory
# - Analytics enabled
```

## Session Management API

### Creating Sessions

```python
from voxon import Voxon

voxon = Voxon(config)

# Create a new session
session = voxon.create_session(
    user_id="user_123",
    config=SessionConfig(
        instructions="You are a friendly tutor",
        memory_mode="persistent",
        enable_learning=True
    )
)

# Or use preset
session = voxon.create_session(
    user_id="user_456",
    preset=SessionPresets.EDUCATIONAL
)
```

### Session Operations

```python
# Start a conversation within session
conversation = await session.start_conversation()

# Send messages
response = await conversation.send_message("Hello!")

# Access session info
print(f"Session ID: {session.session_id}")
print(f"User ID: {session.user_id}")
print(f"Active: {session.is_active}")

# Save session state
await session.save()

# Load existing session
session = await voxon.load_session(session_id="sess_abc123")

# Resume conversation with context
conversation = await session.resume_conversation()
# AI remembers previous conversations

# Close session
await session.close()
```

### Memory Management

```python
# Access conversation memory
memory = session.get_memory()
print(f"Total memories: {len(memory.entries)}")
print(f"Memory usage: {memory.token_count}/{memory.max_tokens}")

# Add explicit memory
session.add_memory(
    content="User prefers formal language",
    category="preference",
    importance=0.9
)

# Search memories
relevant = session.search_memory("language preference")

# Clear specific memories
session.clear_memory(category="temporary")

# Export memory for backup
memory_export = session.export_memory()
with open("session_memory.json", "w") as f:
    json.dump(memory_export, f)
```

## Multi-Conversation Sessions

### Conversation Continuity

```python
# First conversation
async with session.start_conversation() as conv:
    await conv.send_message("My name is Alice")
    await conv.send_message("I'm interested in quantum physics")

# Later conversation (same session)
async with session.start_conversation() as conv:
    response = await conv.send_message("What's my name and interest?")
    # AI: "Your name is Alice and you're interested in quantum physics"
```

### Conversation Branching

```python
# Main conversation
main_conv = await session.start_conversation(
    conversation_id="main",
    parent_id=None
)

# Branch for specific topic
branch_conv = await session.start_conversation(
    conversation_id="physics_deep_dive",
    parent_id="main"
)

# Return to main conversation
main_conv = await session.resume_conversation("main")
# Context from branch is available but scoped
```

## Session Analytics

### Real-time Analytics

```python
# Get session analytics
analytics = session.get_analytics()

print(f"""
Session Analytics:
- Total conversations: {analytics['conversation_count']}
- Total messages: {analytics['message_count']}
- Average conversation duration: {analytics['avg_duration_seconds']}s
- Topics discussed: {analytics['topics']}
- User satisfaction: {analytics['satisfaction_score']}/5
- Most active hours: {analytics['active_hours']}
""")

# Conversation-specific analytics
conv_analytics = conversation.get_analytics()
print(f"""
Conversation Analytics:
- Duration: {conv_analytics['duration_seconds']}s
- Turn count: {conv_analytics['turn_count']}
- Interruptions: {conv_analytics['interruption_count']}
- Context injections: {conv_analytics['injection_count']}
- Response time: {conv_analytics['avg_response_time_ms']}ms
""")
```

### Analytics Events

```python
# Subscribe to analytics events
session.events.on('analytics.milestone', lambda e: 
    print(f"Milestone reached: {e.milestone}"))

# Track custom metrics
session.track_metric('task_completed', 1)
session.track_metric('user_satisfaction', 4.5)

# Get custom metrics
custom_metrics = session.get_custom_metrics()
```

## User Adaptation

### Learning from Interactions

```python
# Enable adaptation
session.config.enable_adaptation = True

# System automatically learns:
# - Speaking patterns
# - Topic preferences  
# - Interaction style
# - Optimal response length

# Access learned preferences
preferences = session.get_learned_preferences()
print(f"""
Learned Preferences:
- Preferred speaking pace: {preferences['speaking_pace']}
- Average message length: {preferences['avg_message_length']}
- Formal/Casual: {preferences['formality_score']}
- Technical level: {preferences['technical_level']}
""")

# Override specific adaptations
session.set_preference('formality', 'always_formal')
```

### Preference Profiles

```python
# Export preference profile
profile = session.export_preference_profile()

# Apply profile to new session
new_session = voxon.create_session(
    user_id="user_789",
    preference_profile=profile
)

# Merge profiles
combined_profile = SessionProfile.merge([profile1, profile2])
```

## Session Persistence

### Storage Backends

```python
# File-based storage (default)
voxon = Voxon(
    config=VoxonConfig(
        session_storage="file",
        session_storage_path="./sessions"
    )
)

# Database storage
voxon = Voxon(
    config=VoxonConfig(
        session_storage="database",
        session_database_url="postgresql://..."
    )
)

# Redis storage
voxon = Voxon(
    config=VoxonConfig(
        session_storage="redis",
        session_redis_url="redis://..."
    )
)

# Custom storage
class CustomSessionStorage(SessionStorage):
    async def save(self, session_data: Dict) -> None:
        # Custom implementation
        pass
    
    async def load(self, session_id: str) -> Dict:
        # Custom implementation
        pass

voxon = Voxon(
    config=VoxonConfig(
        session_storage=CustomSessionStorage()
    )
)
```

### Session Data Structure

```python
{
    "session_id": "sess_abc123",
    "user_id": "user_123",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T11:30:00Z",
    "config": {
        "instructions": "...",
        "voice": "nova",
        "memory_mode": "persistent"
    },
    "memory": {
        "entries": [
            {
                "content": "User name is Alice",
                "timestamp": "2024-01-15T10:05:00Z",
                "importance": 0.9,
                "category": "identity"
            }
        ],
        "token_count": 2500
    },
    "analytics": {
        "conversation_count": 3,
        "message_count": 45,
        "total_duration_seconds": 1800
    },
    "preferences": {
        "formality": 0.7,
        "verbosity": 0.5,
        "technical_level": 0.8
    }
}
```

## Advanced Session Features

### Session Templates

```python
# Define reusable session template
template = SessionTemplate(
    name="medical_consultation",
    base_config=SessionConfig(
        instructions="You are a medical assistant...",
        memory_mode="persistent",
        enable_analytics=True
    ),
    required_context=["patient_id", "appointment_type"],
    memory_categories=["medical_history", "symptoms", "medications"]
)

# Create session from template
session = voxon.create_session_from_template(
    template=template,
    context={
        "patient_id": "P12345",
        "appointment_type": "follow_up"
    }
)
```

### Session Sharing

```python
# Share session between users (with permissions)
shared_session = session.create_shared_view(
    shared_with_user_id="doctor_456",
    permissions=["read", "add_notes"],
    expire_after_hours=24
)

# Access shared session
shared = await voxon.access_shared_session(
    share_token=shared_session.token,
    user_id="doctor_456"
)
```

### Session Migrations

```python
# Migrate session between environments
export_data = await session.export_full()

# On different system
imported_session = await voxon.import_session(
    export_data,
    user_id_mapping={"old_123": "new_456"}
)
```

## Monitoring and Debugging

### Session Events

```python
# Monitor all session events
session.events.on("*", lambda e: 
    logger.info(f"Session event: {e.type}"))

# Specific session events
session.events.on("memory.added", handle_memory_update)
session.events.on("conversation.started", handle_conversation_start)
session.events.on("preference.learned", handle_preference_update)
```

### Session Debugging

```python
# Enable debug mode
session.enable_debug()

# Get debug information
debug_info = session.get_debug_info()
print(f"""
Debug Information:
- Memory snapshots: {len(debug_info['memory_snapshots'])}
- Event history: {len(debug_info['events'])}
- State transitions: {debug_info['state_transitions']}
- Error count: {debug_info['error_count']}
""")

# Export debug logs
debug_logs = session.export_debug_logs()
```

## Best Practices

### 1. Session Lifecycle Management

```python
# Good: Proper session cleanup
try:
    session = await voxon.create_session(user_id="123")
    conversation = await session.start_conversation()
    # ... use conversation ...
finally:
    await session.save()
    await session.close()

# Better: Use context managers
async with voxon.session(user_id="123") as session:
    async with session.conversation() as conv:
        await conv.send_message("Hello")
```

### 2. Memory Optimization

```python
# Good: Manage memory size
if session.memory_usage > 0.8:  # 80% full
    # Compress old memories
    session.compress_memory(keep_recent_days=7)
    
# Good: Categorize memories
session.add_memory(
    content="Prefers morning appointments",
    category="scheduling",
    expires_after_days=30
)
```

### 3. Privacy and Security

```python
# Good: Sanitize sensitive data
session.add_memory(
    content="User phone is [REDACTED]",
    category="contact",
    sensitive=True
)

# Good: Encryption for storage
voxon = Voxon(
    config=VoxonConfig(
        session_encryption_key=os.getenv("SESSION_KEY"),
        session_encryption_method="AES256"
    )
)
```

### 4. Performance Optimization

```python
# Good: Lazy load session data
session = await voxon.load_session(
    session_id="123",
    lazy_load=True  # Don't load memory until needed
)

# Good: Batch operations
with session.batch_update():
    session.add_memory(...)
    session.update_preference(...)
    session.track_metric(...)
# All updates saved together
```

## Troubleshooting

### Common Issues

1. **Session not persisting**
   - Check storage backend configuration
   - Verify save() is called
   - Check storage permissions

2. **Memory growing too large**
   - Implement memory compression
   - Set appropriate TTLs
   - Use memory categories

3. **Slow session loading**
   - Enable lazy loading
   - Implement caching
   - Optimize storage queries

4. **Analytics not accurate**
   - Ensure events are being tracked
   - Check timezone settings
   - Verify metric calculations