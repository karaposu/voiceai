# Event System Documentation

## Overview

The VoiceAI system uses a comprehensive event-driven architecture that enables real-time communication between components. This document covers the complete event system used across all modules.

## Event Systems by Module

### VoxEngine Events

VoxEngine provides the most comprehensive event system using the modern EventType enum.

#### Event Registration

```python
from voxengine import EventType

# Basic registration
engine.events.on(EventType.TEXT_OUTPUT, handler)

# With priority (higher priority = earlier execution)
engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, handler, priority=10)

# With filter
engine.events.on(EventType.TEXT_OUTPUT, handler, 
                filter=lambda e: len(e.text) > 100)

# One-time handler
engine.events.once(EventType.CONNECTION_ESTABLISHED, handler)

# Wildcard handler (receives all events)
engine.events.on("*", universal_handler)

# Remove handler
handler_id = engine.events.on(EventType.TEXT_OUTPUT, handler)
engine.events.off(handler_id)
```

#### Complete Event Type Reference

##### Connection Events
| Event | Description | Data |
|-------|-------------|------|
| `CONNECTION_STARTING` | Connection attempt initiated | `{timestamp, source}` |
| `CONNECTION_ESTABLISHED` | Successfully connected | `{connection_id, latency_ms}` |
| `CONNECTION_FAILED` | Connection attempt failed | `{error, retry_count}` |
| `CONNECTION_LOST` | Connection dropped | `{reason, last_activity}` |
| `CONNECTION_CLOSED` | Normal disconnection | `{duration_seconds}` |

##### Audio Events
| Event | Description | Data |
|-------|-------------|------|
| `AUDIO_INPUT_STARTED` | Microphone activated | `{device_id, sample_rate}` |
| `AUDIO_INPUT_CHUNK` | Audio data captured | `{audio_data, duration_ms}` |
| `AUDIO_INPUT_STOPPED` | Microphone deactivated | `{total_duration_ms}` |
| `AUDIO_OUTPUT_STARTED` | Speaker activated | `{device_id}` |
| `AUDIO_OUTPUT_CHUNK` | Audio data for playback | `{audio_data, sample_rate, channels}` |
| `AUDIO_OUTPUT_STOPPED` | Speaker deactivated | `{reason}` |

##### Text Events
| Event | Description | Data |
|-------|-------------|------|
| `TEXT_INPUT` | Text sent to AI | `{text, timestamp}` |
| `TEXT_OUTPUT` | Text from AI | `{text, is_partial, message_id}` |

##### Conversation Events
| Event | Description | Data |
|-------|-------------|------|
| `CONVERSATION_STARTED` | New conversation | `{conversation_id, mode}` |
| `CONVERSATION_TURN_DETECTED` | Speaker change | `{from_role, to_role, timestamp}` |
| `CONVERSATION_INTERRUPTED` | User interrupted | `{interruption_point}` |
| `CONVERSATION_ENDED` | Conversation complete | `{duration, message_count}` |
| `CONVERSATION_UPDATED` | State changed | `{state, changes}` |

##### Response Events
| Event | Description | Data |
|-------|-------------|------|
| `RESPONSE_STARTED` | AI began response | `{response_id, trigger}` |
| `RESPONSE_COMPLETED` | AI finished | `{response_id, duration_ms}` |
| `RESPONSE_CANCELLED` | Response aborted | `{reason, partial_text}` |

##### Function Events
| Event | Description | Data |
|-------|-------------|------|
| `FUNCTION_CALL_INVOKED` | Function requested | `{function_name, arguments, call_id}` |
| `FUNCTION_CALL_COMPLETED` | Function executed | `{call_id, result}` |
| `FUNCTION_CALL_FAILED` | Function error | `{call_id, error}` |

##### Error Events
| Event | Description | Data |
|-------|-------------|------|
| `ERROR_GENERAL` | General error | `{error, error_message, recoverable, context}` |
| `ERROR_AUDIO` | Audio error | `{error_type, device_info}` |
| `ERROR_NETWORK` | Network error | `{error_type, retry_available}` |

### Voxon Events

Voxon emits orchestration-level events:

```python
# Event registration
coordinator.events.on('injection.completed', handler)
coordinator.events.on('mode.changed', handler)
coordinator.events.on('state.enhanced', handler)
```

#### Voxon Event Types

| Event | Description | Data |
|-------|-------------|------|
| `injection.started` | Context injection began | `{context_id, timing}` |
| `injection.completed` | Context injected | `{context_id, success}` |
| `injection.failed` | Injection failed | `{context_id, reason}` |
| `mode.changed` | VAD/response mode changed | `{old_mode, new_mode}` |
| `state.enhanced` | State enrichment complete | `{enhancements}` |
| `engines.synchronized` | Engines in sync | `{sync_time_ms}` |

### ContextWeaver Events

ContextWeaver emits detection and learning events:

```python
# Internal events (for monitoring)
weaver.events.on('detection.cycle', handler)
weaver.events.on('learning.update', handler)
```

#### ContextWeaver Event Types

| Event | Description | Data |
|-------|-------------|------|
| `detection.started` | Detection cycle began | `{detector_count}` |
| `detection.completed` | Detection finished | `{duration_ms, results}` |
| `context.selected` | Context chosen | `{context_id, score}` |
| `learning.update` | Learning occurred | `{pattern_type, success}` |
| `strategy.decision` | Strategy decided | `{decision, factors}` |

## Event Flow Patterns

### Typical Conversation Flow

```
1. CONNECTION_ESTABLISHED (VoxEngine)
   ↓
2. CONVERSATION_STARTED (VoxEngine)
   ↓
3. AUDIO_INPUT_STARTED (VoxEngine)
   ↓
4. AUDIO_INPUT_CHUNK × many (VoxEngine)
   ↓
5. detection.cycle × many (ContextWeaver)
   ↓
6. CONVERSATION_TURN_DETECTED (VoxEngine)
   ↓
7. injection.started (Voxon)
   ↓
8. TEXT_INPUT (VoxEngine) [context injection]
   ↓
9. RESPONSE_STARTED (VoxEngine)
   ↓
10. TEXT_OUTPUT (VoxEngine)
    ↓
11. AUDIO_OUTPUT_CHUNK × many (VoxEngine)
    ↓
12. RESPONSE_COMPLETED (VoxEngine)
```

### Context Injection Flow

```
1. state.enhanced (Voxon)
   ↓
2. detection.started (ContextWeaver)
   ↓
3. detection.completed (ContextWeaver)
   ↓
4. context.selected (ContextWeaver)
   ↓
5. injection.started (Voxon)
   ↓
6. TEXT_INPUT (VoxEngine)
   ↓
7. injection.completed (Voxon)
```

## Advanced Event Handling

### Event Metrics

```python
# Get event system metrics
metrics = engine.events.get_metrics()
print(f"Total events emitted: {metrics['events_emitted']}")
print(f"Total events handled: {metrics['events_handled']}")
print(f"Active handlers: {metrics['handler_count']}")
print(f"Average handling time: {metrics['avg_handling_time_ms']}ms")
```

### Event History

```python
# Get recent events of specific type
history = engine.events.get_history(
    event_type=EventType.TEXT_OUTPUT,
    limit=10
)

for event in history:
    print(f"{event.timestamp}: {event.type} - {event.data}")

# Get all events in time range
from datetime import datetime, timedelta
start_time = datetime.now() - timedelta(minutes=5)
events = engine.events.get_history(
    start_time=start_time,
    end_time=datetime.now()
)
```

### Event Filtering and Transformation

```python
# Complex event filtering
def complex_filter(event):
    return (event.type == EventType.TEXT_OUTPUT and 
            len(event.text) > 100 and 
            'error' not in event.text.lower())

engine.events.on(EventType.TEXT_OUTPUT, handler, filter=complex_filter)

# Event transformation
def transform_audio_event(event):
    # Add computed properties
    event.duration_seconds = event.duration_ms / 1000
    event.data_size_kb = len(event.audio_data) / 1024
    return event

engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, 
                handler, 
                transform=transform_audio_event)
```

### Event Batching

```python
# Batch events for processing
batch = []

def batch_handler(event):
    batch.append(event)
    if len(batch) >= 10:
        process_batch(batch)
        batch.clear()

engine.events.on(EventType.AUDIO_INPUT_CHUNK, batch_handler)
```

## Best Practices

### 1. Event Handler Design

```python
# Good: Async handlers for I/O operations
async def async_handler(event):
    await database.log_event(event)
    await notify_service.send(event)

engine.events.on(EventType.TEXT_OUTPUT, async_handler)

# Good: Fast synchronous handlers for simple operations
def sync_handler(event):
    metrics.increment('text_outputs')
    logger.debug(f"Text: {event.text[:50]}...")

engine.events.on(EventType.TEXT_OUTPUT, sync_handler)
```

### 2. Error Handling in Events

```python
def safe_handler(event):
    try:
        process_event(event)
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        # Don't re-raise - let other handlers continue

engine.events.on(EventType.TEXT_OUTPUT, safe_handler)
```

### 3. Event Priority Management

```python
# Critical handlers run first
engine.events.on(EventType.ERROR_GENERAL, error_handler, priority=100)
engine.events.on(EventType.CONNECTION_LOST, reconnect_handler, priority=90)

# Logging runs last
engine.events.on("*", log_handler, priority=1)
```

### 4. Resource Cleanup

```python
# Always remove handlers when done
handler_ids = []

def setup_handlers():
    handler_ids.append(
        engine.events.on(EventType.TEXT_OUTPUT, handler1)
    )
    handler_ids.append(
        engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, handler2)
    )

def cleanup_handlers():
    for handler_id in handler_ids:
        engine.events.off(handler_id)
    handler_ids.clear()
```

## Migration Guide

### From Callbacks to Events

```python
# Old callback style (deprecated)
engine.on_text_response = lambda text: print(text)
engine.on_audio_response = lambda audio: play(audio)
engine.on_error = lambda err: logger.error(err)

# New event style (recommended)
engine.events.on(EventType.TEXT_OUTPUT, 
                lambda e: print(e.text))
engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, 
                lambda e: play(e.audio_data))
engine.events.on(EventType.ERROR_GENERAL, 
                lambda e: logger.error(e.error_message))
```

### Advantages of Event System

1. **Multiple handlers** per event type
2. **Priority control** for handler execution order
3. **Filtering** to reduce handler calls
4. **One-time handlers** for setup tasks
5. **Wildcard handlers** for monitoring
6. **Event history** for debugging
7. **Metrics** for performance monitoring
8. **Type safety** with EventType enum

## Debugging Events

### Enable Event Logging

```python
import logging

# Log all events
logging.getLogger('voxengine.events').setLevel(logging.DEBUG)
logging.getLogger('voxon.events').setLevel(logging.DEBUG)
logging.getLogger('contextweaver.events').setLevel(logging.DEBUG)

# Or use wildcard handler
def debug_handler(event):
    print(f"[{event.timestamp}] {event.type}: {event.data}")

engine.events.on("*", debug_handler)
```

### Event Inspector

```python
# Inspect event system state
state = engine.events.get_state()
print(f"Handlers by event type:")
for event_type, handlers in state['handlers'].items():
    print(f"  {event_type}: {len(handlers)} handlers")

print(f"\nEvent queue size: {state['queue_size']}")
print(f"Processing: {state['is_processing']}")
```

## Performance Considerations

1. **Handler execution time**: Keep handlers fast (<10ms)
2. **Async vs sync**: Use async for I/O, sync for computation
3. **Event filtering**: Filter early to reduce handler calls
4. **Memory usage**: Clear event history periodically
5. **Handler count**: Remove unused handlers

```python
# Monitor handler performance
metrics = engine.events.get_handler_metrics()
for handler_id, stats in metrics.items():
    if stats['avg_execution_time_ms'] > 10:
        logger.warning(f"Slow handler {handler_id}: {stats['avg_execution_time_ms']}ms")
```