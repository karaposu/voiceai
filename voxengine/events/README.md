# Voice Engine Event System

The Voice Engine now features a comprehensive event-driven architecture that provides flexible, type-safe event handling for all voice interactions.

## Overview

The event system replaces the previous callback-based approach with a modern event emitter pattern, while maintaining backward compatibility with existing callbacks.

## Key Features

- **Type-safe events**: Strongly typed event classes for different event types
- **Flexible handling**: Support for both async and sync event handlers
- **Priority system**: Handlers can have priorities for execution order
- **Event filtering**: Filter events based on custom criteria
- **Global handlers**: Handle all events with wildcard subscriptions
- **Metrics**: Built-in metrics for monitoring event flow
- **Thread-safe**: Handles events from different threads/contexts
- **Backward compatible**: Legacy callbacks still work alongside events

## Event Types

### Connection Events
- `CONNECTION_STARTING`: Connection attempt initiated
- `CONNECTION_ESTABLISHED`: Successfully connected
- `CONNECTION_FAILED`: Connection attempt failed
- `CONNECTION_LOST`: Connection dropped unexpectedly
- `CONNECTION_CLOSED`: Connection closed normally

### Audio Events
- `AUDIO_INPUT_STARTED`: Audio input began
- `AUDIO_INPUT_CHUNK`: Audio input data received
- `AUDIO_INPUT_STOPPED`: Audio input ended
- `AUDIO_OUTPUT_STARTED`: Audio output began
- `AUDIO_OUTPUT_CHUNK`: Audio output data received
- `AUDIO_OUTPUT_STOPPED`: Audio output ended

### Text Events
- `TEXT_INPUT`: Text message sent
- `TEXT_OUTPUT`: Text response received
- `TEXT_STREAM_STARTED`: Text streaming began
- `TEXT_STREAM_CHUNK`: Partial text received
- `TEXT_STREAM_COMPLETED`: Text streaming completed

### Conversation Events
- `CONVERSATION_STARTED`: Conversation began
- `CONVERSATION_TURN_DETECTED`: Speaker turn detected
- `CONVERSATION_INTERRUPTED`: Conversation interrupted
- `CONVERSATION_ENDED`: Conversation ended

### Other Events
- `RESPONSE_STARTED`: AI response started
- `RESPONSE_COMPLETED`: AI response completed
- `FUNCTION_CALL_INVOKED`: Function call requested
- `ERROR_GENERAL`: General error occurred
- `VAD_SPEECH_STARTED`: Voice activity detected
- `VAD_SPEECH_ENDED`: Voice activity ended

## Usage Examples

### Basic Event Handling

```python
from voxengine import VoiceEngine, EventType

engine = VoiceEngine(api_key="...")

# Handle text output
engine.events.on(EventType.TEXT_OUTPUT, 
                lambda event: print(f"AI: {event.text}"))

# Handle audio output
engine.events.on(EventType.AUDIO_OUTPUT_CHUNK,
                lambda event: play_audio(event.audio_data))

# Handle errors
engine.events.on(EventType.ERROR_GENERAL,
                lambda event: logger.error(f"Error: {event.error_message}"))
```

### Using Both Events and Callbacks

```python
# Modern event-based approach
engine.events.on(EventType.TEXT_OUTPUT, handle_text_event)

# Legacy callback approach (still supported)
engine.on_text_response = handle_text_callback

# Both will be called when text is received
```

### Advanced Features

```python
# Priority handlers (higher priority = earlier execution)
engine.events.on(EventType.TEXT_OUTPUT, urgent_handler, priority=10)
engine.events.on(EventType.TEXT_OUTPUT, normal_handler, priority=5)

# Filtered events
def only_long_texts(event):
    return len(event.text) > 100

engine.events.on(EventType.TEXT_OUTPUT, 
                handle_long_text, 
                filter=only_long_texts)

# One-time handlers
engine.events.once(EventType.CONNECTION_ESTABLISHED,
                  lambda e: print("Connected!"))

# Global handler for all events
engine.events.on("*", lambda event: logger.debug(f"Event: {event.type}"))

# Remove handlers
handler_id = engine.events.on(EventType.TEXT_OUTPUT, handler)
engine.events.off(handler_id)
```

### Async Handlers

```python
async def async_handler(event):
    await process_event(event)
    await save_to_database(event)

engine.events.on(EventType.TEXT_OUTPUT, async_handler)
```

### Event Metrics

```python
# Get event system metrics
metrics = engine.events.get_metrics()
print(f"Events emitted: {metrics['events_emitted']}")
print(f"Events handled: {metrics['events_handled']}")
print(f"Active handlers: {metrics['handler_count']}")
```

## Implementation Details

The event system consists of:

1. **Event Types** (`event_types.py`): Defines all event types and event classes
2. **Event Emitter** (`event_emitter.py`): Core event emitter implementation
3. **Integration**: Voice Engine emits events for all significant operations

The system is designed to handle events from different threads and contexts, automatically queuing events when no event loop is available and processing them when one becomes available.

## Migration Guide

To migrate from callbacks to events:

```python
# Old way
engine.on_text_response = lambda text: print(text)

# New way
engine.events.on(EventType.TEXT_OUTPUT, 
                lambda event: print(event.text))
```

Both approaches work and can be used together during migration.