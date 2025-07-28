# VoxEngine Interface Documentation

## Overview

VoxEngine is a high-performance Python framework for OpenAI's Realtime API, providing a clean, intuitive interface for building real-time voice applications. This document details all public APIs, methods, and events available in VoxEngine.

## Table of Contents

1. [Core Components](#core-components)
2. [VoiceEngine API](#voiceengine-api)
3. [Event System](#event-system)
4. [Configuration](#configuration)
5. [Providers](#providers)
6. [Audio Management](#audio-management)
7. [Session Management](#session-management)
8. [Error Handling](#error-handling)
9. [Metrics and Monitoring](#metrics-and-monitoring)

## Core Components

### VoiceEngine

The main entry point for all voice interactions.

```python
from voxengine import VoiceEngine, VoiceEngineConfig

# Create engine with configuration
engine = VoiceEngine(
    api_key="your-api-key",
    mode="fast",  # or "big", "provider"
    # ... other config options
)
```

## VoiceEngine API

### Initialization

#### `VoiceEngine.__init__()`

```python
def __init__(
    self,
    api_key: Optional[str] = None,
    config: Optional[VoiceEngineConfig] = None,
    mode: Literal["fast", "big", "provider"] = "fast",
    **kwargs
) -> None
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key. Can also be set via environment variable `OPENAI_API_KEY`
- `config` (VoiceEngineConfig, optional): Complete configuration object. If provided, other parameters are ignored
- `mode` (str): Engine mode - "fast" for low latency, "big" for features, "provider" for custom providers
- `**kwargs`: Additional configuration options passed to VoiceEngineConfig

**Example:**
```python
# Simple initialization
engine = VoiceEngine(api_key="sk-...")

# With custom config
config = VoiceEngineConfig(
    api_key="sk-...",
    mode="fast",
    voice="alloy",
    sample_rate=24000
)
engine = VoiceEngine(config=config)
```

### Connection Management

#### `async connect(retry_count: int = 3) -> None`

Establishes connection to the OpenAI Realtime API.

**Parameters:**
- `retry_count` (int): Number of connection attempts before failing (default: 3)

**Raises:**
- `EngineError`: If connection fails after all retry attempts

**Events Emitted:**
- `CONNECTION_STARTING`: When connection attempt begins
- `CONNECTION_ESTABLISHED`: When successfully connected
- `CONNECTION_FAILED`: On connection failure

**Example:**
```python
try:
    await engine.connect()
    print("Connected successfully")
except EngineError as e:
    print(f"Connection failed: {e}")
```

#### `async disconnect() -> None`

Gracefully disconnects from the API and cleans up resources.

**Events Emitted:**
- `CONNECTION_CLOSED`: When disconnection is complete

**Example:**
```python
await engine.disconnect()
```

### Audio Control

#### `async start_listening() -> None`

Starts capturing audio from the configured input device.

**Raises:**
- `EngineError`: If not connected

**Events Emitted:**
- `AUDIO_INPUT_STARTED`: When audio capture begins

**Example:**
```python
await engine.start_listening()
```

#### `async stop_listening() -> None`

Stops audio capture.

**Events Emitted:**
- `AUDIO_INPUT_STOPPED`: When audio capture ends

### Communication Methods

#### `async send_audio(audio_data: bytes) -> None`

Sends raw audio data to the AI for processing.

**Parameters:**
- `audio_data` (bytes): Raw PCM audio data (16-bit, mono, 24kHz by default)

**Raises:**
- `EngineError`: If not connected

**Example:**
```python
# Send audio chunk
await engine.send_audio(audio_chunk)
```

#### `async send_text(text: str) -> None`

Sends a text message to the AI.

**Parameters:**
- `text` (str): Text message to send

**Raises:**
- `EngineError`: If not connected

**Events Emitted:**
- `TEXT_INPUT`: When text is sent

**Example:**
```python
await engine.send_text("Hello, how are you?")
```

#### `async send_recorded_audio(audio_data: bytes, auto_respond: bool = True) -> None`

Sends a complete audio recording (e.g., from push-to-talk).

**Parameters:**
- `audio_data` (bytes): Complete audio recording
- `auto_respond` (bool): Whether to automatically trigger AI response (default: True)

**Example:**
```python
# Send complete recording
with open("recording.wav", "rb") as f:
    audio_data = f.read()
await engine.send_recorded_audio(audio_data)
```

#### `async interrupt() -> None`

Interrupts the current AI response.

**Events Emitted:**
- `CONVERSATION_INTERRUPTED`: When interruption occurs

**Example:**
```python
# User starts speaking, interrupt AI
await engine.interrupt()
```

### Convenience Methods

#### `async text_2_audio_response(text: str, timeout: float = 30.0) -> bytes`

Converts text to speech and returns the complete audio.

**Parameters:**
- `text` (str): Text to convert to speech
- `timeout` (float): Maximum time to wait for response (default: 30 seconds)

**Returns:**
- `bytes`: Complete audio data

**Raises:**
- `EngineError`: If timeout or no audio received

**Example:**
```python
audio = await engine.text_2_audio_response("Hello world")
# Play or save the audio
```

### Properties

#### `is_connected -> bool`

Returns True if engine is connected to the API.

```python
if engine.is_connected:
    await engine.send_text("Hello")
```

#### `is_listening -> bool`

Returns True if engine is actively capturing audio.

```python
if not engine.is_listening:
    await engine.start_listening()
```

#### `events -> EventEmitter`

Access to the event system for registering handlers.

```python
engine.events.on(EventType.TEXT_OUTPUT, handle_text)
```

## Event System

VoxEngine uses a comprehensive event-driven architecture. You can handle events using either the modern event system or legacy callbacks.

### Modern Event System

#### Registering Event Handlers

```python
from voxengine import EventType

# Basic handler
engine.events.on(EventType.TEXT_OUTPUT, 
                lambda event: print(f"AI: {event.text}"))

# With priority (higher = earlier execution)
engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, 
                handle_audio, 
                priority=10)

# With filter
engine.events.on(EventType.TEXT_OUTPUT,
                handle_long_text,
                filter=lambda e: len(e.text) > 100)

# One-time handler
engine.events.once(EventType.CONNECTION_ESTABLISHED,
                  lambda e: print("Connected!"))

# Handle all events
engine.events.on("*", lambda e: logger.debug(f"Event: {e.type}"))
```

#### Removing Handlers

```python
# Get handler ID when registering
handler_id = engine.events.on(EventType.TEXT_OUTPUT, handler)

# Remove later
engine.events.off(handler_id)
```

### Event Types

#### Connection Events

- **CONNECTION_STARTING**: Connection attempt initiated
  ```python
  {
    type: EventType.CONNECTION_STARTING,
    timestamp: float,
    source: str
  }
  ```

- **CONNECTION_ESTABLISHED**: Successfully connected
  ```python
  {
    type: EventType.CONNECTION_ESTABLISHED,
    connection_id: Optional[str],
    latency_ms: Optional[float]
  }
  ```

- **CONNECTION_FAILED**: Connection attempt failed
  ```python
  {
    type: EventType.CONNECTION_FAILED,
    error: Exception,
    retry_count: int
  }
  ```

- **CONNECTION_LOST**: Connection dropped unexpectedly
- **CONNECTION_CLOSED**: Connection closed normally

#### Audio Events

- **AUDIO_INPUT_STARTED**: Audio input began
- **AUDIO_INPUT_CHUNK**: Audio input data received
  ```python
  {
    type: EventType.AUDIO_INPUT_CHUNK,
    audio_data: bytes,
    duration_ms: float
  }
  ```
- **AUDIO_INPUT_STOPPED**: Audio input ended

- **AUDIO_OUTPUT_STARTED**: Audio output began
- **AUDIO_OUTPUT_CHUNK**: Audio output data received
  ```python
  {
    type: EventType.AUDIO_OUTPUT_CHUNK,
    audio_data: bytes,
    sample_rate: int,
    channels: int
  }
  ```
- **AUDIO_OUTPUT_STOPPED**: Audio output ended

#### Text Events

- **TEXT_INPUT**: Text message sent
  ```python
  {
    type: EventType.TEXT_INPUT,
    text: str
  }
  ```

- **TEXT_OUTPUT**: Text response received
  ```python
  {
    type: EventType.TEXT_OUTPUT,
    text: str,
    is_partial: bool
  }
  ```

#### Conversation Events

- **CONVERSATION_STARTED**: Conversation began
- **CONVERSATION_TURN_DETECTED**: Speaker turn detected
- **CONVERSATION_INTERRUPTED**: Conversation interrupted
- **CONVERSATION_ENDED**: Conversation ended

#### Response Events

- **RESPONSE_STARTED**: AI response started
- **RESPONSE_COMPLETED**: AI response completed
- **RESPONSE_CANCELLED**: Response cancelled

#### Function Events

- **FUNCTION_CALL_INVOKED**: Function call requested
  ```python
  {
    type: EventType.FUNCTION_CALL_INVOKED,
    function_name: str,
    arguments: Dict[str, Any],
    call_id: Optional[str]
  }
  ```

#### Error Events

- **ERROR_GENERAL**: General error occurred
  ```python
  {
    type: EventType.ERROR_GENERAL,
    error: Exception,
    error_message: str,
    recoverable: bool
  }
  ```

### Legacy Callbacks

For backward compatibility, VoxEngine still supports callback-style event handling:

```python
# Audio response callback
engine.on_audio_response = lambda audio: play_audio(audio)

# Text response callback
engine.on_text_response = lambda text: print(f"AI: {text}")

# Error callback
engine.on_error = lambda error: logger.error(error)

# Function call callback
engine.on_function_call = lambda call: handle_function(call)

# Response done callback
engine.on_response_done = lambda: print("Response complete")
```

## Configuration

### VoiceEngineConfig

Complete configuration for VoiceEngine.

```python
@dataclass
class VoiceEngineConfig:
    # API Configuration
    api_key: str
    provider: str = "openai"
    
    # Mode selection
    mode: Literal["fast", "big", "provider"] = "fast"
    
    # Audio settings
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 24000
    chunk_duration_ms: int = 100
    
    # Features
    vad_enabled: bool = True
    vad_type: Literal["client", "server"] = "client"
    vad_threshold: float = 0.02
    vad_speech_start_ms: int = 100
    vad_speech_end_ms: int = 500
    
    # Voice settings
    voice: str = "alloy"
    language: Optional[str] = None
    
    # Performance
    latency_mode: Literal["ultra_low", "balanced", "quality"] = "balanced"
    
    # Features (for big lane)
    enable_transcription: bool = False
    enable_functions: bool = False
    enable_multi_provider: bool = False
    
    # Advanced
    log_level: str = "INFO"
    save_audio: bool = False
    audio_save_path: Optional[Path] = None
    
    # Additional provider-specific config
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Configuration Examples

```python
# Ultra-low latency configuration
config = VoiceEngineConfig(
    api_key="sk-...",
    mode="fast",
    latency_mode="ultra_low",
    vad_threshold=0.01,
    chunk_duration_ms=50
)

# Feature-rich configuration
config = VoiceEngineConfig(
    api_key="sk-...",
    mode="big",
    enable_transcription=True,
    enable_functions=True,
    voice="nova",
    save_audio=True,
    audio_save_path=Path("./recordings")
)

# Custom provider configuration
config = VoiceEngineConfig(
    api_key="sk-...",
    mode="provider",
    provider="custom",
    metadata={
        "endpoint": "wss://custom.api.com",
        "model": "custom-model-v1"
    }
)
```

## Providers

### Available Providers

- **OpenAI** (default): Official OpenAI Realtime API
- **Mock**: For testing without API calls
- **Custom**: Implement your own provider

### Provider Interface

Custom providers must implement the `BaseProvider` interface:

```python
class BaseProvider(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to provider"""
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        
    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio to provider"""
        
    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text to provider"""
```

## Audio Management

### Audio Configuration

```python
# Configure audio devices
config = VoiceEngineConfig(
    input_device=1,  # Microphone device ID
    output_device=2,  # Speaker device ID
    sample_rate=24000,  # Sample rate in Hz
    chunk_duration_ms=100  # Audio chunk size
)
```

### Voice Activity Detection (VAD)

```python
# Configure VAD
config = VoiceEngineConfig(
    vad_enabled=True,
    vad_type="client",  # Client-side VAD
    vad_threshold=0.02,  # Energy threshold
    vad_speech_start_ms=100,  # Speech start delay
    vad_speech_end_ms=500  # Speech end delay
)
```

## Session Management

### Session Configuration

```python
from voxengine import SessionConfig, SessionPresets

# Use preset
config = SessionPresets.CONVERSATION

# Or custom config
config = SessionConfig(
    instructions="You are a helpful assistant",
    voice="alloy",
    temperature=0.8,
    max_output_tokens=4096
)
```

## Error Handling

### Exception Types

```python
from voxengine import (
    EngineError,        # Base exception
    ConnectionError,    # Connection issues
    AuthenticationError,# Auth failures
    AudioError,        # Audio processing errors
    StreamError        # Stream errors
)
```

### Error Handling Examples

```python
try:
    await engine.connect()
except AuthenticationError:
    print("Invalid API key")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except EngineError as e:
    print(f"Engine error: {e}")
```

## Metrics and Monitoring

### Getting Metrics

```python
# Get engine state
state = engine.get_state()
print(f"Current state: {state}")

# Get event system metrics
metrics = engine.events.get_metrics()
print(f"Events emitted: {metrics['events_emitted']}")
print(f"Events handled: {metrics['events_handled']}")
print(f"Active handlers: {metrics['handler_count']}")
```

### Event History

```python
# Get recent events
history = engine.events.get_history(
    event_type=EventType.TEXT_OUTPUT,
    limit=10
)

for event in history:
    print(f"{event.timestamp}: {event.type}")
```

## Context Manager Support

VoxEngine supports Python's context manager protocol for automatic resource management:

```python
async with engine:
    # Engine is automatically connected
    await engine.start_listening()
    # ... do work ...
# Engine is automatically disconnected
```

## Factory Methods

### `VoiceEngine.create_simple(api_key: str) -> VoiceEngine`

Creates a pre-configured engine with sensible defaults for quick prototyping.

```python
engine = VoiceEngine.create_simple(api_key="sk-...")
# Equivalent to VoiceEngine with default fast mode configuration
```

## Performance Metrics

### `get_metrics() -> Dict[str, Any]`

Returns detailed performance metrics:

```python
metrics = engine.get_metrics()
# Returns:
{
    "audio": {
        "capture_rate": float,  # Audio chunks per second
        "playback_latency": float,  # Playback delay in ms
        "buffer_health": float  # Buffer status 0-1
    },
    "connection": {
        "latency": float,  # WebSocket latency
        "uptime": float,  # Connection uptime in seconds
    },
    "processing": {
        "vad_performance": float,  # VAD processing time
        "audio_processing": float  # Audio pipeline latency
    }
}
```

### `get_usage() -> Usage`

Returns API usage statistics:

```python
usage = await engine.get_usage()
print(f"Audio seconds: {usage.audio_seconds}")
print(f"Text tokens: {usage.text_tokens}")
print(f"Total sessions: {usage.total_sessions}")
```

### `estimate_cost() -> Cost`

Estimates the cost based on current usage:

```python
cost = await engine.estimate_cost()
print(f"Audio cost: ${cost.audio_cost:.2f}")
print(f"Text cost: ${cost.text_cost:.2f}")
print(f"Total cost: ${cost.total:.2f}")
```

## Installation

```bash
pip install voxengine
```

## Requirements

- Python 3.8+
- `sounddevice` for audio I/O
- `websockets` for API connection  
- `numpy` for audio processing
- `python-dotenv` for environment variables (optional)

## Performance Characteristics

### Fast Lane Mode
- **Audio capture to API**: < 10ms
- **API to audio playback**: < 10ms  
- **Total round-trip latency**: < 50ms
- **CPU usage**: < 5%
- **Memory footprint**: < 50MB
- **Connection overhead**: Minimal (single WebSocket)

### Big Lane Mode (Coming Soon)
- **Total latency**: 100-200ms (feature processing overhead)
- **CPU usage**: 10-15% (audio pipeline processing)
- **Memory footprint**: 100-150MB (buffering and features)
- **Additional features**: Transcription, multi-provider, effects

## Complete Examples

```python
import asyncio
from voxengine import VoiceEngine, EventType

async def main():
    engine = VoiceEngine(api_key="sk-...")
    
    # Set up event handlers
    engine.events.on(EventType.AUDIO_OUTPUT_CHUNK,
                    lambda e: play_audio(e.audio_data))
    engine.events.on(EventType.TEXT_OUTPUT,
                    lambda e: print(f"AI: {e.text}"))
    engine.events.on(EventType.ERROR_GENERAL,
                    lambda e: print(f"Error: {e.error_message}"))
    
    # Connect and start
    await engine.connect()
    await engine.start_listening()
    
    print("Voice assistant ready. Press Ctrl+C to stop.")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await engine.disconnect()

asyncio.run(main())
```

### Push-to-Talk Implementation

```python
async def push_to_talk_mode():
    engine = VoiceEngine(api_key="sk-...")
    await engine.connect()
    
    # Record audio while key is pressed
    recording = []
    
    def on_key_press():
        recording.clear()
        asyncio.create_task(engine.start_listening())
    
    def on_key_release():
        asyncio.create_task(handle_recording())
    
    async def handle_recording():
        await engine.stop_listening()
        if recording:
            audio_data = b''.join(recording)
            await engine.send_recorded_audio(audio_data)
    
    # Set up audio capture
    engine.events.on(EventType.AUDIO_INPUT_CHUNK,
                    lambda e: recording.append(e.audio_data))
    
    # Run with keyboard listener
    # ... keyboard handling code ...
```

### Multi-Language Assistant

```python
async def multilingual_assistant():
    config = VoiceEngineConfig(
        api_key="sk-...",
        voice="nova",
        language="es",  # Spanish
        enable_transcription=True
    )
    
    engine = VoiceEngine(config=config)
    
    # Handle transcriptions
    engine.events.on(EventType.TEXT_OUTPUT,
                    lambda e: print(f"Transcription: {e.text}"))
    
    await engine.connect()
    await engine.start_listening()
```

## Best Practices

1. **Always handle errors**: Use try/except blocks around connection and API calls
2. **Clean up resources**: Always call `disconnect()` when done
3. **Use events over callbacks**: The event system is more flexible and powerful
4. **Monitor metrics**: Track event metrics in production
5. **Configure appropriately**: Choose the right mode (fast/big) for your use case
6. **Handle connection loss**: Implement reconnection logic for production apps

## Migration from Callbacks to Events

```python
# Old callback style
engine.on_text_response = lambda text: print(text)

# New event style (recommended)
engine.events.on(EventType.TEXT_OUTPUT, 
                lambda event: print(event.text))

# Both work during migration
```

## Troubleshooting

### Common Issues

1. **No audio output**: Check output device configuration
2. **High latency**: Use "fast" mode and "ultra_low" latency setting
3. **Connection drops**: Implement reconnection with exponential backoff
4. **No events received**: Ensure event handlers are registered before connecting

### Debug Mode

```python
# Enable debug logging
config = VoiceEngineConfig(
    api_key="sk-...",
    log_level="DEBUG"
)

# Monitor all events
engine.events.on("*", lambda e: print(f"[DEBUG] {e.type}: {e.data}"))