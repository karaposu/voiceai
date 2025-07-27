# Provider Interface Documentation

## Overview

The Provider Interface is a core abstraction in VoiceEngine V2 that enables support for multiple AI voice providers (OpenAI, Anthropic, etc.) through a unified interface. This design allows VoiceEngine to work with any provider without changing the core engine logic.

## Architecture

```
┌─────────────────┐
│   VoiceEngine   │
└────────┬────────┘
         │
    ┌────▼────┐
    │BaseProvider│ (Abstract Interface)
    └────┬────┘
         │
    ┌────┴─────────────┬─────────────┬──────────────┐
    │                  │             │              │
┌───▼────┐     ┌──────▼──────┐ ┌────▼────┐  ┌─────▼─────┐
│OpenAI  │     │  Anthropic  │ │  Mock   │  │  Custom   │
│Provider│     │  Provider   │ │Provider │  │ Provider  │
└────────┘     └─────────────┘ └─────────┘  └───────────┘
```

## BaseProvider Interface

The `BaseProvider` abstract class defines the contract that all providers must implement:

### Core Methods

#### Connection Management

```python
async def connect(self) -> bool:
    """Establish connection to the provider."""
    
async def disconnect(self) -> None:
    """Disconnect from the provider."""
    
def is_connected(self) -> bool:
    """Check if provider is connected."""
```

#### Message Handling

```python
async def send_message(self, message_type: MessageType, data: Dict[str, Any]) -> None:
    """Send a generic message to the provider."""
    
async def send_audio(self, audio_data: bytes) -> None:
    """Send audio data to the provider."""
    
async def send_text(self, text: str, role: str = "user") -> None:
    """Send text message to the provider."""
```

#### Response Control

```python
async def create_response(self, modalities: List[str] = None) -> None:
    """Request a response from the AI."""
    
async def interrupt(self) -> None:
    """Interrupt the current response."""
```

#### Configuration

```python
async def update_session(self, **kwargs) -> None:
    """Update session configuration."""
    
def get_audio_format(self) -> AudioFormat:
    """Get the audio format required by this provider."""
    
def get_capabilities(self) -> Dict[str, Any]:
    """Get provider capabilities."""
```

#### Event Handling

```python
async def events(self) -> AsyncIterator[ProviderEvent]:
    """Async iterator for provider events."""
    
def on_event(self, event_type: str, handler: Callable) -> None:
    """Register an event handler."""
```

## Standard Types

### MessageType Enum

Defines standard message types across all providers:

- `SESSION_CREATE` - Create a new session
- `SESSION_UPDATE` - Update session parameters
- `CONVERSATION_ITEM_CREATE` - Add conversation item
- `AUDIO_BUFFER_APPEND` - Append audio to buffer
- `RESPONSE_CREATE` - Request AI response
- `RESPONSE_CANCEL` - Cancel current response
- `AUDIO_CHUNK` - Audio data chunk event
- `TEXT_CHUNK` - Text data chunk event

### ConnectionState Enum

- `DISCONNECTED` - Not connected
- `CONNECTING` - Connection in progress
- `CONNECTED` - Successfully connected
- `RECONNECTING` - Attempting to reconnect
- `ERROR` - Connection error state

### ProviderEvent

Standard event structure:

```python
@dataclass
class ProviderEvent:
    type: str                    # Event type
    data: Dict[str, Any]        # Event data
    timestamp: float            # Unix timestamp
    provider: str               # Provider name
    session_id: Optional[str]   # Session identifier
    error: Optional[str]        # Error message if any
```

### AudioFormat

Audio format specification:

```python
@dataclass
class AudioFormat:
    encoding: str = "pcm16"     # Audio encoding
    sample_rate: int = 24000    # Sample rate in Hz
    channels: int = 1           # Number of channels
    frame_size: Optional[int]   # Frame size in bytes
```

## Implementing a Provider

To implement a new provider:

1. **Create a Config Class**
```python
class MyProviderConfig(ProviderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add provider-specific config
```

2. **Implement the Provider**
```python
class MyProvider(BaseProvider):
    async def connect(self) -> bool:
        # Establish connection
        # Set self.state = ConnectionState.CONNECTED
        # Generate self.session_id
        return True
        
    # Implement all abstract methods...
```

3. **Handle Protocol Translation**
- Map generic MessageType to provider-specific format
- Convert provider events to standard ProviderEvent
- Handle provider-specific audio formats

## Provider Examples

### MockProvider

A testing provider that simulates AI behavior:
- No network calls
- Configurable latency and error rates
- Records all interactions
- Generates mock responses

### OpenAIProvider

Real implementation for OpenAI's Realtime API:
- WebSocket connection management
- Protocol translation (e.g., `session.create` → `session.created`)
- Audio format: PCM16 @ 24kHz
- Supports text and audio modalities

## Usage Example

```python
# Create provider
config = OpenAIConfig(api_key="your-key")
provider = OpenAIProvider(config)

# Connect
await provider.connect()

# Send messages
await provider.send_text("Hello, AI!")
await provider.send_audio(audio_bytes)

# Handle events
async for event in provider.events():
    if event.type == "text_chunk":
        print(event.data["text"])
    elif event.type == "audio_chunk":
        # Process audio
        pass

# Disconnect
await provider.disconnect()
```

## Testing Providers

Use the provided smoke tests:

```bash
# Test mock provider
python voicechatengine/v2/smoke_tests/test_provider_interface.py

# Test OpenAI provider (requires API key)
OPENAI_API_KEY=your-key python voicechatengine/v2/smoke_tests/test_openai_provider.py
```

## Error Handling

Providers define specific error types:

- `ConnectionError` - Connection-related issues
- `MessageError` - Message sending failures
- `AudioError` - Audio processing errors

Always handle these appropriately:

```python
try:
    await provider.connect()
except ConnectionError as e:
    # Handle connection failure
    pass
```

## Best Practices

1. **State Management**: Always track connection state
2. **Event Handling**: Use async iterators for events
3. **Error Recovery**: Implement reconnection logic
4. **Resource Cleanup**: Always disconnect properly
5. **Protocol Translation**: Keep provider-specific logic isolated
6. **Testing**: Use MockProvider for unit tests

## Future Extensions

The provider interface is designed to support:
- Function calling
- Tool use
- Multi-modal inputs/outputs
- Custom protocols
- Advanced features (VAD, turn detection, etc.)