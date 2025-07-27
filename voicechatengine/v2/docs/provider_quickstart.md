# Provider Interface Quick Start

## Installation

The provider interface is part of VoiceEngine V2. Import the necessary components:

```python
from voicechatengine.v2.providers import (
    MockProvider, MockConfig,
    OpenAIProvider, OpenAIConfig
)
```

## Basic Usage

### 1. Mock Provider (For Testing)

```python
import asyncio

async def test_mock_provider():
    # Configure mock provider
    config = MockConfig(
        api_key="test-key",
        simulate_latency=True,
        response_delay=0.5
    )
    
    # Create and connect
    provider = MockProvider(config)
    connected = await provider.connect()
    print(f"Connected: {connected}")
    
    # Send a message
    await provider.send_text("Hello, AI!", role="user")
    
    # Handle events
    event_count = 0
    async for event in provider.events():
        print(f"Event: {event.type}")
        event_count += 1
        if event_count > 5:
            break
    
    # Disconnect
    await provider.disconnect()

# Run
asyncio.run(test_mock_provider())
```

### 2. OpenAI Provider (Production)

```python
import asyncio
import os

async def test_openai_provider():
    # Configure OpenAI provider
    config = OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-realtime-preview",
        voice="nova"
    )
    
    # Create and connect
    provider = OpenAIProvider(config)
    if await provider.connect():
        print(f"Connected to OpenAI: {provider.session_id}")
        
        # Update session settings
        await provider.update_session(
            temperature=0.7,
            instructions="You are a helpful assistant."
        )
        
        # Send text and get response
        await provider.send_text("What's the weather like?")
        await provider.create_response(modalities=["text", "audio"])
        
        # Process response events
        async for event in provider.events():
            if event.type == "text_chunk":
                print(event.data["text"], end="", flush=True)
            elif event.type == "response.completed":
                print("\n[Response complete]")
                break
        
        await provider.disconnect()

# Run (requires OPENAI_API_KEY environment variable)
asyncio.run(test_openai_provider())
```

## Advanced Examples

### Event Handlers

```python
async def setup_event_handlers(provider):
    # Define handlers
    async def on_audio(event):
        audio_hex = event.data.get("audio", "")
        audio_bytes = bytes.fromhex(audio_hex)
        print(f"Received {len(audio_bytes)} bytes of audio")
    
    def on_error(event):
        print(f"Error: {event.error}")
    
    # Register handlers
    provider.on_event("audio_chunk", on_audio)
    provider.on_event("error", on_error)
```

### Streaming Audio

```python
async def stream_audio_to_provider(provider, audio_source):
    """Stream audio chunks to provider"""
    chunk_size = 1024
    
    while True:
        chunk = await audio_source.read(chunk_size)
        if not chunk:
            break
            
        await provider.send_audio(chunk)
        
    # Commit audio buffer (if supported)
    await provider.send_message(
        MessageType.AUDIO_BUFFER_COMMIT,
        {}
    )
```

### Error Handling

```python
async def robust_connection(provider, max_retries=3):
    """Connect with retry logic"""
    for attempt in range(max_retries):
        try:
            if await provider.connect():
                return True
        except ConnectionError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return False
```

### Custom Provider Implementation

```python
from voicechatengine.v2.providers.base import BaseProvider, ProviderConfig

class CustomProvider(BaseProvider):
    async def connect(self) -> bool:
        # Your connection logic
        self.state = ConnectionState.CONNECTED
        self.session_id = "custom_session_123"
        return True
    
    async def send_text(self, text: str, role: str = "user") -> None:
        # Your text handling logic
        await self.send_message(
            MessageType.CONVERSATION_ITEM_CREATE,
            {"text": text, "role": role}
        )
    
    # Implement other required methods...
```

## Testing Your Implementation

### Unit Test with Mock Provider

```python
async def test_my_voice_engine():
    # Use mock provider for testing
    provider = MockProvider(MockConfig(api_key="test"))
    await provider.connect()
    
    # Your engine logic here
    engine = VoiceEngine(provider)
    await engine.process_user_input("Hello")
    
    # Verify interactions
    history = provider.get_message_history()
    assert len(history) > 0
    
    await provider.disconnect()
```

### Integration Test

```python
# Run smoke tests
python voicechatengine/v2/smoke_tests/test_provider_interface.py
```

## Common Patterns

### 1. Provider Factory

```python
def create_provider(provider_type: str, **kwargs):
    if provider_type == "openai":
        return OpenAIProvider(OpenAIConfig(**kwargs))
    elif provider_type == "mock":
        return MockProvider(MockConfig(**kwargs))
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
```

### 2. Event Processing Pipeline

```python
async def process_provider_events(provider):
    async for event in provider.events():
        # Route events to appropriate handlers
        if event.type.startswith("audio"):
            await handle_audio_event(event)
        elif event.type.startswith("text"):
            await handle_text_event(event)
        elif event.type == "error":
            await handle_error(event)
```

### 3. Context Manager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def provider_session(provider):
    try:
        await provider.connect()
        yield provider
    finally:
        await provider.disconnect()

# Usage
async with provider_session(provider) as p:
    await p.send_text("Hello!")
```

## Next Steps

1. Run the smoke tests to verify your setup
2. Experiment with the MockProvider for development
3. Implement your own provider for custom protocols
4. Integrate with VoiceEngine for full functionality

For more details, see the [full documentation](provider_interface.md).