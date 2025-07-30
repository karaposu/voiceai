# Provider System Documentation

## Overview

VoxEngine uses a provider abstraction system that allows it to work with different voice AI backends. This enables flexibility in choosing the best provider for your use case and makes it easy to switch providers or implement custom solutions.

## Available Providers

### 1. OpenAI Provider (Default)

The official OpenAI Realtime API provider with full feature support.

```python
config = VoiceEngineConfig(
    api_key="sk-...",
    provider="openai",
    mode="fast"
)
```

**Features:**
- Real-time bidirectional audio streaming
- Server-side and client-side VAD
- Multiple voices (alloy, echo, fable, onyx, nova, shimmer)
- Function calling support
- Low latency mode

**Requirements:**
- OpenAI API key with Realtime API access
- Stable WebSocket connection

### 2. Mock Provider

For testing and development without making actual API calls.

```python
config = VoiceEngineConfig(
    provider="mock",
    mode="fast"
)
```

**Features:**
- Simulates all API responses
- Configurable response delays
- Predefined response patterns
- No API costs
- Offline development

**Use Cases:**
- Unit testing
- UI development
- Demo applications
- API limit situations

### 3. Custom Providers

Implement your own provider for any voice AI service.

```python
config = VoiceEngineConfig(
    provider="custom",
    mode="provider",
    metadata={
        "endpoint": "wss://your-api.com/realtime",
        "auth_type": "bearer",
        "model": "custom-voice-v1"
    }
)
```

## Provider Interface

All providers must implement the `BaseProvider` abstract class:

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseProvider(ABC):
    """Base interface for all voice providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.session_id = None
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to provider"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection cleanly"""
        pass
    
    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to provider"""
        pass
    
    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text message to provider"""
        pass
    
    @abstractmethod
    async def interrupt(self) -> None:
        """Interrupt current response"""
        pass
    
    @abstractmethod
    async def configure_session(self, config: Dict[str, Any]) -> None:
        """Configure provider session"""
        pass
    
    # Optional methods
    async def get_usage(self) -> Optional[Dict[str, Any]]:
        """Get usage statistics if available"""
        return None
    
    async def estimate_cost(self) -> Optional[float]:
        """Estimate cost if available"""
        return None
```

## Implementing a Custom Provider

### Example: Azure Cognitive Services Provider

```python
from voxengine.providers.base import BaseProvider
from voxengine import EventType
import websockets
import json

class AzureProvider(BaseProvider):
    """Azure Cognitive Services Speech provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config['metadata']['endpoint']
        self.api_key = config['api_key']
        self.region = config['metadata'].get('region', 'eastus')
        
    async def connect(self):
        """Connect to Azure Speech Services"""
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'X-Region': self.region
        }
        
        self.connection = await websockets.connect(
            self.endpoint,
            extra_headers=headers
        )
        
        # Send initial configuration
        await self._configure_azure_session()
        
        # Start listening for responses
        self._start_response_handler()
        
        # Emit connection event
        self.emit_event(EventType.CONNECTION_ESTABLISHED, {
            'provider': 'azure',
            'region': self.region
        })
    
    async def send_audio(self, audio_data: bytes):
        """Send audio to Azure"""
        # Azure expects specific audio format
        formatted_audio = self._format_audio_for_azure(audio_data)
        
        message = {
            'type': 'audio',
            'audio': formatted_audio.hex()
        }
        
        await self.connection.send(json.dumps(message))
    
    async def send_text(self, text: str):
        """Send text for TTS"""
        message = {
            'type': 'text',
            'text': text,
            'voice': self.config.get('voice', 'en-US-JennyNeural')
        }
        
        await self.connection.send(json.dumps(message))
        
    def _format_audio_for_azure(self, audio_data: bytes) -> bytes:
        """Convert audio to Azure's expected format"""
        # Implementation depends on Azure's requirements
        return audio_data
```

### Registering a Custom Provider

```python
from voxengine import register_provider
from my_providers import AzureProvider, GoogleProvider

# Register providers
register_provider("azure", AzureProvider)
register_provider("google", GoogleProvider)

# Use custom provider
config = VoiceEngineConfig(
    provider="azure",
    api_key="your-azure-key",
    metadata={
        "endpoint": "wss://speech.azure.com/realtime",
        "region": "westus2"
    }
)
```

## Provider Configuration

### Provider-Specific Metadata

Each provider can have custom configuration in the `metadata` field:

```python
# OpenAI specific
metadata = {
    "model": "gpt-4-turbo",
    "temperature": 0.7,
    "max_tokens": 150
}

# AWS Transcribe specific
metadata = {
    "region": "us-east-1",
    "language_code": "en-US",
    "vocabulary_name": "medical-terms"
}

# Google Cloud Speech specific
metadata = {
    "language_code": "en-US",
    "enable_word_time_offsets": True,
    "model": "latest_long"
}
```

## Provider Events

Providers emit standard events that VoxEngine handles:

```python
class CustomProvider(BaseProvider):
    async def _handle_provider_message(self, message):
        """Handle messages from provider"""
        
        if message['type'] == 'transcription':
            self.emit_event(EventType.TEXT_OUTPUT, {
                'text': message['text'],
                'is_partial': message.get('partial', False)
            })
            
        elif message['type'] == 'audio':
            self.emit_event(EventType.AUDIO_OUTPUT_CHUNK, {
                'audio_data': message['audio'],
                'sample_rate': 24000
            })
            
        elif message['type'] == 'error':
            self.emit_event(EventType.ERROR_GENERAL, {
                'error': Exception(message['error']),
                'recoverable': message.get('recoverable', False)
            })
```

## Provider Selection Strategy

### Mode-Based Selection

VoxEngine selects providers based on the configured mode:

```python
# Fast mode - optimized for latency
if config.mode == "fast":
    # Use lightweight provider features
    # Disable transcription, minimize processing
    
# Big mode - full features
elif config.mode == "big":
    # Enable all provider capabilities
    # Transcription, emotion detection, etc.
    
# Provider mode - custom implementation
elif config.mode == "provider":
    # Use provider-specific optimizations
```

### Dynamic Provider Switching

```python
# Start with one provider
engine = VoiceEngine(config)
await engine.connect()

# Switch providers mid-session
await engine.switch_provider("azure", new_config)
# Maintains conversation context
```

## Provider Comparison

| Feature | OpenAI | Azure | Google | AWS | Custom |
|---------|---------|--------|---------|------|---------|
| Real-time streaming | ✅ | ✅ | ✅ | ✅ | Depends |
| Server VAD | ✅ | ✅ | ❌ | ✅ | Depends |
| Multiple voices | ✅ | ✅ | ✅ | ✅ | Depends |
| Function calling | ✅ | ❌ | ❌ | ❌ | Depends |
| Emotion detection | ❌ | ✅ | ✅ | ❌ | Depends |
| Custom vocabulary | ❌ | ✅ | ✅ | ✅ | Depends |
| Cost per minute | $$$ | $$ | $$ | $ | Varies |

## Best Practices

### 1. Provider Abstraction

```python
# Good: Use provider-agnostic code
await engine.send_text("Hello")
await engine.send_audio(audio_data)

# Bad: Provider-specific code in application
if provider == "openai":
    await engine._send_openai_message(...)
```

### 2. Configuration Management

```python
# Good: Centralized provider configs
PROVIDER_CONFIGS = {
    "openai": {
        "api_key": os.getenv("OPENAI_KEY"),
        "mode": "fast"
    },
    "azure": {
        "api_key": os.getenv("AZURE_KEY"),
        "metadata": {...}
    }
}

# Select based on environment
provider = os.getenv("VOICE_PROVIDER", "openai")
config = PROVIDER_CONFIGS[provider]
```

### 3. Error Handling

```python
# Handle provider-specific errors
try:
    await engine.connect()
except ProviderError as e:
    if e.provider == "openai" and e.code == "rate_limit":
        # Switch to backup provider
        await engine.switch_provider("azure")
    else:
        raise
```

### 4. Testing with Mock Provider

```python
# Development/testing configuration
if os.getenv("ENVIRONMENT") == "test":
    config = VoiceEngineConfig(provider="mock")
else:
    config = VoiceEngineConfig(provider="openai")

# Mock provider responses
mock_responses = {
    "Hello": "Hi there! How can I help?",
    "What's the weather?": "It's sunny today!"
}
```

## Provider Development Guide

### Step 1: Implement BaseProvider

```python
class MyProvider(BaseProvider):
    # Implement all abstract methods
    pass
```

### Step 2: Handle Connection Lifecycle

```python
async def connect(self):
    # 1. Establish connection
    # 2. Authenticate
    # 3. Configure session
    # 4. Start event handlers
    # 5. Emit CONNECTION_ESTABLISHED
```

### Step 3: Implement Audio Pipeline

```python
async def send_audio(self, audio_data: bytes):
    # 1. Validate audio format
    # 2. Apply any required transformations
    # 3. Send to provider
    # 4. Handle backpressure
```

### Step 4: Event Translation

```python
# Translate provider events to VoxEngine events
provider_event = await self.receive_message()
voxengine_event = self.translate_event(provider_event)
self.emit_event(voxengine_event.type, voxengine_event.data)
```

### Step 5: Testing

```python
# Test provider implementation
async def test_provider():
    provider = MyProvider(test_config)
    
    # Test connection
    await provider.connect()
    assert provider.is_connected
    
    # Test audio
    await provider.send_audio(test_audio)
    
    # Test events
    events = []
    provider.on_event = events.append
    await provider.send_text("Test")
    
    assert any(e.type == EventType.TEXT_OUTPUT for e in events)
```

## Future Providers

Planned provider implementations:

1. **Anthropic Claude Voice** - When available
2. **ElevenLabs** - High-quality voice synthesis
3. **Deepgram** - Real-time transcription focus
4. **AssemblyAI** - Advanced speech analytics
5. **Local Whisper** - On-device processing

## Provider Migration

### Migrating from OpenAI to Azure

```python
# Step 1: Update configuration
old_config = VoiceEngineConfig(
    provider="openai",
    api_key="sk-..."
)

new_config = VoiceEngineConfig(
    provider="azure",
    api_key="azure-key",
    metadata={
        "region": "eastus",
        "voice": "en-US-JennyNeural"
    }
)

# Step 2: Map voice names
voice_mapping = {
    "alloy": "en-US-JennyNeural",
    "nova": "en-US-AriaNeural",
    "echo": "en-US-GuyNeural"
}

# Step 3: Update application code (if needed)
# Most code should work unchanged due to abstraction
```