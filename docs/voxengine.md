# VoxEngine Documentation

## What is this for?

VoxEngine is the low-level voice I/O engine that handles real-time audio streaming, voice activity detection (VAD), and communication with voice AI providers (primarily OpenAI's Realtime API). It provides the foundational layer for voice conversations, managing the technical complexities of audio processing, WebRTC connections, and real-time streaming.

## What it requires

### Dependencies
- **Python 3.8+**
- **Core Libraries:**
  - `asyncio` - Asynchronous operations
  - `pyaudio` - Audio I/O (optional, for local audio)
  - `websockets` - WebSocket communication
  - `aiohttp` - HTTP client for API calls
  - `numpy` - Audio processing
  - `pydantic` - Data validation

### API Requirements
- **OpenAI API Key** - Required for OpenAI provider
- **Network connectivity** - For WebSocket connections
- **Audio permissions** - For microphone/speaker access (if using local audio)

### Configuration
```python
@dataclass
class VoiceEngineConfig:
    # API Configuration
    api_key: str                     # Required API key
    provider: str = "openai"         # Voice provider
    
    # Mode selection
    mode: Literal["fast", "big", "provider"] = "fast"
    
    # Audio settings
    sample_rate: int = 24000         # Audio sample rate
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    chunk_duration_ms: int = 100
    
    # Voice settings
    voice: str = "alloy"             # Voice selection
    language: Optional[str] = None   # Language code
    
    # VAD settings
    vad_type: Literal["client", "server"] = "client"
    vad_enabled: bool = True
    vad_threshold: float = 0.02
    vad_speech_start_ms: int = 100
    vad_speech_end_ms: int = 500
    
    # Response control
    response_mode: Literal["automatic", "manual"] = "automatic"
    
    # Performance
    latency_mode: Literal["ultra_low", "balanced", "quality"] = "balanced"
    
    # Big lane features
    enable_transcription: bool = False
    enable_functions: bool = False
    enable_multi_provider: bool = False
    
    # Advanced
    log_level: str = "INFO"
    save_audio: bool = False
    audio_save_path: Optional[Path] = None
    
    # Provider-specific configuration
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Configuration Examples

```python
# Ultra-low latency voice assistant
config = VoiceEngineConfig(
    api_key="sk-...",
    mode="fast",
    latency_mode="ultra_low",
    vad_type="server",
    response_mode="automatic"
)

# High-quality transcription mode
config = VoiceEngineConfig(
    api_key="sk-...",
    mode="big",
    enable_transcription=True,
    latency_mode="quality",
    voice="nova"
)

# Manual response control for complex interactions
config = VoiceEngineConfig(
    api_key="sk-...",
    vad_type="client",
    response_mode="manual",
    vad_speech_end_ms=1000  # Longer silence detection
)

# Custom provider configuration
config = VoiceEngineConfig(
    api_key="custom-key",
    mode="provider",
    provider="azure",
    metadata={
        "endpoint": "wss://azure.cognitiveservices.azure.com",
        "region": "eastus",
        "model": "neural-voice"
    }
)
```

### VAD Modes Explained

**Client VAD (`vad_type="client"`)**
- Voice activity detection happens on the client side
- VoxEngine controls when to send audio to the AI
- More control over silence detection and timing
- Higher latency but more predictable behavior
- Best for: Applications needing precise timing control

**Server VAD (`vad_type="server"`)**
- Voice activity detection by the AI provider (OpenAI)
- Lower latency as processing happens server-side
- Less control over exact timing windows
- AI automatically detects speech end and responds
- Best for: Natural conversations with minimal latency

### Response Control Modes

VoxEngine supports two response control modes that work with VAD:

**Automatic Response Mode**
- AI responds automatically after detecting speech end
- Works best with server VAD for natural flow
- No explicit trigger needed
- Injection windows are very tight (50-200ms)

**Manual Response Mode**
- Requires explicit `response.create` call
- Provides full control over when AI responds
- Allows for longer injection windows
- Can interrupt user to inject context
- Enabled via: `response_mode="manual"`

## Limitations

1. **Provider Support**
   - Currently optimized for OpenAI Realtime API
   - Other providers require adapter implementation

2. **Audio Constraints**
   - Fixed sample rate per session (typically 24kHz)
   - Mono audio only
   - PCM16 or G.711 μ-law encoding

3. **Network Requirements**
   - Requires stable internet connection
   - WebSocket connection must remain open
   - Latency dependent on network quality

4. **Concurrency**
   - One active session per engine instance
   - Multiple engines require separate instances

5. **Platform Limitations**
   - Local audio requires platform-specific audio drivers
   - WebRTC mode requires browser environment

## Possible Use Cases

1. **Voice Assistants**
   - Real-time conversational AI
   - Voice-controlled applications
   - Interactive voice response (IVR) systems

2. **Transcription Services**
   - Live transcription
   - Meeting recording and transcription
   - Voice note taking

3. **Voice Communication**
   - AI-powered call centers
   - Voice chat applications
   - Language learning apps

4. **Accessibility**
   - Screen readers with voice
   - Voice-controlled interfaces
   - Audio descriptions

5. **Entertainment**
   - Interactive storytelling
   - Voice-based games
   - AI companions

## Available Endpoints

### Main Class: `VoiceEngine`

```python
class VoiceEngine:
    # Lifecycle
    async def connect(retry_count: int = 3) -> None
    async def disconnect() -> None
    
    # Audio Control
    async def start_listening() -> None
    async def stop_listening() -> None
    async def send_audio(audio_data: bytes) -> None
    async def send_recorded_audio(audio_data: bytes, auto_respond: bool = True) -> None
    
    # Text/Speech
    async def send_text(text: str) -> None
    async def interrupt() -> None
    async def text_2_audio_response(text: str, timeout: float = 30.0) -> bytes
    
    # State Management
    @property
    def is_connected() -> bool
    @property
    def is_listening() -> bool
    @property
    def conversation_state() -> ConversationState
    def get_state() -> Dict[str, Any]
    
    # Performance & Monitoring
    def get_metrics() -> Dict[str, Any]
    async def get_usage() -> Usage
    async def estimate_cost() -> Cost
    
    # Events
    events: EventEmitter
    
    # Factory Methods
    @classmethod
    def create_simple(cls, api_key: str) -> VoiceEngine
    
    # Context Manager Support
    async def __aenter__() -> VoiceEngine
    async def __aexit__() -> None
```

### Modern Event System

VoxEngine uses a powerful event-driven architecture with the EventType enum:

```python
from voxengine import EventType

# Basic event registration
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

# Remove handler
handler_id = engine.events.on(EventType.TEXT_OUTPUT, handler)
engine.events.off(handler_id)
```

#### Key Event Types

**Connection Events:**
- `EventType.CONNECTION_STARTING` - Connection attempt initiated
- `EventType.CONNECTION_ESTABLISHED` - Successfully connected
- `EventType.CONNECTION_FAILED` - Connection attempt failed
- `EventType.CONNECTION_LOST` - Connection dropped unexpectedly
- `EventType.CONNECTION_CLOSED` - Connection closed normally

**Audio Events:**
- `EventType.AUDIO_INPUT_STARTED` - Audio capture began
- `EventType.AUDIO_INPUT_CHUNK` - Audio input data received
- `EventType.AUDIO_INPUT_STOPPED` - Audio capture ended
- `EventType.AUDIO_OUTPUT_STARTED` - Audio playback began
- `EventType.AUDIO_OUTPUT_CHUNK` - Audio output data received
- `EventType.AUDIO_OUTPUT_STOPPED` - Audio playback ended

**Text Events:**
- `EventType.TEXT_INPUT` - Text message sent to AI
- `EventType.TEXT_OUTPUT` - Text response from AI

**Conversation Events:**
- `EventType.CONVERSATION_STARTED` - New conversation began
- `EventType.CONVERSATION_TURN_DETECTED` - Speaker turn change
- `EventType.CONVERSATION_INTERRUPTED` - User interrupted AI
- `EventType.CONVERSATION_ENDED` - Conversation concluded

**Response Events:**
- `EventType.RESPONSE_STARTED` - AI began generating response
- `EventType.RESPONSE_COMPLETED` - AI finished response
- `EventType.RESPONSE_CANCELLED` - Response generation cancelled

**Function Events:**
- `EventType.FUNCTION_CALL_INVOKED` - AI requested function call

**Error Events:**
- `EventType.ERROR_GENERAL` - General error occurred

#### Event Data Structures

```python
# Connection event
{
    'type': EventType.CONNECTION_ESTABLISHED,
    'connection_id': 'conn_abc123',
    'latency_ms': 45.2,
    'timestamp': 1234567890.123
}

# Audio chunk event
{
    'type': EventType.AUDIO_OUTPUT_CHUNK,
    'audio_data': b'...',  # Raw audio bytes
    'sample_rate': 24000,
    'channels': 1,
    'duration_ms': 100
}

# Text output event
{
    'type': EventType.TEXT_OUTPUT,
    'text': 'Hello, how can I help you?',
    'is_partial': False,
    'message_id': 'msg_xyz789'
}

# Error event
{
    'type': EventType.ERROR_GENERAL,
    'error': Exception('Connection timeout'),
    'error_message': 'Connection timeout after 30s',
    'recoverable': True,
    'context': {...}
}
```

#### Legacy Event System (Deprecated)

For backward compatibility, the old callback system is still supported:

```python
# Old style - still works but deprecated
engine.on_audio_response = lambda audio: play_audio(audio)
engine.on_text_response = lambda text: print(f"AI: {text}")
engine.on_error = lambda error: logger.error(error)
```

## Interfaces

### Core Interfaces

1. **StreamProtocol**
   ```python
   class StreamProtocol:
       async def connect()
       async def disconnect()
       async def send(data: bytes)
       async def receive() -> bytes
   ```

2. **AudioProtocol**
   ```python
   class AudioProtocol:
       def start_recording()
       def stop_recording()
       def start_playback()
       def stop_playback()
       def get_audio_data() -> bytes
   ```

3. **ProviderProtocol**
   ```python
   class ProviderProtocol:
       async def initialize(config: dict)
       async def create_session() -> Session
       async def send_audio(audio: bytes)
       async def send_text(text: str)
   ```

## Integration Points

### 1. **Voxon Integration**
- VoxEngine is managed by Voxon's EngineCoordinator
- Provides voice I/O capabilities to higher layers
- Emits events consumed by Voxon

### 2. **ContextWeaver Integration**
- Receives text injections from ContextWeaver
- Provides conversation state for context decisions
- Timing coordination through event system

### 3. **Audio Sources**
- **Local Audio**: Microphone/speaker via PyAudio
- **WebRTC**: Browser audio streams
- **File Input**: Pre-recorded audio files
- **Network Streams**: Remote audio sources

### 4. **State Management**
- Exports AudioState for low-level audio info
- Exports ConnectionState for connection status
- Conversation state managed by Voxon

### 5. **VAD Mode and Injection Timing**

The relationship between VAD modes and context injection timing:

**Server VAD + Automatic Response**
- Tightest injection windows (50-100ms)
- Must inject immediately when silence detected
- Risk of collision with AI response
- Requires aggressive injection strategy

**Server VAD + Manual Response**  
- Controlled injection windows
- Can delay AI response for injection
- Better for complex context injection
- Allows strategic timing

**Client VAD + Automatic Response**
- Moderate injection windows (200-500ms)
- Local control over silence detection
- More predictable timing
- Good balance for most applications

**Client VAD + Manual Response**
- Maximum control and flexibility
- Longest injection windows (up to 2000ms)
- Can coordinate complex multi-context injection
- Best for specialized applications

### Example Integration
```python
# Voxon creates and manages VoxEngine
voice_engine = VoiceEngine(config)
coordinator.voice_engine = voice_engine

# ContextWeaver injects through VoxEngine
await voice_engine.send_text(context.information)

# Events flow to Voxon
voice_engine.events.on('response.text', handle_response)
```

## Edge Cases Covered

1. **Connection Failures**
   - Automatic reconnection with exponential backoff
   - Graceful degradation on repeated failures
   - Clear error events with context

2. **Audio Interruptions**
   - Handles microphone disconnection
   - Manages speaker availability changes
   - Buffers audio during temporary issues

3. **Network Issues**
   - WebSocket reconnection logic
   - Message queuing during disconnection
   - Timeout handling for API calls

4. **Concurrency**
   - Thread-safe audio operations
   - Async-safe event emissions
   - Protected state modifications

5. **Resource Management**
   - Proper cleanup on disconnect
   - Memory-efficient audio buffering
   - Automatic resource release

6. **VAD Edge Cases**
   - Handles VAD mode switches mid-conversation
   - Manages silence timeout variations
   - Adapts to noisy environments

7. **Error Recovery**
   - Continues operation after non-fatal errors
   - Provides detailed error information
   - Maintains conversation context

## API Method Details

### Audio Methods

#### `async send_recorded_audio(audio_data: bytes, auto_respond: bool = True)`
Sends a complete audio recording (useful for push-to-talk).
```python
# Record audio, then send
audio_data = record_audio()  # Your recording logic
await engine.send_recorded_audio(audio_data, auto_respond=True)
```

#### `async text_2_audio_response(text: str, timeout: float = 30.0) -> bytes`
Converts text to speech and returns complete audio data.
```python
# Get audio for text
audio = await engine.text_2_audio_response("Hello world")
# Save or play the audio
with open("response.wav", "wb") as f:
    f.write(audio)
```

### Monitoring Methods

#### `get_metrics() -> Dict[str, Any]`
Returns detailed performance metrics.
```python
metrics = engine.get_metrics()
print(f"Audio latency: {metrics['audio']['playback_latency']}ms")
print(f"Connection uptime: {metrics['connection']['uptime']}s")
```

#### `async get_usage() -> Usage`
Returns API usage statistics.
```python
usage = await engine.get_usage()
print(f"Audio seconds used: {usage.audio_seconds}")
print(f"Text tokens used: {usage.text_tokens}")
```

#### `async estimate_cost() -> Cost`
Estimates costs based on current usage.
```python
cost = await engine.estimate_cost()
print(f"Estimated cost: ${cost.total:.2f}")
```

## Example Usage

### Basic Voice Assistant
```python
from voxengine import VoiceEngine, VoiceEngineConfig, EventType

# Configure engine
config = VoiceEngineConfig(
    api_key="sk-...",
    vad_type="server",
    voice="nova",
    response_mode="automatic"
)
engine = VoiceEngine(config)

# Set up event handlers
engine.events.on(EventType.TEXT_OUTPUT,
                lambda e: print(f"AI: {e.text}"))
engine.events.on(EventType.ERROR_GENERAL,
                lambda e: print(f"Error: {e.error_message}"))

# Connect and start
await engine.connect()
await engine.start_listening()

# Let it run...
await asyncio.sleep(60)

# Cleanup
await engine.disconnect()
```

### Context Manager Usage
```python
# Automatic resource management
async with VoiceEngine.create_simple("sk-...") as engine:
    await engine.start_listening()
    await engine.send_text("Hello!")
    # ... do work ...
# Automatically disconnects
```

### Push-to-Talk Mode
```python
engine = VoiceEngine(config)
await engine.connect()

# On key press
recording = []
engine.events.on(EventType.AUDIO_INPUT_CHUNK,
                lambda e: recording.append(e.audio_data))
await engine.start_listening()

# On key release
await engine.stop_listening()
audio_data = b''.join(recording)
await engine.send_recorded_audio(audio_data)
```

### Advanced Event Handling
```python
# Priority-based handlers
engine.events.on(EventType.TEXT_OUTPUT, log_handler, priority=1)
engine.events.on(EventType.TEXT_OUTPUT, display_handler, priority=10)

# Filtered handlers
engine.events.on(EventType.TEXT_OUTPUT,
                handle_questions,
                filter=lambda e: '?' in e.text)

# One-time setup
engine.events.once(EventType.CONNECTION_ESTABLISHED,
                  lambda e: print(f"Connected in {e.latency_ms}ms"))

# Monitor all events
engine.events.on("*", lambda e: logger.debug(f"{e.type}: {e}"))
```