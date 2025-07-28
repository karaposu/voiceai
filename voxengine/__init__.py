# here is voxengine/__init__.py

"""
voxengine - Modern Python framework for OpenAI's Realtime API

"""

__version__ = "0.2.0"

# Core imports for smoke tests
from voxengine.voice_engine import VoiceEngine, VoiceEngineConfig
from voxstream.core.processor import StreamProcessor as AudioProcessor
from voxstream import VoxStream as AudioEngine
from voxstream.core.stream import (
    create_fast_lane_engine,
    create_big_lane_engine,
    create_adaptive_engine,
)
from voxstream.config.types import StreamMetrics as ProcessingMetrics
# ProcessingStrategy is not available in voxstream, define it locally if needed
from enum import Enum
class ProcessingStrategy(Enum):
    ZERO_COPY = "zero_copy"
    FAST_LANE = "fast_lane"
    BALANCED = "balanced"
    QUALITY = "quality"
from voxengine.session import SessionConfig, SessionPresets
from voxengine.config import Identity, IDENTITIES
from voxengine.core.exceptions import (
    RealtimeError,
    ConnectionError,
    AuthenticationError,
    AudioError,
    StreamError,
    EngineError,
)


# Message protocol (used by smoke tests)
from voxengine.core.message_protocol import (
    ClientMessageType,
    ServerMessageType,
    MessageFactory,
    MessageValidator,
    ProtocolInfo,
)

# Audio types (used by smoke tests)
from voxstream.config.types import (
    AudioFormat,
    StreamConfig as AudioConfig,
    ProcessingMode,
    BufferConfig,
    AudioConstants,
    VADConfig,
    VADType,
)
from voxstream.voice.vad import VoiceState

# Stream protocol (used by smoke tests)
from voxengine.core.stream_protocol import (
    StreamEvent,
    StreamEventType,
    StreamState,
)



from voxstream.io.manager import (
    AudioManager,
    AudioManagerConfig,
    # AudioComponentState,
    # create_audio_manager,
)

# Session manager (used by smoke tests)
from voxengine.session.session_manager import SessionManager


__all__ = [
    "__version__",
    "VoiceEngine",
    "VoiceEngineConfig",
    "AudioProcessor",
    "AudioEngine",
    "ProcessingMetrics",
    "ProcessingStrategy",
    "create_fast_lane_engine",
    "create_big_lane_engine",
    "create_adaptive_engine",
    "SessionConfig",
    "SessionPresets",
    "Identity",
    "IDENTITIES",
    "RealtimeError",
    "ConnectionError",
    "AuthenticationError",
    "AudioError",
    "StreamError",
    "EngineError",
    # "Tool",
    # "TurnDetectionConfig", 
    # "TranscriptionConfig",
    # "AudioFormatType",
    # "ModalityType",
    # "VoiceType",
    "ClientMessageType",
    "ServerMessageType",
    "MessageFactory",
    "MessageValidator",
    "ProtocolInfo",
    "AudioFormat",
    "AudioConfig",
    "ProcessingMode",
    "BufferConfig",
    "AudioConstants",
    "VADConfig",
    "VADType",
    "StreamEvent",
    "StreamEventType",
    "StreamState",
    "SessionManager",
    "AudioManager",
    "AudioManagerConfig",
    # "AudioComponentState",
    # "create_audio_manager",
]