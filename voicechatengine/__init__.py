# here is voicechatengine/__init__.py

"""
voicechatengine - Modern Python framework for OpenAI's Realtime API

"""

__version__ = "0.2.0"

# Core imports for smoke tests
from voicechatengine.voice_engine import VoiceEngine, VoiceEngineConfig
from voicechatengine.audioengine.audioengine.audio_processor import AudioProcessor
from voicechatengine.audioengine.audioengine.audio_engine import (
    AudioEngine,
    ProcessingMetrics,
    ProcessingStrategy,
    create_fast_lane_engine,
    create_big_lane_engine,
    create_adaptive_engine,
)
from voicechatengine.session import SessionConfig, SessionPresets
from voicechatengine.config import Identity, IDENTITIES
from voicechatengine.core.exceptions import (
    RealtimeError,
    ConnectionError,
    AuthenticationError,
    AudioError,
    StreamError,
    EngineError,
)


# Message protocol (used by smoke tests)
from voicechatengine.core.message_protocol import (
    ClientMessageType,
    ServerMessageType,
    MessageFactory,
    MessageValidator,
    ProtocolInfo,
)

# Audio types (used by smoke tests)
from voicechatengine.audioengine.audioengine.audio_types import (
    AudioFormat,
    AudioConfig,
    ProcessingMode,
    BufferConfig,
    AudioConstants,
    VADConfig,
    VADType,
)

# Stream protocol (used by smoke tests)
from voicechatengine.core.stream_protocol import (
    StreamEvent,
    StreamEventType,
    StreamState,
)



from voicechatengine.audioengine.audioengine.audio_manager import (
    AudioManager,
    AudioManagerConfig,
    # AudioComponentState,
    # create_audio_manager,
)

# Session manager (used by smoke tests)
from voicechatengine.session.session_manager import SessionManager


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