"""
Event Types for Vox Engine

Comprehensive event types for all voice engine operations including
connection, audio, conversation, and timing events.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import time


class EventType(str, Enum):
    """All event types in the voice engine"""
    
    # Connection Events
    CONNECTION_STARTING = "connection.starting"
    CONNECTION_ESTABLISHED = "connection.established"
    CONNECTION_FAILED = "connection.failed"
    CONNECTION_LOST = "connection.lost"
    CONNECTION_CLOSED = "connection.closed"
    CONNECTION_RECONNECTING = "connection.reconnecting"
    
    # Audio Input Events
    AUDIO_INPUT_STARTED = "audio.input.started"
    AUDIO_INPUT_CHUNK = "audio.input.chunk"
    AUDIO_INPUT_STOPPED = "audio.input.stopped"
    AUDIO_INPUT_ERROR = "audio.input.error"
    
    # Audio Output Events
    AUDIO_OUTPUT_STARTED = "audio.output.started"
    AUDIO_OUTPUT_CHUNK = "audio.output.chunk"
    AUDIO_OUTPUT_STOPPED = "audio.output.stopped"
    AUDIO_OUTPUT_ERROR = "audio.output.error"
    
    # Text Events
    TEXT_INPUT = "text.input"
    TEXT_OUTPUT = "text.output"
    TEXT_STREAM_STARTED = "text.stream.started"
    TEXT_STREAM_CHUNK = "text.stream.chunk"
    TEXT_STREAM_COMPLETED = "text.stream.completed"
    
    # Conversation Events
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_TURN_DETECTED = "conversation.turn_detected"
    CONVERSATION_INTERRUPTED = "conversation.interrupted"
    CONVERSATION_ENDED = "conversation.ended"
    
    # Voice Activity Detection (VAD) Events
    VAD_SPEECH_STARTED = "vad.speech_started"
    VAD_SPEECH_ENDED = "vad.speech_ended"
    VAD_SILENCE_DETECTED = "vad.silence_detected"
    VAD_NOISE_DETECTED = "vad.noise_detected"
    
    # Response Events
    RESPONSE_STARTED = "response.started"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_CANCELLED = "response.cancelled"
    RESPONSE_ERROR = "response.error"
    
    # Function Call Events
    FUNCTION_CALL_INVOKED = "function.invoked"
    FUNCTION_CALL_COMPLETED = "function.completed"
    FUNCTION_CALL_FAILED = "function.failed"
    
    # Timing Events
    TIMING_TURN_DETECTION = "timing.turn_detection"
    TIMING_RESPONSE_LATENCY = "timing.response_latency"
    TIMING_FIRST_BYTE = "timing.first_byte"
    TIMING_CONTEXT_WINDOW = "timing.context_window"
    
    # State Events
    STATE_CHANGED = "state.changed"
    STATE_ERROR = "state.error"
    
    # Metrics Events
    METRICS_UPDATED = "metrics.updated"
    METRICS_AUDIO_LEVEL = "metrics.audio_level"
    METRICS_NETWORK_QUALITY = "metrics.network_quality"
    
    # Error Events
    ERROR_GENERAL = "error.general"
    ERROR_NETWORK = "error.network"
    ERROR_AUDIO = "error.audio"
    ERROR_PROVIDER = "error.provider"
    
    # System Events
    SYSTEM_READY = "system.ready"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_WARNING = "system.warning"


@dataclass
class Event:
    """Base event class for all voice engine events"""
    
    type: EventType
    timestamp: float = field(default_factory=time.time)
    data: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event data"""
        if self.data is None:
            self.data = {}
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error event"""
        return self.error is not None or "error" in self.type.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "source": self.source,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata
        }


# Specific Event Classes

@dataclass
class AudioEvent(Event):
    """Audio-specific event"""
    audio_data: Optional[bytes] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_ms: Optional[float] = None


@dataclass
class TextEvent(Event):
    """Text-specific event"""
    text: str = ""
    is_partial: bool = False
    language: Optional[str] = None


@dataclass
class ConnectionEvent(Event):
    """Connection-specific event"""
    connection_id: Optional[str] = None
    retry_count: int = 0
    latency_ms: Optional[float] = None


@dataclass
class ConversationEvent(Event):
    """Conversation-specific event"""
    turn_id: Optional[str] = None
    speaker: Optional[str] = None  # "user" or "assistant"
    duration_ms: Optional[float] = None


@dataclass
class TimingEvent(Event):
    """Timing-specific event"""
    duration_ms: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class FunctionCallEvent(Event):
    """Function call event"""
    function_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    call_id: Optional[str] = None


@dataclass
class MetricsEvent(Event):
    """Metrics event"""
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    period_seconds: Optional[float] = None


@dataclass
class ErrorEvent(Event):
    """Error event with additional context"""
    error_code: Optional[str] = None
    error_message: str = ""
    recoverable: bool = True
    retry_after_seconds: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.error and not self.error_message:
            self.error_message = str(self.error)