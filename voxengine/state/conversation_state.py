"""
Conversation State Management

Clean, immutable state representation for voice conversations.
Designed to be thread-safe and integrate seamlessly with the event system.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


class ConversationStatus(str, Enum):
    """Conversation lifecycle states"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class SpeakerRole(str, Enum):
    """Speaker roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    """Immutable message representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: SpeakerRole = SpeakerRole.USER
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Audio-specific fields
    audio_data: Optional[bytes] = None
    audio_duration_ms: Optional[float] = None
    
    # Response tracking
    is_complete: bool = True
    is_interrupted: bool = False
    
    def evolve(self, **changes) -> 'Message':
        """Create a new message with specified changes"""
        return replace(self, **changes)


@dataclass(frozen=True)
class Turn:
    """Represents a conversation turn (user input + assistant response)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_message: Optional[Message] = None
    assistant_message: Optional[Message] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if turn has both user and assistant messages"""
        return self.user_message is not None and self.assistant_message is not None
    
    @property
    def is_active(self) -> bool:
        """Check if turn is currently active"""
        return self.user_message is not None and self.assistant_message is None
    
    def evolve(self, **changes) -> 'Turn':
        """Create a new turn with specified changes"""
        return replace(self, **changes)


@dataclass(frozen=True)
class AudioState:
    """Audio-specific state"""
    is_listening: bool = False
    is_playing: bool = False
    input_device_id: Optional[int] = None
    output_device_id: Optional[int] = None
    
    # VAD state
    vad_active: bool = False
    last_speech_timestamp: Optional[datetime] = None
    silence_duration_ms: float = 0.0
    
    # Audio metrics
    input_volume_db: float = -60.0
    output_volume_db: float = -60.0
    audio_latency_ms: float = 0.0
    
    def evolve(self, **changes) -> 'AudioState':
        """Create a new state with specified changes"""
        return replace(self, **changes)


@dataclass(frozen=True)
class ConnectionState:
    """Connection-specific state"""
    is_connected: bool = False
    connection_id: Optional[str] = None
    connected_at: Optional[datetime] = None
    provider: str = "openai"
    
    # Connection metrics
    latency_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0
    reconnect_count: int = 0
    
    def evolve(self, **changes) -> 'ConnectionState':
        """Create a new state with specified changes"""
        return replace(self, **changes)


@dataclass(frozen=True)
class ConversationMetrics:
    """Conversation metrics and statistics"""
    total_turns: int = 0
    total_messages: int = 0
    total_duration_ms: float = 0.0
    
    # Timing metrics
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    
    # Interaction metrics
    interruption_count: int = 0
    silence_count: int = 0
    topic_changes: int = 0
    
    # Cost tracking
    audio_seconds_used: float = 0.0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    
    def evolve(self, **changes) -> 'ConversationMetrics':
        """Create a new state with specified changes"""
        return replace(self, **changes)


@dataclass(frozen=True)
class ConversationState:
    """
    Complete conversation state - immutable and thread-safe.
    
    This represents the entire state of a voice conversation at any point in time.
    All fields are immutable - use the `evolve()` method to create new states.
    """
    # Identity
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    
    # Status
    status: ConversationStatus = ConversationStatus.IDLE
    started_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)
    
    # Sub-states
    connection: ConnectionState = field(default_factory=ConnectionState)
    audio: AudioState = field(default_factory=AudioState)
    metrics: ConversationMetrics = field(default_factory=ConversationMetrics)
    
    # Conversation data
    messages: List[Message] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)
    current_turn: Optional[Turn] = None
    
    # Context and configuration
    context: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Feature flags
    transcription_enabled: bool = False
    functions_enabled: bool = False
    context_injection_enabled: bool = False
    
    def evolve(self, **changes) -> 'ConversationState':
        """Create a new state with specified changes (immutable update)"""
        return replace(self, **changes)
    
    @property
    def is_active(self) -> bool:
        """Check if conversation is currently active"""
        return self.status in [
            ConversationStatus.CONNECTED,
            ConversationStatus.LISTENING,
            ConversationStatus.PROCESSING,
            ConversationStatus.RESPONDING
        ]
    
    @property
    def message_count(self) -> int:
        """Total number of messages"""
        return len(self.messages)
    
    @property
    def turn_count(self) -> int:
        """Total number of completed turns"""
        return len([t for t in self.turns if t.is_complete])
    
    @property
    def last_message(self) -> Optional[Message]:
        """Get the last message in the conversation"""
        return self.messages[-1] if self.messages else None
    
    @property
    def last_user_message(self) -> Optional[Message]:
        """Get the last user message"""
        for msg in reversed(self.messages):
            if msg.role == SpeakerRole.USER:
                return msg
        return None
    
    @property
    def last_assistant_message(self) -> Optional[Message]:
        """Get the last assistant message"""
        for msg in reversed(self.messages):
            if msg.role == SpeakerRole.ASSISTANT:
                return msg
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "message_count": self.message_count,
            "turn_count": self.turn_count,
            "is_active": self.is_active,
            "connection": {
                "is_connected": self.connection.is_connected,
                "provider": self.connection.provider,
                "latency_ms": self.connection.latency_ms
            },
            "audio": {
                "is_listening": self.audio.is_listening,
                "is_playing": self.audio.is_playing,
                "vad_active": self.audio.vad_active
            },
            "metrics": {
                "total_turns": self.metrics.total_turns,
                "total_duration_ms": self.metrics.total_duration_ms,
                "interruption_count": self.metrics.interruption_count
            }
        }