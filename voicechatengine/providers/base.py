"""
Base Provider Interface for VoiceEngine V2

This module defines the abstract interface that all voice AI providers must implement.
It ensures provider-agnostic operation of the VoiceEngine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Callable, AsyncIterator, List
import asyncio


class MessageType(Enum):
    """Standard message types across all providers"""
    # Session management
    SESSION_CREATE = "session.create"
    SESSION_UPDATE = "session.update"
    SESSION_DELETE = "session.delete"
    
    # Conversation
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    
    # Audio
    AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    
    # Response
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"
    
    # Real-time events
    AUDIO_CHUNK = "audio.chunk"
    TEXT_CHUNK = "text.chunk"
    FUNCTION_CALL = "function.call"
    
    # Control
    INTERRUPT = "interrupt"
    PING = "ping"


class ConnectionState(Enum):
    """Provider connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ProviderConfig:
    """Base configuration for providers"""
    api_key: str
    model: str = "default"
    url: Optional[str] = None
    voice: str = "alloy"
    language: str = "en"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30
    retry_attempts: int = 3
    enable_logging: bool = True


@dataclass
class ProviderEvent:
    """Standard event structure from providers"""
    type: str
    data: Dict[str, Any]
    timestamp: float
    provider: str
    session_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AudioFormat:
    """Audio format specification"""
    encoding: str = "pcm16"  # pcm16, mp3, opus
    sample_rate: int = 24000
    channels: int = 1
    frame_size: Optional[int] = None


class BaseProvider(ABC):
    """
    Abstract base class for all voice AI providers.
    
    This class defines the interface that all providers must implement
    to work with VoiceEngine V2.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.session_id: Optional[str] = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the provider.
        
        Returns:
            bool: True if connection successful
        """
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the provider."""
        pass
        
    @abstractmethod
    async def send_message(self, message_type: MessageType, data: Dict[str, Any]) -> None:
        """
        Send a message to the provider.
        
        Args:
            message_type: Type of message to send
            data: Message data
        """
        pass
        
    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to the provider.
        
        Args:
            audio_data: Raw audio bytes
        """
        pass
        
    @abstractmethod
    async def send_text(self, text: str, role: str = "user") -> None:
        """
        Send text message to the provider.
        
        Args:
            text: Text content
            role: Message role (user, assistant, system)
        """
        pass
        
    @abstractmethod
    async def interrupt(self) -> None:
        """Interrupt the current response."""
        pass
        
    @abstractmethod
    async def create_response(self, modalities: List[str] = None) -> None:
        """
        Request a response from the AI.
        
        Args:
            modalities: List of response types (e.g., ["text", "audio"])
        """
        pass
        
    @abstractmethod
    def get_audio_format(self) -> AudioFormat:
        """Get the audio format required by this provider."""
        pass
        
    @abstractmethod
    async def events(self) -> AsyncIterator[ProviderEvent]:
        """
        Async iterator for provider events.
        
        Yields:
            ProviderEvent: Events from the provider
        """
        pass
        
    # Common functionality implemented in base class
    
    def on_event(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].remove(handler)
            
    async def _emit_event(self, event: ProviderEvent) -> None:
        """Emit an event to registered handlers."""
        handlers = self._event_handlers.get(event.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log error but don't stop event propagation
                print(f"Error in event handler: {e}")
                
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self.state == ConnectionState.CONNECTED
        
    async def ensure_connected(self) -> None:
        """Ensure provider is connected, attempt connection if not."""
        if not self.is_connected():
            success = await self.connect()
            if not success:
                raise ConnectionError(f"Failed to connect to {self.__class__.__name__}")
                
    @abstractmethod
    async def update_session(self, **kwargs) -> None:
        """
        Update session configuration.
        
        Args:
            **kwargs: Session parameters to update
        """
        pass
        
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get provider capabilities.
        
        Returns:
            Dict containing capability information
        """
        pass


class ProviderError(Exception):
    """Base exception for provider errors"""
    pass


class ConnectionError(ProviderError):
    """Connection-related errors"""
    pass


class MessageError(ProviderError):
    """Message sending errors"""
    pass


class AudioError(ProviderError):
    """Audio-related errors"""
    pass