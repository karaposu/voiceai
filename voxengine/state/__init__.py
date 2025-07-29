"""
VoxEngine State Management

Clean, thread-safe state management for voice conversations.
"""

from .conversation_state import (
    ConversationState,
    ConversationStatus,
    ConversationMetrics,
    Message,
    Turn,
    AudioState,
    ConnectionState,
    SpeakerRole
)

from .state_manager import (
    StateManager,
    StateUpdate
)

__all__ = [
    # State classes
    "ConversationState",
    "ConversationStatus",
    "ConversationMetrics",
    "Message",
    "Turn",
    "AudioState", 
    "ConnectionState",
    "SpeakerRole",
    
    # Manager
    "StateManager",
    "StateUpdate"
]