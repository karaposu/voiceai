"""
Voxon State Management

High-level conversation state management moved from voxengine.
AudioState and ConnectionState remain in voxengine as they are low-level engine states.
"""

from .conversation_state import (
    ConversationState,
    ConversationStatus,
    SpeakerRole,
    Message,
    Turn,
    ConversationMetrics
)

from .state_manager import StateManager, StateUpdate

__all__ = [
    # High-level conversation states
    'ConversationState',
    'ConversationStatus',
    'SpeakerRole',
    'Message',
    'Turn',
    'ConversationMetrics',
    # Manager
    'StateManager',
    'StateUpdate'
]