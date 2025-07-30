"""
Voxon - Conversation Intelligence Layer

High-level orchestration for voice conversations.
Manages VoxEngine and ContextWeaver to create
intelligent, context-aware voice experiences.

Now includes conversation state, session management, and identity configuration
moved from voxengine for better separation of concerns.
"""

# Core orchestration
from .orchestrator import Voxon, VoxonConfig, EngineCoordinator
from .conversation import (
    Conversation,
    ConversationManager,
    ConversationTemplate
)

# State management (moved from voxengine)
from .state import (
    ConversationState, ConversationStatus, SpeakerRole,
    Message, Turn, ConversationMetrics,
    StateManager
)

# Session management (moved from voxengine)
from .session import (
    SessionConfig, SessionPresets, ConfigManager,
    SessionState, Session, SessionManager
)

# Identity management (moved from voxengine)
from .identity import (
    Identity, IDENTITIES,
    DEFAULT_ASSISTANT, VOICE_ASSISTANT, CUSTOMER_SERVICE
)

__all__ = [
    # Orchestration
    "Voxon",
    "VoxonConfig",
    "EngineCoordinator",
    "Conversation",
    "ConversationManager",
    "ConversationTemplate",
    # State
    'ConversationState',
    'ConversationStatus',
    'SpeakerRole',
    'Message',
    'Turn',
    'ConversationMetrics',
    'StateManager',
    # Session
    'SessionConfig',
    'SessionPresets',
    'ConfigManager',
    'SessionState',
    'Session',
    'SessionManager',
    # Identity
    'Identity',
    'IDENTITIES',
    'DEFAULT_ASSISTANT',
    'VOICE_ASSISTANT',
    'CUSTOMER_SERVICE'
]