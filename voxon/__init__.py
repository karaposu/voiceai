"""
Voxon - Conversation Intelligence Layer

High-level orchestration for voice conversations.
Manages VoxEngine and ContextInjectionEngine to create
intelligent, context-aware voice experiences.
"""

from .orchestrator import Voxon, VoxonConfig
from .conversation import (
    Conversation,
    ConversationManager,
    ConversationTemplate
)

__all__ = [
    "Voxon",
    "VoxonConfig", 
    "Conversation",
    "ConversationManager",
    "ConversationTemplate"
]