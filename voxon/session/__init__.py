# voxon/session/__init__.py

"""
Session Management Package

Provides high-level conversation configuration and runtime state management.
"""

from .session import (
    SessionConfig,
    SessionPresets,
    ConfigManager,
    # Re-export temporary types until proper models are implemented
    AudioFormatType,
    ModalityType,
    VoiceType,
    ToolChoiceType,
    TurnDetectionConfig,
    TranscriptionConfig,
    Tool
)

from .session_manager import (
    SessionState,
    Session,
    SessionManager
)

__all__ = [
    # From session.py
    'SessionConfig',
    'SessionPresets',
    'ConfigManager',
    'AudioFormatType',
    'ModalityType',
    'VoiceType',
    'ToolChoiceType',
    'TurnDetectionConfig',
    'TranscriptionConfig',
    'Tool',
    
    # From session_manager.py
    'SessionState',
    'Session',
    'SessionManager',
]