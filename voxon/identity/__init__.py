"""
Voxon Identity Management

Voice assistant personalities and configurations.
"""

from .identity import (
    Identity,
    IDENTITIES,
    DEFAULT_ASSISTANT,
    VOICE_ASSISTANT,
    TRANSCRIPTION_SERVICE,
    CONVERSATIONAL_AI,
    CUSTOMER_SERVICE,
    AUDIO_ONLY_ASSISTANT
)

__all__ = [
    'Identity',
    'IDENTITIES',
    'DEFAULT_ASSISTANT',
    'VOICE_ASSISTANT', 
    'TRANSCRIPTION_SERVICE',
    'CONVERSATIONAL_AI',
    'CUSTOMER_SERVICE',
    'AUDIO_ONLY_ASSISTANT'
]