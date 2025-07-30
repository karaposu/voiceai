"""
VoxEngine Config - Compatibility Bridge

This module now imports from voxon.identity to maintain backward compatibility
while the identity/personality configuration has been moved to the higher-level voxon layer.

NOTE: This is a transitional module. New code should import directly from voxon.identity.
"""

# Re-export everything from voxon.identity for backward compatibility
from voxon.identity import (
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