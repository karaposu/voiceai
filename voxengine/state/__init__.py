"""
VoxEngine State

Contains low-level engine states (AudioState, ConnectionState) and
re-exports high-level conversation states from voxon for compatibility.
"""

# Low-level engine states (belong in voxengine)
from .audio_state import AudioState
from .connection_state import ConnectionState

# Note: To avoid circular imports, high-level conversation states should be
# imported directly from voxon.state when needed

__all__ = [
    # Low-level engine states
    "AudioState",
    "ConnectionState"
]