"""
Audio State

Low-level audio state management for VoxEngine.
Tracks audio devices, VAD activity, and audio metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional
from datetime import datetime


@dataclass(frozen=True)
class AudioState:
    """Audio-specific state for the voice engine"""
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