"""
Silence Detector

Detects periods of silence that might be good for context injection.
"""

from datetime import datetime, timedelta
from typing import Optional
from .base import BaseDetector, DetectionResult


class SilenceDetector(BaseDetector):
    """Detect silence periods for context injection"""
    
    def __init__(
        self, 
        silence_threshold_ms: int = 2000,
        min_confidence: float = 0.8
    ):
        super().__init__(threshold=min_confidence)
        self.silence_threshold_ms = silence_threshold_ms
        self.silence_start: Optional[datetime] = None
    
    async def detect(self, state) -> DetectionResult:
        """Detect if we're in a suitable silence period"""
        
        # Check if VAD is active (someone speaking)
        if hasattr(state, 'audio') and state.audio.vad_active:
            # Reset silence tracking
            self.silence_start = None
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": "voice_active"}
            )
        
        # Track silence duration
        now = datetime.now()
        
        if self.silence_start is None:
            self.silence_start = now
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=now,
                metadata={"reason": "silence_just_started"}
            )
        
        # Calculate silence duration
        silence_duration = (now - self.silence_start).total_seconds() * 1000
        
        if silence_duration >= self.silence_threshold_ms:
            # Calculate confidence based on duration
            confidence = min(1.0, silence_duration / (self.silence_threshold_ms * 2))
            
            return DetectionResult(
                detected=True,
                confidence=confidence,
                timestamp=now,
                metadata={
                    "silence_duration_ms": silence_duration,
                    "threshold_ms": self.silence_threshold_ms
                }
            )
        
        return DetectionResult(
            detected=False,
            confidence=silence_duration / self.silence_threshold_ms,
            timestamp=now,
            metadata={"silence_duration_ms": silence_duration}
        )