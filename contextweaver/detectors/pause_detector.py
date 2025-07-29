"""
Pause Detector

Detects natural pauses in conversation for context injection.
"""

from datetime import datetime
from typing import List, Optional
from .base import BaseDetector, DetectionResult


class PauseDetector(BaseDetector):
    """Detect natural conversation pauses"""
    
    def __init__(
        self,
        pause_patterns: Optional[List[str]] = None,
        min_pause_ms: int = 500,
        max_pause_ms: int = 2000
    ):
        super().__init__()
        self.pause_patterns = pause_patterns or [
            "um", "uh", "hmm", "well", "so", "let me think"
        ]
        self.min_pause_ms = min_pause_ms
        self.max_pause_ms = max_pause_ms
        self.last_speech_end: Optional[datetime] = None
    
    async def detect(self, state) -> DetectionResult:
        """Detect natural pauses in conversation"""
        
        # Check recent messages for pause indicators
        if hasattr(state, 'messages') and state.messages:
            last_message = state.messages[-1]
            
            # Check if last message contains pause patterns
            if hasattr(last_message, 'content'):
                content_lower = last_message.content.lower()
                for pattern in self.pause_patterns:
                    if pattern in content_lower:
                        return DetectionResult(
                            detected=True,
                            confidence=0.85,
                            timestamp=datetime.now(),
                            metadata={
                                "pattern_found": pattern,
                                "message": last_message.content
                            }
                        )
        
        # Check timing between turns
        if hasattr(state, 'current_turn') and state.current_turn:
            if state.current_turn.user_message and not state.current_turn.assistant_message:
                # User has spoken, waiting for assistant
                elapsed = (datetime.now() - state.current_turn.started_at).total_seconds() * 1000
                
                if self.min_pause_ms <= elapsed <= self.max_pause_ms:
                    confidence = 1.0 - abs(elapsed - 1000) / 1000  # Peak at 1 second
                    return DetectionResult(
                        detected=True,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        metadata={
                            "pause_duration_ms": elapsed,
                            "type": "turn_transition"
                        }
                    )
        
        return DetectionResult(
            detected=False,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={"reason": "no_pause_detected"}
        )