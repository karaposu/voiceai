"""
Silence Detector

Enhanced silence detection with VAD state integration and predictive capabilities.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from collections import deque
import statistics
from .base import BaseDetector, DetectionResult


class SilenceDetector(BaseDetector):
    """
    Enhanced silence detector with VAD state awareness.
    
    Features:
    - VAD state integration (client/server modes)
    - Predictive silence detection
    - Adaptive thresholds based on conversation patterns
    - Multi-level confidence scoring
    """
    
    def __init__(
        self, 
        silence_threshold_ms: int = 2000,
        min_confidence: float = 0.8,
        enable_prediction: bool = True,
        history_size: int = 10
    ):
        super().__init__(threshold=min_confidence)
        self.silence_threshold_ms = silence_threshold_ms
        self.silence_start: Optional[datetime] = None
        self.enable_prediction = enable_prediction
        
        # VAD state tracking
        self.vad_mode: Optional[str] = None  # 'client' or 'server'
        self.vad_history = deque(maxlen=history_size)
        self.last_vad_change: Optional[datetime] = None
        
        # Adaptive thresholds
        self.adaptive_threshold_ms = silence_threshold_ms
        self.silence_durations: List[float] = []
        self.max_history = 20
        
        # Prediction state
        self.speech_segments: List[Dict[str, Any]] = []
        self.predicted_silence_end: Optional[datetime] = None
    
    async def detect(self, state) -> DetectionResult:
        """Enhanced silence detection with VAD awareness"""
        
        # Update VAD mode if available
        if hasattr(state, 'vad_mode'):
            self.vad_mode = state.vad_mode
        
        # Get current VAD state
        vad_active = False
        vad_confidence = 0.0
        
        if hasattr(state, 'audio'):
            vad_active = getattr(state.audio, 'vad_active', False)
            vad_confidence = getattr(state.audio, 'vad_confidence', 0.0)
        
        # Track VAD changes
        now = datetime.now()
        self._track_vad_change(vad_active, vad_confidence, now)
        
        # If someone is speaking, reset silence
        if vad_active:
            if self.silence_start:
                # Record silence duration for adaptation
                duration = (now - self.silence_start).total_seconds() * 1000
                self._record_silence_duration(duration)
            
            self.silence_start = None
            self.predicted_silence_end = None
            
            # Track speech segment
            if not self.speech_segments or self.speech_segments[-1].get('end'):
                self.speech_segments.append({
                    'start': now,
                    'end': None,
                    'vad_confidence': vad_confidence
                })
            
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=now,
                metadata={
                    "reason": "voice_active",
                    "vad_confidence": vad_confidence,
                    "vad_mode": self.vad_mode
                }
            )
        
        # End current speech segment
        if self.speech_segments and not self.speech_segments[-1].get('end'):
            self.speech_segments[-1]['end'] = now
            self.speech_segments[-1]['duration_ms'] = (
                (now - self.speech_segments[-1]['start']).total_seconds() * 1000
            )
        
        # Start tracking silence if needed
        if self.silence_start is None:
            self.silence_start = now
            
            # Predict silence duration if enabled
            if self.enable_prediction:
                self.predicted_silence_end = self._predict_silence_end(now)
            
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=now,
                metadata={
                    "reason": "silence_just_started",
                    "predicted_duration_ms": self._get_predicted_duration(),
                    "vad_mode": self.vad_mode
                }
            )
        
        # Calculate silence duration
        silence_duration = (now - self.silence_start).total_seconds() * 1000
        
        # Use adaptive threshold
        threshold = self._get_adaptive_threshold()
        
        # Calculate multi-factor confidence
        confidence = self._calculate_confidence(
            silence_duration, 
            threshold,
            vad_confidence
        )
        
        # Check if silence is sufficient
        if silence_duration >= threshold and confidence >= self.threshold:
            return DetectionResult(
                detected=True,
                confidence=confidence,
                timestamp=now,
                metadata={
                    "silence_duration_ms": silence_duration,
                    "threshold_ms": threshold,
                    "adaptive_threshold": self.adaptive_threshold_ms != self.silence_threshold_ms,
                    "vad_mode": self.vad_mode,
                    "prediction_accuracy": self._get_prediction_accuracy()
                }
            )
        
        # Not ready yet, but provide progress
        return DetectionResult(
            detected=False,
            confidence=confidence,
            timestamp=now,
            metadata={
                "silence_duration_ms": silence_duration,
                "threshold_ms": threshold,
                "progress": silence_duration / threshold,
                "time_remaining_ms": max(0, threshold - silence_duration),
                "vad_mode": self.vad_mode
            }
        )
    
    def _track_vad_change(self, vad_active: bool, confidence: float, timestamp: datetime):
        """Track VAD state changes"""
        if self.vad_history:
            last_state = self.vad_history[-1]
            if last_state['active'] != vad_active:
                self.last_vad_change = timestamp
        
        self.vad_history.append({
            'active': vad_active,
            'confidence': confidence,
            'timestamp': timestamp
        })
    
    def _record_silence_duration(self, duration_ms: float):
        """Record silence duration for adaptation"""
        if duration_ms > 100:  # Ignore very short silences
            self.silence_durations.append(duration_ms)
            if len(self.silence_durations) > self.max_history:
                self.silence_durations.pop(0)
            
            # Update adaptive threshold
            if len(self.silence_durations) >= 5:
                # Use median to avoid outliers
                median_silence = statistics.median(self.silence_durations)
                # Blend with original threshold
                self.adaptive_threshold_ms = int(
                    0.7 * median_silence + 0.3 * self.silence_threshold_ms
                )
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive threshold based on VAD mode and history"""
        base_threshold = self.adaptive_threshold_ms
        
        # Adjust for VAD mode
        if self.vad_mode == "server":
            # Server VAD needs faster detection
            base_threshold *= 0.7
        elif self.vad_mode == "client":
            # Client VAD can be more relaxed
            base_threshold *= 1.1
        
        # Adjust based on recent patterns
        if len(self.vad_history) >= 3:
            # Check for rapid VAD changes (unstable detection)
            recent_changes = sum(
                1 for i in range(1, min(5, len(self.vad_history)))
                if self.vad_history[-i]['active'] != self.vad_history[-i-1]['active']
            )
            if recent_changes >= 3:
                # Unstable VAD, increase threshold
                base_threshold *= 1.3
        
        return base_threshold
    
    def _calculate_confidence(
        self, 
        silence_duration: float, 
        threshold: float,
        vad_confidence: float
    ) -> float:
        """Calculate multi-factor confidence score"""
        # Base confidence from duration
        duration_confidence = min(1.0, silence_duration / (threshold * 1.5))
        
        # VAD stability bonus
        vad_stability = 1.0
        if len(self.vad_history) >= 3:
            # Check if VAD has been consistently silent
            consistent_silent = all(
                not h['active'] for h in list(self.vad_history)[-3:]
            )
            vad_stability = 1.2 if consistent_silent else 0.9
        
        # Prediction bonus (if enabled and accurate)
        prediction_bonus = 1.0
        if self.enable_prediction and self.predicted_silence_end:
            accuracy = self._get_prediction_accuracy()
            if accuracy > 0.8:
                prediction_bonus = 1.1
        
        # Combine factors
        confidence = duration_confidence * vad_stability * prediction_bonus
        
        # Apply VAD confidence penalty if available
        if vad_confidence > 0:
            confidence *= (1 - vad_confidence * 0.5)
        
        return min(1.0, confidence)
    
    def _predict_silence_end(self, silence_start: datetime) -> Optional[datetime]:
        """Predict when silence will end based on patterns"""
        if len(self.speech_segments) < 3:
            return None
        
        # Analyze recent speech patterns
        recent_segments = self.speech_segments[-5:]
        
        # Calculate average pause duration between segments
        pauses = []
        for i in range(1, len(recent_segments)):
            if recent_segments[i-1].get('end') and recent_segments[i].get('start'):
                pause = (recent_segments[i]['start'] - recent_segments[i-1]['end']).total_seconds() * 1000
                pauses.append(pause)
        
        if pauses:
            avg_pause = statistics.mean(pauses)
            # Predict silence will last about as long as average
            return silence_start + timedelta(milliseconds=avg_pause)
        
        return None
    
    def _get_predicted_duration(self) -> Optional[float]:
        """Get predicted silence duration in ms"""
        if self.predicted_silence_end and self.silence_start:
            return (self.predicted_silence_end - self.silence_start).total_seconds() * 1000
        return None
    
    def _get_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy based on history"""
        # This would track actual vs predicted durations
        # For now, return a placeholder
        return 0.85 if self.enable_prediction else 0.0
    
    def update_vad_mode(self, vad_mode: str, auto_response: bool):
        """Update detector for VAD mode changes"""
        self.vad_mode = vad_mode
        
        # Adjust thresholds based on mode
        if vad_mode == "server" and auto_response:
            # Need very fast detection
            self.silence_threshold_ms = min(1000, self.silence_threshold_ms)
        elif vad_mode == "client" and not auto_response:
            # Can be more relaxed
            self.silence_threshold_ms = max(2000, self.silence_threshold_ms)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "vad_mode": self.vad_mode,
            "current_threshold_ms": self._get_adaptive_threshold(),
            "base_threshold_ms": self.silence_threshold_ms,
            "adaptive_threshold_ms": self.adaptive_threshold_ms,
            "average_silence_ms": statistics.mean(self.silence_durations) if self.silence_durations else 0,
            "silence_count": len(self.silence_durations),
            "speech_segments": len(self.speech_segments),
            "vad_changes": len(self.vad_history),
            "prediction_enabled": self.enable_prediction
        }